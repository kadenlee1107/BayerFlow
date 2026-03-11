#include "../include/dng_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>

/* ---- TIFF constants ---- */
#define TIFF_BYTE      1
#define TIFF_ASCII     2
#define TIFF_SHORT     3
#define TIFF_LONG      4
#define TIFF_RATIONAL  5
#define TIFF_SBYTE     6
#define TIFF_UNDEFINED 7
#define TIFF_SSHORT    8
#define TIFF_SLONG     9
#define TIFF_SRATIONAL 10
#define TIFF_FLOAT     11
#define TIFF_DOUBLE    12

/* Tag IDs */
#define TAG_NEW_SUBFILE_TYPE  254
#define TAG_IMAGE_WIDTH       256
#define TAG_IMAGE_LENGTH      257
#define TAG_BITS_PER_SAMPLE   258
#define TAG_COMPRESSION       259
#define TAG_STRIP_OFFSETS     273
#define TAG_ROWS_PER_STRIP    278
#define TAG_STRIP_BYTE_COUNTS 279
#define TAG_SUB_IFDS          330
#define TAG_TILE_WIDTH        322
#define TAG_TILE_LENGTH       323
#define TAG_TILE_OFFSETS      324
#define TAG_TILE_BYTE_COUNTS  325
#define TAG_CFA_REPEAT_DIM    33421
#define TAG_CFA_PATTERN2      33422
#define TAG_DNG_VERSION       50706
#define TAG_BLACK_LEVEL       50714
#define TAG_WHITE_LEVEL       50717
#define TAG_AS_SHOT_NEUTRAL   50728

#define MAX_TILES 64

/* ---- Byte-order-aware readers ---- */
typedef struct {
    FILE    *fp;
    int      big_endian;  /* 1=MM, 0=II */
    uint8_t *buf;         /* optional: read from memory buffer */
    size_t   buf_len;
} TiffCtx;

static uint16_t tiff_read16(TiffCtx *ctx, long offset) {
    uint8_t b[2];
    if (ctx->buf) {
        if (offset + 2 > (long)ctx->buf_len) return 0;
        memcpy(b, ctx->buf + offset, 2);
    } else {
        fseek(ctx->fp, offset, SEEK_SET);
        if (fread(b, 1, 2, ctx->fp) != 2) return 0;
    }
    if (ctx->big_endian)
        return ((uint16_t)b[0] << 8) | b[1];
    else
        return ((uint16_t)b[1] << 8) | b[0];
}

static uint32_t tiff_read32(TiffCtx *ctx, long offset) {
    uint8_t b[4];
    if (ctx->buf) {
        if (offset + 4 > (long)ctx->buf_len) return 0;
        memcpy(b, ctx->buf + offset, 4);
    } else {
        fseek(ctx->fp, offset, SEEK_SET);
        if (fread(b, 1, 4, ctx->fp) != 4) return 0;
    }
    if (ctx->big_endian)
        return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
               ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
    else
        return ((uint32_t)b[3] << 24) | ((uint32_t)b[2] << 16) |
               ((uint32_t)b[1] << 8)  |  (uint32_t)b[0];
}

/* Read a tag value (handles inline SHORT/LONG vs offset-to-data) */
static uint32_t tiff_tag_value(TiffCtx *ctx, long entry_offset) {
    uint16_t type  = tiff_read16(ctx, entry_offset + 2);
    uint32_t count = tiff_read32(ctx, entry_offset + 4);
    long val_off   = entry_offset + 8;

    if (type == TIFF_SHORT && count == 1)
        return tiff_read16(ctx, val_off);
    if (type == TIFF_LONG && count == 1)
        return tiff_read32(ctx, val_off);
    if (type == TIFF_SHORT && count <= 2)
        return tiff_read16(ctx, val_off); /* first value */
    /* For larger values, return the offset to the data */
    return tiff_read32(ctx, val_off);
}

/* ---- DNG metadata parsed from IFD ---- */
typedef struct {
    uint32_t width, height;
    uint16_t bits_per_sample;
    uint16_t compression;
    uint32_t strip_offset;
    uint32_t strip_byte_count;
    uint32_t rows_per_strip;
    /* Tile support */
    uint32_t tile_width, tile_height;
    uint32_t tile_offsets[MAX_TILES];
    uint32_t tile_byte_counts[MAX_TILES];
    int      num_tiles;
    int      cfa_pattern;       /* DNG_CFA_* */
    uint16_t black_level;
    uint16_t white_level;
    int      has_white_level;
    uint32_t sub_ifd_offset;    /* offset to SubIFD (tag 330) */
    int      is_full_res;       /* NewSubFileType == 0 */
    int      byte_order;        /* 0=LE, 1=BE */
    /* AsShotNeutral: R, G, B as rationals */
    float    as_shot_neutral[3];
    int      has_as_shot_neutral;
} DngIFDInfo;

/* Parse CFA pattern from 4 bytes: e.g. {0,1,1,2} = RGGB */
static int parse_cfa_pattern(TiffCtx *ctx, long data_offset) {
    uint8_t p[4];
    if (ctx->buf) {
        if (data_offset + 4 > (long)ctx->buf_len) return DNG_CFA_RGGB;
        memcpy(p, ctx->buf + data_offset, 4);
    } else {
        fseek(ctx->fp, data_offset, SEEK_SET);
        if (fread(p, 1, 4, ctx->fp) != 4) return DNG_CFA_RGGB;
    }
    /* DNG CFA values: 0=Red, 1=Green, 2=Blue */
    if (p[0] == 0 && p[1] == 1 && p[2] == 1 && p[3] == 2) return DNG_CFA_RGGB;
    if (p[0] == 1 && p[1] == 0 && p[2] == 2 && p[3] == 1) return DNG_CFA_GRBG;
    if (p[0] == 1 && p[1] == 2 && p[2] == 0 && p[3] == 1) return DNG_CFA_GBRG;
    if (p[0] == 2 && p[1] == 1 && p[2] == 1 && p[3] == 0) return DNG_CFA_BGGR;
    return DNG_CFA_RGGB; /* fallback */
}

/* Parse AsShotNeutral (tag 50728): array of 3 RATIONALs */
static void parse_as_shot_neutral(TiffCtx *ctx, long entry_offset, DngIFDInfo *info) {
    uint16_t type  = tiff_read16(ctx, entry_offset + 2);
    uint32_t count = tiff_read32(ctx, entry_offset + 4);
    if (count < 3) return;

    long data_off = tiff_read32(ctx, entry_offset + 8);

    if (type == TIFF_RATIONAL) {
        for (int i = 0; i < 3; i++) {
            uint32_t num = tiff_read32(ctx, data_off + i * 8);
            uint32_t den = tiff_read32(ctx, data_off + i * 8 + 4);
            info->as_shot_neutral[i] = (den > 0) ? (float)num / (float)den : 1.0f;
        }
        info->has_as_shot_neutral = 1;
    } else if (type == TIFF_SRATIONAL) {
        for (int i = 0; i < 3; i++) {
            int32_t num = (int32_t)tiff_read32(ctx, data_off + i * 8);
            int32_t den = (int32_t)tiff_read32(ctx, data_off + i * 8 + 4);
            info->as_shot_neutral[i] = (den != 0) ? (float)num / (float)den : 1.0f;
        }
        info->has_as_shot_neutral = 1;
    }
}

/* Read an array of LONG or SHORT values from a tag */
static void read_tag_array(TiffCtx *ctx, long eo, uint32_t *out, int max_count, int *actual_count) {
    uint16_t type  = tiff_read16(ctx, eo + 2);
    uint32_t count = tiff_read32(ctx, eo + 4);
    if ((int)count > max_count) count = max_count;
    *actual_count = (int)count;

    /* If data fits in 4-byte value field (1 LONG or <=2 SHORTs), read inline */
    size_t elem_size = (type == TIFF_SHORT) ? 2 : 4;
    size_t total_bytes = count * elem_size;
    long data_off;
    if (total_bytes <= 4)
        data_off = eo + 8; /* inline */
    else
        data_off = (long)tiff_read32(ctx, eo + 8); /* offset to data */

    for (uint32_t i = 0; i < count; i++) {
        if (type == TIFF_SHORT)
            out[i] = tiff_read16(ctx, data_off + (long)i * 2);
        else
            out[i] = tiff_read32(ctx, data_off + (long)i * 4);
    }
}

/* Parse a single IFD and populate info struct */
static int parse_ifd(TiffCtx *ctx, long ifd_offset, DngIFDInfo *info) {
    uint16_t entry_count = tiff_read16(ctx, ifd_offset);
    if (entry_count == 0 || entry_count > 500) return -1;

    info->is_full_res = 1; /* assume full-res unless NewSubFileType says otherwise */
    info->rows_per_strip = 0xFFFFFFFF;
    info->byte_order = ctx->big_endian;

    for (int i = 0; i < entry_count; i++) {
        long eo = ifd_offset + 2 + (long)i * 12;
        uint16_t tag = tiff_read16(ctx, eo);

        switch (tag) {
        case TAG_NEW_SUBFILE_TYPE:
            info->is_full_res = (tiff_tag_value(ctx, eo) == 0) ? 1 : 0;
            break;
        case TAG_IMAGE_WIDTH:
            info->width = tiff_tag_value(ctx, eo);
            break;
        case TAG_IMAGE_LENGTH:
            info->height = tiff_tag_value(ctx, eo);
            break;
        case TAG_BITS_PER_SAMPLE:
            info->bits_per_sample = (uint16_t)tiff_tag_value(ctx, eo);
            break;
        case TAG_COMPRESSION:
            info->compression = (uint16_t)tiff_tag_value(ctx, eo);
            break;
        case TAG_STRIP_OFFSETS: {
            uint16_t type = tiff_read16(ctx, eo + 2);
            uint32_t count = tiff_read32(ctx, eo + 4);
            if (count == 1) {
                info->strip_offset = tiff_tag_value(ctx, eo);
            } else {
                /* Multiple strips: read first offset */
                uint32_t data_off = tiff_read32(ctx, eo + 8);
                if (type == TIFF_SHORT)
                    info->strip_offset = tiff_read16(ctx, data_off);
                else
                    info->strip_offset = tiff_read32(ctx, data_off);
            }
            break;
        }
        case TAG_ROWS_PER_STRIP:
            info->rows_per_strip = tiff_tag_value(ctx, eo);
            break;
        case TAG_STRIP_BYTE_COUNTS: {
            uint16_t type = tiff_read16(ctx, eo + 2);
            uint32_t count = tiff_read32(ctx, eo + 4);
            if (count == 1) {
                info->strip_byte_count = tiff_tag_value(ctx, eo);
            } else {
                uint32_t data_off = tiff_read32(ctx, eo + 8);
                uint32_t total = 0;
                for (uint32_t s = 0; s < count; s++) {
                    if (type == TIFF_SHORT)
                        total += tiff_read16(ctx, data_off + s * 2);
                    else
                        total += tiff_read32(ctx, data_off + s * 4);
                }
                info->strip_byte_count = total;
            }
            break;
        }
        case TAG_TILE_WIDTH:
            info->tile_width = tiff_tag_value(ctx, eo);
            break;
        case TAG_TILE_LENGTH:
            info->tile_height = tiff_tag_value(ctx, eo);
            break;
        case TAG_TILE_OFFSETS:
            read_tag_array(ctx, eo, info->tile_offsets, MAX_TILES, &info->num_tiles);
            break;
        case TAG_TILE_BYTE_COUNTS: {
            int n = 0;
            read_tag_array(ctx, eo, info->tile_byte_counts, MAX_TILES, &n);
            /* num_tiles already set by TAG_TILE_OFFSETS (or set it here if that came first) */
            if (info->num_tiles == 0) info->num_tiles = n;
            break;
        }
        case TAG_SUB_IFDS:
            info->sub_ifd_offset = tiff_tag_value(ctx, eo);
            break;
        case TAG_CFA_PATTERN2: {
            uint32_t count = tiff_read32(ctx, eo + 4);
            long data_off;
            if (count <= 4)
                data_off = eo + 8; /* inline */
            else
                data_off = tiff_read32(ctx, eo + 8);
            info->cfa_pattern = parse_cfa_pattern(ctx, data_off);
            break;
        }
        case TAG_BLACK_LEVEL: {
            uint16_t bl_type = tiff_read16(ctx, eo + 2);
            uint32_t bl_count = tiff_read32(ctx, eo + 4);
            if (bl_type == TIFF_RATIONAL) {
                uint32_t data_off = tiff_read32(ctx, eo + 8);
                uint32_t num = tiff_read32(ctx, data_off);
                uint32_t den = tiff_read32(ctx, data_off + 4);
                info->black_level = (den > 0) ? (uint16_t)(num / den) : 0;
            } else if (bl_count > 1) {
                /* Multiple values (one per CFA channel): read from offset, take first */
                size_t elem_size = (bl_type == TIFF_SHORT) ? 2 : 4;
                size_t total_bytes = bl_count * elem_size;
                long data_off;
                if (total_bytes <= 4)
                    data_off = eo + 8; /* inline */
                else
                    data_off = (long)tiff_read32(ctx, eo + 8);
                if (bl_type == TIFF_SHORT)
                    info->black_level = tiff_read16(ctx, data_off);
                else
                    info->black_level = (uint16_t)tiff_read32(ctx, data_off);
            } else {
                info->black_level = (uint16_t)tiff_tag_value(ctx, eo);
            }
            break;
        }
        case TAG_WHITE_LEVEL:
            info->white_level = (uint16_t)tiff_tag_value(ctx, eo);
            info->has_white_level = 1;
            break;
        case TAG_AS_SHOT_NEUTRAL:
            parse_as_shot_neutral(ctx, eo, info);
            break;
        }
    }
    return 0;
}

/* Check if IFD has image data (strips or tiles) */
static int ifd_has_image_data(const DngIFDInfo *info) {
    return (info->width > 0) && (info->strip_offset > 0 || info->num_tiles > 0);
}

/* Parse a DNG file and find the full-resolution raw IFD.
 * Checks IFD0 and SubIFDs for NewSubFileType==0. */
static int parse_dng_file(const char *path, DngIFDInfo *info) {
    memset(info, 0, sizeof(*info));
    info->cfa_pattern = DNG_CFA_RGGB;

    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    TiffCtx ctx = { .fp = fp, .big_endian = 0, .buf = NULL, .buf_len = 0 };

    /* Read TIFF header */
    uint8_t hdr[8];
    if (fread(hdr, 1, 8, fp) != 8) { fclose(fp); return -1; }

    if (hdr[0] == 'I' && hdr[1] == 'I')
        ctx.big_endian = 0;
    else if (hdr[0] == 'M' && hdr[1] == 'M')
        ctx.big_endian = 1;
    else { fclose(fp); return -1; }

    uint16_t magic = tiff_read16(&ctx, 2);
    if (magic != 42) { fclose(fp); return -1; }

    uint32_t ifd0_offset = tiff_read32(&ctx, 4);

    /* Parse IFD0 */
    DngIFDInfo ifd0_info;
    memset(&ifd0_info, 0, sizeof(ifd0_info));
    ifd0_info.cfa_pattern = DNG_CFA_RGGB;

    if (parse_ifd(&ctx, ifd0_offset, &ifd0_info) != 0) { fclose(fp); return -1; }

    /* Copy AsShotNeutral from IFD0 (it's usually in the main IFD) */
    if (ifd0_info.has_as_shot_neutral) {
        info->has_as_shot_neutral = 1;
        memcpy(info->as_shot_neutral, ifd0_info.as_shot_neutral, sizeof(info->as_shot_neutral));
    }

    /* If IFD0 is full-res and has image data, use it */
    if (ifd0_info.is_full_res && ifd_has_image_data(&ifd0_info)) {
        float neutral[3];
        int had_neutral = info->has_as_shot_neutral;
        if (had_neutral) memcpy(neutral, info->as_shot_neutral, sizeof(neutral));

        *info = ifd0_info;
        if (had_neutral && !info->has_as_shot_neutral) {
            info->has_as_shot_neutral = 1;
            memcpy(info->as_shot_neutral, neutral, sizeof(neutral));
        }
        /* Default WhiteLevel if not present */
        if (!info->has_white_level)
            info->white_level = (uint16_t)((1 << info->bits_per_sample) - 1);
        fclose(fp);
        return 0;
    }

    /* Otherwise, follow SubIFD to find full-resolution raw image */
    if (ifd0_info.sub_ifd_offset > 0) {
        DngIFDInfo sub_info;
        memset(&sub_info, 0, sizeof(sub_info));
        sub_info.cfa_pattern = DNG_CFA_RGGB;

        if (parse_ifd(&ctx, ifd0_info.sub_ifd_offset, &sub_info) == 0 &&
            ifd_has_image_data(&sub_info)) {
            float neutral[3];
            int had_neutral = info->has_as_shot_neutral;
            if (had_neutral) memcpy(neutral, info->as_shot_neutral, sizeof(neutral));

            *info = sub_info;
            if (had_neutral && !info->has_as_shot_neutral) {
                info->has_as_shot_neutral = 1;
                memcpy(info->as_shot_neutral, neutral, sizeof(neutral));
            }
            if (!info->has_white_level)
                info->white_level = (uint16_t)((1 << info->bits_per_sample) - 1);
            fclose(fp);
            return 0;
        }
    }

    /* Fallback: use IFD0 data even if not marked full-res */
    if (ifd_has_image_data(&ifd0_info)) {
        float neutral[3];
        int had_neutral = info->has_as_shot_neutral;
        if (had_neutral) memcpy(neutral, info->as_shot_neutral, sizeof(neutral));

        *info = ifd0_info;
        if (had_neutral && !info->has_as_shot_neutral) {
            info->has_as_shot_neutral = 1;
            memcpy(info->as_shot_neutral, neutral, sizeof(neutral));
        }
        if (!info->has_white_level)
            info->white_level = (uint16_t)((1 << info->bits_per_sample) - 1);
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return -1;
}

/* ===================================================================
 *  LJPEG-92 (Lossless JPEG) Decoder
 *  Handles SOF3/DHT/SOS markers, Huffman decoding, predictors 1-7.
 * =================================================================== */

#define LJ_SOI  0xFFD8
#define LJ_SOF3 0xFFC3
#define LJ_DHT  0xFFC4
#define LJ_SOS  0xFFDA
#define LJ_EOI  0xFFD9
#define LJ_MAX_COMP  4
#define LJ_MAX_HUFF  4

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
    uint32_t bits;
    int nbits;
} LJBitReader;

typedef struct {
    int count[17];       /* codes of each length 1-16 */
    uint16_t code[17];   /* first code of each length */
    int offset[17];      /* index into vals[] */
    uint8_t vals[256];
    int num_vals;
    int valid;
} LJHuff;

static void lj_br_init(LJBitReader *br, const uint8_t *data, size_t len) {
    br->data = data; br->len = len; br->pos = 0;
    br->bits = 0; br->nbits = 0;
}

static inline void lj_fill(LJBitReader *br) {
    while (br->nbits <= 24 && br->pos < br->len) {
        uint8_t b = br->data[br->pos++];
        if (b == 0xFF && br->pos < br->len) {
            if (br->data[br->pos] == 0x00) {
                br->pos++; /* byte stuffing: skip the 0x00 */
            } else {
                /* Marker byte — put 0xFF back and stop filling */
                br->pos--;
                return;
            }
        }
        br->bits = (br->bits << 8) | b;
        br->nbits += 8;
    }
}

static inline int lj_get(LJBitReader *br, int n) {
    if (n == 0) return 0;
    while (br->nbits < n) lj_fill(br);
    br->nbits -= n;
    return (int)((br->bits >> br->nbits) & ((1u << n) - 1));
}

static int lj_huff_build(LJHuff *h, const uint8_t *bits16, const uint8_t *vals, int total) {
    memset(h, 0, sizeof(*h));
    for (int i = 1; i <= 16; i++)
        h->count[i] = bits16[i - 1];
    h->num_vals = total;
    if (total > 256) return -1;
    memcpy(h->vals, vals, total);

    uint16_t code = 0;
    int idx = 0;
    for (int len = 1; len <= 16; len++) {
        h->code[len] = code;
        h->offset[len] = idx;
        idx += h->count[len];
        code = (uint16_t)((code + h->count[len]) << 1);
    }
    h->valid = 1;
    return 0;
}

static int lj_huff_decode(LJBitReader *br, const LJHuff *h) {
    int code = 0;
    for (int len = 1; len <= 16; len++) {
        code = (code << 1) | lj_get(br, 1);
        int idx = code - (int)h->code[len];
        if (idx >= 0 && idx < h->count[len])
            return h->vals[h->offset[len] + idx];
    }
    return -1;
}

static inline int lj_decode_diff(LJBitReader *br, int cat) {
    if (cat == 0) return 0;
    if (cat == 16) return 32768;
    int val = lj_get(br, cat);
    /* Sign extension: if MSB is 0, value is negative */
    if (val < (1 << (cat - 1)))
        val -= (1 << cat) - 1;
    return val;
}

static uint16_t lj_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

/* Decode a single LJPEG-92 tile/strip into output buffer.
 * output must hold at least out_width * out_height uint16 values.
 * Returns 0 on success, -1 on error. */
static int ljpeg_decode_tile(const uint8_t *jpeg, size_t jpeg_len,
                             uint16_t *output, int out_width, int out_height) {
    /* Parse JPEG markers */
    int precision = 0, jwidth = 0, jheight = 0, ncomp = 0;
    int comp_huff[LJ_MAX_COMP] = {0};
    LJHuff huff[LJ_MAX_HUFF];
    int predictor = 1, point_transform = 0;
    memset(huff, 0, sizeof(huff));

    size_t pos = 0;

    /* Expect SOI */
    if (jpeg_len < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) return -1;
    pos = 2;

    while (pos + 1 < jpeg_len) {
        /* Find next marker */
        if (jpeg[pos] != 0xFF) { pos++; continue; }
        while (pos + 1 < jpeg_len && jpeg[pos + 1] == 0xFF) pos++;
        if (pos + 1 >= jpeg_len) break;

        int marker = jpeg[pos + 1];
        pos += 2;

        if (marker == 0xD9) break; /* EOI */
        if (marker == 0xDA) {      /* SOS — parse header then decode */
            if (pos + 2 > jpeg_len) return -1;
            uint16_t sos_len = lj_be16(jpeg + pos);
            if (pos + sos_len > jpeg_len) return -1;
            const uint8_t *sos = jpeg + pos + 2;

            int ns = sos[0];
            if (ns > LJ_MAX_COMP || ns < 1) return -1;

            for (int i = 0; i < ns; i++) {
                /* sos[1 + i*2] = component selector */
                /* sos[1 + i*2 + 1] = Td:Ta (DC:AC table selectors) */
                int td = (sos[1 + i * 2 + 1] >> 4) & 0x0F;
                if (td >= LJ_MAX_HUFF) td = 0;
                comp_huff[i] = td;
            }
            predictor      = sos[1 + ns * 2];
            /* sos[1 + ns * 2 + 1] = Se (should be 0) */
            point_transform = sos[1 + ns * 2 + 2] & 0x0F;

            pos += sos_len; /* now at entropy data */
            break; /* exit marker loop, proceed to decode */
        }

        /* Non-SOS segment: read length and parse */
        if (pos + 2 > jpeg_len) return -1;
        uint16_t seg_len = lj_be16(jpeg + pos);
        if (pos + seg_len > jpeg_len) return -1;
        const uint8_t *seg = jpeg + pos + 2;
        size_t seg_data_len = seg_len - 2;

        if (marker == 0xC3) { /* SOF3 — Lossless Huffman */
            if (seg_data_len < 6) return -1;
            precision = seg[0];
            jheight   = lj_be16(seg + 1);
            jwidth    = lj_be16(seg + 3);
            ncomp     = seg[5];
            if (ncomp > LJ_MAX_COMP) return -1;
            /* Component definitions follow but we only need count */
        } else if (marker == 0xC4) { /* DHT */
            size_t off = 0;
            while (off < seg_data_len) {
                if (off >= seg_data_len) break;
                int tc_th = seg[off++];
                int th = tc_th & 0x0F;
                if (th >= LJ_MAX_HUFF) { return -1; }

                if (off + 16 > seg_data_len) return -1;
                uint8_t bits16[16];
                memcpy(bits16, seg + off, 16);
                off += 16;

                int total = 0;
                for (int i = 0; i < 16; i++) total += bits16[i];
                if (off + total > seg_data_len) return -1;

                lj_huff_build(&huff[th], bits16, seg + off, total);
                off += total;
            }
        }
        /* Skip other markers (DRI, etc.) */
        pos += seg_len;
    }

    if (jwidth == 0 || jheight == 0 || ncomp == 0) return -1;
    if (precision == 0) precision = 16;

    /* Validate Huffman tables */
    for (int c = 0; c < ncomp; c++) {
        if (!huff[comp_huff[c]].valid) {
            /* Fallback to table 0 */
            comp_huff[c] = 0;
            if (!huff[0].valid) return -1;
        }
    }

    /* Allocate component buffers */
    uint16_t *comp_buf[LJ_MAX_COMP] = {NULL};
    for (int c = 0; c < ncomp; c++) {
        comp_buf[c] = (uint16_t *)calloc((size_t)jwidth * jheight, sizeof(uint16_t));
        if (!comp_buf[c]) {
            for (int j = 0; j < c; j++) free(comp_buf[j]);
            return -1;
        }
    }

    /* Decode entropy-coded segment */
    LJBitReader br;
    lj_br_init(&br, jpeg + pos, jpeg_len - pos);

    int mask = (1 << precision) - 1;
    int pred_reset = 1 << (precision - point_transform - 1);

    for (int y = 0; y < jheight; y++) {
        for (int x = 0; x < jwidth; x++) {
            for (int c = 0; c < ncomp; c++) {
                int cat = lj_huff_decode(&br, &huff[comp_huff[c]]);
                if (cat < 0) cat = 0;
                int diff = lj_decode_diff(&br, cat);

                /* Prediction */
                int pred;
                if (x == 0 && y == 0) {
                    pred = pred_reset;
                } else if (y == 0) {
                    pred = comp_buf[c][x - 1]; /* left */
                } else if (x == 0) {
                    pred = comp_buf[c][(size_t)(y - 1) * jwidth]; /* above */
                } else {
                    int a  = comp_buf[c][(size_t)y * jwidth + x - 1];         /* left */
                    int b  = comp_buf[c][(size_t)(y - 1) * jwidth + x];       /* above */
                    int cc_val = comp_buf[c][(size_t)(y - 1) * jwidth + x - 1]; /* upper-left */
                    switch (predictor) {
                    case 1: pred = a; break;
                    case 2: pred = b; break;
                    case 3: pred = cc_val; break;
                    case 4: pred = a + b - cc_val; break;
                    case 5: pred = a + ((b - cc_val) >> 1); break;
                    case 6: pred = b + ((a - cc_val) >> 1); break;
                    case 7: pred = (a + b) >> 1; break;
                    default: pred = a; break;
                    }
                }

                int val = (pred + diff) & mask;
                if (point_transform > 0) val <<= point_transform;
                comp_buf[c][(size_t)y * jwidth + x] = (uint16_t)val;
            }
        }
    }

    /* Map decoded LJPEG data → tile output.
     *
     * The LJPEG dimensions may differ from the TIFF tile dimensions due to
     * "linearization" (common in Blackmagic CinemaDNG):
     *   LJPEG: 5760×1620 (1 comp)  →  Tile: 2880×3240
     *   Each LJPEG row packs multiple tile rows end-to-end.
     *
     * Also handle multi-component de-interleaving:
     *   LJPEG: 1440×3240 (2 comp)  →  Tile: 2880×3240
     *   pixel[x*2+c] = comp[c][x]
     */
    int deinterleaved_w = jwidth * ncomp; /* total pixels per LJPEG row after de-interleave */
    int row_ratio = (out_width > 0) ? deinterleaved_w / out_width : 1;
    if (row_ratio < 1) row_ratio = 1;

    for (int y = 0; y < jheight; y++) {
        for (int x = 0; x < jwidth; x++) {
            for (int c = 0; c < ncomp; c++) {
                int linear_x = x * ncomp + c; /* position within the deinterleaved row */
                int segment = linear_x / out_width;   /* which tile row within this LJPEG row */
                int tile_x  = linear_x % out_width;   /* column within that tile row */
                int tile_y  = y * row_ratio + segment;
                if (tile_y < out_height && tile_x < out_width)
                    output[(size_t)tile_y * out_width + tile_x] =
                        comp_buf[c][(size_t)y * jwidth + x];
            }
        }
    }

    for (int c = 0; c < ncomp; c++) free(comp_buf[c]);
    return 0;
}

/* ---- Directory scanning and sorting ---- */

static int str_ends_with_ci(const char *str, const char *suffix) {
    size_t slen = strlen(str);
    size_t xlen = strlen(suffix);
    if (slen < xlen) return 0;
    for (size_t i = 0; i < xlen; i++) {
        char a = str[slen - xlen + i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b) return 0;
    }
    return 1;
}

/* Natural sort comparison: treats embedded digit runs as numbers */
static int natural_cmp(const void *a, const void *b) {
    const char *sa = *(const char **)a;
    const char *sb = *(const char **)b;
    while (*sa && *sb) {
        if (*sa >= '0' && *sa <= '9' && *sb >= '0' && *sb <= '9') {
            while (*sa == '0') sa++;
            while (*sb == '0') sb++;
            const char *na = sa, *nb = sb;
            while (*sa >= '0' && *sa <= '9') sa++;
            while (*sb >= '0' && *sb <= '9') sb++;
            int la = (int)(sa - na), lb = (int)(sb - nb);
            if (la != lb) return la - lb;
            int d = strncmp(na, nb, la);
            if (d != 0) return d;
        } else {
            if (*sa != *sb) return (unsigned char)*sa - (unsigned char)*sb;
            sa++;
            sb++;
        }
    }
    return (unsigned char)*sa - (unsigned char)*sb;
}

static int scan_dng_directory(const char *dir_path, char ***out_list, int *out_count) {
    DIR *d = opendir(dir_path);
    if (!d) return -1;

    int capacity = 256;
    int count = 0;
    char **list = (char **)malloc(capacity * sizeof(char *));
    if (!list) { closedir(d); return -1; }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!str_ends_with_ci(ent->d_name, ".dng")) continue;

        if (count >= capacity) {
            capacity *= 2;
            char **tmp = (char **)realloc(list, capacity * sizeof(char *));
            if (!tmp) break;
            list = tmp;
        }

        size_t plen = strlen(dir_path) + strlen(ent->d_name) + 2;
        char *full = (char *)malloc(plen);
        if (!full) break;
        snprintf(full, plen, "%s/%s", dir_path, ent->d_name);
        list[count++] = full;
    }
    closedir(d);

    if (count == 0) {
        free(list);
        return -1;
    }

    qsort(list, count, sizeof(char *), natural_cmp);

    *out_list = list;
    *out_count = count;
    return 0;
}

/* Get directory from path (which may be a file or directory) */
static int resolve_dng_dir(const char *path, char *dir_out, size_t dir_size) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;

    if (S_ISDIR(st.st_mode)) {
        snprintf(dir_out, dir_size, "%s", path);
    } else {
        const char *last_slash = strrchr(path, '/');
        if (!last_slash) return -1;
        size_t len = last_slash - path;
        if (len >= dir_size) return -1;
        memcpy(dir_out, path, len);
        dir_out[len] = '\0';
    }
    return 0;
}

/* ---- Read raw Bayer data from a single DNG file ---- */

/* Read uncompressed data from a single offset into bayer_out.
 * Handles byte-swapping for big-endian files and bit-depth unpacking. */
static int read_uncompressed_data(FILE *fp, int byte_order, long data_offset,
                                  uint16_t *out, size_t pixels, int bits_per_sample) {
    if (bits_per_sample == 16) {
        fseek(fp, data_offset, SEEK_SET);
        size_t bytes = pixels * 2;
        uint8_t *raw = (uint8_t *)malloc(bytes);
        if (!raw) return -1;
        if (fread(raw, 1, bytes, fp) != bytes) { free(raw); return -1; }

        if (byte_order) { /* big-endian */
            for (size_t i = 0; i < pixels; i++)
                out[i] = ((uint16_t)raw[i*2] << 8) | raw[i*2+1];
        } else {
            memcpy(out, raw, bytes);
        }
        free(raw);
    } else if (bits_per_sample == 14) {
        fseek(fp, data_offset, SEEK_SET);
        size_t packed_bytes = (pixels * 14 + 7) / 8;
        uint8_t *raw = (uint8_t *)malloc(packed_bytes);
        if (!raw) return -1;
        if (fread(raw, 1, packed_bytes, fp) != packed_bytes) { free(raw); return -1; }

        size_t bit_pos = 0;
        for (size_t i = 0; i < pixels; i++) {
            size_t byte_idx = bit_pos / 8;
            int bit_offset = (int)(bit_pos % 8);
            uint32_t val = ((uint32_t)raw[byte_idx] << 16);
            if (byte_idx + 1 < packed_bytes) val |= ((uint32_t)raw[byte_idx + 1] << 8);
            if (byte_idx + 2 < packed_bytes) val |= (uint32_t)raw[byte_idx + 2];
            out[i] = (uint16_t)((val >> (10 - bit_offset)) & 0x3FFF);
            bit_pos += 14;
        }
        free(raw);
    } else if (bits_per_sample == 12) {
        fseek(fp, data_offset, SEEK_SET);
        size_t packed_bytes = (pixels * 3 + 1) / 2;
        uint8_t *raw = (uint8_t *)malloc(packed_bytes);
        if (!raw) return -1;
        if (fread(raw, 1, packed_bytes, fp) != packed_bytes) { free(raw); return -1; }

        for (size_t i = 0; i < pixels / 2; i++) {
            uint8_t b0 = raw[i * 3];
            uint8_t b1 = raw[i * 3 + 1];
            uint8_t b2 = raw[i * 3 + 2];
            out[i * 2]     = ((uint16_t)b0 << 4) | (b1 >> 4);
            out[i * 2 + 1] = ((uint16_t)(b1 & 0x0F) << 8) | b2;
        }
        if (pixels % 2) {
            size_t i = pixels / 2;
            out[pixels - 1] = ((uint16_t)raw[i * 3] << 4) | (raw[i * 3 + 1] >> 4);
        }
        free(raw);
    } else {
        /* Fallback: treat as 16-bit */
        fprintf(stderr, "dng_reader: unusual bits_per_sample=%d, trying 16-bit\n", bits_per_sample);
        fseek(fp, data_offset, SEEK_SET);
        size_t bytes = pixels * 2;
        if (fread(out, 1, bytes, fp) != bytes) return -1;
    }
    return 0;
}

/* Apply CFA pattern remapping to RGGB */
static void remap_cfa_to_rggb(uint16_t *bayer_out, int width, int height, int cfa_pattern) {
    if (cfa_pattern == DNG_CFA_RGGB) return;

    int row_shift = 0, col_shift = 0;
    switch (cfa_pattern) {
    case DNG_CFA_GRBG: col_shift = 1; break;
    case DNG_CFA_GBRG: row_shift = 1; break;
    case DNG_CFA_BGGR: row_shift = 1; col_shift = 1; break;
    }

    if (row_shift || col_shift) {
        size_t pixels = (size_t)width * height;
        uint16_t *tmp = (uint16_t *)malloc(pixels * sizeof(uint16_t));
        if (tmp) {
            memcpy(tmp, bayer_out, pixels * sizeof(uint16_t));
            memset(bayer_out, 0, pixels * sizeof(uint16_t));
            for (int y = 0; y < height - row_shift; y++) {
                for (int x = 0; x < width - col_shift; x++) {
                    bayer_out[(size_t)y * width + x] =
                        tmp[(size_t)(y + row_shift) * width + (x + col_shift)];
                }
            }
            free(tmp);
        }
    }
}

static int read_dng_frame(const char *path, uint16_t *bayer_out,
                          int width, int height, int bits_per_sample,
                          int compression, int cfa_pattern) {
    DngIFDInfo info;
    if (parse_dng_file(path, &info) != 0) return -1;

    if ((int)info.width != width || (int)info.height != height) {
        fprintf(stderr, "dng_reader: dimension mismatch in %s: expected %dx%d, got %dx%d\n",
                path, width, height, info.width, info.height);
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    size_t pixels = (size_t)width * height;

    if (info.compression == 1) {
        /* ---- Uncompressed ---- */
        if (info.num_tiles > 0) {
            /* Tile-based uncompressed */
            int tw = info.tile_width;
            int th = info.tile_height;
            int tiles_across = (width + tw - 1) / tw;

            for (int t = 0; t < info.num_tiles; t++) {
                int tile_col = t % tiles_across;
                int tile_row = t / tiles_across;
                int tx = tile_col * tw;
                int ty = tile_row * th;
                int actual_tw = (tx + tw > width)  ? (width - tx)  : tw;
                int actual_th = (ty + th > height) ? (height - ty) : th;

                /* Read tile into temp buffer */
                size_t tile_pixels = (size_t)tw * th;
                uint16_t *tile_buf = (uint16_t *)calloc(tile_pixels, sizeof(uint16_t));
                if (!tile_buf) { fclose(fp); return -1; }

                if (read_uncompressed_data(fp, info.byte_order, info.tile_offsets[t],
                                           tile_buf, tile_pixels, info.bits_per_sample) != 0) {
                    free(tile_buf);
                    fclose(fp);
                    return -1;
                }

                /* Copy tile into full image */
                for (int y = 0; y < actual_th; y++) {
                    memcpy(&bayer_out[(size_t)(ty + y) * width + tx],
                           &tile_buf[(size_t)y * tw],
                           actual_tw * sizeof(uint16_t));
                }
                free(tile_buf);
            }
        } else {
            /* Strip-based uncompressed (original path) */
            if (read_uncompressed_data(fp, info.byte_order, info.strip_offset,
                                       bayer_out, pixels, info.bits_per_sample) != 0) {
                fclose(fp);
                return -1;
            }
        }
    } else if (info.compression == 7) {
        /* ---- LJPEG-92 compressed ---- */
        if (info.num_tiles > 0) {
            /* Tile-based LJPEG */
            int tw = info.tile_width;
            int th = info.tile_height;
            int tiles_across = (width + tw - 1) / tw;

            for (int t = 0; t < info.num_tiles; t++) {
                int tile_col = t % tiles_across;
                int tile_row = t / tiles_across;
                int tx = tile_col * tw;
                int ty = tile_row * th;
                int actual_tw = (tx + tw > width)  ? (width - tx)  : tw;
                int actual_th = (ty + th > height) ? (height - ty) : th;

                /* Read compressed tile data */
                uint32_t tile_bytes = info.tile_byte_counts[t];
                uint8_t *jpeg_data = (uint8_t *)malloc(tile_bytes);
                if (!jpeg_data) { fclose(fp); return -1; }

                fseek(fp, info.tile_offsets[t], SEEK_SET);
                if (fread(jpeg_data, 1, tile_bytes, fp) != tile_bytes) {
                    free(jpeg_data);
                    fclose(fp);
                    return -1;
                }

                /* Decode LJPEG into temp buffer */
                uint16_t *tile_buf = (uint16_t *)calloc((size_t)tw * th, sizeof(uint16_t));
                if (!tile_buf) { free(jpeg_data); fclose(fp); return -1; }

                if (ljpeg_decode_tile(jpeg_data, tile_bytes, tile_buf, tw, th) != 0) {
                    fprintf(stderr, "dng_reader: LJPEG decode failed for tile %d in %s\n", t, path);
                    free(tile_buf);
                    free(jpeg_data);
                    fclose(fp);
                    return -1;
                }
                free(jpeg_data);

                /* Copy decoded tile into full image */
                for (int y = 0; y < actual_th; y++) {
                    memcpy(&bayer_out[(size_t)(ty + y) * width + tx],
                           &tile_buf[(size_t)y * tw],
                           actual_tw * sizeof(uint16_t));
                }
                free(tile_buf);
            }
        } else if (info.strip_offset > 0) {
            /* Strip-based LJPEG */
            uint32_t strip_bytes = info.strip_byte_count;
            if (strip_bytes == 0) { fclose(fp); return -1; }

            uint8_t *jpeg_data = (uint8_t *)malloc(strip_bytes);
            if (!jpeg_data) { fclose(fp); return -1; }

            fseek(fp, info.strip_offset, SEEK_SET);
            if (fread(jpeg_data, 1, strip_bytes, fp) != strip_bytes) {
                free(jpeg_data);
                fclose(fp);
                return -1;
            }

            if (ljpeg_decode_tile(jpeg_data, strip_bytes, bayer_out, width, height) != 0) {
                fprintf(stderr, "dng_reader: LJPEG decode failed for %s\n", path);
                free(jpeg_data);
                fclose(fp);
                return -1;
            }
            free(jpeg_data);
        } else {
            fprintf(stderr, "dng_reader: LJPEG file has neither tiles nor strips: %s\n", path);
            fclose(fp);
            return -1;
        }
    } else {
        fprintf(stderr, "dng_reader: unsupported compression %d in %s\n",
                info.compression, path);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    /* Scale to 16-bit: the ProRes RAW encoder expects 16-bit input values.
     * DNG data is typically 12 or 14 bit — left-shift to fill 16 bits. */
    if (bits_per_sample < 16) {
        int shift = 16 - bits_per_sample;
        for (size_t i = 0; i < pixels; i++)
            bayer_out[i] = (uint16_t)((uint32_t)bayer_out[i] << shift);
    }

    /* Remap CFA pattern to RGGB if needed */
    remap_cfa_to_rggb(bayer_out, width, height, cfa_pattern);

    return 0;
}

/* ---- Public API ---- */

int dng_reader_open(DngReader *r, const char *path) {
    memset(r, 0, sizeof(*r));

    if (resolve_dng_dir(path, r->dir_path, sizeof(r->dir_path)) != 0)
        return -1;

    if (scan_dng_directory(r->dir_path, &r->file_list, &r->file_count) != 0)
        return -1;

    r->frame_count = r->file_count;

    DngIFDInfo info;
    if (parse_dng_file(r->file_list[0], &info) != 0) {
        fprintf(stderr, "dng_reader: failed to parse first DNG: %s\n", r->file_list[0]);
        dng_reader_close(r);
        return -1;
    }

    r->width = info.width;
    r->height = info.height;
    r->bits_per_sample = info.bits_per_sample;
    r->compression = info.compression;
    r->cfa_pattern = info.cfa_pattern;
    r->black_level = info.black_level;
    r->white_level = info.white_level;
    r->frames_read = 0;

    fprintf(stderr, "dng_reader: opened %d frames, %dx%d, %d-bit, compression=%d, CFA=%d, BL=%d WL=%d\n",
            r->frame_count, r->width, r->height, r->bits_per_sample,
            r->compression, r->cfa_pattern, r->black_level, r->white_level);

    if (info.num_tiles > 0)
        fprintf(stderr, "dng_reader: tile layout %dx%d, %d tiles\n",
                info.tile_width, info.tile_height, info.num_tiles);

    return 0;
}

int dng_reader_read_frame(DngReader *r, uint16_t *bayer_out) {
    if (r->frames_read >= r->frame_count) return -1;

    int ret = read_dng_frame(r->file_list[r->frames_read], bayer_out,
                             r->width, r->height, r->bits_per_sample,
                             r->compression, r->cfa_pattern);
    if (ret == 0)
        r->frames_read++;
    return ret;
}

void dng_reader_close(DngReader *r) {
    if (r->file_list) {
        for (int i = 0; i < r->file_count; i++)
            free(r->file_list[i]);
        free(r->file_list);
        r->file_list = NULL;
    }
    r->file_count = 0;
    r->frame_count = 0;
    r->frames_read = 0;
}

int dng_reader_probe_frame_count(const char *path) {
    char dir[1024];
    if (resolve_dng_dir(path, dir, sizeof(dir)) != 0) return -1;

    char **list = NULL;
    int count = 0;
    if (scan_dng_directory(dir, &list, &count) != 0) return -1;

    for (int i = 0; i < count; i++) free(list[i]);
    free(list);
    return count;
}

int dng_reader_probe_dimensions(const char *path, int *width, int *height) {
    char dir[1024];
    if (resolve_dng_dir(path, dir, sizeof(dir)) != 0) return -1;

    char **list = NULL;
    int count = 0;
    if (scan_dng_directory(dir, &list, &count) != 0) return -1;

    DngIFDInfo info;
    int ret = parse_dng_file(list[0], &info);

    for (int i = 0; i < count; i++) free(list[i]);
    free(list);

    if (ret != 0) return -1;
    *width = info.width;
    *height = info.height;
    return 0;
}

int dng_reader_probe_wb_gains(const char *path, float *r_gain, float *b_gain) {
    *r_gain = 1.0f;
    *b_gain = 1.0f;

    char dir[1024];
    if (resolve_dng_dir(path, dir, sizeof(dir)) != 0) return -1;

    char **list = NULL;
    int count = 0;
    if (scan_dng_directory(dir, &list, &count) != 0) return -1;

    DngIFDInfo info;
    int ret = parse_dng_file(list[0], &info);

    for (int i = 0; i < count; i++) free(list[i]);
    free(list);

    if (ret != 0 || !info.has_as_shot_neutral) return -1;

    if (info.as_shot_neutral[0] > 0 && info.as_shot_neutral[1] > 0 &&
        info.as_shot_neutral[2] > 0) {
        *r_gain = info.as_shot_neutral[1] / info.as_shot_neutral[0];
        *b_gain = info.as_shot_neutral[1] / info.as_shot_neutral[2];
        return 0;
    }

    return -1;
}
