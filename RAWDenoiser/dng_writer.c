#include "../include/dng_writer.h"
#include "../include/dng_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- TIFF type/tag constants (mirrored from dng_reader.c) ---- */
#define TIFF_BYTE      1
#define TIFF_ASCII     2
#define TIFF_SHORT     3
#define TIFF_LONG      4
#define TIFF_RATIONAL  5
#define TIFF_UNDEFINED 7

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
#define TAG_BLACK_LEVEL       50714
#define TAG_WHITE_LEVEL       50717

/* ---- Captured IFD entry ---- */
typedef struct {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint8_t  value_bytes[4];   /* raw 4-byte inline value field */
    uint8_t *ext_data;         /* heap copy of external data (NULL if inline) */
    uint32_t ext_data_size;
} CapturedEntry;

/* ---- DngWriter ---- */
struct DngWriter {
    int big_endian;             /* 0=II (little), 1=MM (big) */

    CapturedEntry *ifd0;
    int            ifd0_count;

    CapturedEntry *sub;
    int            sub_count;
    int            has_sub;
    int            raw_in_sub;  /* 1 if raw image data lives in SubIFD, 0 if in IFD0 */
};

/* ---- Byte-order I/O helpers ---- */

static uint16_t rd16(FILE *fp, int be) {
    uint8_t b[2];
    if (fread(b, 1, 2, fp) != 2) return 0;
    return be ? ((uint16_t)b[0] << 8) | b[1]
              : ((uint16_t)b[1] << 8) | b[0];
}

static uint32_t rd32(FILE *fp, int be) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return 0;
    return be ? ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
                ((uint32_t)b[2] << 8) | b[3]
              : ((uint32_t)b[3] << 24) | ((uint32_t)b[2] << 16) |
                ((uint32_t)b[1] << 8) | b[0];
}

static void wr16(FILE *fp, uint16_t v, int be) {
    uint8_t b[2];
    if (be) { b[0] = v >> 8; b[1] = v & 0xFF; }
    else    { b[0] = v & 0xFF; b[1] = v >> 8; }
    fwrite(b, 1, 2, fp);
}

static void wr32(FILE *fp, uint32_t v, int be) {
    uint8_t b[4];
    if (be) { b[0] = v >> 24; b[1] = (v >> 16) & 0xFF; b[2] = (v >> 8) & 0xFF; b[3] = v & 0xFF; }
    else    { b[0] = v & 0xFF; b[1] = (v >> 8) & 0xFF; b[2] = (v >> 16) & 0xFF; b[3] = v >> 24; }
    fwrite(b, 1, 4, fp);
}

/* Size in bytes of one element of a TIFF type */
static uint32_t tiff_type_size(uint16_t type) {
    switch (type) {
    case TIFF_BYTE: case TIFF_ASCII: case 6/*SBYTE*/: case TIFF_UNDEFINED: return 1;
    case TIFF_SHORT: case 8/*SSHORT*/: return 2;
    case TIFF_LONG: case 9/*SLONG*/: case 11/*FLOAT*/: return 4;
    case TIFF_RATIONAL: case 10/*SRATIONAL*/: case 12/*DOUBLE*/: return 8;
    default: return 1;
    }
}

/* ---- Capture one IFD from file ---- */

static int capture_ifd(FILE *fp, int be, long ifd_offset,
                       CapturedEntry **out_entries, int *out_count,
                       uint32_t *out_sub_ifd_offset) {
    fseek(fp, ifd_offset, SEEK_SET);
    uint16_t n = rd16(fp, be);
    if (n == 0 || n > 500) return -1;

    CapturedEntry *entries = calloc(n, sizeof(CapturedEntry));
    if (!entries) return -1;

    for (int i = 0; i < n; i++) {
        long eo = ifd_offset + 2 + (long)i * 12;
        fseek(fp, eo, SEEK_SET);

        entries[i].tag   = rd16(fp, be);
        entries[i].type  = rd16(fp, be);
        entries[i].count = rd32(fp, be);
        if (fread(entries[i].value_bytes, 1, 4, fp) != 4) {
            free(entries);
            return -1;
        }

        /* Check if SubIFD pointer */
        if (entries[i].tag == TAG_SUB_IFDS && out_sub_ifd_offset) {
            /* Read the offset value */
            if (entries[i].type == TIFF_LONG && entries[i].count == 1) {
                if (be)
                    *out_sub_ifd_offset = ((uint32_t)entries[i].value_bytes[0] << 24) |
                                          ((uint32_t)entries[i].value_bytes[1] << 16) |
                                          ((uint32_t)entries[i].value_bytes[2] << 8) |
                                           entries[i].value_bytes[3];
                else
                    *out_sub_ifd_offset = ((uint32_t)entries[i].value_bytes[3] << 24) |
                                          ((uint32_t)entries[i].value_bytes[2] << 16) |
                                          ((uint32_t)entries[i].value_bytes[1] << 8) |
                                           entries[i].value_bytes[0];
            }
        }

        /* If data doesn't fit in 4-byte value field, read external data */
        uint32_t data_size = entries[i].count * tiff_type_size(entries[i].type);
        if (data_size > 4) {
            uint32_t data_off;
            if (be)
                data_off = ((uint32_t)entries[i].value_bytes[0] << 24) |
                           ((uint32_t)entries[i].value_bytes[1] << 16) |
                           ((uint32_t)entries[i].value_bytes[2] << 8) |
                            entries[i].value_bytes[3];
            else
                data_off = ((uint32_t)entries[i].value_bytes[3] << 24) |
                           ((uint32_t)entries[i].value_bytes[2] << 16) |
                           ((uint32_t)entries[i].value_bytes[1] << 8) |
                            entries[i].value_bytes[0];

            entries[i].ext_data = malloc(data_size);
            entries[i].ext_data_size = data_size;
            if (entries[i].ext_data) {
                fseek(fp, data_off, SEEK_SET);
                if (fread(entries[i].ext_data, 1, data_size, fp) != data_size) {
                    /* Non-fatal: zero-fill */
                    memset(entries[i].ext_data, 0, data_size);
                }
            }
        }
    }

    *out_entries = entries;
    *out_count = n;
    return 0;
}

/* ---- Check if an IFD has image data (strips or tiles) ---- */
static int ifd_has_image_data(CapturedEntry *entries, int count) {
    for (int i = 0; i < count; i++) {
        if (entries[i].tag == TAG_STRIP_OFFSETS ||
            entries[i].tag == TAG_TILE_OFFSETS)
            return 1;
    }
    return 0;
}

/* ---- Public API ---- */

DngWriter *dng_writer_open(const char *template_dng_path) {
    FILE *fp = fopen(template_dng_path, "rb");
    if (!fp) return NULL;

    /* Read TIFF header */
    uint8_t hdr[8];
    if (fread(hdr, 1, 8, fp) != 8) { fclose(fp); return NULL; }

    int be;
    if (hdr[0] == 'M' && hdr[1] == 'M') be = 1;
    else if (hdr[0] == 'I' && hdr[1] == 'I') be = 0;
    else { fclose(fp); return NULL; }

    uint16_t magic = be ? ((uint16_t)hdr[2] << 8) | hdr[3]
                        : ((uint16_t)hdr[3] << 8) | hdr[2];
    if (magic != 42) { fclose(fp); return NULL; }

    uint32_t ifd0_off = be ? ((uint32_t)hdr[4] << 24) | ((uint32_t)hdr[5] << 16) |
                             ((uint32_t)hdr[6] << 8)  | hdr[7]
                           : ((uint32_t)hdr[7] << 24) | ((uint32_t)hdr[6] << 16) |
                             ((uint32_t)hdr[5] << 8)  | hdr[4];

    DngWriter *w = calloc(1, sizeof(DngWriter));
    if (!w) { fclose(fp); return NULL; }
    w->big_endian = be;

    /* Capture IFD0 */
    uint32_t sub_ifd_offset = 0;
    if (capture_ifd(fp, be, ifd0_off, &w->ifd0, &w->ifd0_count, &sub_ifd_offset) != 0) {
        fclose(fp); free(w); return NULL;
    }

    /* Capture SubIFD if present */
    if (sub_ifd_offset > 0) {
        if (capture_ifd(fp, be, sub_ifd_offset, &w->sub, &w->sub_count, NULL) == 0) {
            w->has_sub = 1;
            /* Check which IFD has the raw image data */
            w->raw_in_sub = ifd_has_image_data(w->sub, w->sub_count);
        }
    }
    if (!w->raw_in_sub)
        w->raw_in_sub = 0; /* raw data is in IFD0 */

    fclose(fp);
    return w;
}

static void free_entries(CapturedEntry *entries, int count) {
    for (int i = 0; i < count; i++)
        free(entries[i].ext_data);
    free(entries);
}

void dng_writer_close(DngWriter *w) {
    if (!w) return;
    free_entries(w->ifd0, w->ifd0_count);
    if (w->has_sub) free_entries(w->sub, w->sub_count);
    free(w);
}

/* ---- Tags to skip (tile-related, replaced by strip) ---- */
static int is_tile_tag(uint16_t tag) {
    return tag == TAG_TILE_WIDTH || tag == TAG_TILE_LENGTH ||
           tag == TAG_TILE_OFFSETS || tag == TAG_TILE_BYTE_COUNTS;
}

/* ---- Encode a uint32 into 4 bytes in the specified byte order ---- */
static void encode32(uint8_t out[4], uint32_t v, int be) {
    if (be) {
        out[0] = (v >> 24) & 0xFF; out[1] = (v >> 16) & 0xFF;
        out[2] = (v >> 8) & 0xFF;  out[3] = v & 0xFF;
    } else {
        out[0] = v & 0xFF;         out[1] = (v >> 8) & 0xFF;
        out[2] = (v >> 16) & 0xFF; out[3] = (v >> 24) & 0xFF;
    }
}

static void encode16_in32(uint8_t out[4], uint16_t v, int be) {
    memset(out, 0, 4);
    if (be) { out[0] = (v >> 8) & 0xFF; out[1] = v & 0xFF; }
    else    { out[0] = v & 0xFF;        out[1] = (v >> 8) & 0xFF; }
}

/* ---- Write one IFD (possibly patching image-data tags) ----
 * Returns the file offset just after the IFD's external data blobs.
 * image_data_offset: where the raw image data will start.
 * is_raw_ifd: 1 if this IFD should have its image-data tags patched. */
static long write_ifd(FILE *fp, int be,
                      CapturedEntry *entries, int count,
                      int is_raw_ifd,
                      int width, int height,
                      uint32_t image_data_offset,
                      uint32_t image_data_size,
                      int source_bps,
                      uint32_t sub_ifd_file_offset /* 0 if no SubIFD fixup needed */) {
    /* Filter out tile tags from raw IFD, count output entries */
    int out_count = 0;
    for (int i = 0; i < count; i++) {
        if (is_raw_ifd && is_tile_tag(entries[i].tag)) continue;
        out_count++;
    }

    /* Check if we need to ADD strip tags (source used tiles, had no strips) */
    int has_strip_offsets = 0, has_strip_byte_counts = 0, has_rows_per_strip = 0;
    int has_compression = 0, has_bps = 0, has_white_level = 0;
    if (is_raw_ifd) {
        for (int i = 0; i < count; i++) {
            if (is_tile_tag(entries[i].tag)) continue;
            if (entries[i].tag == TAG_STRIP_OFFSETS)     has_strip_offsets = 1;
            if (entries[i].tag == TAG_STRIP_BYTE_COUNTS) has_strip_byte_counts = 1;
            if (entries[i].tag == TAG_ROWS_PER_STRIP)    has_rows_per_strip = 1;
            if (entries[i].tag == TAG_COMPRESSION)       has_compression = 1;
            if (entries[i].tag == TAG_BITS_PER_SAMPLE)   has_bps = 1;
            if (entries[i].tag == TAG_WHITE_LEVEL)       has_white_level = 1;
        }
        if (!has_strip_offsets)     out_count++;
        if (!has_strip_byte_counts) out_count++;
        if (!has_rows_per_strip)    out_count++;
        if (!has_compression)       out_count++;
        if (!has_bps)               out_count++;
        if (!has_white_level)       out_count++;
    }

    /* IFD layout: 2-byte count + out_count * 12 bytes + 4-byte next-IFD pointer */
    long ifd_start = ftell(fp);
    long ext_data_start = ifd_start + 2 + (long)out_count * 12 + 4;

    /* First pass: compute where each external blob goes */
    long ext_cursor = ext_data_start;

    /* Write IFD entry count */
    wr16(fp, (uint16_t)out_count, be);

    int bit_shift = (source_bps < 16) ? (16 - source_bps) : 0;

    /* Write entries */
    for (int i = 0; i < count; i++) {
        if (is_raw_ifd && is_tile_tag(entries[i].tag)) continue;

        CapturedEntry e = entries[i]; /* work on a copy */

        /* ---- Patch image-data tags for raw IFD ---- */
        if (is_raw_ifd) {
            switch (e.tag) {
            case TAG_COMPRESSION:
                /* Force uncompressed */
                e.type = TIFF_SHORT; e.count = 1;
                encode16_in32(e.value_bytes, 1, be);
                free(e.ext_data); e.ext_data = NULL; e.ext_data_size = 0;
                break;
            case TAG_BITS_PER_SAMPLE:
                e.type = TIFF_SHORT; e.count = 1;
                encode16_in32(e.value_bytes, 16, be);
                free(e.ext_data); e.ext_data = NULL; e.ext_data_size = 0;
                break;
            case TAG_STRIP_OFFSETS:
                e.type = TIFF_LONG; e.count = 1;
                encode32(e.value_bytes, image_data_offset, be);
                free(e.ext_data); e.ext_data = NULL; e.ext_data_size = 0;
                break;
            case TAG_STRIP_BYTE_COUNTS:
                e.type = TIFF_LONG; e.count = 1;
                encode32(e.value_bytes, image_data_size, be);
                free(e.ext_data); e.ext_data = NULL; e.ext_data_size = 0;
                break;
            case TAG_ROWS_PER_STRIP:
                e.type = TIFF_LONG; e.count = 1;
                encode32(e.value_bytes, (uint32_t)height, be);
                free(e.ext_data); e.ext_data = NULL; e.ext_data_size = 0;
                break;
            case TAG_BLACK_LEVEL:
                /* Scale BlackLevel from source bit depth to 16-bit.
                 * For external data (multi-value): DON'T modify ext_data here
                 * because it's a shared pointer to the template's data.
                 * The blob-writing phase handles it with a proper copy.
                 * Only handle the inline case (single value in value_bytes). */
                if (bit_shift > 0 && !(e.ext_data && e.ext_data_size > 0)) {
                    /* Inline BlackLevel (single value, fits in 4-byte field) */
                    if (e.type == TIFF_SHORT) {
                        uint16_t v;
                        if (be) v = ((uint16_t)e.value_bytes[0] << 8) | e.value_bytes[1];
                        else    v = ((uint16_t)e.value_bytes[1] << 8) | e.value_bytes[0];
                        v = (uint16_t)((uint32_t)v << bit_shift);
                        encode16_in32(e.value_bytes, v, be);
                    } else if (e.type == TIFF_LONG) {
                        uint32_t v;
                        if (be) v = ((uint32_t)e.value_bytes[0]<<24)|((uint32_t)e.value_bytes[1]<<16)|
                                    ((uint32_t)e.value_bytes[2]<<8)|e.value_bytes[3];
                        else    v = ((uint32_t)e.value_bytes[3]<<24)|((uint32_t)e.value_bytes[2]<<16)|
                                    ((uint32_t)e.value_bytes[1]<<8)|e.value_bytes[0];
                        v <<= bit_shift;
                        encode32(e.value_bytes, v, be);
                    }
                }
                break;
            case TAG_WHITE_LEVEL:
                /* Set to 65535 for 16-bit */
                e.type = TIFF_LONG; e.count = 1;
                encode32(e.value_bytes, 65535, be);
                free(e.ext_data); e.ext_data = NULL; e.ext_data_size = 0;
                break;
            }
        }

        /* Fix up SubIFD pointer in IFD0 */
        if (e.tag == TAG_SUB_IFDS && sub_ifd_file_offset > 0) {
            encode32(e.value_bytes, sub_ifd_file_offset, be);
        }

        /* Write the 12-byte entry */
        wr16(fp, e.tag, be);
        wr16(fp, e.type, be);
        wr32(fp, e.count, be);

        uint32_t data_size = e.count * tiff_type_size(e.type);
        if (data_size > 4 && e.ext_data) {
            /* Write offset to where ext data will go */
            wr32(fp, (uint32_t)ext_cursor, be);
            ext_cursor += (long)e.ext_data_size;
            /* Align to word boundary */
            if (ext_cursor & 1) ext_cursor++;
        } else {
            /* Inline value */
            fwrite(e.value_bytes, 1, 4, fp);
        }
    }

    /* ---- Append synthetic strip tags if source had tiles ---- */
    if (is_raw_ifd) {
        if (!has_compression) {
            wr16(fp, TAG_COMPRESSION, be);
            wr16(fp, TIFF_SHORT, be);
            wr32(fp, 1, be);
            uint8_t vb[4]; encode16_in32(vb, 1, be); fwrite(vb, 1, 4, fp);
        }
        if (!has_bps) {
            wr16(fp, TAG_BITS_PER_SAMPLE, be);
            wr16(fp, TIFF_SHORT, be);
            wr32(fp, 1, be);
            uint8_t vb[4]; encode16_in32(vb, 16, be); fwrite(vb, 1, 4, fp);
        }
        if (!has_strip_offsets) {
            wr16(fp, TAG_STRIP_OFFSETS, be);
            wr16(fp, TIFF_LONG, be);
            wr32(fp, 1, be);
            wr32(fp, image_data_offset, be);
        }
        if (!has_rows_per_strip) {
            wr16(fp, TAG_ROWS_PER_STRIP, be);
            wr16(fp, TIFF_LONG, be);
            wr32(fp, 1, be);
            wr32(fp, (uint32_t)height, be);
        }
        if (!has_strip_byte_counts) {
            wr16(fp, TAG_STRIP_BYTE_COUNTS, be);
            wr16(fp, TIFF_LONG, be);
            wr32(fp, 1, be);
            wr32(fp, image_data_size, be);
        }
        if (!has_white_level) {
            wr16(fp, TAG_WHITE_LEVEL, be);
            wr16(fp, TIFF_LONG, be);
            wr32(fp, 1, be);
            wr32(fp, 65535, be);
        }
    }

    /* Next IFD pointer = 0 (no more IFDs) */
    wr32(fp, 0, be);

    /* Write external data blobs */
    ext_cursor = ext_data_start;
    for (int i = 0; i < count; i++) {
        if (is_raw_ifd && is_tile_tag(entries[i].tag)) continue;

        CapturedEntry *e = &entries[i];

        /* Recompute ext_data for BlackLevel (we modified the copy above but ext_data
         * is on the original). For BlackLevel scaling we need to reapply to the original. */
        uint32_t data_size = e->count * tiff_type_size(e->type);
        if (data_size > 4 && e->ext_data) {
            fseek(fp, ext_cursor, SEEK_SET);

            /* For BlackLevel in raw IFD, write the scaled version */
            if (is_raw_ifd && e->tag == TAG_BLACK_LEVEL && bit_shift > 0) {
                /* Need to write scaled copy */
                uint8_t *scaled = malloc(e->ext_data_size);
                if (scaled) {
                    memcpy(scaled, e->ext_data, e->ext_data_size);
                    if (e->type == TIFF_SHORT) {
                        for (uint32_t j = 0; j < e->count && j * 2 < e->ext_data_size; j++) {
                            uint16_t v;
                            if (be) v = ((uint16_t)scaled[j*2] << 8) | scaled[j*2+1];
                            else    v = ((uint16_t)scaled[j*2+1] << 8) | scaled[j*2];
                            v = (uint16_t)((uint32_t)v << bit_shift);
                            if (be) { scaled[j*2] = v >> 8; scaled[j*2+1] = v & 0xFF; }
                            else    { scaled[j*2] = v & 0xFF; scaled[j*2+1] = v >> 8; }
                        }
                    } else if (e->type == TIFF_LONG) {
                        for (uint32_t j = 0; j < e->count && j * 4 < e->ext_data_size; j++) {
                            uint32_t v;
                            if (be) v = ((uint32_t)scaled[j*4]<<24)|((uint32_t)scaled[j*4+1]<<16)|
                                        ((uint32_t)scaled[j*4+2]<<8)|scaled[j*4+3];
                            else    v = ((uint32_t)scaled[j*4+3]<<24)|((uint32_t)scaled[j*4+2]<<16)|
                                        ((uint32_t)scaled[j*4+1]<<8)|scaled[j*4];
                            v <<= bit_shift;
                            encode32(scaled + j*4, v, be);
                        }
                    } else if (e->type == TIFF_RATIONAL) {
                        for (uint32_t j = 0; j < e->count && j * 8 + 4 <= e->ext_data_size; j++) {
                            uint32_t num;
                            if (be) num = ((uint32_t)scaled[j*8]<<24)|((uint32_t)scaled[j*8+1]<<16)|
                                          ((uint32_t)scaled[j*8+2]<<8)|scaled[j*8+3];
                            else    num = ((uint32_t)scaled[j*8+3]<<24)|((uint32_t)scaled[j*8+2]<<16)|
                                          ((uint32_t)scaled[j*8+1]<<8)|scaled[j*8];
                            num <<= bit_shift;
                            encode32(scaled + j*8, num, be);
                        }
                    }
                    fwrite(scaled, 1, e->ext_data_size, fp);
                    free(scaled);
                } else {
                    fwrite(e->ext_data, 1, e->ext_data_size, fp);
                }
            } else {
                fwrite(e->ext_data, 1, e->ext_data_size, fp);
            }

            ext_cursor += (long)e->ext_data_size;
            if (ext_cursor & 1) {
                fputc(0, fp);
                ext_cursor++;
            }
        }
    }

    return ext_cursor;
}

/* ---- Undo CFA remap (inverse of remap_cfa_to_rggb in dng_reader.c) ---- */
static void unmap_rggb_to_cfa(const uint16_t *rggb, uint16_t *out,
                               int width, int height, int cfa_pattern) {
    size_t pixels = (size_t)width * height;

    if (cfa_pattern == DNG_CFA_RGGB) {
        memcpy(out, rggb, pixels * sizeof(uint16_t));
        return;
    }

    int row_shift = 0, col_shift = 0;
    switch (cfa_pattern) {
    case DNG_CFA_GRBG: col_shift = 1; break;
    case DNG_CFA_GBRG: row_shift = 1; break;
    case DNG_CFA_BGGR: row_shift = 1; col_shift = 1; break;
    }

    /* Inverse of the remap: original[y+row_shift][x+col_shift] = rggb[y][x]
     * So: out[(y+row_shift)*w + (x+col_shift)] = rggb[y*w + x] */
    memset(out, 0, pixels * sizeof(uint16_t));
    for (int y = 0; y < height - row_shift; y++) {
        for (int x = 0; x < width - col_shift; x++) {
            out[(size_t)(y + row_shift) * width + (x + col_shift)] =
                rggb[(size_t)y * width + x];
        }
    }
}

int dng_writer_write_frame(DngWriter *w, const uint16_t *bayer_data,
                           int width, int height,
                           const char *output_path,
                           int source_bits_per_sample,
                           int source_cfa_pattern) {
    if (!w) return -1;

    size_t pixels = (size_t)width * height;
    uint32_t image_data_size = (uint32_t)(pixels * 2); /* 16-bit per pixel */
    int be = w->big_endian;

    /* Undo CFA remap */
    uint16_t *unmapped = malloc(pixels * sizeof(uint16_t));
    if (!unmapped) return -1;
    unmap_rggb_to_cfa(bayer_data, unmapped, width, height, source_cfa_pattern);

    FILE *fp = fopen(output_path, "wb");
    if (!fp) { free(unmapped); return -1; }

    /* ---- TIFF header ---- */
    if (be) { fputc('M', fp); fputc('M', fp); }
    else    { fputc('I', fp); fputc('I', fp); }
    wr16(fp, 42, be);
    wr32(fp, 8, be); /* IFD0 starts at offset 8 */

    /* ---- Compute layout ----
     * We need to know where the image data goes before writing IFDs,
     * because StripOffsets must point to it. Do a sizing pass first. */

    /* Count entries in each IFD (after filtering) */
    int ifd0_out_count = 0;
    for (int i = 0; i < w->ifd0_count; i++) {
        int is_raw = !w->raw_in_sub;
        if (is_raw && is_tile_tag(w->ifd0[i].tag)) continue;
        ifd0_out_count++;
    }
    /* Add synthetic strip tags if raw is in IFD0 and source had tiles */
    if (!w->raw_in_sub) {
        int has_so = 0, has_sbc = 0, has_rps = 0, has_c = 0, has_b = 0, has_wl = 0;
        for (int i = 0; i < w->ifd0_count; i++) {
            if (is_tile_tag(w->ifd0[i].tag)) continue;
            if (w->ifd0[i].tag == TAG_STRIP_OFFSETS) has_so = 1;
            if (w->ifd0[i].tag == TAG_STRIP_BYTE_COUNTS) has_sbc = 1;
            if (w->ifd0[i].tag == TAG_ROWS_PER_STRIP) has_rps = 1;
            if (w->ifd0[i].tag == TAG_COMPRESSION) has_c = 1;
            if (w->ifd0[i].tag == TAG_BITS_PER_SAMPLE) has_b = 1;
            if (w->ifd0[i].tag == TAG_WHITE_LEVEL) has_wl = 1;
        }
        if (!has_so) ifd0_out_count++;
        if (!has_sbc) ifd0_out_count++;
        if (!has_rps) ifd0_out_count++;
        if (!has_c) ifd0_out_count++;
        if (!has_b) ifd0_out_count++;
        if (!has_wl) ifd0_out_count++;
    }

    /* IFD0 size: 2 + count*12 + 4 */
    long ifd0_size = 2 + (long)ifd0_out_count * 12 + 4;

    /* External data blobs for IFD0 */
    long ifd0_ext_size = 0;
    for (int i = 0; i < w->ifd0_count; i++) {
        int is_raw = !w->raw_in_sub;
        if (is_raw && is_tile_tag(w->ifd0[i].tag)) continue;
        uint32_t ds = w->ifd0[i].count * tiff_type_size(w->ifd0[i].type);
        if (ds > 4 && w->ifd0[i].ext_data) {
            /* Check if this tag gets replaced inline in raw IFD */
            if (is_raw && (w->ifd0[i].tag == TAG_COMPRESSION ||
                           w->ifd0[i].tag == TAG_BITS_PER_SAMPLE ||
                           w->ifd0[i].tag == TAG_STRIP_OFFSETS ||
                           w->ifd0[i].tag == TAG_STRIP_BYTE_COUNTS ||
                           w->ifd0[i].tag == TAG_ROWS_PER_STRIP ||
                           w->ifd0[i].tag == TAG_WHITE_LEVEL))
                continue;
            ifd0_ext_size += w->ifd0[i].ext_data_size;
            if (ifd0_ext_size & 1) ifd0_ext_size++;
        }
    }

    long sub_ifd_file_offset = 0;
    long sub_size = 0;
    long sub_ext_size = 0;
    if (w->has_sub) {
        sub_ifd_file_offset = 8 + ifd0_size + ifd0_ext_size;

        int sub_out_count = 0;
        for (int i = 0; i < w->sub_count; i++) {
            if (w->raw_in_sub && is_tile_tag(w->sub[i].tag)) continue;
            sub_out_count++;
        }
        if (w->raw_in_sub) {
            int has_so = 0, has_sbc = 0, has_rps = 0, has_c = 0, has_b = 0, has_wl = 0;
            for (int i = 0; i < w->sub_count; i++) {
                if (is_tile_tag(w->sub[i].tag)) continue;
                if (w->sub[i].tag == TAG_STRIP_OFFSETS) has_so = 1;
                if (w->sub[i].tag == TAG_STRIP_BYTE_COUNTS) has_sbc = 1;
                if (w->sub[i].tag == TAG_ROWS_PER_STRIP) has_rps = 1;
                if (w->sub[i].tag == TAG_COMPRESSION) has_c = 1;
                if (w->sub[i].tag == TAG_BITS_PER_SAMPLE) has_b = 1;
                if (w->sub[i].tag == TAG_WHITE_LEVEL) has_wl = 1;
            }
            if (!has_so) sub_out_count++;
            if (!has_sbc) sub_out_count++;
            if (!has_rps) sub_out_count++;
            if (!has_c) sub_out_count++;
            if (!has_b) sub_out_count++;
            if (!has_wl) sub_out_count++;
        }

        sub_size = 2 + (long)sub_out_count * 12 + 4;

        for (int i = 0; i < w->sub_count; i++) {
            if (w->raw_in_sub && is_tile_tag(w->sub[i].tag)) continue;
            uint32_t ds = w->sub[i].count * tiff_type_size(w->sub[i].type);
            if (ds > 4 && w->sub[i].ext_data) {
                if (w->raw_in_sub && (w->sub[i].tag == TAG_COMPRESSION ||
                                      w->sub[i].tag == TAG_BITS_PER_SAMPLE ||
                                      w->sub[i].tag == TAG_STRIP_OFFSETS ||
                                      w->sub[i].tag == TAG_STRIP_BYTE_COUNTS ||
                                      w->sub[i].tag == TAG_ROWS_PER_STRIP ||
                                      w->sub[i].tag == TAG_WHITE_LEVEL))
                    continue;
                sub_ext_size += w->sub[i].ext_data_size;
                if (sub_ext_size & 1) sub_ext_size++;
            }
        }
    }

    /* Image data goes after all IFDs + ext data */
    long image_offset;
    if (w->has_sub)
        image_offset = sub_ifd_file_offset + sub_size + sub_ext_size;
    else
        image_offset = 8 + ifd0_size + ifd0_ext_size;

    /* Align to 2-byte boundary */
    if (image_offset & 1) image_offset++;

    /* ---- Write IFD0 ---- */
    write_ifd(fp, be, w->ifd0, w->ifd0_count,
              !w->raw_in_sub, /* is_raw_ifd */
              width, height,
              (uint32_t)image_offset, image_data_size,
              source_bits_per_sample,
              w->has_sub ? (uint32_t)sub_ifd_file_offset : 0);

    /* ---- Write SubIFD ---- */
    if (w->has_sub) {
        fseek(fp, sub_ifd_file_offset, SEEK_SET);
        write_ifd(fp, be, w->sub, w->sub_count,
                  w->raw_in_sub, /* is_raw_ifd */
                  width, height,
                  (uint32_t)image_offset, image_data_size,
                  source_bits_per_sample,
                  0);
    }

    /* ---- Write image data ---- */
    fseek(fp, image_offset, SEEK_SET);

    if (be) {
        /* Big-endian: byte-swap each uint16 */
        uint8_t *swapped = malloc(pixels * 2);
        if (!swapped) { fclose(fp); free(unmapped); return -1; }
        for (size_t i = 0; i < pixels; i++) {
            swapped[i * 2]     = (unmapped[i] >> 8) & 0xFF;
            swapped[i * 2 + 1] = unmapped[i] & 0xFF;
        }
        fwrite(swapped, 1, pixels * 2, fp);
        free(swapped);
    } else {
        /* Little-endian: native on Apple Silicon */
        fwrite(unmapped, sizeof(uint16_t), pixels, fp);
    }

    fclose(fp);
    free(unmapped);

    return 0;
}
