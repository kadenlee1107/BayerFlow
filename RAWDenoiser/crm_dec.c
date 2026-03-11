/*
 * Canon Cinema RAW Light (CRM) Decoder
 *
 * Container: ISOBMFF parser for .CRM files
 * Codec: CRX decoder (Le Gall 5/3 wavelet + adaptive Rice-Golomb)
 *
 * Ported from LibRaw's crx.cpp by Alexey Danilchenko (2018-2019).
 * Original license: LGPL 2.1 / CDDL 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/crm_dec.h"

/* ========================================================================
 * Utility helpers
 * ======================================================================== */

static inline uint32_t rd_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}
static inline uint16_t rd_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}
static inline int64_t rd_be64(const uint8_t *p) {
    return ((int64_t)rd_be32(p) << 32) | rd_be32(p + 4);
}

#define TAG(a,b,c,d) (((uint32_t)(a)<<24)|((uint32_t)(b)<<16)|((uint32_t)(c)<<8)|(d))

#define _abs(x) (((x) ^ ((int32_t)(x) >> 31)) - ((int32_t)(x) >> 31))
#define _min(a,b) ((a) < (b) ? (a) : (b))
#define _constrain(x,l,u) ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))

/* ========================================================================
 * ISOBMFF Container Parser
 * ======================================================================== */

/* Find an atom (box) within [start, end) in the file.
 * Returns file offset of box payload, or -1 if not found.
 * out_size receives the payload size (excluding 8-byte header). */
static int64_t find_atom(FILE *f, int64_t start, int64_t end,
                          uint32_t tag, uint64_t *out_size)
{
    fseeko(f, start, SEEK_SET);
    while (ftello(f) < end - 8) {
        uint8_t hdr[8];
        if (fread(hdr, 1, 8, f) != 8) return -1;
        uint64_t box_size = rd_be32(hdr);
        uint32_t box_tag = rd_be32(hdr + 4);

        int hdr_len = 8;
        if (box_size == 1) {
            /* 64-bit extended size */
            uint8_t ext[8];
            if (fread(ext, 1, 8, f) != 8) return -1;
            box_size = (uint64_t)rd_be32(ext) << 32 | rd_be32(ext + 4);
            hdr_len = 16;
        }
        if (box_size < (uint64_t)hdr_len) return -1;

        if (box_tag == tag) {
            *out_size = box_size - hdr_len;
            return ftello(f);
        }
        fseeko(f, (int64_t)(box_size - hdr_len), SEEK_CUR);
    }
    return -1;
}

/* Find atom tag in moov → uuid hierarchy.
 * CRM files wrap most atoms inside a UUID box. */
static int64_t find_video_trak(FILE *f, int64_t moov_start, int64_t moov_end,
                                int64_t *trak_end)
{
    /* CRM moov structure: uuid(...) + mvhd + trak(video) + trak(audio) + trak(tmcd) + trak(meta)
     * We need the first trak with vmhd inside minf (= video track). */
    fseeko(f, moov_start, SEEK_SET);
    while (ftello(f) < moov_end - 8) {
        uint8_t hdr[8];
        if (fread(hdr, 1, 8, f) != 8) return -1;
        uint64_t box_size = rd_be32(hdr);
        uint32_t box_tag = rd_be32(hdr + 4);
        if (box_size == 1) {
            uint8_t ext[8];
            if (fread(ext, 1, 8, f) != 8) return -1;
            box_size = (uint64_t)rd_be32(ext) << 32 | rd_be32(ext + 4);
        }
        if (box_size < 8) return -1;
        int64_t box_payload = ftello(f);
        int64_t box_end_pos = box_payload + (int64_t)(box_size - 8);

        if (box_tag == TAG('t','r','a','k')) {
            /* Check if this trak has mdia → minf → vmhd (video track) */
            uint64_t mdia_sz;
            int64_t mdia = find_atom(f, box_payload, box_end_pos, TAG('m','d','i','a'), &mdia_sz);
            if (mdia >= 0) {
                uint64_t minf_sz;
                int64_t minf = find_atom(f, mdia, mdia + (int64_t)mdia_sz, TAG('m','i','n','f'), &minf_sz);
                if (minf >= 0) {
                    uint64_t vmhd_sz;
                    int64_t vmhd = find_atom(f, minf, minf + (int64_t)minf_sz, TAG('v','m','h','d'), &vmhd_sz);
                    if (vmhd >= 0) {
                        *trak_end = box_end_pos;
                        return box_payload;
                    }
                }
            }
        }
        fseeko(f, box_end_pos, SEEK_SET);
    }
    return -1;
}

/* Parse CMP1 box from within CRAW sample description.
 * Layout follows LibRaw's crxParseImageHeader exactly:
 *   [0-3]: preamble (marker + hdr_size or version/flags)
 *   [4-5]: codec version (0x100 or 0x200)
 *   [6-7]: reserved
 *   [8-11]: f_width
 *   [12-15]: f_height
 *   [16-19]: tileWidth
 *   [20-23]: tileHeight
 *   [24]: nBits
 *   [25]: [7:4]=nPlanes, [3:0]=cfaLayout
 *   [26]: [7:4]=encType, [3:0]=imageLevels
 *   [27]: [7]=hasTileCols, [6]=hasTileRows
 *   [28-31]: mdatHdrSize (4 bytes)
 *   [32]: [7]=extHeader flag
 */
static int parse_cmp1(const uint8_t *data, int size, CrmCodecParams *params)
{
    if (size < 32) return CRM_ERR_INVALID;

    params->version      = rd_be16(data + 4);
    params->f_width      = (int)rd_be32(data + 8);
    params->f_height     = (int)rd_be32(data + 12);
    params->tile_width   = (int)rd_be32(data + 16);
    params->tile_height  = (int)rd_be32(data + 20);
    params->n_bits       = data[24];
    params->n_planes     = data[25] >> 4;
    params->cfa_layout   = data[25] & 0xF;
    params->enc_type     = data[26] >> 4;
    params->image_levels = data[26] & 0xF;
    params->has_tile_cols = data[27] >> 7;
    params->has_tile_rows = (data[27] >> 6) & 1;
    params->mdat_hdr_size = (int)rd_be32(data + 28);
    params->median_bits  = params->n_bits;  /* default */
    params->sample_precision = 0;

    /* Extended header (v0.200): check for medianBits override */
    if (size >= 33) {
        int extHeader = data[32] >> 7;
        int useMedianBits = 0;
        if (extHeader && size >= 57 && params->n_planes == 4)
            useMedianBits = (data[56] >> 6) & 1;
        if (useMedianBits && size >= 85)
            params->median_bits = data[84];
    }

    /* CMP1 stores full-sensor dimensions; convert to per-plane for Bayer */
    if (params->n_planes == 4) {
        params->f_width     /= 2;
        params->f_height    /= 2;
        params->tile_width  /= 2;
        params->tile_height /= 2;
    }

    /* Validate */
    if ((params->version != 0x100 && params->version != 0x200) || !params->mdat_hdr_size)
        return CRM_ERR_INVALID;
    if (params->n_bits > 15 && params->enc_type != 1) return CRM_ERR_INVALID;
    if (params->n_bits > 16 || params->n_bits < 8) return CRM_ERR_INVALID;
    if (params->f_width < 16 || params->f_height < 16) return CRM_ERR_INVALID;
    if (params->f_width > 0x7FFF || params->f_height > 0x7FFF) return CRM_ERR_INVALID;
    if (params->image_levels > 3) return CRM_ERR_INVALID;

    return CRM_OK;
}

int crm_mov_parse(const char *path, CrmMovInfo *mov)
{
    memset(mov, 0, sizeof(*mov));

    FILE *f = fopen(path, "rb");
    if (!f) return CRM_ERR_IO;

    fseeko(f, 0, SEEK_END);
    int64_t file_size = ftello(f);

    /* Find moov */
    uint64_t moov_size;
    int64_t moov = find_atom(f, 0, file_size, TAG('m','o','o','v'), &moov_size);
    if (moov < 0) { fclose(f); return CRM_ERR_INVALID; }

    /* Find video trak */
    int64_t trak_end;
    int64_t trak = find_video_trak(f, moov, moov + (int64_t)moov_size, &trak_end);
    if (trak < 0) { fclose(f); return CRM_ERR_INVALID; }

    /* Find mdia → mdhd (timescale) */
    uint64_t mdia_sz;
    int64_t mdia = find_atom(f, trak, trak_end, TAG('m','d','i','a'), &mdia_sz);
    if (mdia < 0) { fclose(f); return CRM_ERR_INVALID; }

    double timescale = 0;
    uint64_t mdhd_sz;
    int64_t mdhd = find_atom(f, mdia, mdia + (int64_t)mdia_sz, TAG('m','d','h','d'), &mdhd_sz);
    if (mdhd >= 0 && mdhd_sz >= 24) {
        uint8_t d[32];
        fseeko(f, mdhd, SEEK_SET);
        fread(d, 1, 24, f);
        if (d[0] == 0) {
            timescale = (double)rd_be32(d + 12);
        } else {
            timescale = (double)rd_be32(d + 20);
        }
    }

    /* Find minf → stbl */
    uint64_t minf_sz;
    int64_t minf = find_atom(f, mdia, mdia + (int64_t)mdia_sz, TAG('m','i','n','f'), &minf_sz);
    if (minf < 0) { fclose(f); return CRM_ERR_INVALID; }

    uint64_t stbl_sz;
    int64_t stbl = find_atom(f, minf, minf + (int64_t)minf_sz, TAG('s','t','b','l'), &stbl_sz);
    if (stbl < 0) { fclose(f); return CRM_ERR_INVALID; }

    /* Parse stsd → CRAW entry → CMP1 */
    uint64_t stsd_sz;
    int64_t stsd = find_atom(f, stbl, stbl + (int64_t)stbl_sz, TAG('s','t','s','d'), &stsd_sz);
    if (stsd >= 0 && stsd_sz > 100) {
        /* Read entire stsd for CMP1 parsing */
        size_t stsd_total = (size_t)_min(stsd_sz, 4096);
        uint8_t *stsd_data = malloc(stsd_total);
        if (stsd_data) {
            fseeko(f, stsd, SEEK_SET);
            fread(stsd_data, 1, stsd_total, f);

            /* version(4) + count(4) + CRAW entry starts at offset 8 */
            if (stsd_total > 8) {
                uint32_t entry_size = rd_be32(stsd_data + 8);
                uint32_t entry_tag  = rd_be32(stsd_data + 12);
                (void)entry_size;

                if (entry_tag == TAG('C','R','A','W')) {
                    /* CRAW entry: 86 bytes standard header + variable Canon extension.
                     * Sub-boxes (CMP1, CDI1, CNCV) follow after the extension.
                     * Search for CMP1 tag robustly instead of assuming fixed offset. */
                    size_t craw_start = 8;  /* offset of CRAW entry in stsd data */
                    size_t craw_end = _min(craw_start + entry_size, stsd_total);

                    for (size_t pos = craw_start + 86; pos + 8 < craw_end; pos++) {
                        if (stsd_data[pos+4] == 'C' && stsd_data[pos+5] == 'M' &&
                            stsd_data[pos+6] == 'P' && stsd_data[pos+7] == '1') {
                            uint32_t sub_sz = rd_be32(stsd_data + pos);
                            if (sub_sz >= 40 && pos + sub_sz <= craw_end) {
                                int cmp1_payload_sz = (int)(sub_sz - 8);
                                parse_cmp1(stsd_data + pos + 8, cmp1_payload_sz, &mov->codec);
                            }
                            break;
                        }
                    }

                    /* Extract width/height from CRAW entry
                     * CRAW entry layout: size(4) + 'CRAW'(4) + reserved(6) + refidx(2)
                     * + version(2) + revision(2) + vendor(4) + temporal_q(4) + spatial_q(4)
                     * + width(2) + height(2) + ...
                     * Width at entry_offset + 8 + 6 + 2 + 2 + 2 + 4 + 4 + 4 = +32
                     * Actually the width/height in CRAW stsd are often 0 or nominal.
                     * The real dimensions come from CMP1. */
                }

                /* Save stsd entry for container writer */
                mov->stsd_entry = malloc(stsd_total);
                if (mov->stsd_entry) {
                    memcpy(mov->stsd_entry, stsd_data, stsd_total);
                    mov->stsd_entry_size = (int)stsd_total;
                }
            }
            free(stsd_data);
        }
    }

    /* Full image dimensions = plane dims * 2 (Bayer) if n_planes == 4 */
    if (mov->codec.n_planes == 4) {
        mov->width  = mov->codec.f_width * 2;
        mov->height = mov->codec.f_height * 2;
    } else {
        mov->width  = mov->codec.f_width;
        mov->height = mov->codec.f_height;
    }

    /* Parse stsz */
    uint64_t stsz_sz;
    int64_t stsz = find_atom(f, stbl, stbl + (int64_t)stbl_sz, TAG('s','t','s','z'), &stsz_sz);
    if (stsz < 0 || stsz_sz < 12) { fclose(f); crm_mov_free(mov); return CRM_ERR_INVALID; }

    uint8_t stsz_hdr[12];
    fseeko(f, stsz, SEEK_SET);
    if (fread(stsz_hdr, 1, 12, f) != 12) { fclose(f); crm_mov_free(mov); return CRM_ERR_IO; }

    uint32_t uniform_size = rd_be32(stsz_hdr + 4);
    uint32_t sample_count = rd_be32(stsz_hdr + 8);

    mov->frame_count = (int)sample_count;
    mov->frame_sizes = calloc(sample_count, sizeof(int32_t));
    if (!mov->frame_sizes) { fclose(f); crm_mov_free(mov); return CRM_ERR_ALLOC; }

    if (uniform_size > 0) {
        for (uint32_t i = 0; i < sample_count; i++)
            mov->frame_sizes[i] = (int32_t)uniform_size;
    } else {
        uint8_t *sz_data = malloc(sample_count * 4);
        if (!sz_data) { fclose(f); crm_mov_free(mov); return CRM_ERR_ALLOC; }
        if (fread(sz_data, 4, sample_count, f) != sample_count) {
            free(sz_data); fclose(f); crm_mov_free(mov); return CRM_ERR_IO;
        }
        for (uint32_t i = 0; i < sample_count; i++)
            mov->frame_sizes[i] = (int32_t)rd_be32(sz_data + i * 4);
        free(sz_data);
    }

    /* Parse co64 (CRM uses 64-bit chunk offsets) */
    uint64_t co64_sz;
    int64_t co64 = find_atom(f, stbl, stbl + (int64_t)stbl_sz, TAG('c','o','6','4'), &co64_sz);
    int use_co64 = 1;
    if (co64 < 0) {
        co64 = find_atom(f, stbl, stbl + (int64_t)stbl_sz, TAG('s','t','c','o'), &co64_sz);
        use_co64 = 0;
        if (co64 < 0) { fclose(f); crm_mov_free(mov); return CRM_ERR_INVALID; }
    }

    uint8_t co_hdr[8];
    fseeko(f, co64, SEEK_SET);
    if (fread(co_hdr, 1, 8, f) != 8) { fclose(f); crm_mov_free(mov); return CRM_ERR_IO; }
    uint32_t chunk_count = rd_be32(co_hdr + 4);

    int64_t *chunk_offsets = malloc(chunk_count * sizeof(int64_t));
    if (!chunk_offsets) { fclose(f); crm_mov_free(mov); return CRM_ERR_ALLOC; }

    for (uint32_t i = 0; i < chunk_count; i++) {
        if (use_co64) {
            uint8_t buf[8];
            if (fread(buf, 1, 8, f) != 8) { free(chunk_offsets); fclose(f); crm_mov_free(mov); return CRM_ERR_IO; }
            chunk_offsets[i] = rd_be64(buf);
        } else {
            uint8_t buf[4];
            if (fread(buf, 1, 4, f) != 4) { free(chunk_offsets); fclose(f); crm_mov_free(mov); return CRM_ERR_IO; }
            chunk_offsets[i] = (int64_t)rd_be32(buf);
        }
    }

    /* Parse stsc (sample-to-chunk mapping) */
    uint64_t stsc_sz;
    int64_t stsc = find_atom(f, stbl, stbl + (int64_t)stbl_sz, TAG('s','t','s','c'), &stsc_sz);

    mov->frame_offsets = calloc(sample_count, sizeof(int64_t));
    if (!mov->frame_offsets) { free(chunk_offsets); fclose(f); crm_mov_free(mov); return CRM_ERR_ALLOC; }

    if (stsc >= 0 && stsc_sz >= 8) {
        uint8_t stsc_hdr[8];
        fseeko(f, stsc, SEEK_SET);
        if (fread(stsc_hdr, 1, 8, f) != 8) { free(chunk_offsets); fclose(f); crm_mov_free(mov); return CRM_ERR_IO; }
        uint32_t stsc_count = rd_be32(stsc_hdr + 4);

        typedef struct { uint32_t first_chunk, spc, desc; } StscE;
        StscE *entries = malloc(stsc_count * sizeof(StscE));
        if (!entries) { free(chunk_offsets); fclose(f); crm_mov_free(mov); return CRM_ERR_ALLOC; }

        for (uint32_t i = 0; i < stsc_count; i++) {
            uint8_t buf[12];
            if (fread(buf, 1, 12, f) != 12) {
                free(entries); free(chunk_offsets); fclose(f); crm_mov_free(mov); return CRM_ERR_IO;
            }
            entries[i].first_chunk = rd_be32(buf);
            entries[i].spc = rd_be32(buf + 4);
            entries[i].desc = rd_be32(buf + 8);
        }

        uint32_t sample_idx = 0;
        for (uint32_t chunk = 0; chunk < chunk_count && sample_idx < sample_count; chunk++) {
            uint32_t spc = 1;
            for (uint32_t s = 0; s < stsc_count; s++) {
                if (chunk + 1 >= entries[s].first_chunk)
                    spc = entries[s].spc;
            }
            int64_t offset = chunk_offsets[chunk];
            for (uint32_t s = 0; s < spc && sample_idx < sample_count; s++) {
                mov->frame_offsets[sample_idx] = offset;
                offset += mov->frame_sizes[sample_idx];
                sample_idx++;
            }
        }
        free(entries);
    } else {
        for (uint32_t i = 0; i < sample_count && i < chunk_count; i++)
            mov->frame_offsets[i] = chunk_offsets[i];
    }
    free(chunk_offsets);

    /* Compute fps from stts + timescale */
    if (timescale > 0) {
        uint64_t stts_sz;
        int64_t stts = find_atom(f, stbl, stbl + (int64_t)stbl_sz, TAG('s','t','t','s'), &stts_sz);
        if (stts >= 0 && stts_sz >= 16) {
            uint8_t stts_data[16];
            fseeko(f, stts, SEEK_SET);
            if (fread(stts_data, 1, 16, f) == 16) {
                uint32_t delta = rd_be32(stts_data + 12);
                if (delta > 0)
                    mov->fps = timescale / (double)delta;
            }
        }
    }

    fclose(f);

    return CRM_OK;
}

void crm_mov_free(CrmMovInfo *mov)
{
    if (!mov) return;
    free(mov->frame_offsets);
    free(mov->frame_sizes);
    free(mov->stsd_entry);
    memset(mov, 0, sizeof(*mov));
}

int crm_mov_read_frame(const char *path, const CrmMovInfo *mov,
                       int frame_idx, uint8_t **packet_out, size_t *size_out)
{
    if (frame_idx < 0 || frame_idx >= mov->frame_count)
        return CRM_ERR_INVALID;

    FILE *f = fopen(path, "rb");
    if (!f) return CRM_ERR_IO;

    size_t pkt_size = (size_t)mov->frame_sizes[frame_idx];
    int64_t pkt_offset = mov->frame_offsets[frame_idx];

    uint8_t *pkt = malloc(pkt_size);
    if (!pkt) { fclose(f); return CRM_ERR_ALLOC; }

    fseeko(f, pkt_offset, SEEK_SET);
    if (fread(pkt, 1, pkt_size, f) != pkt_size) {
        free(pkt); fclose(f); return CRM_ERR_IO;
    }

    fclose(f);
    *packet_out = pkt;
    *size_out = pkt_size;
    return CRM_OK;
}

/* ========================================================================
 * CRX Codec Structures (ported from LibRaw crx.cpp)
 * ======================================================================== */

typedef struct {
    const uint8_t *data;
    size_t data_size;
    size_t cur_pos;
    uint32_t bit_data;
    int32_t bits_left;
} CrxBitstream;

typedef struct {
    CrxBitstream bs;
    int16_t subband_width;
    int16_t subband_height;
    int32_t rounded_bits_mask;
    int32_t rounded_bits;
    int16_t cur_line;
    int32_t *line_buf0;
    int32_t *line_buf1;
    int32_t *line_buf2;
    int32_t s_param;
    int32_t k_param;
    int32_t *param_data;
    int32_t *non_progr_data;
    int supports_partial;
} CrxBandParam;

typedef struct {
    int32_t *subband0_buf;
    int32_t *subband1_buf;
    int32_t *subband2_buf;
    int32_t *subband3_buf;
    int32_t *line_buf[8];
    int16_t cur_line;
    int16_t cur_h;
    int8_t flt_tap_h;
    int16_t height;
    int16_t width;
} CrxWaveletTransform;

typedef struct {
    CrxBandParam *band_param;
    int64_t mdat_offset;
    uint8_t *band_buf;
    uint16_t width;
    uint16_t height;
    int32_t q_param;
    int32_t k_param;
    int32_t q_step_base;
    uint32_t q_step_mult;
    int supports_partial;
    int32_t band_size;
    int64_t data_size;
    int64_t data_offset;
    short row_start_addon;
    short row_end_addon;
    short col_start_addon;
    short col_end_addon;
    short level_shift;
} CrxSubband;

typedef struct {
    uint8_t *comp_buf;
    CrxSubband *sub_bands;
    CrxWaveletTransform *wvlt_transform;
    int8_t comp_number;
    int64_t data_offset;
    int32_t comp_size;
    int supports_partial;
    int32_t rounded_bits_mask;
    int8_t tile_flag;
} CrxPlaneComp;

typedef struct {
    uint32_t *q_step_tbl;
    int width;
    int height;
} CrxQStep;

typedef struct {
    CrxPlaneComp *comps;
    int8_t tile_flag;
    int8_t tile_number;
    int64_t data_offset;
    int32_t tile_size;
    uint16_t width;
    uint16_t height;
    int has_qp_data;
    CrxQStep *q_step;
    uint32_t mdat_qp_data_size;
    uint16_t mdat_extra_size;
} CrxTile;

typedef struct {
    uint8_t n_planes;
    uint16_t plane_width;
    uint16_t plane_height;
    uint8_t sample_precision;
    uint8_t median_bits;
    uint8_t subband_count;
    uint8_t levels;
    uint8_t n_bits;
    uint8_t enc_type;
    uint8_t tile_cols;
    uint8_t tile_rows;
    CrxTile *tiles;
    uint64_t mdat_offset;
    uint64_t mdat_size;
    int16_t *out_bufs[4];
    int16_t *plane_buf;
    const uint8_t *input_data;
    size_t input_size;
} CrxImage;

enum {
    E_HAS_TILES_ON_THE_RIGHT  = 1,
    E_HAS_TILES_ON_THE_LEFT   = 2,
    E_HAS_TILES_ON_THE_BOTTOM = 4,
    E_HAS_TILES_ON_THE_TOP    = 8
};

/* Decoder context */
struct CrmDecContext {
    CrmCodecParams params;
    CrxImage image;
    int initialized;
    /* Pooled allocations */
    void *alloc_pool;
};

/* ========================================================================
 * Constant Tables
 * ======================================================================== */

static int32_t ex_coef_num_tbl[144] = {
    1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1,
    1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static int32_t crx_q_step_tbl[8] = {0x28, 0x2D, 0x33, 0x39, 0x40, 0x48, 0, 0};

static uint32_t JS[32] = {
    1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8,
    0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80,
    0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000
};

static uint32_t J_tbl[32] = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
};

/* ========================================================================
 * Bitstream Reader (in-memory buffer, replacing LibRaw's file-based I/O)
 * ======================================================================== */

static inline void crx_bs_init(CrxBitstream *bs, const uint8_t *data, size_t size)
{
    bs->data = data;
    bs->data_size = size;
    bs->cur_pos = 0;
    bs->bit_data = 0;
    bs->bits_left = 0;
}

static inline uint32_t crx_bs_read_be32(CrxBitstream *bs)
{
    uint32_t v = ((uint32_t)bs->data[bs->cur_pos] << 24) |
                 ((uint32_t)bs->data[bs->cur_pos+1] << 16) |
                 ((uint32_t)bs->data[bs->cur_pos+2] << 8) |
                  bs->data[bs->cur_pos+3];
    bs->cur_pos += 4;
    return v;
}

static inline int crx_bs_get_zeros(CrxBitstream *bs)
{
    uint32_t nonZeroBit = 0;
    int32_t result = 0;

    if (bs->bit_data) {
        nonZeroBit = 31 - __builtin_clz(bs->bit_data);
        result = 31 - nonZeroBit;
        /* Use 64-bit shift to avoid UB when nonZeroBit==0 (shift by 32) */
        bs->bit_data = (uint32_t)((uint64_t)bs->bit_data << (32 - nonZeroBit));
        bs->bits_left -= (32 - nonZeroBit);
    } else {
        uint32_t bitsLeft = bs->bits_left;
        while (1) {
            if (bs->cur_pos + 4 <= bs->data_size) {
                uint32_t nextData = crx_bs_read_be32(bs);
                if (nextData) {
                    nonZeroBit = 31 - __builtin_clz(nextData);
                    result = bitsLeft + 31 - nonZeroBit;
                    bs->bit_data = (uint32_t)((uint64_t)nextData << (32 - nonZeroBit));
                    bs->bits_left = nonZeroBit;
                    return result;
                }
                bitsLeft += 32;
            } else if (bs->cur_pos < bs->data_size) {
                uint32_t nextData = bs->data[bs->cur_pos++];
                if (nextData) {
                    nonZeroBit = 31 - __builtin_clz(nextData);
                    result = bitsLeft + 7 - nonZeroBit;
                    bs->bit_data = (uint32_t)((uint64_t)nextData << (32 - nonZeroBit));
                    bs->bits_left = nonZeroBit;
                    return result;
                }
                bitsLeft += 8;
            } else {
                return bitsLeft;
            }
        }
    }
    return result;
}

static inline uint32_t crx_bs_get_bits(CrxBitstream *bs, int bits)
{
    int bitsLeft = bs->bits_left;
    uint32_t bitData = bs->bit_data;
    uint32_t result;

    if (bitsLeft < bits) {
        if (bs->cur_pos + 4 <= bs->data_size) {
            uint32_t nextWord = crx_bs_read_be32(bs);
            bs->bits_left = 32 - (bits - bitsLeft);
            result = ((nextWord >> bitsLeft) | bitData) >> (32 - bits);
            /* Use 64-bit shift to avoid UB when bits==bitsLeft (shift by 0 is fine but bits-bitsLeft could be 32) */
            bs->bit_data = (uint32_t)((uint64_t)nextWord << (bits - bitsLeft));
            return result;
        }
        do {
            if (bs->cur_pos >= bs->data_size) break;
            bitsLeft += 8;
            bitData |= (uint32_t)bs->data[bs->cur_pos++] << (32 - bitsLeft);
        } while (bitsLeft < bits);
    }
    result = bitData >> (32 - bits);
    bs->bit_data = (uint32_t)((uint64_t)bitData << bits);
    bs->bits_left = bitsLeft - bits;
    return result;
}

/* ========================================================================
 * MED Prediction + K Parameter + Symbol Decode
 * ======================================================================== */

static inline int32_t crx_predict_k(int32_t prevK, int32_t bitCode, int32_t maxVal)
{
    /* Note: LibRaw uses  1 << prevK >> 1  which is  (1 << prevK) >> 1  due to
     * left-to-right associativity. NOT 1 << (prevK >> 1). */
    int32_t newK = prevK - (bitCode < ((1 << prevK) >> 1)) +
                   ((bitCode >> prevK) > 2) + ((bitCode >> prevK) > 5);
    return (!maxVal || newK < maxVal) ? newK : maxVal;
}

static inline void crx_decode_symbol_l1(CrxBandParam *param, int32_t doMedian, int32_t notEOL)
{
    if (doMedian) {
        int32_t delta = param->line_buf0[1] - param->line_buf0[0];
        int32_t symb[4];
        symb[2] = param->line_buf1[0];
        symb[0] = symb[1] = delta + symb[2];
        symb[3] = param->line_buf0[1];
        param->line_buf1[1] = symb[(((param->line_buf0[0] < param->line_buf1[0]) ^ (delta < 0)) << 1) +
                                    ((param->line_buf1[0] < param->line_buf0[1]) ^ (delta < 0))];
    } else {
        param->line_buf1[1] = param->line_buf0[1];
    }

    uint32_t bitCode = crx_bs_get_zeros(&param->bs);
    if (bitCode >= 41)
        bitCode = crx_bs_get_bits(&param->bs, 21);
    else if (param->k_param)
        bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);

    param->line_buf1[1] += -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);

    if (notEOL) {
        int32_t nextDelta = (param->line_buf0[2] - param->line_buf0[1]) << 1;
        bitCode = (bitCode + _abs(nextDelta)) >> 1;
        ++param->line_buf0;
    }

    param->k_param = crx_predict_k(param->k_param, bitCode, 15);
    ++param->line_buf1;
}

static inline void crx_decode_symbol_l1_rounded(CrxBandParam *param, int32_t doSym, int32_t doCode)
{
    int32_t sym = param->line_buf0[1];
    if (doSym) {
        int32_t deltaH = param->line_buf0[1] - param->line_buf0[0];
        int32_t symb[4];
        symb[2] = param->line_buf1[0];
        symb[0] = symb[1] = deltaH + symb[2];
        symb[3] = param->line_buf0[1];
        sym = symb[(((param->line_buf0[0] < param->line_buf1[0]) ^ (deltaH < 0)) << 1) +
                    ((param->line_buf1[0] < param->line_buf0[1]) ^ (deltaH < 0))];
    }

    uint32_t bitCode = crx_bs_get_zeros(&param->bs);
    if (bitCode >= 41)
        bitCode = crx_bs_get_bits(&param->bs, 21);
    else if (param->k_param)
        bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
    int32_t code = -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
    param->line_buf1[1] = param->rounded_bits_mask * 2 * code + (code >> 31) + sym;

    if (doCode) {
        if (param->line_buf0[2] > param->line_buf0[1])
            code = (param->line_buf0[2] - param->line_buf0[1] + param->rounded_bits_mask - 1) >> param->rounded_bits;
        else
            code = -((param->line_buf0[1] - param->line_buf0[2] + param->rounded_bits_mask) >> param->rounded_bits);
        param->k_param = crx_predict_k(param->k_param, (bitCode + 2 * _abs(code)) >> 1, 15);
    } else {
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
    }
    ++param->line_buf1;
}

/* ========================================================================
 * Core Line Decoders (6 variants — faithful port from LibRaw crx.cpp)
 * ======================================================================== */

/* Normal line with MED prediction + RLE */
static int crx_decode_line_core(CrxBandParam *param)
{
    int length = param->subband_width;
    param->line_buf1[0] = param->line_buf0[1];

    for (; length > 1; --length) {
        if (param->line_buf1[0] != param->line_buf0[1] ||
            param->line_buf1[0] != param->line_buf0[2]) {
            crx_decode_symbol_l1(param, 1, 1);
        } else {
            int nSyms = 0;
            if (crx_bs_get_bits(&param->bs, 1)) {
                nSyms = 1;
                while (crx_bs_get_bits(&param->bs, 1)) {
                    nSyms += JS[param->s_param];
                    if (nSyms > length) { nSyms = length; break; }
                    if (param->s_param < 31) ++param->s_param;
                    if (nSyms == length) break;
                }
                if (nSyms < length) {
                    if (J_tbl[param->s_param])
                        nSyms += crx_bs_get_bits(&param->bs, J_tbl[param->s_param]);
                    if (param->s_param > 0) --param->s_param;
                    if (nSyms > length) return -1;
                }
                length -= nSyms;
                param->line_buf0 += nSyms;
                while (nSyms-- > 0) {
                    param->line_buf1[1] = param->line_buf1[0];
                    ++param->line_buf1;
                }
            }
            if (length > 0)
                crx_decode_symbol_l1(param, 0, (length > 1));
        }
    }
    if (length == 1)
        crx_decode_symbol_l1(param, 1, 0);

    param->line_buf1[1] = param->line_buf1[0] + 1;
    return 0;
}

/* Top line (first line, no previous reference) */
static int crx_decode_top_line_core(CrxBandParam *param)
{
    param->line_buf1[0] = 0;
    int32_t length = param->subband_width;

    for (; length > 1; --length) {
        if (param->line_buf1[0])
            param->line_buf1[1] = param->line_buf1[0];
        else {
            int nSyms = 0;
            if (crx_bs_get_bits(&param->bs, 1)) {
                nSyms = 1;
                while (crx_bs_get_bits(&param->bs, 1)) {
                    nSyms += JS[param->s_param];
                    if (nSyms > length) { nSyms = length; break; }
                    if (param->s_param < 31) ++param->s_param;
                    if (nSyms == length) break;
                }
                if (nSyms < length) {
                    if (J_tbl[param->s_param])
                        nSyms += crx_bs_get_bits(&param->bs, J_tbl[param->s_param]);
                    if (param->s_param > 0) --param->s_param;
                    if (nSyms > length) return -1;
                }
                length -= nSyms;
                while (nSyms-- > 0) {
                    param->line_buf1[1] = param->line_buf1[0];
                    ++param->line_buf1;
                }
                if (length <= 0) break;
            }
            param->line_buf1[1] = 0;
        }

        uint32_t bitCode = crx_bs_get_zeros(&param->bs);
        if (bitCode >= 41)
            bitCode = crx_bs_get_bits(&param->bs, 21);
        else if (param->k_param)
            bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
        param->line_buf1[1] += -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        ++param->line_buf1;
    }
    if (length == 1) {
        param->line_buf1[1] = param->line_buf1[0];
        uint32_t bitCode = crx_bs_get_zeros(&param->bs);
        if (bitCode >= 41)
            bitCode = crx_bs_get_bits(&param->bs, 21);
        else if (param->k_param)
            bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
        param->line_buf1[1] += -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        ++param->line_buf1;
    }
    param->line_buf1[1] = param->line_buf1[0] + 1;
    return 0;
}

/* Rounded line (quantized subbands with rounding) */
static int crx_decode_line_rounded(CrxBandParam *param)
{
    int32_t valueReached = 0;
    param->line_buf0[0] = param->line_buf0[1];
    param->line_buf1[0] = param->line_buf0[1];
    int32_t length = param->subband_width;

    for (; length > 1; --length) {
        if (_abs(param->line_buf0[2] - param->line_buf0[1]) > param->rounded_bits_mask) {
            crx_decode_symbol_l1_rounded(param, 1, 1);
            ++param->line_buf0;
            valueReached = 1;
        } else if (valueReached || _abs(param->line_buf0[0] - param->line_buf1[0]) > param->rounded_bits_mask) {
            crx_decode_symbol_l1_rounded(param, 1, 1);
            ++param->line_buf0;
            valueReached = 0;
        } else {
            int nSyms = 0;
            if (crx_bs_get_bits(&param->bs, 1)) {
                nSyms = 1;
                while (crx_bs_get_bits(&param->bs, 1)) {
                    nSyms += JS[param->s_param];
                    if (nSyms > length) { nSyms = length; break; }
                    if (param->s_param < 31) ++param->s_param;
                    if (nSyms == length) break;
                }
                if (nSyms < length) {
                    if (J_tbl[param->s_param])
                        nSyms += crx_bs_get_bits(&param->bs, J_tbl[param->s_param]);
                    if (param->s_param > 0) --param->s_param;
                }
                if (nSyms > length) return -1;
            }
            length -= nSyms;
            param->line_buf0 += nSyms;
            while (nSyms-- > 0) {
                param->line_buf1[1] = param->line_buf1[0];
                ++param->line_buf1;
            }
            if (length > 1) {
                crx_decode_symbol_l1_rounded(param, 0, 1);
                ++param->line_buf0;
                valueReached = _abs(param->line_buf0[1] - param->line_buf0[0]) > param->rounded_bits_mask;
            } else if (length == 1) {
                crx_decode_symbol_l1_rounded(param, 0, 0);
            }
        }
    }
    if (length == 1)
        crx_decode_symbol_l1_rounded(param, 1, 0);

    param->line_buf1[1] = param->line_buf1[0] + 1;
    return 0;
}

/* Top line rounded */
static int crx_decode_top_line_rounded(CrxBandParam *param)
{
    param->line_buf1[0] = 0;
    int32_t length = param->subband_width;

    for (; length > 1; --length) {
        if (_abs(param->line_buf1[0]) > param->rounded_bits_mask)
            param->line_buf1[1] = param->line_buf1[0];
        else {
            int nSyms = 0;
            if (crx_bs_get_bits(&param->bs, 1)) {
                nSyms = 1;
                while (crx_bs_get_bits(&param->bs, 1)) {
                    nSyms += JS[param->s_param];
                    if (nSyms > length) { nSyms = length; break; }
                    if (param->s_param < 31) ++param->s_param;
                    if (nSyms == length) break;
                }
                if (nSyms < length) {
                    if (J_tbl[param->s_param])
                        nSyms += crx_bs_get_bits(&param->bs, J_tbl[param->s_param]);
                    if (param->s_param > 0) --param->s_param;
                    if (nSyms > length) return -1;
                }
            }
            length -= nSyms;
            while (nSyms-- > 0) {
                param->line_buf1[1] = param->line_buf1[0];
                ++param->line_buf1;
            }
            if (length <= 0) break;
            param->line_buf1[1] = 0;
        }

        uint32_t bitCode = crx_bs_get_zeros(&param->bs);
        if (bitCode >= 41)
            bitCode = crx_bs_get_bits(&param->bs, 21);
        else if (param->k_param)
            bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
        int32_t sVal = -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
        param->line_buf1[1] += param->rounded_bits_mask * 2 * sVal + (sVal >> 31);
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        ++param->line_buf1;
    }
    if (length == 1) {
        uint32_t bitCode = crx_bs_get_zeros(&param->bs);
        if (bitCode >= 41)
            bitCode = crx_bs_get_bits(&param->bs, 21);
        else if (param->k_param)
            bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
        int32_t sVal = -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
        param->line_buf1[1] += param->rounded_bits_mask * 2 * sVal + (sVal >> 31);
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        ++param->line_buf1;
    }
    param->line_buf1[1] = param->line_buf1[0] + 1;
    return 0;
}

/* Top line no-ref (progressive/partial subbands) */
static int crx_decode_top_line_no_ref(CrxBandParam *param)
{
    param->line_buf0[0] = 0;
    param->line_buf1[0] = 0;
    int32_t length = param->subband_width;

    for (; length > 1; --length) {
        if (param->line_buf1[0]) {
            uint32_t bitCode = crx_bs_get_zeros(&param->bs);
            if (bitCode >= 41)
                bitCode = crx_bs_get_bits(&param->bs, 21);
            else if (param->k_param)
                bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
            param->line_buf1[1] = -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
            param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        } else {
            int nSyms = 0;
            if (crx_bs_get_bits(&param->bs, 1)) {
                nSyms = 1;
                while (crx_bs_get_bits(&param->bs, 1)) {
                    nSyms += JS[param->s_param];
                    if (nSyms > length) { nSyms = length; break; }
                    if (param->s_param < 31) ++param->s_param;
                    if (nSyms == length) break;
                }
                if (nSyms < length) {
                    if (J_tbl[param->s_param])
                        nSyms += crx_bs_get_bits(&param->bs, J_tbl[param->s_param]);
                    if (param->s_param > 0) --param->s_param;
                    if (nSyms > length) return -1;
                }
            }
            length -= nSyms;
            while (nSyms-- > 0) {
                param->line_buf2[0] = 0;
                param->line_buf1[1] = 0;
                ++param->line_buf1;
                ++param->line_buf2;
            }
            if (length <= 0) break;

            uint32_t bitCode = crx_bs_get_zeros(&param->bs);
            if (bitCode >= 41)
                bitCode = crx_bs_get_bits(&param->bs, 21);
            else if (param->k_param)
                bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
            param->line_buf1[1] = -(int32_t)((bitCode + 1) & 1) ^ (int32_t)((bitCode + 1) >> 1);
            param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        }
        param->line_buf2[0] = param->k_param;
        ++param->line_buf2;
        ++param->line_buf1;
    }
    if (length == 1) {
        uint32_t bitCode = crx_bs_get_zeros(&param->bs);
        if (bitCode >= 41)
            bitCode = crx_bs_get_bits(&param->bs, 21);
        else if (param->k_param)
            bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
        param->line_buf1[1] = -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        param->line_buf2[0] = param->k_param;
        ++param->line_buf1;
    }
    param->line_buf1[1] = 0;
    return 0;
}

/* Normal line no-ref (index-based, matches LibRaw crxDecodeLineNoRefPrevLine) */
static int crx_decode_line_no_ref(CrxBandParam *param)
{
    int32_t i = 0;
    for (; i < param->subband_width - 1; i++) {
        if (param->line_buf0[i + 2] | param->line_buf0[i + 1] | param->line_buf1[i]) {
            uint32_t bitCode = crx_bs_get_zeros(&param->bs);
            if (bitCode >= 41)
                bitCode = crx_bs_get_bits(&param->bs, 21);
            else if (param->k_param)
                bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
            param->line_buf1[i + 1] = -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
            param->k_param = crx_predict_k(param->k_param, bitCode, 0);
            if (param->line_buf2[i + 1] - param->k_param <= 1) {
                if (param->k_param >= 15) param->k_param = 15;
            } else {
                ++param->k_param;
            }
        } else {
            int nSyms = 0;
            if (crx_bs_get_bits(&param->bs, 1)) {
                nSyms = 1;
                if (i != param->subband_width - 1) {
                    while (crx_bs_get_bits(&param->bs, 1)) {
                        nSyms += JS[param->s_param];
                        if (i + nSyms > param->subband_width) { nSyms = param->subband_width - i; break; }
                        if (param->s_param < 31) ++param->s_param;
                        if (i + nSyms == param->subband_width) break;
                    }
                    if (i + nSyms < param->subband_width) {
                        if (J_tbl[param->s_param])
                            nSyms += crx_bs_get_bits(&param->bs, J_tbl[param->s_param]);
                        if (param->s_param > 0) --param->s_param;
                    }
                    if (i + nSyms > param->subband_width)
                        return -1;
                }
            } else if (i > param->subband_width) {
                return -1;
            }

            if (nSyms > 0) {
                memset(param->line_buf1 + i + 1, 0, nSyms * sizeof(int32_t));
                memset(param->line_buf2 + i, 0, nSyms * sizeof(int32_t));
                i += nSyms;
            }

            if (i >= param->subband_width - 1) {
                if (i == param->subband_width - 1) {
                    uint32_t bitCode = crx_bs_get_zeros(&param->bs);
                    if (bitCode >= 41) bitCode = crx_bs_get_bits(&param->bs, 21);
                    else if (param->k_param) bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
                    param->line_buf1[i + 1] = -(int32_t)((bitCode + 1) & 1) ^ (int32_t)((bitCode + 1) >> 1);
                    param->k_param = crx_predict_k(param->k_param, bitCode, 15);
                    param->line_buf2[i] = param->k_param;
                }
                continue;
            } else {
                uint32_t bitCode = crx_bs_get_zeros(&param->bs);
                if (bitCode >= 41) bitCode = crx_bs_get_bits(&param->bs, 21);
                else if (param->k_param) bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
                param->line_buf1[i + 1] = -(int32_t)((bitCode + 1) & 1) ^ (int32_t)((bitCode + 1) >> 1);
                param->k_param = crx_predict_k(param->k_param, bitCode, 0);
                if (param->line_buf2[i + 1] - param->k_param <= 1) {
                    if (param->k_param >= 15) param->k_param = 15;
                } else {
                    ++param->k_param;
                }
            }
        }
        param->line_buf2[i] = param->k_param;
    }
    if (i == param->subband_width - 1) {
        int32_t bitCode = crx_bs_get_zeros(&param->bs);
        if (bitCode >= 41) bitCode = crx_bs_get_bits(&param->bs, 21);
        else if (param->k_param) bitCode = crx_bs_get_bits(&param->bs, param->k_param) | (bitCode << param->k_param);
        param->line_buf1[i + 1] = -(bitCode & 1) ^ (bitCode >> 1);
        param->k_param = crx_predict_k(param->k_param, bitCode, 15);
        param->line_buf2[i] = param->k_param;
    }
    return 0;
}

/* ========================================================================
 * Line Decoder Wrapper (manages buffer pointers, dispatches to core decoders)
 * Matches LibRaw's crxDecodeLine(CrxBandParam *param, uint8_t *bandBuf)
 * ======================================================================== */

static int crx_decode_line_wrapper(CrxBandParam *param, uint8_t *bandBuf)
{
    if (!param || !bandBuf) return -1;
    if (param->cur_line >= param->subband_height) return -1;

    int32_t lineLength = param->subband_width + 2;

    if (param->cur_line == 0) {
        param->s_param = 0;
        param->k_param = 0;
        if (param->supports_partial) {
            if (param->rounded_bits_mask <= 0) {
                param->line_buf0 = param->param_data;
                param->line_buf1 = param->line_buf0 + lineLength;
                int32_t *lineBuf = param->line_buf1 + 1;
                if (crx_decode_top_line_core(param)) return -1;
                memcpy(bandBuf, lineBuf, param->subband_width * sizeof(int32_t));
                ++param->cur_line;
            } else {
                param->rounded_bits = 1;
                if (param->rounded_bits_mask & ~1) {
                    while (param->rounded_bits_mask >> param->rounded_bits)
                        ++param->rounded_bits;
                }
                param->line_buf0 = param->param_data;
                param->line_buf1 = param->line_buf0 + lineLength;
                int32_t *lineBuf = param->line_buf1 + 1;
                if (crx_decode_top_line_rounded(param)) return -1;
                memcpy(bandBuf, lineBuf, param->subband_width * sizeof(int32_t));
                ++param->cur_line;
            }
        } else {
            param->line_buf2 = param->non_progr_data;
            param->line_buf0 = param->param_data;
            param->line_buf1 = param->line_buf0 + lineLength;
            int32_t *lineBuf = param->line_buf1 + 1;
            if (crx_decode_top_line_no_ref(param)) return -1;
            memcpy(bandBuf, lineBuf, param->subband_width * sizeof(int32_t));
            ++param->cur_line;
        }
    } else if (!param->supports_partial) {
        param->line_buf2 = param->non_progr_data;
        if (param->cur_line & 1) {
            param->line_buf1 = param->param_data;
            param->line_buf0 = param->line_buf1 + lineLength;
        } else {
            param->line_buf0 = param->param_data;
            param->line_buf1 = param->line_buf0 + lineLength;
        }
        int32_t *lineBuf = param->line_buf1 + 1;
        if (crx_decode_line_no_ref(param)) return -1;
        memcpy(bandBuf, lineBuf, param->subband_width * sizeof(int32_t));
        ++param->cur_line;
    } else if (param->rounded_bits_mask <= 0) {
        if (param->cur_line & 1) {
            param->line_buf1 = param->param_data;
            param->line_buf0 = param->line_buf1 + lineLength;
        } else {
            param->line_buf0 = param->param_data;
            param->line_buf1 = param->line_buf0 + lineLength;
        }
        int32_t *lineBuf = param->line_buf1 + 1;
        if (crx_decode_line_core(param)) return -1;
        memcpy(bandBuf, lineBuf, param->subband_width * sizeof(int32_t));
        ++param->cur_line;
    } else {
        if (param->cur_line & 1) {
            param->line_buf1 = param->param_data;
            param->line_buf0 = param->line_buf1 + lineLength;
        } else {
            param->line_buf0 = param->param_data;
            param->line_buf1 = param->line_buf0 + lineLength;
        }
        int32_t *lineBuf = param->line_buf1 + 1;
        if (crx_decode_line_rounded(param)) return -1;
        memcpy(bandBuf, lineBuf, param->subband_width * sizeof(int32_t));
        ++param->cur_line;
    }
    return 0;
}

/* ========================================================================
 * Subband Decode with Dequantization
 * ======================================================================== */

static inline int crx_get_subband_row(CrxSubband *band, int row)
{
    return row < band->row_start_addon ? 0
         : (row < band->height - band->row_end_addon ? row - band->row_start_addon
            : band->height - band->row_end_addon - band->row_start_addon - 1);
}

static int crx_decode_line_with_iquant(CrxSubband *band, CrxQStep *qStep)
{
    if (!band->data_size) {
        memset(band->band_buf, 0, band->band_size);
        return 0;
    }

    if (band->supports_partial && !qStep) {
        uint32_t bitCode = crx_bs_get_zeros(&band->band_param->bs);
        if (bitCode >= 23)
            bitCode = crx_bs_get_bits(&band->band_param->bs, 8);
        else if (band->k_param)
            bitCode = crx_bs_get_bits(&band->band_param->bs, band->k_param) | (bitCode << band->k_param);
        band->q_param += -(int32_t)(bitCode & 1) ^ (int32_t)(bitCode >> 1);
        band->k_param = crx_predict_k(band->k_param, bitCode, 0);
        if (band->k_param > 7)
            return -1;
    }

    if (crx_decode_line_wrapper(band->band_param, band->band_buf))
        return -1;

    if (band->width <= 0) return 0;

    int32_t *bandBuf = (int32_t *)band->band_buf;

    if (qStep) {
        uint32_t *qStepTbl = &qStep->q_step_tbl[qStep->width * crx_get_subband_row(band, band->band_param->cur_line - 1)];
        for (int i = 0; i < band->col_start_addon; ++i) {
            int32_t qv = band->q_step_base + ((qStepTbl[0] * band->q_step_mult) >> 3);
            bandBuf[i] *= _constrain(qv, 1, 0x168000);
        }
        for (int i = band->col_start_addon; i < band->width - band->col_end_addon; ++i) {
            int32_t qv = band->q_step_base +
                ((qStepTbl[(i - band->col_start_addon) >> band->level_shift] * band->q_step_mult) >> 3);
            bandBuf[i] *= _constrain(qv, 1, 0x168000);
        }
        int lastIdx = (band->width - band->col_end_addon - band->col_start_addon - 1) >> band->level_shift;
        for (int i = band->width - band->col_end_addon; i < band->width; ++i) {
            int32_t qv = band->q_step_base + ((qStepTbl[lastIdx] * band->q_step_mult) >> 3);
            bandBuf[i] *= _constrain(qv, 1, 0x168000);
        }
    } else {
        int32_t qScale;
        if (band->q_param / 6 >= 6)
            qScale = crx_q_step_tbl[band->q_param % 6] * (1 << ((band->q_param / 6 - 6) & 0x1f));
        else
            qScale = crx_q_step_tbl[band->q_param % 6] >> (6 - band->q_param / 6);
        if (qScale != 1)
            for (int32_t i = 0; i < band->width; ++i)
                bandBuf[i] *= qScale;
    }
    return 0;
}

/* ========================================================================
 * Wavelet Inverse Transform (Le Gall 5/3)
 * ======================================================================== */

static void crx_horizontal_53(int32_t *lineBufLA, int32_t *lineBufLB,
                               CrxWaveletTransform *w, uint32_t tileFlag)
{
    int32_t *b0 = w->subband0_buf;
    int32_t *b1 = w->subband1_buf;
    int32_t *b2 = w->subband2_buf;
    int32_t *b3 = w->subband3_buf;

    if (w->width <= 1) {
        lineBufLA[0] = b0[0];
        lineBufLB[0] = b2[0];
        return;
    }

    if (tileFlag & E_HAS_TILES_ON_THE_LEFT) {
        lineBufLA[0] = b0[0] - ((b1[0] + b1[1] + 2) >> 2);
        lineBufLB[0] = b2[0] - ((b3[0] + b3[1] + 2) >> 2);
        ++b1; ++b3;
    } else {
        lineBufLA[0] = b0[0] - ((b1[0] + 1) >> 1);
        lineBufLB[0] = b2[0] - ((b3[0] + 1) >> 1);
    }
    ++b0; ++b2;

    for (int i = 0; i < w->width - 3; i += 2) {
        int32_t dA = b0[0] - ((b1[0] + b1[1] + 2) >> 2);
        lineBufLA[1] = b1[0] + ((dA + lineBufLA[0]) >> 1);
        lineBufLA[2] = dA;
        int32_t dB = b2[0] - ((b3[0] + b3[1] + 2) >> 2);
        lineBufLB[1] = b3[0] + ((dB + lineBufLB[0]) >> 1);
        lineBufLB[2] = dB;
        ++b0; ++b1; ++b2; ++b3;
        lineBufLA += 2; lineBufLB += 2;
    }

    if (tileFlag & E_HAS_TILES_ON_THE_RIGHT) {
        int32_t dA = b0[0] - ((b1[0] + b1[1] + 2) >> 2);
        lineBufLA[1] = b1[0] + ((dA + lineBufLA[0]) >> 1);
        int32_t dB = b2[0] - ((b3[0] + b3[1] + 2) >> 2);
        lineBufLB[1] = b3[0] + ((dB + lineBufLB[0]) >> 1);
        if (w->width & 1) { lineBufLA[2] = dA; lineBufLB[2] = dB; }
    } else if (w->width & 1) {
        lineBufLA[1] = b1[0] + ((lineBufLA[0] + b0[0] - ((b1[0] + 1) >> 1)) >> 1);
        lineBufLA[2] = b0[0] - ((b1[0] + 1) >> 1);
        lineBufLB[1] = b3[0] + ((lineBufLB[0] + b2[0] - ((b3[0] + 1) >> 1)) >> 1);
        lineBufLB[2] = b2[0] - ((b3[0] + 1) >> 1);
    } else {
        lineBufLA[1] = lineBufLA[0] + b1[0];
        lineBufLB[1] = lineBufLB[0] + b3[0];
    }
}

/* Single-band horizontal 53 (L only, no H) — used in initialize edge cases */
static void crx_horizontal_53_single(int32_t *lineBuf, int32_t *band0Buf, int32_t *band1Buf,
                                      CrxWaveletTransform *w, uint32_t tileFlag)
{
    if (w->width <= 1) { lineBuf[0] = band0Buf[0]; return; }
    if (tileFlag & E_HAS_TILES_ON_THE_LEFT) {
        lineBuf[0] = band0Buf[0] - ((band1Buf[0] + band1Buf[1] + 2) >> 2);
        ++band1Buf;
    } else {
        lineBuf[0] = band0Buf[0] - ((band1Buf[0] + 1) >> 1);
    }
    ++band0Buf;
    for (int i = 0; i < w->width - 3; i += 2) {
        int32_t d = band0Buf[0] - ((band1Buf[0] + band1Buf[1] + 2) >> 2);
        lineBuf[1] = band1Buf[0] + ((d + lineBuf[0]) >> 1);
        lineBuf[2] = d;
        ++band0Buf; ++band1Buf; lineBuf += 2;
    }
    if (tileFlag & E_HAS_TILES_ON_THE_RIGHT) {
        int32_t d = band0Buf[0] - ((band1Buf[0] + band1Buf[1] + 2) >> 2);
        lineBuf[1] = band1Buf[0] + ((d + lineBuf[0]) >> 1);
        if (w->width & 1) lineBuf[2] = d;
    } else if (w->width & 1) {
        int32_t d = band0Buf[0] - ((band1Buf[0] + 1) >> 1);
        lineBuf[1] = band1Buf[0] + ((d + lineBuf[0]) >> 1);
        lineBuf[2] = d;
    } else {
        lineBuf[1] = band1Buf[0] + lineBuf[0];
    }
}

static int32_t *crx_idwt53_filter_get_line(CrxPlaneComp *comp, int32_t level)
{
    int32_t *result = comp->wvlt_transform[level]
        .line_buf[(comp->wvlt_transform[level].flt_tap_h - comp->wvlt_transform[level].cur_h + 5) % 5 + 3];
    comp->wvlt_transform[level].cur_h--;
    return result;
}

static int crx_idwt53_filter_decode(CrxPlaneComp *comp, int32_t level, CrxQStep *qStep);
static int crx_idwt53_filter_transform(CrxPlaneComp *comp, uint32_t level);

static int crx_idwt53_filter_decode(CrxPlaneComp *comp, int32_t level, CrxQStep *qStep)
{
    if (comp->wvlt_transform[level].cur_h) return 0;

    CrxSubband *sband = comp->sub_bands + 3 * level;
    CrxQStep *qStepLevel = qStep ? qStep + level : 0;

    if (comp->wvlt_transform[level].height - 3 <= comp->wvlt_transform[level].cur_line &&
        !(comp->tile_flag & E_HAS_TILES_ON_THE_BOTTOM)) {
        if (comp->wvlt_transform[level].height & 1) {
            if (level) {
                if (crx_idwt53_filter_decode(comp, level - 1, qStep)) return -1;
            } else if (crx_decode_line_with_iquant(sband, qStepLevel)) return -1;
            if (crx_decode_line_with_iquant(sband + 1, qStepLevel)) return -1;
        }
    } else {
        if (level) {
            if (crx_idwt53_filter_decode(comp, level - 1, qStep)) return -1;
        } else if (crx_decode_line_with_iquant(sband, qStepLevel)) return -1;
        if (crx_decode_line_with_iquant(sband + 1, qStepLevel) ||
            crx_decode_line_with_iquant(sband + 2, qStepLevel) ||
            crx_decode_line_with_iquant(sband + 3, qStepLevel))
            return -1;
    }
    return 0;
}

static int crx_idwt53_filter_transform(CrxPlaneComp *comp, uint32_t level)
{
    CrxWaveletTransform *w = comp->wvlt_transform + level;
    if (w->cur_h) return 0;

    if (w->cur_line >= w->height - 3) {
        if (!(comp->tile_flag & E_HAS_TILES_ON_THE_BOTTOM)) {
            if (w->height & 1) {
                if (level) {
                    if (!w[-1].cur_h && crx_idwt53_filter_transform(comp, level - 1)) return -1;
                    w->subband0_buf = crx_idwt53_filter_get_line(comp, level - 1);
                }
                int32_t *lineBufH0 = w->line_buf[w->flt_tap_h + 3];
                int32_t *lineBufH1 = w->line_buf[(w->flt_tap_h + 1) % 5 + 3];
                int32_t *lineBufH2 = w->line_buf[(w->flt_tap_h + 2) % 5 + 3];
                int32_t *lineBufL0 = w->line_buf[0];
                int32_t *lineBufL1 = w->line_buf[1];
                w->line_buf[1] = w->line_buf[2];
                w->line_buf[2] = lineBufL1;
                crx_horizontal_53_single(lineBufL0, w->subband0_buf, w->subband1_buf, w, comp->tile_flag);
                lineBufL0 = w->line_buf[0];
                lineBufL1 = w->line_buf[1];
                for (int32_t i = 0; i < w->width; i++) {
                    int32_t d = lineBufL0[i] - ((lineBufL1[i] + 1) >> 1);
                    lineBufH1[i] = lineBufL1[i] + ((d + lineBufH0[i]) >> 1);
                    lineBufH2[i] = d;
                }
                w->cur_h += 3; w->cur_line += 3;
                w->flt_tap_h = (w->flt_tap_h + 3) % 5;
            } else {
                int32_t *lineBufL2 = w->line_buf[2];
                int32_t *lineBufH0 = w->line_buf[w->flt_tap_h + 3];
                int32_t *lineBufH1 = w->line_buf[(w->flt_tap_h + 1) % 5 + 3];
                w->line_buf[1] = lineBufL2;
                w->line_buf[2] = w->line_buf[1];
                for (int32_t i = 0; i < w->width; i++)
                    lineBufH1[i] = lineBufH0[i] + lineBufL2[i];
                w->cur_h += 2; w->cur_line += 2;
                w->flt_tap_h = (w->flt_tap_h + 2) % 5;
            }
        }
    } else {
        if (level) {
            if (!w[-1].cur_h && crx_idwt53_filter_transform(comp, level - 1)) return -1;
            w->subband0_buf = crx_idwt53_filter_get_line(comp, level - 1);
        }
        int32_t *lineBufL0 = w->line_buf[0];
        int32_t *lineBufL1 = w->line_buf[1];
        int32_t *lineBufL2 = w->line_buf[2];
        int32_t *lineBufH0 = w->line_buf[w->flt_tap_h + 3];
        int32_t *lineBufH1 = w->line_buf[(w->flt_tap_h + 1) % 5 + 3];
        int32_t *lineBufH2 = w->line_buf[(w->flt_tap_h + 2) % 5 + 3];
        w->line_buf[1] = w->line_buf[2];
        w->line_buf[2] = lineBufL1;
        crx_horizontal_53(lineBufL0, lineBufL1, w, comp->tile_flag);
        lineBufL0 = w->line_buf[0];
        lineBufL1 = w->line_buf[1];
        lineBufL2 = w->line_buf[2];
        for (int32_t i = 0; i < w->width; i++) {
            int32_t d = lineBufL0[i] - ((lineBufL2[i] + lineBufL1[i] + 2) >> 2);
            lineBufH1[i] = lineBufL1[i] + ((d + lineBufH0[i]) >> 1);
            lineBufH2[i] = d;
        }
        if (w->cur_line >= w->height - 3 && (w->height & 1)) {
            w->cur_h += 3; w->cur_line += 3;
            w->flt_tap_h = (w->flt_tap_h + 3) % 5;
        } else {
            w->cur_h += 2; w->cur_line += 2;
            w->flt_tap_h = (w->flt_tap_h + 2) % 5;
        }
    }
    return 0;
}

static int crx_idwt53_filter_initialize(CrxPlaneComp *comp, int32_t level, CrxQStep *qStep)
{
    if (level == 0) return 0;

    for (int curLevel = 0, curBand = 0; curLevel < level; curLevel++, curBand += 3) {
        CrxQStep *qStepLevel = qStep ? qStep + curLevel : 0;
        CrxWaveletTransform *w = comp->wvlt_transform + curLevel;

        if (curLevel)
            w->subband0_buf = crx_idwt53_filter_get_line(comp, curLevel - 1);
        else if (crx_decode_line_with_iquant(comp->sub_bands + curBand, qStepLevel))
            return -1;

        int32_t *lineBufH0 = w->line_buf[w->flt_tap_h + 3];

        if (w->height > 1) {
            if (crx_decode_line_with_iquant(comp->sub_bands + curBand + 1, qStepLevel) ||
                crx_decode_line_with_iquant(comp->sub_bands + curBand + 2, qStepLevel) ||
                crx_decode_line_with_iquant(comp->sub_bands + curBand + 3, qStepLevel))
                return -1;

            int32_t *lineBufL0 = w->line_buf[0];
            int32_t *lineBufL2 = w->line_buf[2];

            if (comp->tile_flag & E_HAS_TILES_ON_THE_TOP) {
                crx_horizontal_53(lineBufL0, w->line_buf[1], w, comp->tile_flag);
                if (crx_decode_line_with_iquant(comp->sub_bands + curBand + 3, qStepLevel) ||
                    crx_decode_line_with_iquant(comp->sub_bands + curBand + 2, qStepLevel))
                    return -1;
                crx_horizontal_53_single(lineBufL2, w->subband2_buf, w->subband3_buf, w, comp->tile_flag);
                int32_t *L0 = w->line_buf[0];
                int32_t *L1 = w->line_buf[1];
                for (int32_t i = 0; i < w->width; i++)
                    lineBufH0[i] = L0[i] - ((L1[i] + lineBufL2[i] + 2) >> 2);
            } else {
                crx_horizontal_53(lineBufL0, lineBufL2, w, comp->tile_flag);
                int32_t *L0 = w->line_buf[0];
                for (int i = 0; i < w->width; i++)
                    lineBufH0[i] = L0[i] - ((lineBufL2[i] + 1) >> 1);
            }

            if (crx_idwt53_filter_decode(comp, curLevel, qStep) ||
                crx_idwt53_filter_transform(comp, curLevel))
                return -1;
        } else {
            if (crx_decode_line_with_iquant(comp->sub_bands + curBand + 1, qStepLevel))
                return -1;
            crx_horizontal_53_single(lineBufH0, w->subband0_buf, w->subband1_buf, w, comp->tile_flag);
            ++w->cur_line; ++w->cur_h;
            w->flt_tap_h = (w->flt_tap_h + 1) % 5;
        }
    }
    return 0;
}

/* ========================================================================
 * Plane Line Conversion (pixel clamping and optional color transform)
 * ======================================================================== */

static void crx_convert_plane_line(CrxImage *img, int imageRow, int imageCol,
                                    int plane, int32_t *lineData, int lineLength)
{
    if (lineData) {
        uint64_t rawOffset = 4 * img->plane_width * imageRow + 2 * imageCol;
        if (img->enc_type == 1) {
            int32_t maxVal = 1 << (img->n_bits - 1);
            int32_t minVal = -maxVal; --maxVal;
            for (int i = 0; i < lineLength; i++)
                img->out_bufs[plane][rawOffset + 2 * i] = _constrain(lineData[i], minVal, maxVal);
        } else if (img->enc_type == 3) {
            rawOffset = plane * img->plane_width * img->plane_height + img->plane_width * imageRow + imageCol;
            for (int i = 0; i < lineLength; i++)
                img->plane_buf[rawOffset + i] = lineData[i];
        } else if (img->n_planes == 4) {
            int32_t median = 1 << (img->n_bits - 1);
            int32_t maxVal = (1 << img->n_bits) - 1;
            for (int i = 0; i < lineLength; i++)
                img->out_bufs[plane][rawOffset + 2 * i] = _constrain(median + lineData[i], 0, maxVal);
        } else if (img->n_planes == 1) {
            int32_t maxVal = (1 << img->n_bits) - 1;
            int32_t median = 1 << (img->n_bits - 1);
            rawOffset = img->plane_width * imageRow + imageCol;
            for (int i = 0; i < lineLength; i++)
                img->out_bufs[0][rawOffset + i] = _constrain(median + lineData[i], 0, maxVal);
        }
    } else if (img->enc_type == 3 && img->plane_buf) {
        int32_t planeSize = img->plane_width * img->plane_height;
        int16_t *p0 = img->plane_buf + imageRow * img->plane_width;
        int16_t *p1 = p0 + planeSize;
        int16_t *p2 = p1 + planeSize;
        int16_t *p3 = p2 + planeSize;
        int32_t median = (1 << (img->median_bits - 1)) << 10;
        int32_t maxVal = (1 << img->median_bits) - 1;
        uint32_t rawOff = 4 * img->plane_width * imageRow;
        for (int i = 0; i < img->plane_width; i++) {
            int32_t gr = median + (p0[i] << 10) - 168 * p1[i] - 585 * p3[i];
            if (gr < 0) gr = -(((_abs(gr) + 512) >> 9) & ~1);
            else gr = ((_abs(gr) + 512) >> 9) & ~1;
            int32_t val = (median + (p0[i] << 10) + 1510 * p3[i] + 512) >> 10;
            img->out_bufs[0][rawOff + 2 * i] = _constrain(val, 0, maxVal);
            val = (p2[i] + gr + 1) >> 1;
            img->out_bufs[1][rawOff + 2 * i] = _constrain(val, 0, maxVal);
            val = (gr - p2[i] + 1) >> 1;
            img->out_bufs[2][rawOff + 2 * i] = _constrain(val, 0, maxVal);
            val = (median + (p0[i] << 10) + 1927 * p1[i] + 512) >> 10;
            img->out_bufs[3][rawOff + 2 * i] = _constrain(val, 0, maxVal);
        }
    }
}

/* ========================================================================
 * Subband Setup + Param Init
 * ======================================================================== */

static void crx_setup_subband_idx(int version, CrxSubband *band, int level,
                                   short colStartIdx, short bandWidthExCoef,
                                   short rowStartIdx, short bandHeightExCoef)
{
    if (version == 0x200) {
        band->row_start_addon = rowStartIdx;
        band->row_end_addon = bandHeightExCoef;
        band->col_start_addon = colStartIdx;
        band->col_end_addon = bandWidthExCoef;
        band->level_shift = 3 - level;
    } else {
        band->row_start_addon = 0;
        band->row_end_addon = 0;
        band->col_start_addon = 0;
        band->col_end_addon = 0;
        band->level_shift = 0;
    }
}

static int crx_process_subbands(int version, CrxImage *img, CrxTile *tile, CrxPlaneComp *comp)
{
    CrxSubband *band = comp->sub_bands + img->subband_count - 1;
    uint32_t bw = tile->width, bh = tile->height;
    int32_t bwEx = 0, bhEx = 0;

    if (img->levels) {
        int32_t *rowExCoef = ex_coef_num_tbl + 0x30 * (img->levels - 1) + 6 * (tile->width & 7);
        int32_t *colExCoef = ex_coef_num_tbl + 0x30 * (img->levels - 1) + 6 * (tile->height & 7);

        for (int level = 0; level < img->levels; ++level) {
            int32_t wOdd = bw & 1, hOdd = bh & 1;
            bw = (wOdd + bw) >> 1; bh = (hOdd + bh) >> 1;
            int32_t bwEx0 = 0, bwEx1 = 0, bhEx0 = 0, bhEx1 = 0;
            int32_t colStart = 0, rowStart = 0;

            if (tile->tile_flag & E_HAS_TILES_ON_THE_RIGHT)  { bwEx0 = rowExCoef[2*level]; bwEx1 = rowExCoef[2*level+1]; }
            if (tile->tile_flag & E_HAS_TILES_ON_THE_LEFT)   { ++bwEx0; colStart = 1; }
            if (tile->tile_flag & E_HAS_TILES_ON_THE_BOTTOM) { bhEx0 = colExCoef[2*level]; bhEx1 = colExCoef[2*level+1]; }
            if (tile->tile_flag & E_HAS_TILES_ON_THE_TOP)    { ++bhEx0; rowStart = 1; }

            band[0].width = bw + bwEx0 - wOdd;
            band[0].height = bh + bhEx0 - hOdd;
            crx_setup_subband_idx(version, band, level+1, colStart, bwEx0-colStart, rowStart, bhEx0-rowStart);

            band[-1].width = bw + bwEx1;
            band[-1].height = bh + bhEx0 - hOdd;
            crx_setup_subband_idx(version, band-1, level+1, 0, bwEx1, rowStart, bhEx0-rowStart);

            band[-2].width = bw + bwEx0 - wOdd;
            band[-2].height = bh + bhEx1;
            crx_setup_subband_idx(version, band-2, level+1, colStart, bwEx0-colStart, 0, bhEx1);

            band -= 3;
        }
        bwEx = bhEx = 0;
        if (tile->tile_flag & E_HAS_TILES_ON_THE_RIGHT)  bwEx = rowExCoef[2*img->levels - 1];
        if (tile->tile_flag & E_HAS_TILES_ON_THE_BOTTOM) bhEx = colExCoef[2*img->levels - 1];
    }
    band->width = bwEx + bw;
    band->height = bhEx + bh;
    if (img->levels)
        crx_setup_subband_idx(version, band, img->levels, 0, bwEx, 0, bhEx);

    return 0;
}

static int crx_param_init(CrxImage *img, CrxBandParam **param,
                           uint64_t subbandMdatOffset, uint64_t subbandDataSize,
                           uint32_t subbandWidth, uint32_t subbandHeight,
                           int supportsPartial, uint32_t roundedBitsMask)
{
    int32_t progrDataSize = supportsPartial ? 0 : sizeof(int32_t) * subbandWidth;
    int32_t paramLength = 2 * subbandWidth + 4;
    uint8_t *paramBuf = calloc(1, sizeof(CrxBandParam) + sizeof(int32_t) * paramLength + progrDataSize);
    if (!paramBuf) return -1;

    *param = (CrxBandParam *)paramBuf;
    paramBuf += sizeof(CrxBandParam);

    (*param)->param_data = (int32_t *)paramBuf;
    (*param)->non_progr_data = progrDataSize ? (*param)->param_data + paramLength : 0;
    (*param)->subband_width = subbandWidth;
    (*param)->subband_height = subbandHeight;
    (*param)->rounded_bits = 0;
    (*param)->cur_line = 0;
    (*param)->rounded_bits_mask = roundedBitsMask;
    (*param)->supports_partial = supportsPartial;

    /* Initialize in-memory bitstream pointing to the subband data */
    crx_bs_init(&(*param)->bs,
                img->input_data + subbandMdatOffset,
                (size_t)subbandDataSize);

    return 0;
}

static int crx_setup_subband_data(CrxImage *img, CrxPlaneComp *comp, const CrxTile *tile, uint64_t mdatOffset)
{
    long compDataSize = 0;
    long waveletDataOffset = 0;
    long compCoeffDataOffset = 0;
    int32_t toSubbands = 3 * img->levels + 1;
    CrxSubband *subbands = comp->sub_bands;

    for (int32_t i = 0; i < toSubbands; i++) {
        subbands[i].band_size = subbands[i].width * sizeof(int32_t);
        compDataSize += subbands[i].band_size;
    }

    if (img->levels) {
        waveletDataOffset = (compDataSize + 7) & ~7;
        compDataSize = (sizeof(CrxWaveletTransform) * img->levels + waveletDataOffset + 7) & ~7;
        compCoeffDataOffset = compDataSize;
        for (int level = 0; level < img->levels; ++level) {
            if (level < img->levels - 1)
                compDataSize += 8 * sizeof(int32_t) * comp->sub_bands[3 * (level + 1) + 2].width;
            else
                compDataSize += 8 * sizeof(int32_t) * tile->width;
        }
    }

    comp->comp_buf = malloc(compDataSize);
    if (!comp->comp_buf) return -1;
    memset(comp->comp_buf, 0, compDataSize);

    uint64_t subbandMdatOff = img->mdat_offset + mdatOffset;
    uint8_t *subbandBuf = comp->comp_buf;

    for (int32_t i = 0; i < toSubbands; i++) {
        subbands[i].band_buf = subbandBuf;
        subbandBuf += subbands[i].band_size;
        subbands[i].mdat_offset = subbandMdatOff + subbands[i].data_offset;
    }

    if (img->levels) {
        CrxWaveletTransform *wt = (CrxWaveletTransform *)(comp->comp_buf + waveletDataOffset);
        int32_t *pd = (int32_t *)(comp->comp_buf + compCoeffDataOffset);
        comp->wvlt_transform = wt;
        wt[0].subband0_buf = (int32_t *)subbands[0].band_buf;

        for (int level = 0; level < img->levels; ++level) {
            int32_t band = 3 * level + 1;
            int32_t tw;
            if (level >= img->levels - 1) { wt[level].height = tile->height; tw = tile->width; }
            else { wt[level].height = subbands[band + 3].height; tw = subbands[band + 4].width; }
            wt[level].width = tw;
            for (int b = 0; b < 8; b++) { wt[level].line_buf[b] = pd; pd += tw; }
            wt[level].cur_line = 0; wt[level].cur_h = 0; wt[level].flt_tap_h = 0;
            wt[level].subband1_buf = (int32_t *)subbands[band].band_buf;
            wt[level].subband2_buf = (int32_t *)subbands[band + 1].band_buf;
            wt[level].subband3_buf = (int32_t *)subbands[band + 2].band_buf;
        }
    }

    for (int32_t i = 0; i < toSubbands; i++) {
        if (subbands[i].data_size) {
            int supportsPartial = 0;
            uint32_t roundedBitsMask = 0;
            if (comp->supports_partial && i == 0) {
                roundedBitsMask = comp->rounded_bits_mask;
                supportsPartial = 1;
            }
            if (crx_param_init(img, &subbands[i].band_param, subbands[i].mdat_offset,
                               subbands[i].data_size, subbands[i].width, subbands[i].height,
                               supportsPartial, roundedBitsMask))
                return -1;
        }
    }
    return 0;
}

/* ========================================================================
 * Frame Header Parsing (FF01/FF02/FF03 and FF11/FF12/FF13)
 * ======================================================================== */

static int crx_read_subband_headers(CrxImage *img, CrxPlaneComp *comp,
                                     uint8_t **dataPtr, int32_t *dataSize)
{
    if (!img->subband_count) return 0;
    int32_t subbandOffset = 0;
    CrxSubband *band = comp->sub_bands;

    for (int cur = 0; cur < img->subband_count; cur++, band++) {
        if (*dataSize < 4) return -1;
        int hdrSign = rd_be16(*dataPtr);
        int hdrSize = rd_be16(*dataPtr + 2);
        if (*dataSize < hdrSize + 4) return -1;
        if ((hdrSign != 0xFF03 || hdrSize != 8) && (hdrSign != 0xFF13 || hdrSize != 16))
            return -1;

        int32_t subbandSize = (int32_t)rd_be32(*dataPtr + 4);

        if (cur != ((*dataPtr)[8] & 0xF0) >> 4) {
            band->data_size = subbandSize;
            return -1;
        }

        band->data_offset = subbandOffset;
        band->k_param = 0;
        band->band_param = 0;
        band->band_buf = 0;
        band->band_size = 0;

        if (hdrSign == 0xFF03) {
            uint32_t bitData = rd_be32(*dataPtr + 8);
            band->data_size = subbandSize - (bitData & 0x7FFFF);
            band->supports_partial = (bitData & 0x8000000) != 0;
            band->q_param = (bitData >> 19) & 0xFF;
            band->q_step_base = 0;
            band->q_step_mult = 0;
        } else {
            if (rd_be16(*dataPtr + 8) & 0xFFF) return -1;
            if (rd_be16(*dataPtr + 18)) return -1;
            band->supports_partial = 0;
            band->q_param = 0;
            band->data_size = subbandSize - rd_be16(*dataPtr + 16);
            band->q_step_base = (int32_t)rd_be32(*dataPtr + 12);
            band->q_step_mult = rd_be16(*dataPtr + 10);
        }
        subbandOffset += subbandSize;
        *dataPtr += hdrSize + 4;
        *dataSize -= hdrSize + 4;
    }
    return 0;
}

/* QP data decode functions for v0.200 */
static uint32_t crx_read_qp(CrxBitstream *bs, int32_t kParam)
{
    uint32_t qp = crx_bs_get_zeros(bs);
    if (qp >= 23) qp = crx_bs_get_bits(bs, 8);
    else if (kParam) qp = crx_bs_get_bits(bs, kParam) | (qp << kParam);
    return qp;
}

static int32_t crx_prediction_val(int32_t left, int32_t top, int32_t deltaH, int32_t deltaV)
{
    int32_t symb[4] = {left + deltaH, left + deltaH, left, top};
    return symb[(((deltaV < 0) ^ (deltaH < 0)) << 1) + ((left < top) ^ (deltaH < 0))];
}

static void crx_decode_golomb_top(CrxBitstream *bs, int32_t width, int32_t *lineBuf, int32_t *kParam)
{
    lineBuf[0] = 0;
    while (width-- > 0) {
        lineBuf[1] = lineBuf[0];
        uint32_t qp = crx_read_qp(bs, *kParam);
        lineBuf[1] += -(int32_t)(qp & 1) ^ (int32_t)(qp >> 1);
        *kParam = crx_predict_k(*kParam, qp, 7);
        ++lineBuf;
    }
    lineBuf[1] = lineBuf[0] + 1;
}

static void crx_decode_golomb_normal(CrxBitstream *bs, int32_t width,
                                      int32_t *lineBuf0, int32_t *lineBuf1, int32_t *kParam)
{
    lineBuf1[0] = lineBuf0[1];
    int32_t deltaH = lineBuf0[1] - lineBuf0[0];
    while (width-- > 0) {
        lineBuf1[1] = crx_prediction_val(lineBuf1[0], lineBuf0[1], deltaH, lineBuf0[0] - lineBuf1[0]);
        uint32_t qp = crx_read_qp(bs, *kParam);
        lineBuf1[1] += -(int32_t)(qp & 1) ^ (int32_t)(qp >> 1);
        if (width) {
            deltaH = lineBuf0[2] - lineBuf0[1];
            *kParam = crx_predict_k(*kParam, (qp + 2 * _abs(deltaH)) >> 1, 7);
            ++lineBuf0;
        } else {
            *kParam = crx_predict_k(*kParam, qp, 7);
        }
        ++lineBuf1;
    }
    lineBuf1[1] = lineBuf1[0] + 1;
}

static int crx_make_q_step(CrxImage *img, CrxTile *tile, int32_t *qpTable)
{
    if (img->levels > 3 || img->levels < 1) return -1;
    int qpW = (tile->width >> 3) + ((tile->width & 7) != 0);
    int qpH = (tile->height >> 1) + (tile->height & 1);
    int qpH4 = (tile->height >> 2) + ((tile->height & 3) != 0);
    int qpH8 = (tile->height >> 3) + ((tile->height & 7) != 0);
    uint32_t totalH = qpH;
    if (img->levels > 1) totalH += qpH4;
    if (img->levels > 2) totalH += qpH8;

    tile->q_step = malloc(totalH * qpW * sizeof(uint32_t) + img->levels * sizeof(CrxQStep));
    if (!tile->q_step) return -1;

    uint32_t *qStepTbl = (uint32_t *)(tile->q_step + img->levels);
    CrxQStep *qs = tile->q_step;

    switch (img->levels) {
    case 3:
        qs->q_step_tbl = qStepTbl; qs->width = qpW; qs->height = qpH8;
        for (int r = 0; r < qpH8; ++r) {
            int r0 = qpW * _min(4*r, qpH-1), r1 = qpW * _min(4*r+1, qpH-1);
            int r2 = qpW * _min(4*r+2, qpH-1), r3 = qpW * _min(4*r+3, qpH-1);
            for (int c = 0; c < qpW; ++c, ++qStepTbl) {
                int32_t qv = qpTable[r0++] + qpTable[r1++] + qpTable[r2++] + qpTable[r3++];
                qv = ((qv < 0) * 3 + qv) >> 2;
                *qStepTbl = (qv/6 >= 6) ? crx_q_step_tbl[qv%6] << ((qv/6-6)&0x1f) : crx_q_step_tbl[qv%6] >> (6-qv/6);
            }
        }
        ++qs; /* fall through */
    case 2:
        qs->q_step_tbl = qStepTbl; qs->width = qpW; qs->height = qpH4;
        for (int r = 0; r < qpH4; ++r) {
            int r0 = qpW * _min(2*r, qpH-1), r1 = qpW * _min(2*r+1, qpH-1);
            for (int c = 0; c < qpW; ++c, ++qStepTbl) {
                int32_t qv = (qpTable[r0++] + qpTable[r1++]) / 2;
                *qStepTbl = (qv/6 >= 6) ? crx_q_step_tbl[qv%6] << ((qv/6-6)&0x1f) : crx_q_step_tbl[qv%6] >> (6-qv/6);
            }
        }
        ++qs; /* fall through */
    case 1:
        qs->q_step_tbl = qStepTbl; qs->width = qpW; qs->height = qpH;
        for (int r = 0; r < qpH; ++r)
            for (int c = 0; c < qpW; ++c, ++qStepTbl, ++qpTable) {
                *qStepTbl = (*qpTable/6 >= 6) ? crx_q_step_tbl[*qpTable%6] << ((*qpTable/6-6)&0x1f)
                                               : crx_q_step_tbl[*qpTable%6] >> (6 - *qpTable/6);
            }
        break;
    }
    return 0;
}

static int crx_read_image_headers(int version, CrxImage *img, uint8_t *mdatPtr, int32_t mdatHdrSize)
{
    int nTiles = img->tile_rows * img->tile_cols;
    if (!nTiles) return -1;

    if (!img->tiles) {
        img->tiles = calloc(sizeof(CrxTile) * nTiles +
                            sizeof(CrxPlaneComp) * nTiles * img->n_planes +
                            sizeof(CrxSubband) * nTiles * img->n_planes * img->subband_count, 1);
        if (!img->tiles) return -1;

        CrxTile *tile = img->tiles;
        CrxPlaneComp *comps = (CrxPlaneComp *)(tile + nTiles);
        CrxSubband *bands = (CrxSubband *)(comps + img->n_planes * nTiles);

        int tileW = img->plane_width / img->tile_cols;
        int tileH = img->plane_height / img->tile_rows;
        if (tileW < 0x16) tileW = img->plane_width;
        if (tileH < 0x16) tileH = img->plane_height;

        for (int t = 0; t < nTiles; t++, tile++) {
            tile->tile_flag = 0;
            tile->tile_number = t;
            tile->tile_size = 0;
            tile->comps = comps + t * img->n_planes;

            if ((t + 1) % img->tile_cols) {
                tile->width = tileW;
                if (img->tile_cols > 1) {
                    tile->tile_flag = E_HAS_TILES_ON_THE_RIGHT;
                    if (t % img->tile_cols) tile->tile_flag |= E_HAS_TILES_ON_THE_LEFT;
                }
            } else {
                tile->width = img->plane_width - tileW * (img->tile_cols - 1);
                if (img->tile_cols > 1) tile->tile_flag = E_HAS_TILES_ON_THE_LEFT;
            }
            if (t < nTiles - img->tile_cols) {
                tile->height = tileH;
                if (img->tile_rows > 1) {
                    tile->tile_flag |= E_HAS_TILES_ON_THE_BOTTOM;
                    if (t >= img->tile_cols) tile->tile_flag |= E_HAS_TILES_ON_THE_TOP;
                }
            } else {
                tile->height = img->plane_height - tileH * (img->tile_rows - 1);
                if (img->tile_rows > 1) tile->tile_flag |= E_HAS_TILES_ON_THE_TOP;
            }

            CrxPlaneComp *comp = tile->comps;
            CrxSubband *band = bands + t * img->n_planes * img->subband_count;
            for (int c = 0; c < img->n_planes; c++, comp++) {
                comp->comp_number = c;
                comp->supports_partial = 1;
                comp->tile_flag = tile->tile_flag;
                comp->sub_bands = band;
                comp->comp_buf = 0;
                comp->wvlt_transform = 0;
                for (int b = 0; b < img->subband_count; b++, band++) {
                    band->supports_partial = 0;
                    band->q_param = 4;
                    band->band_param = 0;
                    band->data_size = 0;
                }
            }
        }
    }

    /* Parse tile/comp/subband headers */
    uint32_t tileOffset = 0;
    int32_t dataSize = mdatHdrSize;
    uint8_t *dataPtr = mdatPtr;
    CrxTile *tile = img->tiles;

    for (int t = 0; t < nTiles; ++t, ++tile) {
        if (dataSize < 4) return -1;
        int hdrSign = rd_be16(dataPtr);
        int hdrSize = rd_be16(dataPtr + 2);
        if ((hdrSign != 0xFF01 || hdrSize != 8) && (hdrSign != 0xFF11 || (hdrSize != 8 && hdrSize != 16)))
            return -1;
        if (dataSize < hdrSize + 4) return -1;
        int tailSign = rd_be16(dataPtr + 10);
        if ((hdrSize == 8 && tailSign) || (hdrSize == 16 && tailSign != 0x4000))
            return -1;
        if (rd_be16(dataPtr + 8) != (unsigned)t) return -1;

        dataSize -= hdrSize + 4;
        tile->tile_size = (int32_t)rd_be32(dataPtr + 4);
        tile->data_offset = tileOffset;
        tile->q_step = 0;

        if (hdrSize == 16) {
            if (rd_be16(dataPtr + 18) != 0) return -1;
            tile->has_qp_data = 1;
            tile->mdat_qp_data_size = rd_be32(dataPtr + 12);
            tile->mdat_extra_size = rd_be16(dataPtr + 16);
        } else {
            tile->has_qp_data = 0;
            tile->mdat_qp_data_size = 0;
            tile->mdat_extra_size = 0;
        }

        dataPtr += hdrSize + 4;
        tileOffset += tile->tile_size;

        uint32_t compOffset = 0;
        CrxPlaneComp *comp = tile->comps;
        for (int c = 0; c < img->n_planes; ++c, ++comp) {
            if (dataSize < 0xC) return -1;
            hdrSign = rd_be16(dataPtr);
            hdrSize = rd_be16(dataPtr + 2);
            if ((hdrSign != 0xFF02 && hdrSign != 0xFF12) || hdrSize != 8) return -1;
            if (c != dataPtr[8] >> 4) return -1;
            if ((rd_be16(dataPtr + 9) & 0xFFFF00) >> 8 != 0) { /* check 3 bytes at 9,10,11 are 0 */ }

            comp->comp_size = (int32_t)rd_be32(dataPtr + 4);
            int32_t compHdrRoundedBits = (dataPtr[8] >> 1) & 3;
            comp->supports_partial = (dataPtr[8] & 8) != 0;
            comp->data_offset = compOffset;
            comp->tile_flag = tile->tile_flag;
            compOffset += comp->comp_size;
            dataSize -= 0xC;
            dataPtr += 0xC;

            comp->rounded_bits_mask = 0;
            if (compHdrRoundedBits) {
                if (img->levels || !comp->supports_partial) return -1;
                comp->rounded_bits_mask = 1 << (compHdrRoundedBits - 1);
            }

            if (crx_read_subband_headers(img, comp, &dataPtr, &dataSize) ||
                crx_process_subbands(version, img, tile, comp))
                return -1;
        }
    }

    /* v0.200: decode QP data for tiles */
    if (version != 0x200) return 0;

    tile = img->tiles;
    for (int t = 0; t < nTiles; ++t, ++tile) {
        if (tile->has_qp_data) {
            CrxBitstream bs;
            crx_bs_init(&bs, img->input_data + img->mdat_offset + tile->data_offset,
                        tile->mdat_qp_data_size);

            unsigned qpW = (tile->width >> 3) + ((tile->width & 7) != 0);
            unsigned qpH = (tile->height >> 1) + (tile->height & 1);
            unsigned long totalQP = qpH * qpW;

            int32_t *qpTable = calloc(totalQP + 2 * (qpW + 2), sizeof(int32_t));
            if (!qpTable) return -1;

            int32_t *qpLineBuf = qpTable + totalQP;
            int32_t *qpCur = qpTable;
            int32_t kParam = 0;

            for (unsigned r = 0; r < qpH; ++r) {
                int32_t *qpLine0 = r & 1 ? qpLineBuf + qpW + 2 : qpLineBuf;
                int32_t *qpLine1 = r & 1 ? qpLineBuf : qpLineBuf + qpW + 2;
                if (r) crx_decode_golomb_normal(&bs, qpW, qpLine0, qpLine1, &kParam);
                else   crx_decode_golomb_top(&bs, qpW, qpLine1, &kParam);
                for (unsigned c = 0; c < qpW; ++c)
                    *qpCur++ = qpLine1[c + 1] + 4;
            }

            int rc = crx_make_q_step(img, tile, qpTable);
            free(qpTable);
            if (rc) return -1;
        }
    }
    return 0;
}

/* ========================================================================
 * High-Level Decode
 * ======================================================================== */

static int crx_decode_plane(CrxImage *img, uint32_t planeNumber)
{
    int imageRow = 0;
    for (int tRow = 0; tRow < img->tile_rows; tRow++) {
        int imageCol = 0;
        for (int tCol = 0; tCol < img->tile_cols; tCol++) {
            CrxTile *tile = img->tiles + tRow * img->tile_cols + tCol;
            CrxPlaneComp *comp = tile->comps + planeNumber;
            uint64_t tileMdatOff = tile->data_offset + tile->mdat_qp_data_size +
                                   tile->mdat_extra_size + comp->data_offset;

            if (crx_setup_subband_data(img, comp, tile, tileMdatOff))
                return -1;

            if (img->levels) {
                if (crx_idwt53_filter_initialize(comp, img->levels, tile->q_step))
                    return -1;
                for (int i = 0; i < tile->height; ++i) {
                    if (crx_idwt53_filter_decode(comp, img->levels - 1, tile->q_step))
                        return -1;
                    if (crx_idwt53_filter_transform(comp, img->levels - 1))
                        return -1;
                    int32_t *lineData = crx_idwt53_filter_get_line(comp, img->levels - 1);
                    crx_convert_plane_line(img, imageRow + i, imageCol, planeNumber, lineData, tile->width);
                }
            } else {
                if (!comp->sub_bands->data_size) {
                    memset(comp->sub_bands->band_buf, 0, comp->sub_bands->band_size);
                    return 0;
                }
                for (int i = 0; i < tile->height; ++i) {
                    if (crx_decode_line_wrapper(comp->sub_bands->band_param, comp->sub_bands->band_buf))
                        return -1;
                    crx_convert_plane_line(img, imageRow + i, imageCol, planeNumber,
                                           (int32_t *)comp->sub_bands->band_buf, tile->width);
                }
            }
            imageCol += tile->width;
        }
        imageRow += img->tiles[tRow * img->tile_cols].height;
    }
    return 0;
}

static void crx_free_image_data(CrxImage *img)
{
    if (img->tiles) {
        int nTiles = img->tile_rows * img->tile_cols;
        for (int t = 0; t < nTiles; t++) {
            CrxTile *tile = &img->tiles[t];
            if (tile->comps) {
                for (int p = 0; p < img->n_planes; p++) {
                    CrxPlaneComp *comp = &tile->comps[p];
                    if (comp->comp_buf) { free(comp->comp_buf); comp->comp_buf = 0; }
                    if (comp->sub_bands) {
                        for (int b = 0; b < img->subband_count; b++) {
                            if (comp->sub_bands[b].band_param) {
                                free(comp->sub_bands[b].band_param);
                                comp->sub_bands[b].band_param = 0;
                            }
                        }
                    }
                }
            }
            if (tile->q_step) { free(tile->q_step); tile->q_step = 0; }
        }
        free(img->tiles); img->tiles = 0;
    }
    if (img->plane_buf) { free(img->plane_buf); img->plane_buf = 0; }
}

/* ========================================================================
 * Public API
 * ======================================================================== */

CrmDecContext *crm_dec_alloc(void)
{
    return calloc(1, sizeof(CrmDecContext));
}

int crm_dec_init(CrmDecContext *ctx, const CrmCodecParams *params)
{
    if (!ctx) return CRM_ERR_INVALID;
    ctx->params = *params;
    ctx->initialized = 1;
    return CRM_OK;
}

void crm_dec_free(CrmDecContext *ctx)
{
    if (!ctx) return;
    crx_free_image_data(&ctx->image);
    free(ctx);
}

int crm_dec_decode_frame(CrmDecContext *ctx,
                         const uint8_t *packet, size_t packet_size,
                         uint16_t *output, int out_stride)
{
    if (!ctx || !ctx->initialized) return CRM_ERR_INVALID;

    CrmCodecParams *hdr = &ctx->params;
    int pw = hdr->f_width, ph = hdr->f_height;

    int IncrBitTable[16] = {0,0,0,0, 0,0,0,0, 0,1,1,0, 0,0,1,0};

    CrxImage *img = &ctx->image;
    crx_free_image_data(img);
    memset(img, 0, sizeof(*img));

    img->plane_width = pw;
    img->plane_height = ph;
    img->tile_cols = hdr->has_tile_cols ? (pw + hdr->tile_width/2 - 1) / (hdr->tile_width/2) : 1;
    img->tile_rows = hdr->has_tile_rows ? (ph + hdr->tile_height/2 - 1) / (hdr->tile_height/2) : 1;

    /* If tile dims provided in CMP1, compute tile grid properly.
     * CMP1 tile dimensions are per-plane (same coordinate space as f_width/f_height). */
    {
        int tw = hdr->tile_width, th = hdr->tile_height;
        if (tw > 0 && tw < pw) img->tile_cols = (pw + tw - 1) / tw;
        else img->tile_cols = 1;
        if (th > 0 && th < ph) img->tile_rows = (ph + th - 1) / th;
        else img->tile_rows = 1;
    }

    img->levels = hdr->image_levels;
    img->subband_count = 3 * img->levels + 1;
    img->n_planes = hdr->n_planes;
    img->n_bits = hdr->n_bits;
    img->enc_type = hdr->enc_type;
    img->sample_precision = hdr->n_bits + IncrBitTable[4 * hdr->enc_type + 2] + 1;
    img->mdat_offset = hdr->mdat_hdr_size;
    img->mdat_size = packet_size;
    img->median_bits = hdr->median_bits;
    img->input_data = packet;
    img->input_size = packet_size;
    img->tiles = 0;
    img->plane_buf = 0;

    /* Allocate output plane buffers */
    int16_t *outBuf = (int16_t *)output;
    int32_t rowSize = 2 * pw;

    if (img->n_planes == 1) {
        img->out_bufs[0] = outBuf;
    } else {
        switch (hdr->cfa_layout) {
        case 0: /* RGGB */
            img->out_bufs[0] = outBuf;
            img->out_bufs[1] = outBuf + 1;
            img->out_bufs[2] = outBuf + rowSize;
            img->out_bufs[3] = outBuf + rowSize + 1;
            break;
        case 1: /* GRBG */
            img->out_bufs[1] = outBuf;
            img->out_bufs[0] = outBuf + 1;
            img->out_bufs[3] = outBuf + rowSize;
            img->out_bufs[2] = outBuf + rowSize + 1;
            break;
        default: /* fallback to RGGB */
            img->out_bufs[0] = outBuf;
            img->out_bufs[1] = outBuf + 1;
            img->out_bufs[2] = outBuf + rowSize;
            img->out_bufs[3] = outBuf + rowSize + 1;
            break;
        }
    }

    /* Allocate plane buffer for encType==3 */
    if (img->enc_type == 3 && img->n_planes == 4 && img->n_bits > 8) {
        img->plane_buf = malloc(ph * pw * img->n_planes * ((img->sample_precision + 7) >> 3));
        if (!img->plane_buf) return CRM_ERR_ALLOC;
    }

    /* Read frame headers */
    if ((int)packet_size < hdr->mdat_hdr_size)
        return CRM_ERR_INVALID;

    if (crx_read_image_headers(hdr->version, img, (uint8_t *)packet, hdr->mdat_hdr_size))
        return CRM_ERR_BITSTREAM;

    /* Decode all planes */
    for (int plane = 0; plane < img->n_planes; ++plane) {
        if (crx_decode_plane(img, plane))
            return CRM_ERR_BITSTREAM;
    }

    /* encType==3: finalize color transform */
    if (img->enc_type == 3) {
        for (int row = 0; row < ph; ++row)
            crx_convert_plane_line(img, row, 0, 0, NULL, 0);
    }

    /* Convert from CRX int16 output to uint16 Bayer in caller's buffer.
     * CRX outputs interleaved int16 with 2-pixel stride per plane.
     * The out_bufs already point into 'output', so data is in place.
     * Just need to expand from n_bits to 16-bit. */
    int full_w = pw * 2, full_h = ph * 2;
    if (img->n_planes == 4) {
        int shift = 16 - hdr->n_bits;
        for (int y = 0; y < full_h; y++) {
            uint16_t *row = output + y * out_stride;
            for (int x = 0; x < full_w; x++) {
                int16_t v = (int16_t)row[x];
                uint16_t u = (v < 0) ? 0 : (uint16_t)v;
                row[x] = u << shift;
            }
        }
    }

    crx_free_image_data(img);
    return CRM_OK;
}

int crm_dec_probe(const uint8_t *packet, size_t packet_size,
                  int *width, int *height)
{
    if (packet_size < 12) { *width = *height = 0; return CRM_ERR_INVALID; }
    /* Check for FF01 or FF11 tile header */
    int hdrSign = rd_be16(packet);
    if (hdrSign != 0xFF01 && hdrSign != 0xFF11) {
        *width = *height = 0;
        return CRM_ERR_INVALID;
    }
    /* Dimensions come from CMP1, not from the frame packet */
    *width = *height = 0;
    return CRM_OK;
}
