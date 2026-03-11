/*
 * mxf_reader.c — MXF/ARRIRAW reader
 *
 * Minimal MXF parser that extracts ARRIRAW essence frames from MXF containers.
 * Scans KLV (Key-Length-Value) structure to find picture essence elements,
 * which contain complete ARRIRAW frame data (4096-byte ARRI header + packed
 * 12-bit or 13-bit Bayer pixels).
 *
 * Supports two essence key patterns:
 *   1. Standard GC picture: 060e2b34 0102 01xx 0d01 0301 15xx xxxx
 *   2. ARRI private:        060e2b34 0102 01xx 0f01 0301 01xx xxxx
 *
 * Both patterns contain the same ARRIRAW frame data when the clip was
 * recorded as uncompressed ARRIRAW. The "ARRI" magic at the start of
 * each frame's data confirms it's ARRIRAW (not Codex-compressed).
 *
 * References:
 *   - SMPTE ST 377-1:2019 (MXF file format)
 *   - SMPTE RDD 54:2022 (ARRIRAW in MXF generic container)
 *   - FFmpeg libavformat/mxfdec.c
 */

#include "../include/mxf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── MXF Universal Label constants ───────────────────────────────────── */

/* All SMPTE ULs start with this 4-byte prefix */
static const uint8_t MXF_UL_PREFIX[4] = { 0x06, 0x0E, 0x2B, 0x34 };

/* Partition Pack: first 13 bytes (byte 13 = type: 02=header, 03=body, 04=footer) */
static const uint8_t PARTITION_PACK_PREFIX[13] = {
    0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01,
    0x0D, 0x01, 0x02, 0x01, 0x01
};

/* ── Big-endian helpers ──────────────────────────────────────────────── */

static inline uint32_t read_u32_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static inline uint64_t read_u64_be(const uint8_t *p) {
    return ((uint64_t)read_u32_be(p) << 32) | read_u32_be(p + 4);
}

/* ── BER length decoder ──────────────────────────────────────────────── */

static int64_t decode_ber_length(FILE *f, int *len_size) {
    uint8_t first;
    if (fread(&first, 1, 1, f) != 1) return -1;

    if (first == 0x80) {
        *len_size = 1;
        return -1; /* indefinite length — unsupported */
    }

    if (!(first & 0x80)) {
        *len_size = 1;
        return (int64_t)first;
    }

    int nbytes = first & 0x7F;
    if (nbytes > 8) return -1;

    *len_size = 1 + nbytes;
    int64_t length = 0;
    for (int i = 0; i < nbytes; i++) {
        uint8_t b;
        if (fread(&b, 1, 1, f) != 1) return -1;
        length = (length << 8) | b;
    }

    return length;
}

/* ── Essence key matching ────────────────────────────────────────────── */

/*
 * Check if a 16-byte key is a potential ARRIRAW essence element.
 *
 * Standard GC picture:  06 0E 2B 34 01 02 01 xx 0D 01 03 01 15 xx xx xx
 * ARRI private essence: 06 0E 2B 34 01 02 01 xx 0F 01 03 01 xx xx xx xx
 *
 * We match flexibly on byte 7 (version) and bytes 12-15 (element details).
 */
static int is_arriraw_essence_key(const uint8_t *key) {
    /* Must start with SMPTE UL prefix */
    if (memcmp(key, MXF_UL_PREFIX, 4) != 0) return 0;

    /* Byte 4-5 must be 01 02 (Label, specific registry) */
    if (key[4] != 0x01 || key[5] != 0x02) return 0;

    /* Byte 6 must be 01 */
    if (key[6] != 0x01) return 0;

    /* Byte 7: version — accept any (0x01, 0x0D, etc.) */

    /* Check for standard GC picture: bytes 8-11 = 0D 01 03 01, byte 12 = 0x15 */
    if (key[8] == 0x0D && key[9] == 0x01 && key[10] == 0x03 && key[11] == 0x01
        && key[12] == 0x15) {
        return 1;
    }

    /* Check for ARRI private: bytes 8-11 = 0F 01 03 01 */
    if (key[8] == 0x0F && key[9] == 0x01 && key[10] == 0x03 && key[11] == 0x01) {
        return 1;
    }

    return 0;
}

static int is_partition_pack_key(const uint8_t *key) {
    return memcmp(key, PARTITION_PACK_PREFIX, 13) == 0;
}

/* ── Partition pack parser ───────────────────────────────────────────── */

typedef struct {
    uint32_t kag_size;
    int64_t  header_byte_count;
    int64_t  index_byte_count;
    uint32_t body_sid;
    int64_t  body_offset;
} MxfPartitionInfo;

static int parse_partition_pack(FILE *f, int64_t value_len, MxfPartitionInfo *p) {
    if (value_len < 88) return -1;
    uint8_t buf[88];
    if (fread(buf, 1, 88, f) != 88) return -1;

    p->kag_size          = read_u32_be(buf + 4);
    p->header_byte_count = (int64_t)read_u64_be(buf + 32);
    p->index_byte_count  = (int64_t)read_u64_be(buf + 40);
    p->body_sid          = read_u32_be(buf + 60);
    p->body_offset       = (int64_t)read_u64_be(buf + 52);

    return 0;
}

/* ── KLV scanner ─────────────────────────────────────────────────────── */

/*
 * Scan an MXF file for ARRIRAW essence elements.
 *
 * Strategy:
 *   1. Parse partition packs to find body partitions (BodySID != 0)
 *   2. Skip header metadata (HeaderByteCount) and index (IndexByteCount)
 *   3. Within body essence, look for picture essence KLVs
 *   4. Verify ARRIRAW by checking for "ARRI" magic at start of value
 */
static int scan_mxf_frames(FILE *f, int64_t file_size,
                            int64_t **out_offsets, int64_t **out_sizes,
                            int *out_count) {
    int capacity = 1024;
    int count = 0;
    int64_t *offsets = (int64_t *)malloc(capacity * sizeof(int64_t));
    int64_t *sizes   = (int64_t *)malloc(capacity * sizeof(int64_t));
    if (!offsets || !sizes) {
        free(offsets);
        free(sizes);
        return -1;
    }

    fseeko(f, 0, SEEK_SET);

    while (1) {
        int64_t klv_start = ftello(f);
        if (klv_start < 0 || klv_start >= file_size - 20) break;

        /* Read 16-byte key */
        uint8_t key[16];
        if (fread(key, 1, 16, f) != 16) break;

        /* All valid MXF KLVs start with 06 0E 2B 34 */
        if (memcmp(key, MXF_UL_PREFIX, 4) != 0) {
            /* Scan forward byte-by-byte for next UL prefix (max 4KB) */
            int found = 0;
            int64_t limit = klv_start + 4096;
            if (limit > file_size - 16) limit = file_size - 16;
            for (int64_t scan = klv_start + 1; scan < limit; scan++) {
                fseeko(f, scan, SEEK_SET);
                uint8_t probe[4];
                if (fread(probe, 1, 4, f) != 4) break;
                if (memcmp(probe, MXF_UL_PREFIX, 4) == 0) {
                    fseeko(f, scan, SEEK_SET);
                    found = 1;
                    break;
                }
            }
            if (!found) break;
            continue;
        }

        /* Decode BER length */
        int len_size;
        int64_t value_len = decode_ber_length(f, &len_size);
        if (value_len < 0) {
            fseeko(f, klv_start + 1, SEEK_SET);
            continue;
        }

        int64_t value_start = klv_start + 16 + len_size;
        int64_t next_klv = value_start + value_len;

        /* Check for partition pack — parse to get body info */
        if (is_partition_pack_key(key)) {
            MxfPartitionInfo part;
            fseeko(f, value_start, SEEK_SET);
            if (parse_partition_pack(f, value_len, &part) == 0) {
                /* Skip header/index metadata to reach essence */
                int64_t skip = part.header_byte_count + part.index_byte_count;
                if (skip > 0) {
                    int64_t essence_start = next_klv + skip;
                    /* Align to KAG if needed */
                    if (part.kag_size > 1) {
                        essence_start = ((essence_start + part.kag_size - 1)
                                        / part.kag_size) * part.kag_size;
                    }
                    /* Don't skip past essence if body partition has data */
                    if (part.body_sid != 0 && essence_start < file_size) {
                        fseeko(f, essence_start, SEEK_SET);
                        continue;
                    }
                }
            }
            /* Fall through to skip to next_klv */
        }

        /* Check if this is an ARRIRAW essence element */
        if (is_arriraw_essence_key(key) && value_len > 4096) {
            /* Verify it contains ARRIRAW data by checking for "ARRI" magic */
            uint8_t magic[4];
            fseeko(f, value_start, SEEK_SET);
            if (fread(magic, 1, 4, f) == 4 && memcmp(magic, "ARRI", 4) == 0) {
                /* Found an ARRIRAW frame! */
                if (count >= capacity) {
                    capacity *= 2;
                    int64_t *tmp_o = (int64_t *)realloc(offsets, capacity * sizeof(int64_t));
                    int64_t *tmp_s = (int64_t *)realloc(sizes, capacity * sizeof(int64_t));
                    if (!tmp_o || !tmp_s) {
                        free(tmp_o ? tmp_o : offsets);
                        free(tmp_s ? tmp_s : sizes);
                        return -1;
                    }
                    offsets = tmp_o;
                    sizes = tmp_s;
                }
                offsets[count] = value_start;
                sizes[count] = value_len;
                count++;
            }
        }

        /* Skip to next KLV */
        if (next_klv > file_size) break;
        fseeko(f, next_klv, SEEK_SET);
    }

    if (count == 0) {
        free(offsets);
        free(sizes);
        return -1;
    }

    *out_offsets = offsets;
    *out_sizes = sizes;
    *out_count = count;
    return 0;
}

/* ── Public API ──────────────────────────────────────────────────────── */

int mxf_is_mxf_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint8_t key[16];
    int is_mxf = 0;
    if (fread(key, 1, 16, f) == 16) {
        if (is_partition_pack_key(key))
            is_mxf = 1;
    }
    fclose(f);
    return is_mxf;
}

int mxf_reader_open(MxfReader *r, const char *path) {
    memset(r, 0, sizeof(*r));
    strncpy(r->file_path, path, sizeof(r->file_path) - 1);

    r->file = fopen(path, "rb");
    if (!r->file) {
        fprintf(stderr, "mxf_reader: cannot open %s\n", path);
        return MXF_ERR_IO;
    }

    /* Get file size */
    fseeko(r->file, 0, SEEK_END);
    int64_t file_size = ftello(r->file);
    fseeko(r->file, 0, SEEK_SET);

    fprintf(stderr, "mxf_reader: scanning %s (%.1f MB)...\n",
            path, file_size / (1024.0 * 1024.0));

    /* Scan for ARRIRAW frames */
    if (scan_mxf_frames(r->file, file_size,
                        &r->frame_offsets, &r->frame_sizes,
                        &r->frame_count) != 0) {
        fprintf(stderr, "mxf_reader: no uncompressed ARRIRAW frames in %s\n", path);
        fprintf(stderr, "mxf_reader: This MXF likely uses Codex HDE compression "
                "(lossless, proprietary).\n");
        fprintf(stderr, "mxf_reader: Use DaVinci Resolve to transcode to "
                "ARRIRAW (.ari) or CinemaDNG (.dng) first.\n");
        fclose(r->file);
        r->file = NULL;
        return MXF_ERR_FMT;
    }

    /* Parse first frame's ARRI header for metadata */
    fseeko(r->file, r->frame_offsets[0], SEEK_SET);
    uint8_t hdr[4096];
    if (fread(hdr, 1, 4096, r->file) != 4096) {
        fprintf(stderr, "mxf_reader: failed to read first frame header\n");
        mxf_reader_close(r);
        return MXF_ERR_IO;
    }

    if (ari_parse_header_buf(hdr, &r->info) != ARI_OK) {
        fprintf(stderr, "mxf_reader: failed to parse ARRI header\n");
        mxf_reader_close(r);
        return MXF_ERR_FMT;
    }

    r->width = r->info.width;
    r->height = r->info.height;
    r->bits_per_pixel = r->info.bits_per_pixel;
    r->frames_read = 0;

    fprintf(stderr, "mxf_reader: %d frames, %dx%d, %d-bit, camera=%s, "
            "EI=%u, WB=%uK\n",
            r->frame_count, r->width, r->height, r->bits_per_pixel,
            r->info.camera_model, r->info.exposure_index,
            r->info.white_balance_kelvin);

    return MXF_OK;
}

int mxf_reader_read_frame(MxfReader *r, uint16_t *bayer_out) {
    if (!r->file || r->frames_read >= r->frame_count) return -1;

    int idx = r->frames_read;
    int64_t frame_offset = r->frame_offsets[idx];

    /* Read the ARRI header for this frame */
    fseeko(r->file, frame_offset, SEEK_SET);
    uint8_t hdr[4096];
    if (fread(hdr, 1, 4096, r->file) != 4096) return -1;

    AriFrameInfo frame_info;
    if (ari_parse_header_buf(hdr, &frame_info) != ARI_OK) return -1;

    /* Read packed pixel data */
    fseeko(r->file, frame_offset + frame_info.data_offset, SEEK_SET);
    uint8_t *packed = (uint8_t *)malloc(frame_info.data_size);
    if (!packed) return -1;

    if (fread(packed, 1, frame_info.data_size, r->file) != frame_info.data_size) {
        free(packed);
        return -1;
    }

    int ret = ari_unpack_pixels(packed, frame_info.data_size, &frame_info,
                                bayer_out, r->width);
    free(packed);

    if (ret == ARI_OK)
        r->frames_read++;

    return ret;
}

void mxf_reader_close(MxfReader *r) {
    if (r->file) {
        fclose(r->file);
        r->file = NULL;
    }
    free(r->frame_offsets);
    r->frame_offsets = NULL;
    free(r->frame_sizes);
    r->frame_sizes = NULL;
    r->frame_count = 0;
    r->frames_read = 0;
}

/* ── Probe functions ─────────────────────────────────────────────────── */

int mxf_reader_probe_frame_count(const char *path) {
    MxfReader r;
    if (mxf_reader_open(&r, path) != MXF_OK) return -1;
    int count = r.frame_count;
    mxf_reader_close(&r);
    return count;
}

int mxf_reader_probe_dimensions(const char *path, int *width, int *height) {
    MxfReader r;
    if (mxf_reader_open(&r, path) != MXF_OK) return -1;
    *width = r.width;
    *height = r.height;
    mxf_reader_close(&r);
    return 0;
}
