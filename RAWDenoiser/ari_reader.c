/*
 * ari_reader.c — ARRIRAW (.ari) frame reader
 *
 * Reads ARRI cinema camera raw footage stored as individual .ari frame files.
 * Format: 4096-byte header + packed 12-bit (or 13-bit) uncompressed Bayer data.
 *
 * Bit packing (from dcraw, load_flags=88):
 *   - 32-bit little-endian words
 *   - 12-bit pixels extracted MSB-first from a 64-bit accumulator
 *   - Column pair swap (col ^ 1) applied to correct sensor readout order
 *     → With col^1 swap: GRBG Bayer pattern (dcraw filters=0x61616161)
 *     → Without col^1 swap: stream order = RGGB (what our pipeline expects)
 *   We skip the col^1 swap to get RGGB directly.
 *
 * Supported cameras: ALEXA, ALEXA Mini, ALEXA LF, ALEXA Mini LF, ALEXA 65,
 *                    ALEXA 35, ALEXA 265, AMIRA
 */

#include "../include/ari_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctype.h>

/* ── Little-endian helpers ───────────────────────────────────────────── */

static inline uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline float read_f32_le(const uint8_t *p) {
    uint32_t u = read_u32_le(p);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* ── Header parser (buffer-based) ────────────────────────────────────── */

int ari_parse_header_buf(const uint8_t *hdr, AriFrameInfo *info) {
    memset(info, 0, sizeof(*info));

    /* Magic check: "ARRI" */
    if (memcmp(hdr, "ARRI", 4) != 0)
        return ARI_ERR_FMT;

    /* IDI — Image Data Information (offset 0x10) */
    info->width      = (int)read_u32_le(hdr + 0x14);
    info->height     = (int)read_u32_le(hdr + 0x18);
    info->active_x   = (int)read_u32_le(hdr + 0x24);
    info->active_y   = (int)read_u32_le(hdr + 0x28);
    info->active_w   = (int)read_u32_le(hdr + 0x2C);
    info->active_h   = (int)read_u32_le(hdr + 0x30);
    info->data_offset = read_u32_le(hdr + 0x44);
    info->data_size   = read_u32_le(hdr + 0x48);

    /* Sanity checks */
    if (info->width <= 0 || info->height <= 0 ||
        info->width > 16384 || info->height > 16384)
        return ARI_ERR_FMT;
    if (info->data_offset == 0)
        info->data_offset = 4096; /* default */

    /* Detect 12-bit vs 13-bit from data size */
    size_t expected_12 = (size_t)info->width * info->height * 3 / 2;
    info->bits_per_pixel = (info->data_size > expected_12 + 1024) ? 13 : 12;

    /* ICI — Image Characteristics Info */
    info->white_balance_kelvin = read_u32_le(hdr + 0x5C);
    info->wb_r = read_f32_le(hdr + 0x64);
    info->wb_g = read_f32_le(hdr + 0x68);
    info->wb_b = read_f32_le(hdr + 0x6C);
    info->exposure_index = read_u32_le(hdr + 0x74);
    info->black_level = read_f32_le(hdr + 0x78);
    info->white_level = read_f32_le(hdr + 0x7C);

    /* CDI — Camera Data Information */
    info->sensor_fps  = read_u32_le(hdr + 0x1A0);
    info->project_fps = read_u32_le(hdr + 0x1A4);

    /* Camera model string at offset 0x29C (up to 64 bytes) */
    memcpy(info->camera_model, hdr + 0x29C, 63);
    info->camera_model[63] = '\0';

    return ARI_OK;
}

/* ── Header parser (file-based wrapper) ──────────────────────────────── */

int ari_parse_header(const char *path, AriFrameInfo *info) {
    FILE *f = fopen(path, "rb");
    if (!f) return ARI_ERR_IO;

    uint8_t hdr[4096];
    if (fread(hdr, 1, 4096, f) != 4096) {
        fclose(f);
        return ARI_ERR_IO;
    }
    fclose(f);

    return ari_parse_header_buf(hdr, info);
}

/* ── 12-bit pixel unpacker ───────────────────────────────────────────── */
/*
 * dcraw uses a 64-bit accumulator that reads 32-bit LE words and extracts
 * 12-bit pixels MSB-first. We replicate that logic here.
 *
 * NOTE: dcraw applies col^1 swap (from load_flags=88) which converts
 * the stream's natural RGGB order into GRBG. We skip the swap so our
 * output is RGGB, matching our pipeline's expectation.
 */

static int unpack_12bit(const uint8_t *packed, size_t packed_size,
                        uint16_t *out, int w, int h, int out_stride) {
    const uint8_t *p = packed;
    const uint8_t *pend = packed + packed_size;

    uint64_t bitbuf = 0;
    int vbits = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            /* Refill: load a 32-bit LE word when we have < 12 bits */
            if (vbits < 12) {
                if (p + 4 <= pend) {
                    uint32_t word = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
                    p += 4;
                    bitbuf = (bitbuf << 32) | word;
                    vbits += 32;
                } else {
                    /* Partial read at end of file */
                    while (p < pend && vbits < 12) {
                        bitbuf = (bitbuf << 8) | *p++;
                        vbits += 8;
                    }
                    if (vbits < 12) break;
                }
            }

            /* Extract 12-bit pixel from MSB of accumulator */
            vbits -= 12;
            uint16_t val12 = (uint16_t)((bitbuf >> vbits) & 0xFFF);

            /* Expand 12-bit → 16-bit */
            out[y * out_stride + x] = val12 << 4;
        }
    }

    return ARI_OK;
}

/* ── 13-bit pixel unpacker (ALEXA 35 / ALEXA 265) ────────────────────── */

static int unpack_13bit(const uint8_t *packed, size_t packed_size,
                        uint16_t *out, int w, int h, int out_stride) {
    const uint8_t *p = packed;
    const uint8_t *pend = packed + packed_size;

    uint64_t bitbuf = 0;
    int vbits = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (vbits < 13) {
                if (p + 4 <= pend) {
                    uint32_t word = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
                    p += 4;
                    bitbuf = (bitbuf << 32) | word;
                    vbits += 32;
                } else {
                    while (p < pend && vbits < 13) {
                        bitbuf = (bitbuf << 8) | *p++;
                        vbits += 8;
                    }
                    if (vbits < 13) break;
                }
            }

            vbits -= 13;
            uint16_t val13 = (uint16_t)((bitbuf >> vbits) & 0x1FFF);

            /* Expand 13-bit → 16-bit */
            out[y * out_stride + x] = val13 << 3;
        }
    }

    return ARI_OK;
}

/* ── Read a single .ari frame ────────────────────────────────────────── */

int ari_read_frame(const char *path, const AriFrameInfo *info,
                   uint16_t *bayer_out, int out_stride) {
    FILE *f = fopen(path, "rb");
    if (!f) return ARI_ERR_IO;

    if (fseek(f, info->data_offset, SEEK_SET) != 0) {
        fclose(f);
        return ARI_ERR_IO;
    }

    uint8_t *packed = (uint8_t *)malloc(info->data_size);
    if (!packed) { fclose(f); return ARI_ERR_IO; }

    size_t nread = fread(packed, 1, info->data_size, f);
    fclose(f);

    if (nread != info->data_size) {
        free(packed);
        return ARI_ERR_IO;
    }

    int ret;
    if (info->bits_per_pixel == 13) {
        ret = unpack_13bit(packed, info->data_size,
                           bayer_out, info->width, info->height, out_stride);
    } else {
        ret = unpack_12bit(packed, info->data_size,
                           bayer_out, info->width, info->height, out_stride);
    }

    free(packed);
    return ret;
}

/* ── Directory scanning (folder of .ari files) ───────────────────────── */

static int str_ends_with_ci(const char *str, const char *suffix) {
    size_t slen = strlen(str);
    size_t xlen = strlen(suffix);
    if (slen < xlen) return 0;
    for (size_t i = 0; i < xlen; i++) {
        char c = str[slen - xlen + i];
        if (c >= 'A' && c <= 'Z') c += 32;
        char s = suffix[i];
        if (s >= 'A' && s <= 'Z') s += 32;
        if (c != s) return 0;
    }
    return 1;
}

static int natural_cmp_ari(const void *a, const void *b) {
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

static int scan_ari_directory(const char *dir_path, char ***out_list, int *out_count) {
    DIR *d = opendir(dir_path);
    if (!d) return -1;

    int capacity = 256;
    int count = 0;
    char **list = (char **)malloc(capacity * sizeof(char *));
    if (!list) { closedir(d); return -1; }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!str_ends_with_ci(ent->d_name, ".ari")) continue;

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

    qsort(list, count, sizeof(char *), natural_cmp_ari);

    *out_list = list;
    *out_count = count;
    return 0;
}

static int resolve_ari_dir(const char *path, char *dir_out, size_t dir_size) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;

    if (S_ISDIR(st.st_mode)) {
        snprintf(dir_out, dir_size, "%s", path);
    } else {
        /* Single .ari file → use its parent directory */
        const char *last_slash = strrchr(path, '/');
        if (!last_slash) return -1;
        size_t len = last_slash - path;
        if (len >= dir_size) return -1;
        memcpy(dir_out, path, len);
        dir_out[len] = '\0';
    }
    return 0;
}

/* ── Sequence reader ─────────────────────────────────────────────────── */

int ari_reader_open(AriReader *r, const char *path) {
    memset(r, 0, sizeof(*r));

    if (resolve_ari_dir(path, r->dir_path, sizeof(r->dir_path)) != 0)
        return -1;

    if (scan_ari_directory(r->dir_path, &r->file_list, &r->file_count) != 0)
        return -1;

    r->frame_count = r->file_count;

    /* Parse first frame for metadata */
    if (ari_parse_header(r->file_list[0], &r->info) != ARI_OK) {
        fprintf(stderr, "ari_reader: failed to parse first frame: %s\n",
                r->file_list[0]);
        ari_reader_close(r);
        return -1;
    }

    r->width = r->info.width;
    r->height = r->info.height;
    if (r->width < 2 || r->height < 2 || r->width > 16384 || r->height > 16384) {
        fprintf(stderr, "ari_reader: invalid dimensions %dx%d\n", r->width, r->height);
        ari_reader_close(r);
        return -1;
    }
    r->bits_per_pixel = r->info.bits_per_pixel;
    r->frames_read = 0;

    fprintf(stderr, "ari_reader: opened %d frames, %dx%d, %d-bit, "
            "camera=%s, EI=%u, WB=%uK\n",
            r->frame_count, r->width, r->height, r->bits_per_pixel,
            r->info.camera_model, r->info.exposure_index,
            r->info.white_balance_kelvin);

    return 0;
}

int ari_reader_read_frame(AriReader *r, uint16_t *bayer_out) {
    if (r->frames_read >= r->frame_count) return -1;

    int ret = ari_read_frame(r->file_list[r->frames_read], &r->info,
                             bayer_out, r->width);
    if (ret == ARI_OK)
        r->frames_read++;
    return ret;
}

void ari_reader_close(AriReader *r) {
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

/* ── Buffer-based pixel unpacker (used by MXF reader) ────────────────── */

int ari_unpack_pixels(const uint8_t *packed, size_t packed_size,
                      const AriFrameInfo *info,
                      uint16_t *bayer_out, int out_stride) {
    if (info->bits_per_pixel == 13) {
        return unpack_13bit(packed, packed_size,
                            bayer_out, info->width, info->height, out_stride);
    } else {
        return unpack_12bit(packed, packed_size,
                            bayer_out, info->width, info->height, out_stride);
    }
}

/* ── Probe functions ─────────────────────────────────────────────────── */

int ari_reader_probe_frame_count(const char *path) {
    char dir[4096];
    if (resolve_ari_dir(path, dir, sizeof(dir)) != 0) return -1;

    char **list;
    int count;
    if (scan_ari_directory(dir, &list, &count) != 0) return -1;

    for (int i = 0; i < count; i++) free(list[i]);
    free(list);
    return count;
}

int ari_reader_probe_dimensions(const char *path, int *width, int *height) {
    /* If path is a directory, find first .ari file */
    char dir[4096];
    char first_file[4096];
    struct stat st;

    if (stat(path, &st) != 0) return -1;

    if (S_ISDIR(st.st_mode)) {
        if (resolve_ari_dir(path, dir, sizeof(dir)) != 0) return -1;
        char **list;
        int count;
        if (scan_ari_directory(dir, &list, &count) != 0) return -1;
        strncpy(first_file, list[0], sizeof(first_file) - 1);
        first_file[sizeof(first_file) - 1] = '\0';
        for (int i = 0; i < count; i++) free(list[i]);
        free(list);
    } else {
        strncpy(first_file, path, sizeof(first_file) - 1);
        first_file[sizeof(first_file) - 1] = '\0';
    }

    AriFrameInfo info;
    if (ari_parse_header(first_file, &info) != ARI_OK) return -1;

    *width = info.width;
    *height = info.height;
    return 0;
}
