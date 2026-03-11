#include "../include/mov_reader.h"
#include "prores_raw_dec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Big-endian helpers for MOV atom parsing --- */
static uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

static float read_be_float(const uint8_t *p) {
    uint32_t bits = read_be32(p);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Find atom by 4-char type within a buffer. Returns pointer to atom start. */
static const uint8_t *find_atom(const uint8_t *buf, size_t len, const char *type) {
    size_t pos = 0;
    while (pos + 8 <= len) {
        uint32_t size = read_be32(buf + pos);
        if (size < 8 || pos + size > len) break;
        if (memcmp(buf + pos + 4, type, 4) == 0)
            return buf + pos;
        pos += size;
    }
    return NULL;
}

static uint64_t read_be64(const uint8_t *p) {
    return ((uint64_t)read_be32(p) << 32) | (uint64_t)read_be32(p + 4);
}

/* Walk atom tree in a file to find child atom by 4-char tag within [start, end). */
static long find_atom_file(FILE *f, long start, long end, const char *tag) {
    long pos = start;
    uint8_t h[8];
    while (pos < end - 7) {
        fseek(f, pos, SEEK_SET);
        if (fread(h, 1, 8, f) != 8) return -1;
        uint32_t sz = read_be32(h);
        if (sz < 8 || pos + (long)sz > end) return -1;
        if (memcmp(h + 4, tag, 4) == 0) return pos;
        pos += sz;
    }
    return -1;
}

/* Try to open MOV file natively (no FFmpeg subprocess).
 * Parses moov→trak→mdia→minf→stbl to get frame offsets/sizes.
 * Returns 0 on success, -1 on failure (caller should fall back to FFmpeg). */
static int mov_reader_open_native(MovReader *r) {
    FILE *f = fopen(r->filename, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);

    /* Find moov atom */
    long moov = find_atom_file(f, 0, file_size, "moov");
    if (moov < 0) { fclose(f); return -1; }
    uint8_t h[8];
    fseek(f, moov, SEEK_SET); fread(h, 1, 8, f);
    long moov_end = moov + (long)read_be32(h);

    /* Find first trak */
    long trak = find_atom_file(f, moov + 8, moov_end, "trak");
    if (trak < 0) { fclose(f); return -1; }
    fseek(f, trak, SEEK_SET); fread(h, 1, 8, f);
    long trak_end = trak + (long)read_be32(h);

    /* trak → mdia */
    long mdia = find_atom_file(f, trak + 8, trak_end, "mdia");
    if (mdia < 0) { fclose(f); return -1; }
    fseek(f, mdia, SEEK_SET); fread(h, 1, 8, f);
    long mdia_end = mdia + (long)read_be32(h);

    /* mdia → minf */
    long minf = find_atom_file(f, mdia + 8, mdia_end, "minf");
    if (minf < 0) { fclose(f); return -1; }
    fseek(f, minf, SEEK_SET); fread(h, 1, 8, f);
    long minf_end = minf + (long)read_be32(h);

    /* minf → stbl */
    long stbl = find_atom_file(f, minf + 8, minf_end, "stbl");
    if (stbl < 0) { fclose(f); return -1; }
    fseek(f, stbl, SEEK_SET); fread(h, 1, 8, f);
    long stbl_end = stbl + (long)read_be32(h);

    /* ---- Read stsd for dimensions and codec verification ---- */
    long stsd = find_atom_file(f, stbl + 8, stbl_end, "stsd");
    if (stsd >= 0) {
        /* stsd: [4:size][4:'stsd'][4:ver/flags][4:count] = 16 bytes header */
        fseek(f, stsd + 16, SEEK_SET);
        /* Entry: [4:entry_size][4:codec][6:reserved][2:dri]
         *        [2:ver][2:rev][4:vendor][4:tq][4:sq][2:width][2:height] = 36 bytes */
        uint8_t entry[36];
        if (fread(entry, 1, 36, f) == 36) {
            r->width  = (int)((entry[32] << 8) | entry[33]);
            r->height = (int)((entry[34] << 8) | entry[35]);
        }
    }
    if (r->width <= 0 || r->height <= 0) {
        fclose(f);
        return -1;
    }

    /* ---- Read stsz for per-frame compressed sizes ---- */
    long stsz = find_atom_file(f, stbl + 8, stbl_end, "stsz");
    if (stsz < 0) { fclose(f); return -1; }

    uint8_t stsz_hdr[12];
    fseek(f, stsz + 8, SEEK_SET); /* skip atom size+tag */
    if (fread(stsz_hdr, 1, 12, f) != 12) { fclose(f); return -1; }
    /* [4:ver/flags][4:sample_size][4:count] */
    uint32_t uniform_size = read_be32(stsz_hdr + 4);
    uint32_t sample_count = read_be32(stsz_hdr + 8);
    if (sample_count == 0 || sample_count > 1000000) { fclose(f); return -1; }

    r->frame_count = (int)sample_count;
    r->frame_sizes = (uint32_t *)malloc(sample_count * sizeof(uint32_t));
    if (!r->frame_sizes) { fclose(f); return -1; }

    if (uniform_size > 0) {
        /* All frames are the same size */
        for (uint32_t i = 0; i < sample_count; i++)
            r->frame_sizes[i] = uniform_size;
    } else {
        /* Variable sizes — read per-sample table */
        uint8_t *sz_buf = (uint8_t *)malloc(sample_count * 4);
        if (!sz_buf) { free(r->frame_sizes); r->frame_sizes = NULL; fclose(f); return -1; }
        if (fread(sz_buf, 4, sample_count, f) != sample_count) {
            free(sz_buf); free(r->frame_sizes); r->frame_sizes = NULL; fclose(f); return -1;
        }
        for (uint32_t i = 0; i < sample_count; i++)
            r->frame_sizes[i] = read_be32(sz_buf + i * 4);
        free(sz_buf);
    }

    /* ---- Read co64 or stco for frame offsets ---- */
    r->frame_offsets = (uint64_t *)malloc(sample_count * sizeof(uint64_t));
    if (!r->frame_offsets) {
        free(r->frame_sizes); r->frame_sizes = NULL;
        fclose(f); return -1;
    }

    long co64 = find_atom_file(f, stbl + 8, stbl_end, "co64");
    if (co64 >= 0) {
        uint8_t co64_hdr[8];
        fseek(f, co64 + 8, SEEK_SET);
        if (fread(co64_hdr, 1, 8, f) != 8) goto fail_offsets;
        uint32_t off_count = read_be32(co64_hdr + 4);
        if (off_count != sample_count) goto fail_offsets;

        uint8_t *off_buf = (uint8_t *)malloc(sample_count * 8);
        if (!off_buf) goto fail_offsets;
        if (fread(off_buf, 8, sample_count, f) != sample_count) {
            free(off_buf); goto fail_offsets;
        }
        for (uint32_t i = 0; i < sample_count; i++)
            r->frame_offsets[i] = read_be64(off_buf + i * 8);
        free(off_buf);
    } else {
        long stco = find_atom_file(f, stbl + 8, stbl_end, "stco");
        if (stco < 0) goto fail_offsets;

        uint8_t stco_hdr[8];
        fseek(f, stco + 8, SEEK_SET);
        if (fread(stco_hdr, 1, 8, f) != 8) goto fail_offsets;
        uint32_t off_count = read_be32(stco_hdr + 4);
        if (off_count != sample_count) goto fail_offsets;

        uint8_t *off_buf = (uint8_t *)malloc(sample_count * 4);
        if (!off_buf) goto fail_offsets;
        if (fread(off_buf, 4, sample_count, f) != sample_count) {
            free(off_buf); goto fail_offsets;
        }
        for (uint32_t i = 0; i < sample_count; i++)
            r->frame_offsets[i] = (uint64_t)read_be32(off_buf + i * 4);
        free(off_buf);
    }

    /* Allocate reusable compressed frame buffer (size of largest frame) */
    uint32_t max_size = 0;
    for (uint32_t i = 0; i < sample_count; i++)
        if (r->frame_sizes[i] > max_size) max_size = r->frame_sizes[i];

    r->comp_buf = (uint8_t *)malloc(max_size);
    r->comp_buf_size = (int)max_size;
    if (!r->comp_buf) goto fail_offsets;

    r->file = f;
    r->frames_read = 0;

    fprintf(stderr, "mov_reader: native open %s — %dx%d, %d frames (max frame %u bytes)\n",
            r->filename, r->width, r->height, r->frame_count, max_size);
    return 0;

fail_offsets:
    free(r->frame_offsets); r->frame_offsets = NULL;
    free(r->frame_sizes);   r->frame_sizes = NULL;
    fclose(f);
    return -1;
}

/* Full path to Homebrew ffmpeg/ffprobe — GUI apps don't inherit shell PATH */
#define FFPROBE "/opt/homebrew/bin/ffprobe"
#define FFMPEG  "/opt/homebrew/bin/ffmpeg"

int mov_reader_probe_frame_count(const char *filename) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             FFPROBE " -v error -select_streams v:0 "
             "-count_packets -show_entries stream=nb_read_packets "
             "-of csv=p=0 \"%s\"", filename);

    FILE *p = popen(cmd, "r");
    if (!p) return -1;

    char buf[256];
    int count = -1;
    if (fgets(buf, sizeof(buf), p)) {
        count = atoi(buf);
    }
    pclose(p);
    return count;
}

int mov_reader_probe_dimensions(const char *filename, int *width, int *height) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             FFPROBE " -v error -select_streams v:0 "
             "-show_entries stream=width,height "
             "-of csv=p=0:s=x \"%s\"", filename);

    FILE *p = popen(cmd, "r");
    if (!p) return -1;

    char buf[256];
    if (fgets(buf, sizeof(buf), p)) {
        if (sscanf(buf, "%dx%d", width, height) != 2) {
            pclose(p);
            return -1;
        }
    } else {
        pclose(p);
        return -1;
    }
    pclose(p);
    return 0;
}

int mov_reader_open(MovReader *r, const char *filename) {
    memset(r, 0, sizeof(*r));
    strncpy(r->filename, filename, sizeof(r->filename) - 1);

    /* Try native in-process reader first (no FFmpeg subprocess) */
    if (mov_reader_open_native(r) == 0) {
        return 0;
    }

    fprintf(stderr, "mov_reader: native open failed, falling back to FFmpeg\n");

    /* Probe dimensions */
    if (mov_reader_probe_dimensions(filename, &r->width, &r->height) != 0) {
        fprintf(stderr, "mov_reader: failed to probe dimensions from %s\n", filename);
        return -1;
    }

    /* Probe frame count */
    r->frame_count = mov_reader_probe_frame_count(filename);
    if (r->frame_count <= 0) {
        fprintf(stderr, "mov_reader: failed to probe frame count from %s\n", filename);
        return -1;
    }

    fprintf(stderr, "mov_reader: %s — %dx%d, %d frames (FFmpeg pipe)\n",
            filename, r->width, r->height, r->frame_count);

    /* Open FFmpeg decode pipe */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             FFMPEG " -v error -i \"%s\" -f rawvideo -pix_fmt bayer_rggb16le pipe:1",
             filename);

    r->pipe = popen(cmd, "r");
    if (!r->pipe) {
        fprintf(stderr, "mov_reader: failed to open ffmpeg pipe\n");
        return -1;
    }

    r->frames_read = 0;
    return 0;
}

int mov_reader_read_frame(MovReader *r, uint16_t *bayer_out) {
    /* Native decode path: fseek + fread compressed + decode in-process */
    if (r->file) {
        if (r->frames_read >= r->frame_count) return -1; /* EOF */

        uint32_t comp_size = r->frame_sizes[r->frames_read];
        uint64_t offset    = r->frame_offsets[r->frames_read];

        fseek(r->file, (long)offset, SEEK_SET);
        size_t rd = fread(r->comp_buf, 1, comp_size, r->file);
        if (rd != comp_size) {
            fprintf(stderr, "mov_reader: native short read at frame %d (got %zu of %u)\n",
                    r->frames_read, rd, comp_size);
            return -1;
        }

        if (prores_raw_decode_frame(r->comp_buf, (int)comp_size,
                                     bayer_out, r->width, r->height) != 0) {
            fprintf(stderr, "mov_reader: native decode failed at frame %d\n", r->frames_read);
            return -1;
        }

        r->frames_read++;
        return 0;
    }

    /* FFmpeg pipe fallback */
    if (!r->pipe) return -1;

    size_t frame_bytes = (size_t)r->width * r->height * 2;
    size_t rd = fread(bayer_out, 1, frame_bytes, r->pipe);

    if (rd != frame_bytes) {
        if (rd == 0) return -1;  /* clean EOF */
        fprintf(stderr, "mov_reader: short read at frame %d (got %zu of %zu bytes)\n",
                r->frames_read, rd, frame_bytes);
        return -1;
    }

    r->frames_read++;
    return 0;
}

void mov_reader_close(MovReader *r) {
    if (r->file) {
        fclose(r->file);
        r->file = NULL;
    }
    if (r->pipe) {
        pclose(r->pipe);
        r->pipe = NULL;
    }
    free(r->frame_sizes);   r->frame_sizes = NULL;
    free(r->frame_offsets); r->frame_offsets = NULL;
    free(r->comp_buf);      r->comp_buf = NULL;
    r->comp_buf_size = 0;
}

int mov_reader_probe_wb_gains(const char *filename, float *r_gain, float *b_gain) {
    *r_gain = 1.0f;
    *b_gain = 1.0f;

    FILE *f = fopen(filename, "rb");
    if (!f) return -1;

    /* Walk top-level atoms to find 'moov' */
    uint8_t hdr[8];
    long moov_off = -1;
    uint32_t moov_size = 0;

    while (fread(hdr, 1, 8, f) == 8) {
        uint32_t sz = read_be32(hdr);
        if (sz < 8) break;
        if (memcmp(hdr + 4, "moov", 4) == 0) {
            moov_off = ftell(f) - 8;
            moov_size = sz;
            break;
        }
        if (fseek(f, (long)sz - 8, SEEK_CUR) != 0) break;
    }

    if (moov_off < 0 || moov_size < 16 || moov_size > 16 * 1024 * 1024) {
        fclose(f);
        return -1;
    }

    uint8_t *moov = malloc(moov_size);
    if (!moov) { fclose(f); return -1; }
    fseek(f, moov_off, SEEK_SET);
    if (fread(moov, 1, moov_size, f) != moov_size) {
        free(moov); fclose(f); return -1;
    }
    fclose(f);

    /* moov → meta (directly or via udta) */
    const uint8_t *meta = find_atom(moov + 8, moov_size - 8, "meta");
    if (!meta) {
        const uint8_t *udta = find_atom(moov + 8, moov_size - 8, "udta");
        if (udta) {
            uint32_t udta_sz = read_be32(udta);
            meta = find_atom(udta + 8, udta_sz - 8, "meta");
        }
    }
    if (!meta) { free(moov); return -1; }
    uint32_t meta_size = read_be32(meta);
    if (meta_size < 16) { free(moov); return -1; }

    /* Atomos ProRes RAW: meta children at offset 8 (no version/flags).
     * Standard QuickTime: version/flags at 8, children at 12.
     * Auto-detect by checking where valid child atoms appear. */
    int mc_off = 8;
    if (!find_atom(meta + 8, meta_size - 8, "hdlr") &&
        !find_atom(meta + 8, meta_size - 8, "keys"))
        mc_off = 12;
    const uint8_t *mc = meta + mc_off;
    size_t mc_len = meta_size - mc_off;

    const uint8_t *keys = find_atom(mc, mc_len, "keys");
    const uint8_t *ilst = find_atom(mc, mc_len, "ilst");
    if (!keys || !ilst) { free(moov); return -1; }

    uint32_t keys_size = read_be32(keys);
    uint32_t key_count = read_be32(keys + 12);

    /* Find WB factors key index (1-based) */
    static const char wb_key[] =
        "com.apple.proresraw.whitebalance.bycct.whitebalancefactors";
    int wb_idx = -1;
    size_t kp = 16; /* past size(4) + type(4) + ver/flags(4) + count(4) */
    for (uint32_t i = 0; i < key_count && kp + 8 <= keys_size; i++) {
        uint32_t ksz = read_be32(keys + kp);
        if (ksz < 8) break;
        size_t slen = ksz - 8;
        if (slen == sizeof(wb_key) - 1 &&
            memcmp(keys + kp + 8, wb_key, slen) == 0)
            wb_idx = (int)(i + 1);
        kp += ksz;
    }
    if (wb_idx < 0) { free(moov); return -1; }

    /* Find matching ilst entry */
    uint32_t ilst_size = read_be32(ilst);
    const uint8_t *wb_payload = NULL;
    size_t wb_len = 0;
    size_t ip = 8;
    while (ip + 8 <= ilst_size) {
        uint32_t isz = read_be32(ilst + ip);
        if (isz < 8) break;
        uint32_t idx = read_be32(ilst + ip + 4);
        if ((int)idx == wb_idx && isz > 24) {
            /* [8:entry_hdr][8:data_hdr][4:type][4:locale][payload] */
            wb_payload = ilst + ip + 24;
            wb_len = isz - 24;
            break;
        }
        ip += isz;
    }
    if (!wb_payload || wb_len < 16) { free(moov); return -1; }

    /* Parse WB table: [2:count | 2:version] then entries of [4:CCT][4:R/G][4:B/G] */
    int n_entries = (read_be32(wb_payload) >> 16) & 0xFFFF;
    if (n_entries < 1 || n_entries > 32) { free(moov); return -1; }

    struct { uint32_t cct; float rg, bg; } ent[32];
    for (int i = 0; i < n_entries && 4 + (i + 1) * 12 <= (int)wb_len; i++) {
        const uint8_t *e = wb_payload + 4 + i * 12;
        ent[i].cct = read_be32(e);
        ent[i].rg  = read_be_float(e + 4);
        ent[i].bg  = read_be_float(e + 8);
    }

    /* Interpolate at 5000K (typical daylight) */
    uint32_t target = 5000;
    if (target <= ent[0].cct) {
        *r_gain = ent[0].rg;  *b_gain = ent[0].bg;
    } else if (target >= ent[n_entries - 1].cct) {
        *r_gain = ent[n_entries - 1].rg;  *b_gain = ent[n_entries - 1].bg;
    } else {
        for (int i = 0; i < n_entries - 1; i++) {
            if (ent[i].cct <= target && ent[i + 1].cct >= target) {
                float t = (float)(target - ent[i].cct) /
                          (float)(ent[i + 1].cct - ent[i].cct);
                *r_gain = ent[i].rg + t * (ent[i + 1].rg - ent[i].rg);
                *b_gain = ent[i].bg + t * (ent[i + 1].bg - ent[i].bg);
                break;
            }
        }
    }

    fprintf(stderr, "mov_reader: WB gains at %dK — R/G=%.3f, B/G=%.3f\n",
            target, *r_gain, *b_gain);
    free(moov);
    return 0;
}
