#include "../include/frame_reader.h"
#include "../include/mov_reader.h"
#include "../include/dng_reader.h"
#include "../include/braw_dec.h"
#include "../include/crm_dec.h"
#include "../include/ari_reader.h"
#include "../include/mxf_reader.h"
#include "../include/r3d_reader.h"
#include "../include/cineform_dec.h"
#include "../include/zraw_dec.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctype.h>

/* ---- Format detection ---- */

static int str_ends_with_lower(const char *str, const char *suffix) {
    size_t slen = strlen(str);
    size_t xlen = strlen(suffix);
    if (slen < xlen) return 0;
    for (size_t i = 0; i < xlen; i++) {
        char c = str[slen - xlen + i];
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c != suffix[i]) return 0;
    }
    return 1;
}

/* Peek at first file in a directory to distinguish .ari vs .dng sequences */
static InputFormat detect_dir_format(const char *dir_path) {
    DIR *d = opendir(dir_path);
    if (!d) return FORMAT_CINEMADNG; /* fallback */

    InputFormat result = FORMAT_CINEMADNG; /* default */
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue; /* skip hidden */
        if (str_ends_with_lower(ent->d_name, ".ari")) {
            result = FORMAT_ARRIRAW;
            break;
        }
        if (str_ends_with_lower(ent->d_name, ".dng")) {
            result = FORMAT_CINEMADNG;
            break;
        }
    }
    closedir(d);
    return result;
}

/* Probe MOV codec FourCC to distinguish ProRes RAW vs CineForm.
 * Scans first 64KB and last 64KB (moov may be at end for streaming MOV). */
static int mov_probe_cfhd(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;

    uint8_t buf[65536];

    /* Scan first 64KB */
    size_t n = fread(buf, 1, sizeof(buf), fp);
    for (size_t i = 0; i + 3 < n; i++) {
        if (buf[i] == 'C' && buf[i+1] == 'F' && buf[i+2] == 'H' && buf[i+3] == 'D')
            { fclose(fp); return 1; }
    }

    /* Scan last 64KB (moov-at-end layout) */
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    if (file_size > (long)sizeof(buf)) {
        long tail_off = file_size - (long)sizeof(buf);
        fseek(fp, tail_off, SEEK_SET);
        n = fread(buf, 1, sizeof(buf), fp);
        for (size_t i = 0; i + 3 < n; i++) {
            if (buf[i] == 'C' && buf[i+1] == 'F' && buf[i+2] == 'H' && buf[i+3] == 'D')
                { fclose(fp); return 1; }
        }
    }

    fclose(fp);
    return 0;
}

/* Probe MOV codec FourCC to distinguish ProRes RAW vs ZRAW.
   Scans top-level atoms to find moov, then searches within it for 'zraw' FourCC. */
static int mov_probe_zraw(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;

    /* First try quick scan of first 64KB */
    uint8_t buf[65536];
    size_t n = fread(buf, 1, sizeof(buf), fp);
    for (size_t i = 0; i + 3 < n; i++) {
        if (buf[i] == 'z' && buf[i+1] == 'r' && buf[i+2] == 'a' && buf[i+3] == 'w') {
            fclose(fp);
            return 1;
        }
    }

    /* Quick scan failed — walk top-level atoms to find moov */
    fseeko(fp, 0, SEEK_END);
    int64_t file_size = ftello(fp);
    int64_t pos = 0;
    while (pos < file_size - 8) {
        uint8_t hdr[8];
        fseeko(fp, pos, SEEK_SET);
        if (fread(hdr, 1, 8, fp) != 8) break;
        uint32_t sz = ((uint32_t)hdr[0]<<24)|((uint32_t)hdr[1]<<16)|((uint32_t)hdr[2]<<8)|hdr[3];
        if (sz < 8) break;
        if (hdr[4]=='m' && hdr[5]=='o' && hdr[6]=='o' && hdr[7]=='v') {
            /* Read first 512KB of moov and search for 'zraw' */
            size_t to_read = (sz < 524288) ? sz : 524288;
            uint8_t *moov = (uint8_t *)malloc(to_read);
            if (!moov) break;
            fseeko(fp, pos, SEEK_SET);
            size_t got = fread(moov, 1, to_read, fp);
            for (size_t i = 0; i + 3 < got; i++) {
                if (moov[i]=='z' && moov[i+1]=='r' && moov[i+2]=='a' && moov[i+3]=='w') {
                    free(moov);
                    fclose(fp);
                    return 1;
                }
            }
            free(moov);
            break;
        }
        pos += sz;
    }

    fclose(fp);
    return 0;
}

InputFormat detect_input_format(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0)
        return FORMAT_UNKNOWN;

    /* Directory → check contents for .ari or .dng files */
    if (S_ISDIR(st.st_mode))
        return detect_dir_format(path);

    /* File extension */
    if (str_ends_with_lower(path, ".mov")) {
        if (mov_probe_cfhd(path))
            return FORMAT_CINEFORM;
        if (mov_probe_zraw(path))
            return FORMAT_ZRAW;
        return FORMAT_MOV;
    }
    if (str_ends_with_lower(path, ".dng"))
        return FORMAT_CINEMADNG;
    if (str_ends_with_lower(path, ".braw"))
        return FORMAT_BRAW;
    if (str_ends_with_lower(path, ".crm"))
        return FORMAT_CRM;
    if (str_ends_with_lower(path, ".ari"))
        return FORMAT_ARRIRAW;
    if (str_ends_with_lower(path, ".zraw"))
        return FORMAT_ZRAW;
    if (str_ends_with_lower(path, ".mxf"))
        return FORMAT_MXF;
    if (str_ends_with_lower(path, ".r3d"))
        return FORMAT_R3D;
    if (str_ends_with_lower(path, ".nraw"))
        return FORMAT_R3D;

    return FORMAT_UNKNOWN;
}

/* ---- MovReader wrapper functions ---- */

static int mov_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    MovReader *mr = (MovReader *)r->impl;
    int ret = mov_reader_read_frame(mr, bayer_out);
    if (ret == 0) r->frames_read = mr->frames_read;
    return ret;
}

static void mov_close_wrapper(FrameReader *r) {
    MovReader *mr = (MovReader *)r->impl;
    mov_reader_close(mr);
    free(mr);
    r->impl = NULL;
}

/* ---- DngReader wrapper functions ---- */

static int dng_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    DngReader *dr = (DngReader *)r->impl;
    int ret = dng_reader_read_frame(dr, bayer_out);
    if (ret == 0) r->frames_read = dr->frames_read;
    return ret;
}

static void dng_close_wrapper(FrameReader *r) {
    DngReader *dr = (DngReader *)r->impl;
    dng_reader_close(dr);
    free(dr);
    r->impl = NULL;
}

/* ---- BrawReader wrapper ---- */

typedef struct {
    char path[4096];
    BrawMovInfo mov;
    BrawDecContext dec;
    int width, height;
    int frame_count;
    int frames_read;
} BrawReader;

static int braw_reader_open(BrawReader *br, const char *path) {
    memset(br, 0, sizeof(*br));
    strncpy(br->path, path, sizeof(br->path) - 1);

    if (braw_mov_parse(path, &br->mov) != BRAW_OK)
        return -1;

    if (braw_dec_init(&br->dec) != BRAW_OK) {
        braw_mov_free(&br->mov);
        return -1;
    }

    /* Probe dimensions from first frame */
    uint8_t *pkt;
    size_t pkt_size;
    if (braw_mov_read_frame(path, &br->mov, 0, &pkt, &pkt_size) != BRAW_OK) {
        braw_dec_free(&br->dec);
        braw_mov_free(&br->mov);
        return -1;
    }
    if (braw_dec_probe(pkt, pkt_size, &br->width, &br->height) != BRAW_OK) {
        free(pkt);
        braw_dec_free(&br->dec);
        braw_mov_free(&br->mov);
        return -1;
    }
    free(pkt);

    br->frame_count = br->mov.frame_count;
    br->frames_read = 0;
    return 0;
}

static int braw_reader_read_frame(BrawReader *br, uint16_t *bayer_out) {
    if (br->frames_read >= br->frame_count) return -1;

    uint8_t *pkt;
    size_t pkt_size;
    if (braw_mov_read_frame(br->path, &br->mov, br->frames_read, &pkt, &pkt_size) != BRAW_OK)
        return -1;

    int ret = braw_dec_decode_frame(&br->dec, pkt, pkt_size, bayer_out, br->width, NULL);
    free(pkt);
    if (ret != BRAW_OK) return -1;

    br->frames_read++;
    return 0;
}

static void braw_reader_close(BrawReader *br) {
    braw_dec_free(&br->dec);
    braw_mov_free(&br->mov);
}

static int braw_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    BrawReader *br = (BrawReader *)r->impl;
    int ret = braw_reader_read_frame(br, bayer_out);
    if (ret == 0) r->frames_read = br->frames_read;
    return ret;
}

static void braw_close_wrapper(FrameReader *r) {
    BrawReader *br = (BrawReader *)r->impl;
    braw_reader_close(br);
    free(br);
    r->impl = NULL;
}

/* ---- CrmReader wrapper ---- */

typedef struct {
    char path[4096];
    CrmMovInfo mov;
    CrmDecContext *dec;
    int width, height;
    int frame_count;
    int frames_read;
} CrmReader;

static int crm_reader_open(CrmReader *cr, const char *path) {
    memset(cr, 0, sizeof(*cr));
    strncpy(cr->path, path, sizeof(cr->path) - 1);

    if (crm_mov_parse(path, &cr->mov) != CRM_OK)
        return -1;

    cr->dec = crm_dec_alloc();
    if (!cr->dec) {
        crm_mov_free(&cr->mov);
        return -1;
    }

    if (crm_dec_init(cr->dec, &cr->mov.codec) != CRM_OK) {
        crm_dec_free(cr->dec);
        crm_mov_free(&cr->mov);
        return -1;
    }

    cr->width = cr->mov.width;
    cr->height = cr->mov.height;
    cr->frame_count = cr->mov.frame_count;
    cr->frames_read = 0;
    return 0;
}

static int crm_reader_read_frame(CrmReader *cr, uint16_t *bayer_out) {
    if (cr->frames_read >= cr->frame_count) return -1;

    uint8_t *pkt;
    size_t pkt_size;
    if (crm_mov_read_frame(cr->path, &cr->mov, cr->frames_read, &pkt, &pkt_size) != CRM_OK)
        return -1;

    int ret = crm_dec_decode_frame(cr->dec, pkt, pkt_size, bayer_out, cr->width);
    free(pkt);
    if (ret != CRM_OK) return -1;

    cr->frames_read++;
    return 0;
}

static void crm_reader_close(CrmReader *cr) {
    crm_dec_free(cr->dec);
    crm_mov_free(&cr->mov);
}

static int crm_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    CrmReader *cr = (CrmReader *)r->impl;
    int ret = crm_reader_read_frame(cr, bayer_out);
    if (ret == 0) r->frames_read = cr->frames_read;
    return ret;
}

static void crm_close_wrapper(FrameReader *r) {
    CrmReader *cr = (CrmReader *)r->impl;
    crm_reader_close(cr);
    free(cr);
    r->impl = NULL;
}

/* ---- AriReader wrapper ---- */

static int ari_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    AriReader *ar = (AriReader *)r->impl;
    int ret = ari_reader_read_frame(ar, bayer_out);
    if (ret == 0) r->frames_read = ar->frames_read;
    return ret;
}

static void ari_close_wrapper(FrameReader *r) {
    AriReader *ar = (AriReader *)r->impl;
    ari_reader_close(ar);
    free(ar);
    r->impl = NULL;
}

/* ---- MxfReader wrapper ---- */

static int mxf_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    MxfReader *mr = (MxfReader *)r->impl;
    int ret = mxf_reader_read_frame(mr, bayer_out);
    if (ret == 0) r->frames_read = mr->frames_read;
    return ret;
}

static void mxf_close_wrapper(FrameReader *r) {
    MxfReader *mr = (MxfReader *)r->impl;
    mxf_reader_close(mr);
    free(mr);
    r->impl = NULL;
}

/* ---- R3dReader wrapper ---- */

static int r3d_read_frame_rgb_wrapper(FrameReader *r, uint16_t *rgb_out) {
    R3dReader *rr = (R3dReader *)r->impl;
    int ret = r3d_reader_read_frame_rgb(rr, r->frames_read, rgb_out);
    if (ret == R3D_OK) r->frames_read++;
    return (ret == R3D_OK) ? 0 : -1;
}

static void r3d_close_wrapper(FrameReader *r) {
    R3dReader *rr = (R3dReader *)r->impl;
    r3d_reader_close(rr);
    r->impl = NULL;
}

/* ---- CfReader wrapper ---- */

static int cf_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    CfReader *cr = (CfReader *)r->impl;
    int ret = cf_reader_read_frame(cr, r->frames_read, bayer_out);
    if (ret == CF_OK) r->frames_read++;
    return (ret == CF_OK) ? 0 : -1;
}

static int cf_read_frame_rgb_wrapper(FrameReader *r, uint16_t *rgb_out) {
    CfReader *cr = (CfReader *)r->impl;
    int ret = cf_reader_read_frame_rgb(cr, r->frames_read, rgb_out);
    if (ret == CF_OK) r->frames_read++;
    return (ret == CF_OK) ? 0 : -1;
}

static void cf_close_wrapper(FrameReader *r) {
    CfReader *cr = (CfReader *)r->impl;
    cf_reader_close(cr);
    r->impl = NULL;
}

/* ---- ZrawReader wrapper ---- */

static int zraw_read_frame_wrapper(FrameReader *r, uint16_t *bayer_out) {
    ZrawReader *zr = (ZrawReader *)r->impl;
    int ret = zraw_reader_read_frame(zr, r->frames_read, bayer_out);
    if (ret == ZRAW_OK) r->frames_read++;
    return (ret == ZRAW_OK) ? 0 : -1;
}

static void zraw_close_wrapper(FrameReader *r) {
    ZrawReader *zr = (ZrawReader *)r->impl;
    zraw_reader_close(zr);
    r->impl = NULL;
}

/* ---- Public API ---- */

int frame_reader_open(FrameReader *r, const char *path) {
    memset(r, 0, sizeof(*r));

    InputFormat fmt = detect_input_format(path);
    r->format = fmt;

    switch (fmt) {
    case FORMAT_MOV: {
        MovReader *mr = (MovReader *)calloc(1, sizeof(MovReader));
        if (!mr) return -1;
        if (mov_reader_open(mr, path) != 0) { free(mr); return -1; }
        r->impl = mr;
        r->width = mr->width;
        r->height = mr->height;
        r->frame_count = mr->frame_count;
        r->frames_read = 0;
        r->read_frame = mov_read_frame_wrapper;
        r->close = mov_close_wrapper;
        return 0;
    }
    case FORMAT_CINEMADNG: {
        DngReader *dr = (DngReader *)calloc(1, sizeof(DngReader));
        if (!dr) return -1;
        if (dng_reader_open(dr, path) != 0) { free(dr); return -1; }
        r->impl = dr;
        r->width = dr->width;
        r->height = dr->height;
        r->frame_count = dr->frame_count;
        r->frames_read = 0;
        r->read_frame = dng_read_frame_wrapper;
        r->close = dng_close_wrapper;
        return 0;
    }
    case FORMAT_BRAW: {
        BrawReader *br = (BrawReader *)calloc(1, sizeof(BrawReader));
        if (!br) return -1;
        if (braw_reader_open(br, path) != 0) { free(br); return -1; }
        r->impl = br;
        r->width = br->width;
        r->height = br->height;
        r->frame_count = br->frame_count;
        r->frames_read = 0;
        r->read_frame = braw_read_frame_wrapper;
        r->close = braw_close_wrapper;
        return 0;
    }
    case FORMAT_CRM: {
        CrmReader *cr = (CrmReader *)calloc(1, sizeof(CrmReader));
        if (!cr) return -1;
        if (crm_reader_open(cr, path) != 0) { free(cr); return -1; }
        r->impl = cr;
        r->width = cr->width;
        r->height = cr->height;
        r->frame_count = cr->frame_count;
        r->frames_read = 0;
        r->read_frame = crm_read_frame_wrapper;
        r->close = crm_close_wrapper;
        return 0;
    }
    case FORMAT_ARRIRAW: {
        AriReader *ar = (AriReader *)calloc(1, sizeof(AriReader));
        if (!ar) return -1;
        if (ari_reader_open(ar, path) != 0) { free(ar); return -1; }
        r->impl = ar;
        r->width = ar->width;
        r->height = ar->height;
        r->frame_count = ar->frame_count;
        r->frames_read = 0;
        r->read_frame = ari_read_frame_wrapper;
        r->close = ari_close_wrapper;
        return 0;
    }
    case FORMAT_MXF: {
        MxfReader *mr = (MxfReader *)calloc(1, sizeof(MxfReader));
        if (!mr) return -1;
        if (mxf_reader_open(mr, path) != MXF_OK) { free(mr); return -1; }
        r->impl = mr;
        r->width = mr->width;
        r->height = mr->height;
        r->frame_count = mr->frame_count;
        r->frames_read = 0;
        r->read_frame = mxf_read_frame_wrapper;
        r->close = mxf_close_wrapper;
        return 0;
    }
    case FORMAT_R3D: {
        R3dReader *rr = NULL;
        if (r3d_reader_open(&rr, path) != R3D_OK) return -1;
        R3dInfo info;
        if (r3d_reader_get_info(rr, &info) != R3D_OK) {
            r3d_reader_close(rr);
            return -1;
        }
        r->impl = rr;
        r->width = info.width;
        r->height = info.height;
        r->frame_count = info.frame_count;
        r->frames_read = 0;
        r->is_rgb = 1;
        r->fps = info.fps;
        r->read_frame = NULL;  /* Bayer read not supported for R3D */
        r->read_frame_rgb = r3d_read_frame_rgb_wrapper;
        r->close = r3d_close_wrapper;
        return 0;
    }
    case FORMAT_CINEFORM: {
        CfReader *cr = NULL;
        if (cf_reader_open(&cr, path) != CF_OK) return -1;
        CfInfo info;
        if (cf_reader_get_info(cr, &info) != CF_OK) {
            cf_reader_close(cr);
            return -1;
        }
        r->impl = cr;
        r->width = info.width;
        r->height = info.height;
        r->frame_count = info.frame_count;
        r->frames_read = 0;
        r->fps = info.fps;
        r->close = cf_close_wrapper;
        if (info.is_bayer) {
            r->read_frame = cf_read_frame_wrapper;
        } else {
            /* YUV/RGB CineForm (GoPro): route through RGB denoise pipeline */
            r->is_rgb = 1;
            r->read_frame = NULL;
            r->read_frame_rgb = cf_read_frame_rgb_wrapper;
        }
        return 0;
    }
    case FORMAT_ZRAW: {
        ZrawReader *zr = NULL;
        if (zraw_reader_open(&zr, path) != ZRAW_OK) return -1;
        ZrawInfo info;
        if (zraw_reader_get_info(zr, &info) != ZRAW_OK) {
            zraw_reader_close(zr);
            return -1;
        }
        r->impl = zr;
        r->width = info.width;
        r->height = info.height;
        r->frame_count = info.frame_count;
        r->frames_read = 0;
        r->fps = info.fps;
        r->read_frame = zraw_read_frame_wrapper;
        r->close = zraw_close_wrapper;
        return 0;
    }
    default:
        return -1;
    }
}

int frame_reader_read_frame(FrameReader *r, uint16_t *bayer_out) {
    if (!r->read_frame) return -1;
    return r->read_frame(r, bayer_out);
}

int frame_reader_read_frame_rgb(FrameReader *r, uint16_t *rgb_out) {
    if (!r->read_frame_rgb) return -1;
    return r->read_frame_rgb(r, rgb_out);
}

void frame_reader_close(FrameReader *r) {
    if (r->close) r->close(r);
    r->read_frame = NULL;
    r->close = NULL;
}

int frame_reader_probe_frame_count(const char *path) {
    InputFormat fmt = detect_input_format(path);
    switch (fmt) {
    case FORMAT_MOV:        return mov_reader_probe_frame_count(path);
    case FORMAT_CINEMADNG:  return dng_reader_probe_frame_count(path);
    case FORMAT_BRAW: {
        BrawMovInfo mov;
        if (braw_mov_parse(path, &mov) != BRAW_OK) return -1;
        int count = mov.frame_count;
        braw_mov_free(&mov);
        return count;
    }
    case FORMAT_CRM: {
        CrmMovInfo mov;
        if (crm_mov_parse(path, &mov) != CRM_OK) return -1;
        int count = mov.frame_count;
        crm_mov_free(&mov);
        return count;
    }
    case FORMAT_ARRIRAW:    return ari_reader_probe_frame_count(path);
    case FORMAT_MXF:        return mxf_reader_probe_frame_count(path);
    case FORMAT_R3D:        return r3d_reader_probe_frame_count(path);
    case FORMAT_CINEFORM:   return cf_reader_probe_frame_count(path);
    case FORMAT_ZRAW:       return zraw_reader_probe_frame_count(path);
    default:                return -1;
    }
}

int frame_reader_probe_dimensions(const char *path, int *width, int *height) {
    InputFormat fmt = detect_input_format(path);
    switch (fmt) {
    case FORMAT_MOV:        return mov_reader_probe_dimensions(path, width, height);
    case FORMAT_CINEMADNG:  return dng_reader_probe_dimensions(path, width, height);
    case FORMAT_BRAW: {
        BrawMovInfo mov;
        if (braw_mov_parse(path, &mov) != BRAW_OK) return -1;
        uint8_t *pkt;
        size_t sz;
        if (braw_mov_read_frame(path, &mov, 0, &pkt, &sz) != BRAW_OK) {
            braw_mov_free(&mov);
            return -1;
        }
        int ret = braw_dec_probe(pkt, sz, width, height);
        free(pkt);
        braw_mov_free(&mov);
        return (ret == BRAW_OK) ? 0 : -1;
    }
    case FORMAT_CRM: {
        CrmMovInfo mov;
        if (crm_mov_parse(path, &mov) != CRM_OK) return -1;
        *width = mov.width;
        *height = mov.height;
        crm_mov_free(&mov);
        return 0;
    }
    case FORMAT_ARRIRAW:    return ari_reader_probe_dimensions(path, width, height);
    case FORMAT_MXF:        return mxf_reader_probe_dimensions(path, width, height);
    case FORMAT_R3D:        return r3d_reader_probe_dimensions(path, width, height);
    case FORMAT_CINEFORM:   return cf_reader_probe_dimensions(path, width, height);
    case FORMAT_ZRAW:       return zraw_reader_probe_dimensions(path, width, height);
    default:                return -1;
    }
}

int frame_reader_get_dng_info(const FrameReader *r,
                              int *bits_per_sample,
                              int *cfa_pattern,
                              const char **first_file_path) {
    if (r->format != FORMAT_CINEMADNG || !r->impl) return -1;
    DngReader *dr = (DngReader *)r->impl;
    if (bits_per_sample)  *bits_per_sample = dr->bits_per_sample;
    if (cfa_pattern)      *cfa_pattern = dr->cfa_pattern;
    if (first_file_path)  *first_file_path = (dr->file_count > 0) ? dr->file_list[0] : NULL;
    return 0;
}

int frame_reader_probe_wb_gains(const char *path, float *r_gain, float *b_gain) {
    InputFormat fmt = detect_input_format(path);
    switch (fmt) {
    case FORMAT_MOV:        return mov_reader_probe_wb_gains(path, r_gain, b_gain);
    case FORMAT_CINEMADNG:  return dng_reader_probe_wb_gains(path, r_gain, b_gain);
    case FORMAT_BRAW:
    case FORMAT_CRM:
    case FORMAT_ARRIRAW:
    case FORMAT_MXF:
    case FORMAT_R3D:
    case FORMAT_CINEFORM:
    case FORMAT_ZRAW:
    default:
        *r_gain = 1.0f;
        *b_gain = 1.0f;
        return -1;
    }
}
