/*
 * BRAW MOV Container Writer
 *
 * Streaming pattern: ftyp → mdat (append frames) → moov (at end).
 * Based on mov_writer.c but with BRAW-specific stsd codec tag and timing.
 */

#include "../include/braw_writer.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define BRAW_WRITER_MAX_FRAMES 100000

static void write_be16(uint8_t *buf, uint16_t val) {
    buf[0] = (val >> 8) & 0xFF;
    buf[1] = val & 0xFF;
}

static void write_be32(uint8_t *buf, uint32_t val) {
    buf[0] = (val >> 24) & 0xFF;
    buf[1] = (val >> 16) & 0xFF;
    buf[2] = (val >> 8) & 0xFF;
    buf[3] = val & 0xFF;
}

static void write_be64(uint8_t *buf, uint64_t val) {
    write_be32(buf, val >> 32);
    write_be32(buf + 4, val & 0xFFFFFFFF);
}

static uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

/* Compute timescale and sample duration from fps.
 * Camera BRAW files use simple timescale = round(fps) with delta = 1 for
 * integer frame rates.  Only non-integer (drop-frame) rates need higher
 * timescales. */
static void compute_timing(double fps, int *timescale, int *sample_duration) {
    if (fps > 23.97 && fps < 23.98) {
        *timescale = 24000; *sample_duration = 1001;  /* 23.976 */
    } else if (fps > 23.99 && fps < 24.01) {
        *timescale = 24; *sample_duration = 1;
    } else if (fps > 24.99 && fps < 25.01) {
        *timescale = 25; *sample_duration = 1;
    } else if (fps > 29.96 && fps < 29.98) {
        *timescale = 30000; *sample_duration = 1001;  /* 29.97 */
    } else if (fps > 29.99 && fps < 30.01) {
        *timescale = 30; *sample_duration = 1;
    } else if (fps > 47.94 && fps < 47.96) {
        *timescale = 48000; *sample_duration = 1001;  /* 47.952 */
    } else if (fps > 49.99 && fps < 50.01) {
        *timescale = 50; *sample_duration = 1;
    } else if (fps > 59.93 && fps < 59.95) {
        *timescale = 60000; *sample_duration = 1001;  /* 59.94 */
    } else if (fps > 59.99 && fps < 60.01) {
        *timescale = 60; *sample_duration = 1;
    } else {
        /* Fallback: integer fps if close, otherwise high timescale */
        int ifps = (int)(fps + 0.5);
        if (fps > ifps - 0.01 && fps < ifps + 0.01) {
            *timescale = ifps; *sample_duration = 1;
        } else {
            *timescale = 90000;
            *sample_duration = (int)(90000.0 / fps + 0.5);
        }
    }
}

int braw_writer_open(BrawWriter *w, const char *filename,
                     int width, int height, double fps) {
    memset(w, 0, sizeof(*w));
    w->width = width;
    w->height = height;
    w->fps = fps;
    memcpy(w->codec_tag, "brhq", 4);  /* default: Blackmagic RAW HQ */

    compute_timing(fps, &w->timescale, &w->sample_duration);

    w->fp = fopen(filename, "wb");
    if (!w->fp) {
        fprintf(stderr, "braw_writer: failed to open %s\n", filename);
        return -1;
    }

    w->frame_sizes = (uint32_t *)calloc(BRAW_WRITER_MAX_FRAMES, sizeof(uint32_t));
    w->frame_offsets = (uint64_t *)calloc(BRAW_WRITER_MAX_FRAMES, sizeof(uint64_t));
    if (!w->frame_sizes || !w->frame_offsets) {
        fprintf(stderr, "braw_writer: allocation failed\n");
        fclose(w->fp);
        return -1;
    }

    uint8_t buf[32];

    /* wide + mdat (matches camera BRAW file layout) */
    write_be32(buf, 8);
    memcpy(buf + 4, "wide", 4);
    fwrite(buf, 1, 8, w->fp);

    w->mdat_start = ftell(w->fp);
    write_be32(buf, 0);              /* size=0 means extends to EOF (patched later) */
    memcpy(buf + 4, "mdat", 4);
    fwrite(buf, 1, 8, w->fp);

    /* Pad to 8KB boundary (camera BRAW files align all frames to 0x2000) */
    {
        long pos = ftell(w->fp);
        long next_align = (pos + 0x1FFF) & ~0x1FFFL;
        if (next_align > pos) {
            long pad = next_align - pos;
            uint8_t zero[256];
            memset(zero, 0, sizeof(zero));
            while (pad > 0) {
                int chunk = pad > 256 ? 256 : (int)pad;
                fwrite(zero, 1, chunk, w->fp);
                pad -= chunk;
            }
        }
    }

    w->frame_count = 0;
    return 0;
}

int braw_writer_add_frame(BrawWriter *w, const uint8_t *packet, int packet_size) {
    if (!w->fp) return -1;
    if (w->frame_count >= BRAW_WRITER_MAX_FRAMES) {
        fprintf(stderr, "braw_writer: exceeded max frames\n");
        return -1;
    }

    w->frame_offsets[w->frame_count] = (uint64_t)ftell(w->fp);
    w->frame_sizes[w->frame_count] = (uint32_t)packet_size;

    size_t written = fwrite(packet, 1, packet_size, w->fp);
    if ((int)written != packet_size) {
        fprintf(stderr, "braw_writer: short write at frame %d\n", w->frame_count);
        return -1;
    }

    /* Pad to 8KB boundary after each frame */
    {
        long pos = ftell(w->fp);
        long next_align = (pos + 0x1FFF) & ~0x1FFFL;
        if (next_align > pos) {
            long pad = next_align - pos;
            uint8_t zero[256];
            memset(zero, 0, sizeof(zero));
            while (pad > 0) {
                int chunk = pad > 256 ? 256 : (int)pad;
                fwrite(zero, 1, chunk, w->fp);
                pad -= chunk;
            }
        }
    }

    w->frame_count++;
    return 0;
}

static void write_moov(BrawWriter *w) {
    FILE *f = w->fp;
    uint8_t buf[256];
    int n = w->frame_count;

    /* Use copied stsd entry if available, otherwise build a default */
    int stsd_size = w->stsd_entry ? (16 + w->stsd_entry_size) : 102;
    int stts_size = 24;
    int stsc_size = 28;
    int stsz_size = 20 + n * 4;
    int co64_size = 16 + n * 8;
    int stbl_size = 8 + stsd_size + stts_size + stsc_size + stsz_size + co64_size;
    int vmhd_size = 20;
    int act_mdia_hdlr_size = w->mdia_hdlr ? w->mdia_hdlr_size : 33;
    int act_minf_hdlr_size = w->minf_hdlr ? w->minf_hdlr_size : 33;
    int dref_size = 28;
    int dinf_size = 8 + dref_size;
    int minf_size = 8 + vmhd_size + act_minf_hdlr_size + dinf_size + stbl_size;
    int mdhd_size = 32;
    int mdia_size = 8 + mdhd_size + act_mdia_hdlr_size + minf_size;
    int tkhd_size = 92;
    int elst_size = 28;
    int edts_size = 8 + elst_size;
    int trak_size = 8 + tkhd_size + edts_size + mdia_size;
    int mvhd_size = 108;
    int moov_size = 8 + mvhd_size + trak_size;

    if (w->meta_atom) moov_size += w->meta_atom_size;
    if (w->udta_atom) moov_size += w->udta_atom_size;

    uint32_t total_duration = (uint32_t)n * (uint32_t)w->sample_duration;

    uint32_t matrix[9] = {
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000
    };

    /* moov */
    write_be32(buf, moov_size);
    memcpy(buf + 4, "moov", 4);
    fwrite(buf, 1, 8, f);

    /* mvhd */
    memset(buf, 0, mvhd_size);
    write_be32(buf, mvhd_size);
    memcpy(buf + 4, "mvhd", 4);
    write_be32(buf + 20, w->timescale);
    write_be32(buf + 24, total_duration);
    write_be32(buf + 28, 0x00010000);
    write_be16(buf + 32, 0x0100);
    for (int i = 0; i < 9; i++) write_be32(buf + 44 + i*4, matrix[i]);
    write_be32(buf + 104, 2);
    fwrite(buf, 1, mvhd_size, f);

    /* trak */
    write_be32(buf, trak_size);
    memcpy(buf + 4, "trak", 4);
    fwrite(buf, 1, 8, f);

    /* tkhd */
    memset(buf, 0, tkhd_size);
    write_be32(buf, tkhd_size);
    memcpy(buf + 4, "tkhd", 4);
    write_be32(buf + 8, 0x0000000F);
    write_be32(buf + 20, 1);
    write_be32(buf + 28, total_duration);
    /* volume=0 for video tracks (0x0100 is for audio) */
    for (int i = 0; i < 9; i++) write_be32(buf + 48 + i*4, matrix[i]);
    write_be32(buf + 84, w->width << 16);
    write_be32(buf + 88, w->height << 16);
    fwrite(buf, 1, tkhd_size, f);

    /* edts */
    write_be32(buf, edts_size);
    memcpy(buf + 4, "edts", 4);
    fwrite(buf, 1, 8, f);

    /* elst */
    memset(buf, 0, elst_size);
    write_be32(buf, elst_size);
    memcpy(buf + 4, "elst", 4);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, total_duration);
    write_be32(buf + 20, 0);
    write_be16(buf + 24, 1);
    fwrite(buf, 1, elst_size, f);

    /* tref: only write if we have the referenced track (timecode).
     * We only create a single video track, so skip tref to avoid
     * a dangling reference that causes DaVinci "media offline". */

    /* mdia */
    write_be32(buf, mdia_size);
    memcpy(buf + 4, "mdia", 4);
    fwrite(buf, 1, 8, f);

    /* mdhd */
    memset(buf, 0, mdhd_size);
    write_be32(buf, mdhd_size);
    memcpy(buf + 4, "mdhd", 4);
    write_be32(buf + 20, w->timescale);
    write_be32(buf + 24, total_duration);
    fwrite(buf, 1, mdhd_size, f);

    /* hdlr (media) — use copied or fallback */
    if (w->mdia_hdlr) {
        fwrite(w->mdia_hdlr, 1, w->mdia_hdlr_size, f);
    } else {
        memset(buf, 0, 33);
        write_be32(buf, 33);
        memcpy(buf + 4, "hdlr", 4);
        memcpy(buf + 12, "mhlr", 4);
        memcpy(buf + 16, "vide", 4);
        fwrite(buf, 1, 33, f);
    }

    /* minf */
    write_be32(buf, minf_size);
    memcpy(buf + 4, "minf", 4);
    fwrite(buf, 1, 8, f);

    /* vmhd */
    memset(buf, 0, vmhd_size);
    write_be32(buf, vmhd_size);
    memcpy(buf + 4, "vmhd", 4);
    write_be32(buf + 8, 0x00000001);
    fwrite(buf, 1, vmhd_size, f);

    /* hdlr (data) — use copied or fallback */
    if (w->minf_hdlr) {
        fwrite(w->minf_hdlr, 1, w->minf_hdlr_size, f);
    } else {
        memset(buf, 0, 33);
        write_be32(buf, 33);
        memcpy(buf + 4, "hdlr", 4);
        memcpy(buf + 12, "dhlr", 4);
        memcpy(buf + 16, "alis", 4);
        fwrite(buf, 1, 33, f);
    }

    /* dinf */
    write_be32(buf, dinf_size);
    memcpy(buf + 4, "dinf", 4);
    fwrite(buf, 1, 8, f);

    /* dref */
    memset(buf, 0, dref_size);
    write_be32(buf, dref_size);
    memcpy(buf + 4, "dref", 4);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, 12);
    memcpy(buf + 20, "alis", 4);
    write_be32(buf + 24, 1);
    fwrite(buf, 1, dref_size, f);

    /* stbl */
    write_be32(buf, stbl_size);
    memcpy(buf + 4, "stbl", 4);
    fwrite(buf, 1, 8, f);

    /* stsd — use copied entry if available */
    if (w->stsd_entry) {
        write_be32(buf, stsd_size);
        memcpy(buf + 4, "stsd", 4);
        write_be32(buf + 8, 0);    /* version/flags */
        write_be32(buf + 12, 1);   /* entry count */
        fwrite(buf, 1, 16, f);
        fwrite(w->stsd_entry, 1, w->stsd_entry_size, f);
    } else {
        /* Fallback: construct minimal stsd */
        memset(buf, 0, 102);
        write_be32(buf, 102);
        memcpy(buf + 4, "stsd", 4);
        write_be32(buf + 12, 1);
        write_be32(buf + 16, 86);
        memcpy(buf + 20, w->codec_tag, 4);
        write_be16(buf + 30, 1);
        memcpy(buf + 36, "bmd ", 4);
        write_be32(buf + 44, 0x000003FF);
        write_be16(buf + 48, w->width);
        write_be16(buf + 50, w->height);
        write_be32(buf + 52, 0x00480000);
        write_be32(buf + 56, 0x00480000);
        write_be16(buf + 64, 1);
        buf[66] = 14;
        memcpy(buf + 67, "Blackmagic RAW", 14);
        write_be16(buf + 98, 24);
        write_be16(buf + 100, (uint16_t)-1);
        fwrite(buf, 1, 102, f);
    }

    /* stts */
    write_be32(buf, stts_size);
    memcpy(buf + 4, "stts", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, n);
    write_be32(buf + 20, w->sample_duration);
    fwrite(buf, 1, stts_size, f);

    /* stsc */
    write_be32(buf, stsc_size);
    memcpy(buf + 4, "stsc", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, 1);
    write_be32(buf + 20, 1);
    write_be32(buf + 24, 1);
    fwrite(buf, 1, stsc_size, f);

    /* stsz */
    write_be32(buf, stsz_size);
    memcpy(buf + 4, "stsz", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, n);
    fwrite(buf, 1, 20, f);
    for (int i = 0; i < n; i++) {
        write_be32(buf, w->frame_sizes[i]);
        fwrite(buf, 1, 4, f);
    }

    /* co64 */
    write_be32(buf, co64_size);
    memcpy(buf + 4, "co64", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, n);
    fwrite(buf, 1, 16, f);
    for (int i = 0; i < n; i++) {
        write_be64(buf, w->frame_offsets[i]);
        fwrite(buf, 1, 8, f);
    }

    /* Metadata atoms */
    if (w->meta_atom)
        fwrite(w->meta_atom, 1, w->meta_atom_size, f);
    if (w->udta_atom)
        fwrite(w->udta_atom, 1, w->udta_atom_size, f);
}

/* Helper: copy an atom at file position into a malloc'd buffer */
static uint8_t *copy_atom(FILE *f, long offset, int size) {
    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) return NULL;
    fseek(f, offset, SEEK_SET);
    if ((int)fread(buf, 1, size, f) != size) { free(buf); return NULL; }
    return buf;
}

/* Helper: find a child atom by tag within [start, end) */
static long find_child_atom(FILE *f, long start, long end, const char *tag, uint32_t *out_size) {
    uint8_t hdr[8];
    long pos = start;
    while (pos < end - 8) {
        fseek(f, pos, SEEK_SET);
        if (fread(hdr, 1, 8, f) != 8) break;
        uint32_t sz = read_be32(hdr);
        if (sz < 8 || pos + sz > end) break;
        if (memcmp(hdr + 4, tag, 4) == 0) {
            if (out_size) *out_size = sz;
            return pos;
        }
        pos += sz;
    }
    return -1;
}

int braw_writer_copy_metadata(BrawWriter *w, const char *source_braw) {
    FILE *f = fopen(source_braw, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* Find moov atom */
    uint32_t moov_sz = 0;
    long moov_off = find_child_atom(f, 0, file_size, "moov", &moov_sz);
    if (moov_off < 0) { fclose(f); return -1; }
    long moov_end = moov_off + moov_sz;

    /* Scan moov children for meta, udta, trak */
    uint8_t hdr[8];
    long child = moov_off + 8;
    while (child < moov_end - 8) {
        fseek(f, child, SEEK_SET);
        if (fread(hdr, 1, 8, f) != 8) break;
        uint32_t atom_size = read_be32(hdr);
        if (atom_size < 8 || child + atom_size > moov_end) break;

        if (memcmp(hdr + 4, "meta", 4) == 0 && !w->meta_atom) {
            w->meta_atom_size = (int)atom_size;
            w->meta_atom = copy_atom(f, child, atom_size);
        } else if (memcmp(hdr + 4, "udta", 4) == 0 && !w->udta_atom) {
            w->udta_atom_size = (int)atom_size;
            w->udta_atom = copy_atom(f, child, atom_size);
        } else if (memcmp(hdr + 4, "trak", 4) == 0) {
            long trak_end = child + atom_size;

            /* Extract tref if present */
            uint32_t tref_sz = 0;
            long tref_off = find_child_atom(f, child + 8, trak_end, "tref", &tref_sz);
            if (tref_off >= 0 && !w->tref_atom) {
                w->tref_atom_size = (int)tref_sz;
                w->tref_atom = copy_atom(f, tref_off, tref_sz);
            }

            /* Drill into mdia — only process the video track (handler type "vide") */
            uint32_t mdia_sz = 0;
            long mdia_off = find_child_atom(f, child + 8, trak_end, "mdia", &mdia_sz);
            if (mdia_off >= 0) {
                long mdia_end = mdia_off + mdia_sz;

                /* Check handler type — skip non-video tracks */
                uint32_t hdlr_sz = 0;
                long hdlr_off = find_child_atom(f, mdia_off + 8, mdia_end, "hdlr", &hdlr_sz);
                int is_video = 0;
                if (hdlr_off >= 0 && hdlr_sz >= 20) {
                    uint8_t htype[4];
                    fseek(f, hdlr_off + 16, SEEK_SET);
                    if (fread(htype, 1, 4, f) == 4 && memcmp(htype, "vide", 4) == 0)
                        is_video = 1;
                }

                if (is_video) {
                    /* Copy media hdlr */
                    if (hdlr_off >= 0 && !w->mdia_hdlr) {
                        w->mdia_hdlr_size = (int)hdlr_sz;
                        w->mdia_hdlr = copy_atom(f, hdlr_off, hdlr_sz);
                    }

                    uint32_t minf_sz = 0;
                    long minf_off = find_child_atom(f, mdia_off + 8, mdia_end, "minf", &minf_sz);
                    if (minf_off >= 0) {
                        long minf_end = minf_off + minf_sz;

                        /* Copy data hdlr */
                        uint32_t dhdlr_sz = 0;
                        long dhdlr_off = find_child_atom(f, minf_off + 8, minf_end, "hdlr", &dhdlr_sz);
                        if (dhdlr_off >= 0 && !w->minf_hdlr) {
                            w->minf_hdlr_size = (int)dhdlr_sz;
                            w->minf_hdlr = copy_atom(f, dhdlr_off, dhdlr_sz);
                        }

                        uint32_t stbl_sz = 0;
                        long stbl_off = find_child_atom(f, minf_off + 8, minf_end, "stbl", &stbl_sz);
                        if (stbl_off >= 0) {
                            uint32_t stsd_sz = 0;
                            long stsd_off = find_child_atom(f, stbl_off + 8, stbl_off + stbl_sz, "stsd", &stsd_sz);
                            if (stsd_off >= 0 && stsd_sz > 16 && !w->stsd_entry) {
                                /* Copy the sample entry (everything after stsd header) */
                                int entry_size = (int)stsd_sz - 16;
                                w->stsd_entry_size = entry_size;
                                w->stsd_entry = copy_atom(f, stsd_off + 16, entry_size);
                                /* Extract codec tag from sample entry (offset 4 within entry) */
                                if (w->stsd_entry && entry_size >= 8) {
                                    memcpy(w->codec_tag, w->stsd_entry + 4, 4);
                                }
                            }
                        }
                    }
                }
            }
        }

        child += atom_size;
    }

    fclose(f);
    return 0;
}

int braw_writer_close(BrawWriter *w) {
    if (!w->fp) return -1;

    /* Compute mdat size (includes 8-byte mdat header) */
    long mdat_end = ftell(w->fp);
    uint32_t mdat_size = (uint32_t)(mdat_end - w->mdat_start);

    /* Write moov */
    write_moov(w);

    /* Patch 32-bit mdat size (wide+mdat pattern) */
    uint8_t buf[8];
    fseek(w->fp, w->mdat_start, SEEK_SET);
    write_be32(buf, mdat_size);
    fwrite(buf, 1, 4, w->fp);

    fclose(w->fp);
    w->fp = NULL;

    fprintf(stderr, "braw_writer: wrote %d frames, mdat=%u bytes\n",
            w->frame_count, mdat_size);

    free(w->frame_sizes);
    free(w->frame_offsets);
    free(w->meta_atom);
    free(w->udta_atom);
    free(w->stsd_entry);
    free(w->tref_atom);
    free(w->mdia_hdlr);
    free(w->minf_hdlr);
    w->frame_sizes = NULL;
    w->frame_offsets = NULL;
    w->meta_atom = NULL;
    w->udta_atom = NULL;
    w->stsd_entry = NULL;
    w->tref_atom = NULL;
    w->mdia_hdlr = NULL;
    w->minf_hdlr = NULL;

    return 0;
}
