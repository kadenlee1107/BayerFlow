#include "../include/prores_raw_enc.h"
#include "../include/mov_reader.h"
#include <errno.h>

void write_be16(uint8_t *buf, uint16_t val) {
    buf[0] = (val >> 8) & 0xFF;
    buf[1] = val & 0xFF;
}

void write_be32(uint8_t *buf, uint32_t val) {
    buf[0] = (val >> 24) & 0xFF;
    buf[1] = (val >> 16) & 0xFF;
    buf[2] = (val >> 8) & 0xFF;
    buf[3] = val & 0xFF;
}

void write_be64(uint8_t *buf, uint64_t val) {
    write_be32(buf, val >> 32);
    write_be32(buf + 4, val & 0xFFFFFFFF);
}

int write_mov_file(const char *filename, const uint8_t *frame_data,
                   int frame_size, int width, int height) {
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return -1;
    }
    
    uint8_t buf[1024];
    
    // Calculate sizes
    int stsd_size = 102;  // Fixed size for ProRes RAW stsd
    int stts_size = 24;
    int stsc_size = 28;
    int stsz_size = 24;
    int stco_size = 20;
    int stbl_size = 8 + stsd_size + stts_size + stsc_size + stsz_size + stco_size;
    int vmhd_size = 20;
    int hdlr_size = 44;
    int minf_size = 8 + vmhd_size + hdlr_size + stbl_size;
    int mdhd_size = 32;
    int mdia_size = 8 + mdhd_size + minf_size;
    int tkhd_size = 92;
    int trak_size = 8 + tkhd_size + mdia_size;
    int mvhd_size = 108;
    int moov_size = 8 + mvhd_size + trak_size;
    
    // ftyp atom
    write_be32(buf, 24);
    memcpy(buf + 4, "ftyp", 4);
    memcpy(buf + 8, "qt  ", 4);
    write_be32(buf + 12, 0x20110700);
    memcpy(buf + 16, "qt  ", 4);
    memcpy(buf + 20, "pana", 4);
    fwrite(buf, 1, 24, f);
    
    // moov atom
    write_be32(buf, moov_size);
    memcpy(buf + 4, "moov", 4);
    fwrite(buf, 1, 8, f);
    
    // mvhd
    write_be32(buf, mvhd_size);
    memcpy(buf + 4, "mvhd", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 0);
    write_be32(buf + 20, 90000);
    write_be32(buf + 24, 3003);
    write_be32(buf + 28, 0x00010000);
    write_be16(buf + 32, 0x0100);
    memset(buf + 34, 0, 10);
    
    uint32_t matrix[9] = {
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000
    };
    for (int i = 0; i < 9; i++) {
        write_be32(buf + 44 + i*4, matrix[i]);
    }
    
    write_be32(buf + 80, 0);
    write_be32(buf + 84, 0);
    write_be32(buf + 88, 0);
    write_be32(buf + 92, 0);
    write_be32(buf + 96, 0);
    write_be32(buf + 100, 0);
    write_be32(buf + 104, 2);
    fwrite(buf, 1, mvhd_size, f);
    
    // trak
    write_be32(buf, trak_size);
    memcpy(buf + 4, "trak", 4);
    fwrite(buf, 1, 8, f);
    
    // tkhd
    write_be32(buf, tkhd_size);
    memcpy(buf + 4, "tkhd", 4);
    write_be32(buf + 8, 0x0000000F);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 0);
    write_be32(buf + 20, 1);
    write_be32(buf + 24, 0);
    write_be32(buf + 28, 3003);
    write_be64(buf + 32, 0);
    write_be16(buf + 40, 0);
    write_be16(buf + 42, 0);
    write_be16(buf + 44, 0x0100);
    write_be16(buf + 46, 0);
    
    for (int i = 0; i < 9; i++) {
        write_be32(buf + 48 + i*4, matrix[i]);
    }
    
    write_be32(buf + 84, width << 16);
    write_be32(buf + 88, height << 16);
    fwrite(buf, 1, tkhd_size, f);
    
    // mdia
    write_be32(buf, mdia_size);
    memcpy(buf + 4, "mdia", 4);
    fwrite(buf, 1, 8, f);
    
    // mdhd
    write_be32(buf, mdhd_size);
    memcpy(buf + 4, "mdhd", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 0);
    write_be32(buf + 20, 90000);
    write_be32(buf + 24, 3003);
    write_be16(buf + 28, 0);
    write_be16(buf + 30, 0);
    fwrite(buf, 1, mdhd_size, f);
    
    // minf
    write_be32(buf, minf_size);
    memcpy(buf + 4, "minf", 4);
    fwrite(buf, 1, 8, f);
    
    // vmhd
    write_be32(buf, vmhd_size);
    memcpy(buf + 4, "vmhd", 4);
    write_be32(buf + 8, 0x00000001);
    write_be64(buf + 12, 0);
    fwrite(buf, 1, vmhd_size, f);
    
    // hdlr
    write_be32(buf, hdlr_size);
    memcpy(buf + 4, "hdlr", 4);
    write_be32(buf + 8, 0);
    memcpy(buf + 12, "mhlr", 4);
    memcpy(buf + 16, "vide", 4);
    memset(buf + 20, 0, 12);
    memcpy(buf + 32, "VideoHandler", 12);
    fwrite(buf, 1, hdlr_size, f);
    
    // stbl
    write_be32(buf, stbl_size);
    memcpy(buf + 4, "stbl", 4);
    fwrite(buf, 1, 8, f);
    
    // stsd
    write_be32(buf, stsd_size);
    memcpy(buf + 4, "stsd", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    
    write_be32(buf + 16, 86);  // Sample entry size
    memcpy(buf + 20, "aprh", 4);
    memset(buf + 24, 0, 6);
    write_be16(buf + 30, 1);
    write_be16(buf + 32, 0);
    write_be16(buf + 34, 0);
    memcpy(buf + 36, "appl", 4);
    write_be32(buf + 40, 0);
    write_be32(buf + 44, 0);
    write_be16(buf + 48, width);
    write_be16(buf + 50, height);
    write_be32(buf + 52, 0x00480000);
    write_be32(buf + 56, 0x00480000);
    write_be32(buf + 60, 0);
    write_be16(buf + 64, 1);
    buf[66] = 19;
    memcpy(buf + 67, "Apple ProRes RAW HQ", 19);
    memset(buf + 86, 0, 13);
    write_be16(buf + 98, 24);
    write_be16(buf + 100, -1);
    fwrite(buf, 1, stsd_size, f);
    
    // stts
    write_be32(buf, stts_size);
    memcpy(buf + 4, "stts", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, 1);
    write_be32(buf + 20, 3003);
    fwrite(buf, 1, stts_size, f);
    
    // stsc
    write_be32(buf, stsc_size);
    memcpy(buf + 4, "stsc", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, 1);
    write_be32(buf + 20, 1);
    write_be32(buf + 24, 1);
    fwrite(buf, 1, stsc_size, f);
    
    // stsz
    write_be32(buf, stsz_size);
    memcpy(buf + 4, "stsz", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 1);
    write_be32(buf + 20, frame_size);
    fwrite(buf, 1, stsz_size, f);
    
    // stco
    long stco_pos = ftell(f);
    write_be32(buf, stco_size);
    memcpy(buf + 4, "stco", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, 0);  // Will fix this
    fwrite(buf, 1, stco_size, f);
    
    // mdat
    long mdat_pos = ftell(f);
    write_be32(buf, 8 + frame_size);
    memcpy(buf + 4, "mdat", 4);
    fwrite(buf, 1, 8, f);
    fwrite(frame_data, 1, frame_size, f);
    
    // Fix chunk offset
    fseek(f, stco_pos + 16, SEEK_SET);
    write_be32(buf, (uint32_t)(mdat_pos + 8));
    fwrite(buf, 1, 4, f);
    
    fclose(f);
    
    printf("Wrote MOV file: %s\n", filename);
    printf("  mdat offset: 0x%lx\n", mdat_pos + 8);
    printf("  frame size: %d bytes\n", frame_size);

    return 0;
}

/* ================================================================== */
/*  Streaming multi-frame MOV writer                                   */
/*  Pattern: ftyp → mdat (append frames) → moov (at end)              */
/* ================================================================== */

#define MOV_WRITER_MAX_FRAMES 100000

int mov_writer_open(MovWriter *w, const char *filename, int width, int height) {
    memset(w, 0, sizeof(*w));
    w->width = width;
    w->height = height;

    w->fp = fopen(filename, "wb");
    if (!w->fp) {
        fprintf(stderr, "mov_writer: failed to open %s\n", filename);
        return -1;
    }

    w->frame_sizes = (uint32_t *)calloc(MOV_WRITER_MAX_FRAMES, sizeof(uint32_t));
    w->frame_offsets = (uint64_t *)calloc(MOV_WRITER_MAX_FRAMES, sizeof(uint64_t));
    if (!w->frame_sizes || !w->frame_offsets) {
        fprintf(stderr, "mov_writer: allocation failed\n");
        fclose(w->fp);
        return -1;
    }

    uint8_t buf[32];

    /* ftyp atom */
    write_be32(buf, 24);
    memcpy(buf + 4, "ftyp", 4);
    memcpy(buf + 8, "qt  ", 4);
    write_be32(buf + 12, 0x20110700);
    memcpy(buf + 16, "qt  ", 4);
    memcpy(buf + 20, "pana", 4);
    fwrite(buf, 1, 24, w->fp);

    /* mdat atom — use 64-bit extended size for large files */
    w->mdat_start = ftell(w->fp);
    write_be32(buf, 1);            /* size=1 signals 64-bit extended size */
    memcpy(buf + 4, "mdat", 4);
    write_be64(buf + 8, 0);        /* placeholder for 64-bit size */
    fwrite(buf, 1, 16, w->fp);

    w->frame_count = 0;
    return 0;
}

int mov_writer_add_frame(MovWriter *w, const uint8_t *frame_data, int frame_size) {
    if (!w->fp) return -1;
    if (w->frame_count >= MOV_WRITER_MAX_FRAMES) {
        fprintf(stderr, "mov_writer: exceeded max frames (%d)\n", MOV_WRITER_MAX_FRAMES);
        return -1;
    }

    w->frame_offsets[w->frame_count] = (uint64_t)ftell(w->fp);
    w->frame_sizes[w->frame_count] = (uint32_t)frame_size;

    size_t written = fwrite(frame_data, 1, frame_size, w->fp);
    if ((int)written != frame_size) {
        fprintf(stderr, "mov_writer: short write at frame %d (wrote %zu of %d, ferror=%d errno=%d: %s)\n",
                w->frame_count, written, frame_size, ferror(w->fp), errno, strerror(errno));
        return -1;
    }

    w->frame_count++;
    return 0;
}

static void write_moov_multi(MovWriter *w) {
    FILE *f = w->fp;
    uint8_t buf[1024];
    int n = w->frame_count;

    /*
     * Atom sizes matching reference camera MOV structure:
     * stsd: 128 bytes (112-byte entry + fiel(10) + pasp(16))
     * hdlr (mdia): 33 bytes (mhlr/vide, null-term name)
     * hdlr (minf): 33 bytes (dhlr/alis, null-term name)
     * dinf/dref: 36 bytes (self-reference alis)
     * edts/elst: 36 bytes (edit list)
     */
    int stsd_size = 128;
    int stts_size = 24;
    int stsc_size = 28;
    int stsz_size = 20 + n * 4;
    int co64_size = 16 + n * 8;
    int stbl_size = 8 + stsd_size + stts_size + stsc_size + stsz_size + co64_size;
    int vmhd_size = 20;
    int mdia_hdlr_size = 33;
    int minf_hdlr_size = 33;
    int dref_size = 28;
    int dinf_size = 8 + dref_size;
    int minf_size = 8 + vmhd_size + minf_hdlr_size + dinf_size + stbl_size;
    int mdhd_size = 32;
    int mdia_size = 8 + mdhd_size + mdia_hdlr_size + minf_size;
    int tkhd_size = 92;
    int elst_size = 28;
    int edts_size = 8 + elst_size;
    int trak_size = 8 + tkhd_size + edts_size + mdia_size;
    int mvhd_size = 108;
    int moov_size = 8 + mvhd_size + trak_size;

    /* Add metadata atoms if present */
    if (w->meta_atom)
        moov_size += w->meta_atom_size;
    if (w->udta_atom)
        moov_size += w->udta_atom_size;

    uint32_t total_duration = (uint32_t)n * 3003;

    /* moov */
    write_be32(buf, moov_size);
    memcpy(buf + 4, "moov", 4);
    fwrite(buf, 1, 8, f);

    /* mvhd */
    write_be32(buf, mvhd_size);
    memcpy(buf + 4, "mvhd", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 0);
    write_be32(buf + 20, 90000);          /* timescale */
    write_be32(buf + 24, total_duration); /* duration */
    write_be32(buf + 28, 0x00010000);     /* rate 1.0 */
    write_be16(buf + 32, 0x0100);         /* volume 1.0 */
    memset(buf + 34, 0, 10);

    uint32_t matrix[9] = {
        0x00010000, 0, 0,
        0, 0x00010000, 0,
        0, 0, 0x40000000
    };
    for (int i = 0; i < 9; i++)
        write_be32(buf + 44 + i*4, matrix[i]);

    memset(buf + 80, 0, 24);
    write_be32(buf + 104, 2);  /* next_track_ID */
    fwrite(buf, 1, mvhd_size, f);

    /* trak */
    write_be32(buf, trak_size);
    memcpy(buf + 4, "trak", 4);
    fwrite(buf, 1, 8, f);

    /* tkhd */
    write_be32(buf, tkhd_size);
    memcpy(buf + 4, "tkhd", 4);
    write_be32(buf + 8, 0x0000000F);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 0);
    write_be32(buf + 20, 1);               /* track_ID */
    write_be32(buf + 24, 0);
    write_be32(buf + 28, total_duration);   /* duration */
    write_be64(buf + 32, 0);
    write_be16(buf + 40, 0);
    write_be16(buf + 42, 0);
    write_be16(buf + 44, 0x0100);
    write_be16(buf + 46, 0);
    for (int i = 0; i < 9; i++)
        write_be32(buf + 48 + i*4, matrix[i]);
    write_be32(buf + 84, w->width << 16);
    write_be32(buf + 88, w->height << 16);
    fwrite(buf, 1, tkhd_size, f);

    /* edts */
    write_be32(buf, edts_size);
    memcpy(buf + 4, "edts", 4);
    fwrite(buf, 1, 8, f);

    /* elst — single edit spanning full duration */
    write_be32(buf, elst_size);
    memcpy(buf + 4, "elst", 4);
    write_be32(buf + 8, 0);       /* version + flags */
    write_be32(buf + 12, 1);      /* entry count */
    write_be32(buf + 16, total_duration);  /* segment duration */
    write_be32(buf + 20, 0);      /* media time */
    write_be16(buf + 24, 1);      /* media rate integer */
    write_be16(buf + 26, 0);      /* media rate fraction */
    fwrite(buf, 1, elst_size, f);

    /* mdia */
    write_be32(buf, mdia_size);
    memcpy(buf + 4, "mdia", 4);
    fwrite(buf, 1, 8, f);

    /* mdhd */
    write_be32(buf, mdhd_size);
    memcpy(buf + 4, "mdhd", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 0);
    write_be32(buf + 16, 0);
    write_be32(buf + 20, 90000);
    write_be32(buf + 24, total_duration);
    write_be16(buf + 28, 0);
    write_be16(buf + 30, 0);
    fwrite(buf, 1, mdhd_size, f);

    /* hdlr (media handler — mhlr/vide) */
    write_be32(buf, mdia_hdlr_size);
    memcpy(buf + 4, "hdlr", 4);
    write_be32(buf + 8, 0);       /* version + flags */
    memcpy(buf + 12, "mhlr", 4);  /* component type */
    memcpy(buf + 16, "vide", 4);  /* component subtype */
    memset(buf + 20, 0, 12);      /* manufacturer + flags */
    buf[32] = 0;                   /* null-terminated empty name */
    fwrite(buf, 1, mdia_hdlr_size, f);

    /* minf */
    write_be32(buf, minf_size);
    memcpy(buf + 4, "minf", 4);
    fwrite(buf, 1, 8, f);

    /* vmhd */
    write_be32(buf, vmhd_size);
    memcpy(buf + 4, "vmhd", 4);
    write_be32(buf + 8, 0x00000001);
    write_be64(buf + 12, 0);
    fwrite(buf, 1, vmhd_size, f);

    /* hdlr (data handler — dhlr/alis) */
    write_be32(buf, minf_hdlr_size);
    memcpy(buf + 4, "hdlr", 4);
    write_be32(buf + 8, 0);
    memcpy(buf + 12, "dhlr", 4);  /* data handler */
    memcpy(buf + 16, "alis", 4);  /* alias */
    memset(buf + 20, 0, 12);
    buf[32] = 0;
    fwrite(buf, 1, minf_hdlr_size, f);

    /* dinf */
    write_be32(buf, dinf_size);
    memcpy(buf + 4, "dinf", 4);
    fwrite(buf, 1, 8, f);

    /* dref — self-contained data reference */
    write_be32(buf, dref_size);
    memcpy(buf + 4, "dref", 4);
    write_be32(buf + 8, 0);       /* version + flags */
    write_be32(buf + 12, 1);      /* entry count */
    write_be32(buf + 16, 12);     /* entry size */
    memcpy(buf + 20, "alis", 4);  /* entry type */
    write_be32(buf + 24, 1);      /* flags = self-contained */
    fwrite(buf, 1, dref_size, f);

    /* stbl */
    write_be32(buf, stbl_size);
    memcpy(buf + 4, "stbl", 4);
    fwrite(buf, 1, 8, f);

    /* stsd — matching reference: 128 bytes with fiel + pasp */
    memset(buf, 0, 128);
    write_be32(buf, stsd_size);
    memcpy(buf + 4, "stsd", 4);
    write_be32(buf + 8, 0);       /* version + flags */
    write_be32(buf + 12, 1);      /* entry count */
    write_be32(buf + 16, 112);    /* sample entry size (102 base + fiel(10) + pasp(16) - wait, 86+10+16=112) */
    memcpy(buf + 20, "aprh", 4);  /* codec */
    memset(buf + 24, 0, 6);       /* reserved */
    write_be16(buf + 30, 1);      /* data_ref_index */
    write_be16(buf + 32, 0);      /* version */
    write_be16(buf + 34, 0);      /* revision */
    memcpy(buf + 36, "appl", 4);  /* vendor */
    write_be32(buf + 40, 0);      /* temporal_quality */
    write_be32(buf + 44, 0x000003FF);  /* spatial_quality = 1023 */
    write_be16(buf + 48, w->width);
    write_be16(buf + 50, w->height);
    write_be32(buf + 52, 0x00480000);  /* horiz res 72dpi */
    write_be32(buf + 56, 0x00480000);  /* vert res 72dpi */
    write_be32(buf + 60, 0);           /* data_size */
    write_be16(buf + 64, 1);           /* frame_count */
    buf[66] = 19;                      /* compressor name length */
    memcpy(buf + 67, "Apple ProRes RAW HQ", 19);
    /* bytes 86-98 already zeroed */
    write_be16(buf + 98, 24);          /* depth */
    write_be16(buf + 100, (uint16_t)-1);  /* color_table_id */
    /* fiel atom at offset 102 */
    write_be32(buf + 102, 10);         /* size */
    memcpy(buf + 106, "fiel", 4);
    buf[110] = 1;                      /* progressive */
    buf[111] = 0;
    /* pasp atom at offset 112 */
    write_be32(buf + 112, 16);         /* size */
    memcpy(buf + 116, "pasp", 4);
    write_be32(buf + 120, 1);          /* hSpacing */
    write_be32(buf + 124, 1);          /* vSpacing */
    fwrite(buf, 1, stsd_size, f);

    /* stts — all samples same duration */
    write_be32(buf, stts_size);
    memcpy(buf + 4, "stts", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, n);
    write_be32(buf + 20, 3003);
    fwrite(buf, 1, stts_size, f);

    /* stsc — one chunk per sample */
    write_be32(buf, stsc_size);
    memcpy(buf + 4, "stsc", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, 1);
    write_be32(buf + 16, 1);
    write_be32(buf + 20, 1);
    write_be32(buf + 24, 1);
    fwrite(buf, 1, stsc_size, f);

    /* stsz — per-sample sizes */
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

    /* co64 — per-sample 64-bit chunk offsets */
    write_be32(buf, co64_size);
    memcpy(buf + 4, "co64", 4);
    write_be32(buf + 8, 0);
    write_be32(buf + 12, n);
    fwrite(buf, 1, 16, f);
    for (int i = 0; i < n; i++) {
        write_be64(buf, w->frame_offsets[i]);
        fwrite(buf, 1, 8, f);
    }

    /* Write metadata atoms (meta, udta) if copied from source */
    if (w->meta_atom)
        fwrite(w->meta_atom, 1, w->meta_atom_size, f);
    if (w->udta_atom)
        fwrite(w->udta_atom, 1, w->udta_atom_size, f);
}

/* ================================================================== */
/*  Copy metadata atoms (meta, udta) from a source MOV file            */
/*  Parses the source file to find these atoms inside moov and stores  */
/*  raw copies for later inclusion in the output moov.                 */
/* ================================================================== */

/* Read a big-endian 32-bit value from a buffer */
static uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

int mov_writer_copy_metadata(MovWriter *w, const char *source_mov) {
    FILE *f = fopen(source_mov, "rb");
    if (!f) {
        fprintf(stderr, "mov_writer_copy_metadata: cannot open %s\n", source_mov);
        return -1;
    }

    /* Find moov atom in the top-level file */
    uint8_t hdr[8];
    long moov_offset = -1;
    long moov_size = 0;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    long pos = 0;
    while (pos < file_size - 8) {
        fseek(f, pos, SEEK_SET);
        if (fread(hdr, 1, 8, f) != 8) break;
        uint32_t atom_size = read_be32(hdr);
        if (atom_size < 8) break;
        if (memcmp(hdr + 4, "moov", 4) == 0) {
            moov_offset = pos;
            moov_size = atom_size;
            break;
        }
        pos += atom_size;
    }

    if (moov_offset < 0) {
        fprintf(stderr, "mov_writer_copy_metadata: no moov atom found\n");
        fclose(f);
        return -1;
    }

    /* Scan children of moov for meta and udta atoms */
    long child_pos = moov_offset + 8;
    long moov_end = moov_offset + moov_size;

    while (child_pos < moov_end - 8) {
        fseek(f, child_pos, SEEK_SET);
        if (fread(hdr, 1, 8, f) != 8) break;
        uint32_t atom_size = read_be32(hdr);
        if (atom_size < 8 || child_pos + atom_size > moov_end) break;

        if (memcmp(hdr + 4, "meta", 4) == 0 && !w->meta_atom) {
            w->meta_atom_size = (int)atom_size;
            w->meta_atom = (uint8_t *)malloc(atom_size);
            if (w->meta_atom) {
                fseek(f, child_pos, SEEK_SET);
                if (fread(w->meta_atom, 1, atom_size, f) != atom_size) {
                    free(w->meta_atom);
                    w->meta_atom = NULL;
                    w->meta_atom_size = 0;
                } else {
                    printf("mov_writer: copied meta atom (%d bytes)\n", w->meta_atom_size);
                }
            }
        } else if (memcmp(hdr + 4, "udta", 4) == 0 && !w->udta_atom) {
            w->udta_atom_size = (int)atom_size;
            w->udta_atom = (uint8_t *)malloc(atom_size);
            if (w->udta_atom) {
                fseek(f, child_pos, SEEK_SET);
                if (fread(w->udta_atom, 1, atom_size, f) != atom_size) {
                    free(w->udta_atom);
                    w->udta_atom = NULL;
                    w->udta_atom_size = 0;
                } else {
                    printf("mov_writer: copied udta atom (%d bytes)\n", w->udta_atom_size);
                }
            }
        }

        child_pos += atom_size;
    }

    fclose(f);
    return 0;
}

int mov_writer_close(MovWriter *w) {
    if (!w->fp) return -1;

    /* Patch mdat size: total = 16 (extended header) + sum of all frame data */
    long mdat_end = ftell(w->fp);
    uint64_t mdat_size = (uint64_t)(mdat_end - w->mdat_start);

    /* Write moov at end of file */
    write_moov_multi(w);

    /* Seek back and patch 64-bit extended mdat size (at offset +8 from mdat_start) */
    uint8_t buf[8];
    fseek(w->fp, w->mdat_start + 8, SEEK_SET);
    write_be64(buf, mdat_size);
    fwrite(buf, 1, 8, w->fp);

    fclose(w->fp);
    w->fp = NULL;

    printf("mov_writer: wrote %d frames, mdat=%llu bytes\n",
           w->frame_count, (unsigned long long)mdat_size);

    free(w->frame_sizes);
    free(w->frame_offsets);
    free(w->meta_atom);
    free(w->udta_atom);
    w->frame_sizes = NULL;
    w->frame_offsets = NULL;
    w->meta_atom = NULL;
    w->udta_atom = NULL;

    return 0;
}
