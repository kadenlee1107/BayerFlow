#include "../include/prores_raw_enc.h"
#include "../include/mov_reader.h"
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>

/* Default frame header template (fallback if no source MOV provided) */
static uint8_t frame_header_template[86] = {
    0x00, 0x01, 0x70, 0x65, 0x61, 0x63, 0x16, 0x80, 0x0B, 0xE0, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xF7, 0xFF, 0x3F, 0xCC, 0x20, 0x00, 0x40, 0x06,
    0x00, 0x00, 0x3F, 0x2A, 0x55, 0x97, 0x3E, 0x3B, 0x3B, 0x8B, 0x3D, 0xD1,
    0x64, 0x5C, 0x3E, 0x93, 0xD9, 0x20, 0x3F, 0x37, 0x15, 0x22, 0xBB, 0x80,
    0xD8, 0xDA, 0x3C, 0x1F, 0x7A, 0xDD, 0xBD, 0x8C, 0x0F, 0x2A, 0x3F, 0x92,
    0xE8, 0x3E, 0x41, 0xB8, 0x51, 0xEC, 0x0E, 0xD8, 0x00, 0x38, 0x00, 0xC0,
    0x01, 0x40, 0x01, 0xD7, 0x03, 0x79, 0x07, 0xB2, 0x12, 0x33, 0x2B, 0xCA,
    0x69, 0xB1
};

static uint32_t read_be32_buf(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

/* Walk atom tree to find a child atom by tag within [start, end) */
static long find_atom(FILE *f, long start, long end, const char *tag) {
    long pos = start;
    uint8_t h[8];
    while (pos < end - 7) {
        fseek(f, pos, SEEK_SET);
        if (fread(h, 1, 8, f) != 8) return -1;
        uint32_t sz = read_be32_buf(h);
        if (sz < 8 || pos + sz > end) return -1;
        if (memcmp(h + 4, tag, 4) == 0) return pos;
        pos += sz;
    }
    return -1;
}

int set_frame_header_from_mov(const char *source_mov) {
    FILE *f = fopen(source_mov, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);

    /* Find moov */
    long moov = find_atom(f, 0, file_size, "moov");
    if (moov < 0) { fclose(f); return -1; }
    uint8_t h[8];
    fseek(f, moov, SEEK_SET); fread(h, 1, 8, f);
    long moov_end = moov + (long)read_be32_buf(h);

    /* Find first trak */
    long trak = find_atom(f, moov + 8, moov_end, "trak");
    if (trak < 0) { fclose(f); return -1; }
    fseek(f, trak, SEEK_SET); fread(h, 1, 8, f);
    long trak_end = trak + (long)read_be32_buf(h);

    /* trak → mdia */
    long mdia = find_atom(f, trak + 8, trak_end, "mdia");
    if (mdia < 0) { fclose(f); return -1; }
    fseek(f, mdia, SEEK_SET); fread(h, 1, 8, f);
    long mdia_end = mdia + (long)read_be32_buf(h);

    /* mdia → minf */
    long minf = find_atom(f, mdia + 8, mdia_end, "minf");
    if (minf < 0) { fclose(f); return -1; }
    fseek(f, minf, SEEK_SET); fread(h, 1, 8, f);
    long minf_end = minf + (long)read_be32_buf(h);

    /* minf → stbl */
    long stbl = find_atom(f, minf + 8, minf_end, "stbl");
    if (stbl < 0) { fclose(f); return -1; }
    fseek(f, stbl, SEEK_SET); fread(h, 1, 8, f);
    long stbl_end = stbl + (long)read_be32_buf(h);

    /* stbl → co64 (or stco) */
    long co64 = find_atom(f, stbl + 8, stbl_end, "co64");
    long first_frame_offset = -1;
    if (co64 >= 0) {
        uint8_t buf[12];
        fseek(f, co64 + 8, SEEK_SET);
        if (fread(buf, 1, 8, f) == 8) {
            uint32_t count = read_be32_buf(buf + 4);
            if (count > 0) {
                uint8_t off8[8];
                if (fread(off8, 1, 8, f) == 8) {
                    first_frame_offset = (long)(((uint64_t)read_be32_buf(off8) << 32) |
                                                 read_be32_buf(off8 + 4));
                }
            }
        }
    } else {
        long stco = find_atom(f, stbl + 8, stbl_end, "stco");
        if (stco >= 0) {
            uint8_t buf[8];
            fseek(f, stco + 8, SEEK_SET);
            if (fread(buf, 1, 8, f) == 8) {
                uint32_t count = read_be32_buf(buf + 4);
                if (count > 0) {
                    uint8_t off4[4];
                    if (fread(off4, 1, 4, f) == 4) {
                        first_frame_offset = (long)read_be32_buf(off4);
                    }
                }
            }
        }
    }

    if (first_frame_offset < 0) { fclose(f); return -1; }

    /* Read frame header: frame_size(4) + 'prrf'(4) + header_len(2) + 86 bytes */
    fseek(f, first_frame_offset, SEEK_SET);
    uint8_t frame_hdr[96];
    if (fread(frame_hdr, 1, 96, f) != 96) { fclose(f); return -1; }

    if (memcmp(frame_hdr + 4, "prrf", 4) != 0) {
        fprintf(stderr, "set_frame_header_from_mov: first frame is not prrf\n");
        fclose(f);
        return -1;
    }

    memcpy(frame_header_template, frame_hdr + 10, 86);
    printf("Copied frame header template from %s (first frame at offset %ld)\n",
           source_mov, first_frame_offset);

    fclose(f);
    return 0;
}

/* ================================================================== */
/* Multithreaded tile encoding with dynamic work stealing              */
/* ================================================================== */

#define MAX_TILE_BYTES 16384  /* 16KB per tile max */

/* Shared context for all worker threads (read-only + atomic counter) */
typedef struct {
    const uint16_t *bayer_data;
    int width, height;
    int nb_tw;
    int nb_tiles;
    const int32_t *qmat;
    int scale;
    uint8_t *tile_scratch;   /* base of scratch area (all tiles) */
    int *tile_sizes;         /* output: per-tile encoded sizes */
    atomic_int next_tile;    /* work-stealing counter */
} SharedWorkCtx;

/* Per-thread context */
typedef struct {
    SharedWorkCtx *shared;
    TileScratch scratch;     /* per-thread pre-allocated buffers */
} ThreadCtx;

static void *tile_worker(void *arg) {
    ThreadCtx *tctx = (ThreadCtx *)arg;
    SharedWorkCtx *ctx = tctx->shared;

    int t;
    while ((t = atomic_fetch_add(&ctx->next_tile, 1)) < ctx->nb_tiles) {
        int tx = t % ctx->nb_tw;
        int ty = t / ctx->nb_tw;
        int tile_x = tx * TILE_WIDTH;
        int tile_y = ty * TILE_HEIGHT;

        uint8_t *buf = ctx->tile_scratch + (size_t)t * MAX_TILE_BYTES;
        int size = encode_tile(ctx->bayer_data, ctx->width, ctx->height,
                               tile_x, tile_y, buf,
                               ctx->qmat, ctx->scale, &tctx->scratch);
        ctx->tile_sizes[t] = size;
    }
    return NULL;
}

static int get_num_threads(void) {
    const char *env = getenv("PRORES_THREADS");
    if (env) {
        int n = atoi(env);
        if (n > 0) return n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return (int)n;
    return 4;
}

// Encode complete frame
int encode_frame(const uint16_t *bayer_data, int width, int height,
                 uint8_t *output, int output_size) {

    /* Ensure DCT matrices are initialized before any tile work */
    init_fwd_matrices();

    uint8_t *out_ptr = output;

    // Frame header
    write_be32(out_ptr, 0);  // Frame size (fill in later)
    out_ptr += 4;

    // ProRes RAW frame marker
    write_be32(out_ptr, 0x70727266);  // 'prrf'
    out_ptr += 4;

    // Header length - Version 1 format uses 88 bytes like real camera
    int header_len = 88;
    write_be16(out_ptr, header_len);
    out_ptr += 2;

    // Use frame header template (either from source MOV or default)
    memcpy(out_ptr, frame_header_template, 86);
    write_be16(out_ptr + 6, width);
    write_be16(out_ptr + 8, height);
    out_ptr += 86;

    // Calculate tile layout
    int nb_tw = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int nb_th = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int nb_tiles = nb_tw * nb_th;

    // Encoder quantization: scale=15 → qmat_value=7
    int header_scale = 15;
    int qmat_value = (1 * header_scale) >> 1;
    int32_t qmat_scaled[64];
    for (int i = 0; i < 64; i++) {
        qmat_scaled[i] = default_qmat[i] * qmat_value;
    }

    // Reserve space for tile size table
    uint8_t *tile_size_table = out_ptr;
    out_ptr += nb_tiles * 2;

    // Allocate scratch space for all tiles
    uint8_t *tile_scratch = (uint8_t *)malloc((size_t)nb_tiles * MAX_TILE_BYTES);
    int *tile_sizes = (int *)calloc(nb_tiles, sizeof(int));
    if (!tile_scratch || !tile_sizes) {
        fprintf(stderr, "ERROR: Failed to allocate tile scratch buffers\n");
        free(tile_scratch);
        free(tile_sizes);
        return -1;
    }

    // Setup shared work context with atomic counter
    SharedWorkCtx shared = {
        .bayer_data = bayer_data,
        .width = width,
        .height = height,
        .nb_tw = nb_tw,
        .nb_tiles = nb_tiles,
        .qmat = qmat_scaled,
        .scale = header_scale,
        .tile_scratch = tile_scratch,
        .tile_sizes = tile_sizes,
    };
    atomic_init(&shared.next_tile, 0);

    // Dispatch threads with per-thread scratch buffers
    int num_threads = get_num_threads();
    if (num_threads > nb_tiles) num_threads = nb_tiles;

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadCtx *tctxs = (ThreadCtx *)calloc(num_threads, sizeof(ThreadCtx));

    for (int i = 0; i < num_threads; i++) {
        tctxs[i].shared = &shared;
        /* scratch is zero-initialized by calloc — no further init needed */
        pthread_create(&threads[i], NULL, tile_worker, &tctxs[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(tctxs);

    // Assembly pass: write tile sizes + copy tile data sequentially
    for (int t = 0; t < nb_tiles; t++) {
        int size = tile_sizes[t];
        write_be16(tile_size_table + t * 2, size);

        if ((out_ptr - output) + size > output_size) {
            fprintf(stderr, "ERROR: Output buffer overflow at tile %d/%d\n", t + 1, nb_tiles);
            free(tile_scratch);
            free(tile_sizes);
            return -1;
        }

        memcpy(out_ptr, tile_scratch + (size_t)t * MAX_TILE_BYTES, size);
        out_ptr += size;
    }

    free(tile_scratch);
    free(tile_sizes);

    // Write total frame size at beginning
    int total_size = (int)(out_ptr - output);
    write_be32(output, total_size);

    return total_size;
}
