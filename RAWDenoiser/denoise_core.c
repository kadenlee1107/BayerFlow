#include "include/denoise_bridge.h"
#include "../include/prores_raw_enc.h"
#include "../include/frame_reader.h"
#include "../include/mov_reader.h"
#include "../include/dng_reader.h"
#include "../include/dng_writer.h"
#include "../include/motion_est.h"
#include "../include/of_apple.h"
#include "../include/temporal_filter.h"
#include "../include/spatial_denoise.h"
#include "../include/temporal_techniques.h"
#include "../include/braw_enc.h"
#include "../include/braw_writer.h"
#include "../include/braw_dec.h"
#include "../include/cineform_enc.h"
#include "../include/denoise.h"
#include "../include/rgb_temporal_filter.h"
#include "../include/exr_writer.h"
#include "../include/training_data.h"
/* #include "include/wiener_temporal.h" */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <sys/stat.h>
#include <pthread.h>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <stdatomic.h>

/* ---- Per-stage timing ---- */
static double timer_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Timing accumulators — diagnostics only, not correctness-critical.
 * Single-writer (main denoise loop is sequential per stage), safe for single-clip use. */
static double t_accum_flow     = 0;
static double t_accum_temporal = 0;
static double t_accum_spatial  = 0;
static double t_accum_cnn      = 0;
static double t_accum_encode   = 0;
static double t_accum_io       = 0;
static double t_accum_unsharp  = 0;
static double t_accum_decode   = 0;  /* just read+decode portion of I/O */
static double t_accum_preproc  = 0;  /* dark/hotpix/green portion of I/O */
static int    t_frame_count    = 0;

/* GPU temporal filter (implemented in Swift/Metal via @_cdecl, falls back to CPU) */
extern void temporal_filter_frame_gpu(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma,
    float chroma_boost, float dist_sigma, float flow_tightening,
    const uint16_t *guide,       /* NULL = use center frame; non-NULL = guided NLM */
    float center_weight);        /* 0.3 = low motion (neighbors dominate), 1.0 = high motion */

extern int metal_gpu_available(void);

/* GPU-resident frame ring: frames live in Metal shared memory (MTLBuffer).
 * CPU writes via returned pointers; GPU reads directly — zero frame copy. */
extern void gpu_ring_init(int num_slots, int width, int height);
extern uint16_t *gpu_ring_frame_ptr(int slot);
extern uint16_t *gpu_ring_denoised_ptr(int slot);
extern void temporal_filter_frame_gpu_ring(
    uint16_t *output,
    const int *ring_slots, const int *use_denoised,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma,
    float chroma_boost, float dist_sigma, float flow_tightening,
    const uint16_t *guide, float center_weight);

extern void temporal_filter_vst_bilateral_gpu_ring(
    uint16_t *output,
    const int *ring_slots, const int *use_denoised,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float noise_sigma,
    float black_level, float shot_gain, float read_noise,
    int max_neighbors);

/* Async variant: commits GPU work and returns immediately.
 * Output is NOT valid until temporal_filter_vst_bilateral_gpu_ring_wait() returns 1.
 * Allows overlapping GPU temporal filter with ANE optical flow on the next frame. */
extern void temporal_filter_vst_bilateral_gpu_ring_commit(
    uint16_t *output,
    const int *ring_slots, const int *use_denoised,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float noise_sigma,
    float black_level, float shot_gain, float read_noise,
    int max_neighbors);
extern int temporal_filter_vst_bilateral_gpu_ring_wait(void);

/* ML denoiser (implemented in Swift/CoreML via @_cdecl) */
extern void denoise_frame_ml(
    uint16_t *output,
    const uint16_t **frames,
    int num_frames, int center_idx,
    int width, int height,
    float noise_sigma);

extern int ml_denoiser_available(void);

/* CNN post-filter (implemented in Swift/CoreML via @_cdecl) */
extern void postfilter_frame_cnn(
    uint16_t *output,
    const uint16_t *input,
    int width, int height);

extern int cnn_postfilter_available(void);

/* MPS GPU 4-channel CNN post-filter (faster replacement for CoreML path) */
extern void postfilter_frame_mps(uint16_t *bayer, int width, int height, float blend_factor);
extern void postfilter_frame_mps_shared(int shared_buf_idx, int width, int height, float blend_factor);
extern int mps_postfilter_available(void);
extern void mps_postfilter_set_protect_subjects(int enable, float protection, int invert_mask);
extern void mps_postfilter_set_noise_model(float black_level, float read_noise, float shot_gain);

/* Zero-copy TF→CNN: temporal filter writes to shared MTLBuffer, CNN reads from it */
extern uint16_t *temporal_filter_vst_bilateral_gpu_ring_shared(
    int shared_buf_idx,
    const int *ring_slots, const int *use_denoised,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float noise_sigma,
    float black_level, float shot_gain, float read_noise);

/* ---- Dark frame subtraction ---- */

/* Load a dark frame clip and average all frames into a single reference.
 * Returns malloc'd uint16_t buffer (caller frees), or NULL on failure. */
static uint16_t *load_dark_frame(const char *path, int expected_w, int expected_h) {
    FrameReader dark_reader;
    if (frame_reader_open(&dark_reader, path) != 0) {
        fprintf(stderr, "dark_frame: cannot open '%s'\n", path);
        return NULL;
    }

    if (dark_reader.width != expected_w || dark_reader.height != expected_h) {
        fprintf(stderr, "dark_frame: dimension mismatch — expected %dx%d, got %dx%d\n",
                expected_w, expected_h, dark_reader.width, dark_reader.height);
        frame_reader_close(&dark_reader);
        return NULL;
    }

    size_t pixels = (size_t)expected_w * expected_h;
    size_t frame_bytes = pixels * sizeof(uint16_t);
    int nf = dark_reader.frame_count;
    if (nf <= 0) nf = 1;

    /* Accumulate in uint32 to avoid overflow when averaging */
    uint32_t *accum = (uint32_t *)calloc(pixels, sizeof(uint32_t));
    uint16_t *tmp   = (uint16_t *)malloc(frame_bytes);
    uint16_t *result = (uint16_t *)malloc(frame_bytes);

    if (!accum || !tmp || !result) {
        free(accum); free(tmp); free(result);
        frame_reader_close(&dark_reader);
        return NULL;
    }

    int frames_read = 0;
    for (int i = 0; i < nf; i++) {
        if (frame_reader_read_frame(&dark_reader, tmp) != 0) break;
        for (size_t p = 0; p < pixels; p++)
            accum[p] += tmp[p];
        frames_read++;
    }

    frame_reader_close(&dark_reader);

    if (frames_read == 0) {
        free(accum); free(tmp); free(result);
        return NULL;
    }

    /* Average */
    for (size_t p = 0; p < pixels; p++)
        result[p] = (uint16_t)(accum[p] / frames_read);

    free(accum);
    free(tmp);
    fprintf(stderr, "dark_frame: averaged %d frames from '%s'\n", frames_read, path);
    return result;
}

/* Subtract dark frame from a raw Bayer frame (saturating subtract to zero). */
static void subtract_dark_frame(uint16_t *frame, const uint16_t *dark, int width, int height) {
    size_t pixels = (size_t)width * height;
#ifdef __ARM_NEON__
    uint16x8_t floor_val = vdupq_n_u16(16);
    size_t i = 0;
    for (; i + 8 <= pixels; i += 8) {
        uint16x8_t f = vld1q_u16(frame + i);
        uint16x8_t d = vld1q_u16(dark + i);
        uint16x8_t sub = vqsubq_u16(f, d);
        vst1q_u16(frame + i, vmaxq_u16(sub, floor_val));
    }
    for (; i < pixels; i++) {
        if (frame[i] > dark[i] + 16)
            frame[i] = frame[i] - dark[i];
        else
            frame[i] = 16;
    }
#else
    for (size_t i = 0; i < pixels; i++) {
        if (frame[i] > dark[i] + 16)
            frame[i] = frame[i] - dark[i];
        else
            frame[i] = 16;
    }
#endif
}

/* ---- Auto dark frame estimation ---- */

/* Estimate fixed-pattern noise by computing per-pixel minimum across sampled frames.
 * Hot pixels and dark current offsets are constant across frames, so the minimum
 * captures them while removing signal variation.
 * Also applies a 3x3 median filter to avoid subtracting real dark scene content —
 * true hot pixels are isolated bright points, not spatially correlated.
 *
 * out_dark: pre-allocated buffer (width * height * sizeof(uint16_t))
 * out_hot_count: number of hot pixels detected (can be NULL)
 * Returns 0 on success, -1 on error. */
static int estimate_dark_frame(const char *input_path, uint16_t *out_dark,
                               int *out_hot_count) {
    FrameReader reader;
    if (frame_reader_open(&reader, input_path) != 0) return -1;

    int width  = reader.width;
    int height = reader.height;
    int num_frames = reader.frame_count;
    size_t pixels = (size_t)width * height;
    size_t frame_bytes = pixels * sizeof(uint16_t);

    /* Sample up to 30 evenly-spaced frames for robustness */
    int sample_count = 30;
    if (sample_count > num_frames) sample_count = num_frames;
    int step = num_frames / sample_count;
    if (step < 1) step = 1;

    uint16_t *frame = (uint16_t *)malloc(frame_bytes);
    uint16_t *min_frame = (uint16_t *)malloc(frame_bytes);
    if (!frame || !min_frame) {
        free(frame); free(min_frame);
        frame_reader_close(&reader);
        return -1;
    }

    /* Initialize min_frame with first frame */
    if (frame_reader_read_frame(&reader, min_frame) != 0) {
        free(frame); free(min_frame);
        frame_reader_close(&reader);
        return -1;
    }

    int frames_sampled = 1;
    int frames_read = 1;

    /* Read remaining sampled frames and track per-pixel minimum */
    for (int s = 1; s < sample_count; s++) {
        int target = s * step;
        while (frames_read < target) {
            if (frame_reader_read_frame(&reader, frame) != 0) goto est_done;
            frames_read++;
        }
        if (frame_reader_read_frame(&reader, frame) != 0) break;
        frames_read++;

        for (size_t p = 0; p < pixels; p++) {
            if (frame[p] < min_frame[p])
                min_frame[p] = frame[p];
        }
        frames_sampled++;
    }

est_done:
    frame_reader_close(&reader);
    free(frame);

    /* Now min_frame contains the per-pixel minimum across sampled frames.
     * Hot pixels: those whose minimum is significantly higher than their
     * 3x3 neighborhood median. Only subtract hot pixel offsets, not
     * genuine dark scene content (which would be spatially correlated). */
    int hot_count = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (size_t)y * width + x;
            uint16_t val = min_frame[idx];

            /* Compute median of 3x3 Bayer-aware neighborhood (same color channel: step 2) */
            uint16_t neighbors[9];
            int n = 0;
            for (int dy = -2; dy <= 2; dy += 2) {
                for (int dx = -2; dx <= 2; dx += 2) {
                    int ny = y + dy, nx = x + dx;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        neighbors[n++] = min_frame[(size_t)ny * width + nx];
                    }
                }
            }

            /* Simple insertion sort for median */
            for (int i = 1; i < n; i++) {
                uint16_t key = neighbors[i];
                int j = i - 1;
                while (j >= 0 && neighbors[j] > key) {
                    neighbors[j + 1] = neighbors[j];
                    j--;
                }
                neighbors[j + 1] = key;
            }
            uint16_t median = neighbors[n / 2];

            /* Hot pixel: minimum is >3x the median neighborhood value,
             * and above a minimum threshold (avoid flagging in truly black areas) */
            if (val > median + 200 && val > median * 3 && val > 500) {
                out_dark[idx] = val - median;  /* subtract the excess */
                hot_count++;
            } else {
                out_dark[idx] = 0;
            }
        }
    }

    free(min_frame);

    if (out_hot_count) *out_hot_count = hot_count;
    fprintf(stderr, "dark_frame: auto-estimated from %d frames, found %d hot pixels\n",
            frames_sampled, hot_count);
    return 0;
}

/* ---- Motion analysis ---- */

static int compare_floats(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

int analyze_motion(
    const char         *input_path,
    float              *avg_motion,
    float              *max_motion,
    DenoiseCProgressCB  progress_cb,
    void               *progress_ctx)
{
    *avg_motion = 0;
    *max_motion = 0;

    FrameReader reader;
    if (frame_reader_open(&reader, input_path) != 0)
        return DENOISE_ERR_INPUT_OPEN;

    int width  = reader.width;
    int height = reader.height;
    int num_frames = reader.frame_count;
    int green_w = width / 2;
    int green_h = height / 2;
    size_t frame_bytes = (size_t)width * height * sizeof(uint16_t);
    size_t green_pixels = (size_t)green_w * green_h;

    /* Pick ~25 evenly-spaced pairs */
    int num_pairs = 25;
    if (num_pairs > num_frames - 1) num_pairs = num_frames - 1;
    if (num_pairs <= 0) { frame_reader_close(&reader); return DENOISE_OK; }

    uint16_t *frame_a = (uint16_t *)malloc(frame_bytes);
    uint16_t *frame_b = (uint16_t *)malloc(frame_bytes);
    uint16_t *green_a = (uint16_t *)malloc(green_pixels * sizeof(uint16_t));
    uint16_t *green_b = (uint16_t *)malloc(green_pixels * sizeof(uint16_t));
    float *fx = (float *)malloc(green_pixels * sizeof(float));
    float *fy = (float *)malloc(green_pixels * sizeof(float));
    float *pair_mags = (float *)malloc(num_pairs * sizeof(float));

    if (!frame_a || !frame_b || !green_a || !green_b || !fx || !fy || !pair_mags) {
        free(frame_a); free(frame_b); free(green_a); free(green_b);
        free(fx); free(fy); free(pair_mags);
        frame_reader_close(&reader);
        return DENOISE_ERR_ALLOC;
    }

    /* Read frames and compute flow for each pair */
    int pairs_done = 0;
    int step = (num_frames - 1) / num_pairs;
    if (step < 1) step = 1;

    /* Read first frame */
    if (frame_reader_read_frame(&reader, frame_a) != 0) goto done;
    extract_green_channel(frame_a, width, height, green_a);
    int frames_read = 1;

    for (int p = 0; p < num_pairs; p++) {
        int target = (p + 1) * step;
        if (target >= num_frames) break;

        /* Skip to target frame */
        while (frames_read <= target) {
            if (frame_reader_read_frame(&reader, frame_b) != 0) goto done;
            frames_read++;
        }
        extract_green_channel(frame_b, width, height, green_b);

        /* Compute optical flow */
        if (compute_apple_flow(green_a, green_b, green_w, green_h, fx, fy) != 0) {
            pair_mags[pairs_done++] = 0;
        } else {
            /* Mean flow magnitude */
            double sum = 0;
            for (size_t i = 0; i < green_pixels; i++)
                sum += sqrtf(fx[i] * fx[i] + fy[i] * fy[i]);
            pair_mags[pairs_done++] = (float)(sum / green_pixels);
        }

        /* Swap: frame_b becomes frame_a for next pair */
        uint16_t *tmp;
        tmp = frame_a; frame_a = frame_b; frame_b = tmp;
        tmp = green_a; green_a = green_b; green_b = tmp;

        if (progress_cb && progress_cb(p + 1, num_pairs, progress_ctx) != 0)
            break;
    }

done:
    if (pairs_done > 0) {
        /* Sort to find median and max */
        qsort(pair_mags, pairs_done, sizeof(float), compare_floats);
        *avg_motion = pair_mags[pairs_done / 2];  /* median */
        *max_motion = pair_mags[pairs_done - 1];   /* max */
    }

    free(frame_a); free(frame_b); free(green_a); free(green_b);
    free(fx); free(fy); free(pair_mags);
    frame_reader_close(&reader);
    return DENOISE_OK;
}

/* ---- Frame Pipeline Infrastructure ---- */

#define MAX_WINDOW 16
#define MAX_OF_RADIUS MAX_WINDOW  /* Outer limit for flow storage */
#define OF_COMPUTE_RADIUS 2       /* Compute real Vision OF only for dist ≤ 2; scale for farther.
                                   * Reduces Vision calls from ~14 to 4 (nearest 2 in each direction).
                                   * flow(N→N±k) ≈ (k/2) × flow(N→N±2): valid because (a) optical
                                   * flow is temporally smooth for the motion we encounter, and (b)
                                   * the bilateral range kernel already downweights distant frames
                                   * with large flow, so approximate flows for far frames are fine. */

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  of_done_cond;
    pthread_cond_t  enc_done_cond;
    pthread_cond_t  cnn_done_cond;
    pthread_cond_t  of_work_cond;
    pthread_cond_t  enc_work_cond;
    pthread_cond_t  cnn_work_cond;
    int of_done;
    int enc_done;
    int cnn_done;
    int of_has_work;
    int enc_has_work;
    int cnn_has_work;
    int should_exit;
    int error;
} PipelineSync;

typedef struct {
    PipelineSync *sync;
    const uint16_t *view_green[MAX_WINDOW];
    int frames_loaded;
    int center_idx;
    int green_w, green_h;
    float **fx_out;
    float **fy_out;
    float **fx_pool;  /* pre-allocated buffer pool (avoids per-frame calloc) */
    float **fy_pool;
    double t_accum;   /* thread-local timing */
} OFThreadCtx;

typedef struct {
    PipelineSync *sync;
    const uint16_t *denoised_in;
    int width, height;
    uint8_t *encode_out;
    int encode_buf_size;
    int encoded_size;
    MovWriter *writer;
    /* DNG output (NULL if MOV output) */
    DngWriter *dng_writer;
    char dng_output_dir[1024];
    const char *dng_frame_name; /* filename to use for this frame */
    int source_bps;
    int source_cfa;
    int dng_source_shift;  /* right-shift for DNG→MOV: 16 - source_bps (0 if native MOV) */
    uint16_t *downshift_buf; /* temp buffer for bit-depth downshift (NULL if not needed) */
    /* BRAW output (NULL if not BRAW) */
    BrawWriter *braw_writer;
    BrawEncContext *braw_enc;
    /* CineForm output (NULL if not CFHD) */
    CfMovWriter *cf_writer;
    uint16_t *yuv_y, *yuv_cb, *yuv_cr;
    double t_accum;   /* thread-local timing */
} EncodeThreadCtx;

typedef struct {
    PipelineSync *sync;
    uint16_t *bayer_buf;       /* denoised buffer to apply spatial+CNN on (in-place) */
    int width, height;
    float spatial_str;
    float noise_sigma;
    float cnn_blend;
    int   cnn_two_pass;
    float cnn_blend_pass2;
    int   use_cnn;
    int   shared_buf_idx;      /* ≥0: use zero-copy shared MTLBuffer path */
    float unsharp_amount;      /* 0 = off, 0.3 = subtle, 0.5 = moderate */
    double t_accum_spatial;
    double t_accum_cnn;
    double t_accum_unsharp;
} CnnThreadCtx;

static void *of_thread_func(void *arg) {
    OFThreadCtx *ctx = (OFThreadCtx *)arg;
    PipelineSync *sync = ctx->sync;

    for (;;) {
        pthread_mutex_lock(&sync->mutex);
        while (!sync->of_has_work && !sync->should_exit)
            pthread_cond_wait(&sync->of_work_cond, &sync->mutex);
        if (sync->should_exit) { pthread_mutex_unlock(&sync->mutex); break; }
        sync->of_has_work = 0;
        pthread_mutex_unlock(&sync->mutex);

        double t0 = timer_now();
        int err = 0;

        /* Collect eligible neighbors for batch OF.
         * Only compute real Vision OF for dist ≤ OF_COMPUTE_RADIUS; far neighbors
         * get scaled flow after the batch call (see below). */
        const uint16_t *batch_nbrs[MAX_WINDOW];
        float *batch_fx[MAX_WINDOW], *batch_fy[MAX_WINDOW];
        int batch_idx[MAX_WINDOW];
        int batch_n = 0;

        for (int i = 0; i < ctx->frames_loaded; i++) {
            if (i == ctx->center_idx) {
                ctx->fx_out[i] = NULL;
                ctx->fy_out[i] = NULL;
                continue;
            }
            int dist = abs(i - ctx->center_idx);
            if (dist > MAX_OF_RADIUS) {
                ctx->fx_out[i] = NULL;
                ctx->fy_out[i] = NULL;
                continue;
            }
            ctx->fx_out[i] = ctx->fx_pool[i];
            ctx->fy_out[i] = ctx->fy_pool[i];
            if (dist <= OF_COMPUTE_RADIUS) {
                batch_nbrs[batch_n] = ctx->view_green[i];
                batch_fx[batch_n]   = ctx->fx_out[i];
                batch_fy[batch_n]   = ctx->fy_out[i];
                batch_idx[batch_n]  = i;
                batch_n++;
            }
        }

        if (batch_n > 0) {
            err = compute_apple_flow_batch(ctx->view_green[ctx->center_idx],
                                           batch_nbrs, batch_n,
                                           ctx->green_w, ctx->green_h,
                                           batch_fx, batch_fy);
        }

        /* Scale flow for far neighbors from nearest computed OF in same direction.
         * flow(N→N±k) ≈ (k / nearest_dist) × flow(N→N±nearest_dist). */
        size_t npix_of = (size_t)ctx->green_w * ctx->green_h;
        for (int i = 0; i < ctx->frames_loaded; i++) {
            if (i == ctx->center_idx || !ctx->fx_out[i]) continue;
            int dist = abs(i - ctx->center_idx);
            if (dist <= OF_COMPUTE_RADIUS) continue;
            /* Find nearest computed neighbor in same direction */
            int sign = (i > ctx->center_idx) ? 1 : -1;
            int ref_i = -1;
            int best_dist = INT_MAX;
            for (int b = 0; b < batch_n; b++) {
                int bj = batch_idx[b];
                if ((bj - ctx->center_idx) * sign <= 0) continue;
                int bd = abs(bj - ctx->center_idx);
                if (bd < best_dist) { best_dist = bd; ref_i = bj; }
            }
            if (ref_i < 0 || !ctx->fx_out[ref_i]) {
                memset(ctx->fx_out[i], 0, npix_of * sizeof(float));
                memset(ctx->fy_out[i], 0, npix_of * sizeof(float));
            } else {
                float scale = (float)dist / (float)best_dist;
                for (size_t p = 0; p < npix_of; p++) {
                    ctx->fx_out[i][p] = ctx->fx_out[ref_i][p] * scale;
                    ctx->fy_out[i][p] = ctx->fy_out[ref_i][p] * scale;
                }
            }
        }
        ctx->t_accum += timer_now() - t0;

        pthread_mutex_lock(&sync->mutex);
        if (err) sync->error = DENOISE_ERR_ALLOC;
        sync->of_done = 1;
        pthread_cond_signal(&sync->of_done_cond);
        pthread_mutex_unlock(&sync->mutex);
    }
    return NULL;
}

/* Bayer RGGB 12-bit → YUV 4:2:2 10-bit (bilinear debayer + BT.709 conversion).
 * Input: 12-bit Bayer in 16-bit container (values 0-4095).
 * Output: Y (width×height), Cb (width/2×height), Cr (width/2×height), all 10-bit. */
static void bayer_to_yuv422p10(const uint16_t *bayer, int width, int height,
                                uint16_t *y_out, uint16_t *cb_out, uint16_t *cr_out) {
    /* Bilinear debayer: for each pixel, interpolate missing R/G/B from 2×2 Bayer neighbors.
     * RGGB pattern: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B */
    for (int row = 0; row < height; row++) {
        int r0 = (row > 0) ? row - 1 : row + 1;
        int r1 = (row < height - 1) ? row + 1 : row - 1;

        for (int col = 0; col < width; col += 2) {
            int c1 = (col + 1 < width) ? col + 1 : col - 1;
            int c0_prev = (col > 0) ? col - 1 : col + 1;

            /* Even column: depends on row parity for Bayer phase */
            int R, G, B;
            if ((row & 1) == 0) {
                /* Row even: col even=R, col odd=Gr */
                int px_r = bayer[row * width + col];       /* R */
                int px_g = bayer[row * width + c1];        /* Gr */
                int px_b = bayer[r1 * width + c1];         /* B (diagonal) */
                int px_g2 = bayer[r1 * width + col];       /* Gb */
                R = px_r;
                G = (px_g + px_g2) / 2;
                B = px_b;
            } else {
                /* Row odd: col even=Gb, col odd=B */
                int px_gb = bayer[row * width + col];      /* Gb */
                int px_b  = bayer[row * width + c1];       /* B */
                int px_r  = bayer[r0 * width + col];       /* R (above) — actually Gr's neighbor */
                /* More accurate: get R from even-row, even-col */
                int px_r2 = bayer[r0 * width + c0_prev];   /* R from row above, col-1 */
                int px_g  = bayer[r0 * width + c1];        /* Gr from row above */
                R = (px_r + px_r2) / 2;
                G = (px_gb + px_g) / 2;
                B = px_b;
            }

            /* BT.709 12-bit RGB → 12-bit YCbCr */
            int Y12  = (int)(0.2126f * R + 0.7152f * G + 0.0722f * B);
            int Cb12 = (int)((B - Y12) * 0.5389f) + 2048;
            int Cr12 = (int)((R - Y12) * 0.6350f) + 2048;

            /* Clamp and shift 12-bit → 10-bit */
            if (Y12 < 0) Y12 = 0; if (Y12 > 4095) Y12 = 4095;
            if (Cb12 < 0) Cb12 = 0; if (Cb12 > 4095) Cb12 = 4095;
            if (Cr12 < 0) Cr12 = 0; if (Cr12 > 4095) Cr12 = 4095;

            int y_idx = row * width + col;
            y_out[y_idx]     = (uint16_t)(Y12 >> 2);

            /* Second pixel of the pair */
            int R2, G2, B2;
            if ((row & 1) == 0) {
                R2 = (bayer[row * width + col] + ((col + 2 < width) ? bayer[row * width + col + 2] : bayer[row * width + col])) / 2;
                G2 = bayer[row * width + c1];
                B2 = (bayer[r1 * width + c1] + ((c1 >= 2) ? bayer[r1 * width + c1 - 2] : bayer[r1 * width + c1])) / 2;
            } else {
                R2 = bayer[r0 * width + c1];  /* Gr position — neighbor R */
                int px_r_above = bayer[r0 * width + col];
                R2 = (R2 + px_r_above) / 2;
                G2 = (bayer[row * width + col] + bayer[row * width + c1]) / 2;  /* avg Gb and B neighbors */
                B2 = bayer[row * width + c1];
            }

            int Y12_2  = (int)(0.2126f * R2 + 0.7152f * G2 + 0.0722f * B2);
            int Cb12_2 = (int)((B2 - Y12_2) * 0.5389f) + 2048;
            int Cr12_2 = (int)((R2 - Y12_2) * 0.6350f) + 2048;

            if (Y12_2 < 0) Y12_2 = 0; if (Y12_2 > 4095) Y12_2 = 4095;
            if (Cb12_2 < 0) Cb12_2 = 0; if (Cb12_2 > 4095) Cb12_2 = 4095;
            if (Cr12_2 < 0) Cr12_2 = 0; if (Cr12_2 > 4095) Cr12_2 = 4095;

            y_out[y_idx + 1] = (uint16_t)(Y12_2 >> 2);

            /* 4:2:2 chroma: average the pair */
            cb_out[row * (width / 2) + col / 2] = (uint16_t)(((Cb12 + Cb12_2) / 2) >> 2);
            cr_out[row * (width / 2) + col / 2] = (uint16_t)(((Cr12 + Cr12_2) / 2) >> 2);
        }
    }
}

/* Helper: get the filename (no path) of a source DNG for the given absolute frame index */
static const char *dng_frame_filename(DngReader *dr, int abs_frame_idx) {
    if (!dr || abs_frame_idx < 0 || abs_frame_idx >= dr->file_count) return "frame.dng";
    const char *full = dr->file_list[abs_frame_idx];
    const char *slash = strrchr(full, '/');
    return slash ? slash + 1 : full;
}

/* Helper: write one denoised frame as DNG (inline, not threaded) */
static int write_dng_frame_inline(DngWriter *dng_w, const uint16_t *bayer,
                                  int w, int h, const char *output_dir,
                                  const char *frame_name,
                                  int source_bps, int source_cfa) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s", output_dir, frame_name);
    return dng_writer_write_frame(dng_w, bayer, w, h, path, source_bps, source_cfa);
}

static void *encode_thread_func(void *arg) {
    EncodeThreadCtx *ctx = (EncodeThreadCtx *)arg;
    PipelineSync *sync = ctx->sync;

    for (;;) {
        pthread_mutex_lock(&sync->mutex);
        while (!sync->enc_has_work && !sync->should_exit)
            pthread_cond_wait(&sync->enc_work_cond, &sync->mutex);
        if (sync->should_exit) { pthread_mutex_unlock(&sync->mutex); break; }
        sync->enc_has_work = 0;
        pthread_mutex_unlock(&sync->mutex);

        double t0 = timer_now();
        int enc_err = 0;

        if (ctx->dng_writer) {
            /* DNG output: write denoised Bayer directly */
            char path[2048];
            snprintf(path, sizeof(path), "%s/%s",
                     ctx->dng_output_dir, ctx->dng_frame_name);
            if (dng_writer_write_frame(ctx->dng_writer, ctx->denoised_in,
                                       ctx->width, ctx->height, path,
                                       ctx->source_bps, ctx->source_cfa) != 0)
                enc_err = 1;
        } else if (ctx->braw_writer) {
            /* BRAW output: downshift 16→12 bit, BRAW encode, write packet */
            int npix = ctx->width * ctx->height;
            if (!ctx->downshift_buf) { enc_err = 1; goto enc_done; }
            for (int i = 0; i < npix; i++)
                ctx->downshift_buf[i] = ctx->denoised_in[i] >> 4;
            int encoded = braw_enc_encode_frame(ctx->braw_enc,
                                                 ctx->downshift_buf, ctx->width,
                                                 ctx->encode_out, ctx->encode_buf_size);
            ctx->encoded_size = encoded;
            if (encoded > 0)
                braw_writer_add_frame(ctx->braw_writer, ctx->encode_out, encoded);
            else
                enc_err = 1;
        } else if (ctx->cf_writer) {
            /* CineForm output: Bayer → YUV 4:2:2 10-bit → CFHD encode */
            bayer_to_yuv422p10(ctx->denoised_in, ctx->width, ctx->height,
                               ctx->yuv_y, ctx->yuv_cb, ctx->yuv_cr);
            const uint16_t *planes[3] = { ctx->yuv_y, ctx->yuv_cb, ctx->yuv_cr };
            int strides[3] = { ctx->width, ctx->width / 2, ctx->width / 2 };
            if (cf_mov_writer_add_frame(ctx->cf_writer, planes, strides) != CF_ENC_OK)
                enc_err = 1;
        } else {
            /* MOV output: ProRes RAW encode + mov_writer */
            const uint16_t *enc_src = ctx->denoised_in;
            if (ctx->dng_source_shift > 0 && ctx->downshift_buf) {
                /* DNG→MOV: downshift 16-bit to 12-bit for ProRes RAW encoder */
                int npix = ctx->width * ctx->height;
                int shift = ctx->dng_source_shift;
                for (int i = 0; i < npix; i++)
                    ctx->downshift_buf[i] = ctx->denoised_in[i] >> shift;
                enc_src = ctx->downshift_buf;
            }
            int encoded = encode_frame(enc_src, ctx->width, ctx->height,
                                       ctx->encode_out, ctx->encode_buf_size);
            ctx->encoded_size = encoded;
            if (encoded > 0)
                mov_writer_add_frame(ctx->writer, ctx->encode_out, encoded);
            else
                enc_err = 1;
        }

        enc_done:
        ctx->t_accum += timer_now() - t0;

        pthread_mutex_lock(&sync->mutex);
        if (enc_err) sync->error = DENOISE_ERR_ENCODE;
        sync->enc_done = 1;
        pthread_cond_signal(&sync->enc_done_cond);
        pthread_mutex_unlock(&sync->mutex);
    }
    return NULL;
}

/* Bayer-aware unsharp mask: restores micro-contrast lost during temporal+CNN.
 * Operates per Bayer channel using stride-2 neighbors (same color).
 * amount: 0 = off, 0.3 = subtle, 0.5 = moderate, 1.0 = strong.
 * threshold: minimum detail magnitude to sharpen (avoids amplifying residual noise). */
static void unsharp_mask_bayer(uint16_t *frame, int width, int height,
                               float amount, float noise_sigma) {
    if (amount <= 0) return;

    size_t npix = (size_t)width * height;
    uint16_t *orig = (uint16_t *)malloc(npix * sizeof(uint16_t));
    if (!orig) return;
    memcpy(orig, frame, npix * sizeof(uint16_t));

    /* Soft threshold: smoothly ramp sharpening from 0 at noise floor to
     * full amount above it.  This avoids the hard threshold that caused
     * channel-dependent pass rates (green > R/B) → green color cast. */
    float thresh = noise_sigma * 0.25f;
    if (thresh < 8.0f)   thresh = 8.0f;
    if (thresh > 1000.0f) thresh = 1000.0f;

    /* Process interior pixels (2-pixel border uses same-color neighbors at stride 2) */
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int idx = y * width + x;
            int center = orig[idx];

            /* 8 same-color Bayer neighbors at stride 2 (3x3 in Bayer space) */
            int sum = orig[(y-2)*width + (x-2)] + orig[(y-2)*width + x] + orig[(y-2)*width + (x+2)]
                    + orig[y*width + (x-2)]                              + orig[y*width + (x+2)]
                    + orig[(y+2)*width + (x-2)] + orig[(y+2)*width + x] + orig[(y+2)*width + (x+2)];
            int avg = sum >> 3;  /* /8 */

            int detail = center - avg;
            float abs_d = (float)(detail > 0 ? detail : -detail);

            /* UNet CNN handles signal-dependent denoising via noise map input,
             * so unsharp mask applies uniformly (no shadow ramp needed). */
            float local_amount = amount;

            /* Soft ramp: gain goes from 0 (at detail=0) to full local_amount (at detail>=thresh).
             * All Bayer channels get proportional sharpening → no color cast. */
            float gain = local_amount;
            if (abs_d < thresh) gain *= abs_d / thresh;

            int sharpened = center + (int)(gain * (float)detail);
            if (sharpened < 0) sharpened = 0;
            if (sharpened > 65535) sharpened = 65535;
            frame[idx] = (uint16_t)sharpened;
        }
    }

    free(orig);
}

/* ---- Training Data Patch Extraction ---- */

#define TRAINING_PATCH_SIZE 256
#define TRAINING_PATCHES_PER_FRAME 2

/* Quick variance estimate over a subsampled grid within a patch region.
 * Used to find "interesting" patches (texture, edges) vs flat areas. */
static float quick_patch_variance(const uint16_t *frame, int stride,
                                  int px, int py, int pw, int ph) {
    float sum = 0, sum2 = 0;
    int count = 0;
    /* Sample every 8th pixel for speed */
    for (int y = py; y < py + ph; y += 8) {
        for (int x = px; x < px + pw; x += 8) {
            float v = (float)frame[y * stride + x];
            sum += v;
            sum2 += v * v;
            count++;
        }
    }
    if (count < 2) return 0;
    float mean = sum / count;
    return sum2 / count - mean * mean;
}

/* Returns 1 if patch is valid for training, 0 if corrupted/useless.
 * Catches decoder bugs (clamped pixels), denoiser no-ops, and flat/dead patches
 * before they can poison the training set. */
static int validate_patch(const uint16_t *noisy, const uint16_t *denoised,
                          int patch_w, int patch_h) {
    int total = patch_w * patch_h;
    int noisy_clamped = 0, denoised_clamped = 0;
    int identical = 0;
    int64_t noisy_sum = 0;
    int64_t noisy_sq_sum = 0;

    for (int i = 0; i < total; i++) {
        uint16_t n = noisy[i], d = denoised[i];
        noisy_sum += n;
        noisy_sq_sum += (int64_t)n * n;
        if (n == 64 || n == 65456) noisy_clamped++;     /* 4<<4 or 4091<<4 */
        if (d == 64 || d == 65456) denoised_clamped++;
        if (n == d) identical++;
    }

    /* Reject if >1% pixels at clamp bounds (decoder bug) */
    if (noisy_clamped > total / 100) return 0;
    if (denoised_clamped > total / 100) return 0;

    /* Reject if noisy == denoised for >95% of pixels (denoiser did nothing) */
    if (identical > total * 95 / 100) return 0;

    /* Reject near-zero variance (flat/dead patch — useless for training) */
    double mean = (double)noisy_sum / total;
    double var = (double)noisy_sq_sum / total - mean * mean;
    if (var < 100.0) return 0;

    /* Reject if mean pixel value is unreasonably extreme */
    if (mean < 500.0 || mean > 64000.0) return 0;

    return 1;
}

/* Extract and submit 2 training patches from the current frame.
 * noisy_frame: raw center frame before denoising
 * denoised_frame: temporal-filtered output
 * Called inline after temporal filter, before CNN kicks off. */
static void extract_training_patches(
    const uint16_t *noisy_frame,
    const uint16_t *denoised_frame,
    int width, int height,
    int frame_idx, float noise_sigma, float flow_mag,
    const char *camera_model, int iso)
{
    if (width < TRAINING_PATCH_SIZE + 4 || height < TRAINING_PATCH_SIZE + 4)
        return;

    int max_x = (width - TRAINING_PATCH_SIZE) / 2;
    int max_y = (height - TRAINING_PATCH_SIZE) / 2;
    if (max_x <= 0 || max_y <= 0) return;

    /* Temp buffers for the 256×256 patches */
    size_t patch_pixels = TRAINING_PATCH_SIZE * TRAINING_PATCH_SIZE;
    uint16_t *noisy_patch = (uint16_t *)malloc(patch_pixels * sizeof(uint16_t));
    uint16_t *denoised_patch = (uint16_t *)malloc(patch_pixels * sizeof(uint16_t));
    if (!noisy_patch || !denoised_patch) {
        free(noisy_patch); free(denoised_patch);
        return;
    }

    int positions[TRAINING_PATCHES_PER_FRAME][2];

    /* Patch 1: random position (2px-aligned for Bayer grid) */
    positions[0][0] = (rand() % max_x) * 2;
    positions[0][1] = (rand() % max_y) * 2;

    /* Patch 2: highest-variance region from 8 random candidates */
    float best_var = -1;
    positions[1][0] = positions[0][0];
    positions[1][1] = positions[0][1];
    for (int trial = 0; trial < 8; trial++) {
        int cx = (rand() % max_x) * 2;
        int cy = (rand() % max_y) * 2;
        float var = quick_patch_variance(noisy_frame, width, cx, cy,
                                         TRAINING_PATCH_SIZE, TRAINING_PATCH_SIZE);
        if (var > best_var) {
            best_var = var;
            positions[1][0] = cx;
            positions[1][1] = cy;
        }
    }

    for (int p = 0; p < TRAINING_PATCHES_PER_FRAME; p++) {
        int px = positions[p][0];
        int py = positions[p][1];

        /* Copy patch from full frame (row-by-row) */
        for (int y = 0; y < TRAINING_PATCH_SIZE; y++) {
            memcpy(&noisy_patch[y * TRAINING_PATCH_SIZE],
                   &noisy_frame[(py + y) * width + px],
                   TRAINING_PATCH_SIZE * sizeof(uint16_t));
            memcpy(&denoised_patch[y * TRAINING_PATCH_SIZE],
                   &denoised_frame[(py + y) * width + px],
                   TRAINING_PATCH_SIZE * sizeof(uint16_t));
        }

        /* Validate before submitting — reject corrupted/useless patches */
        if (!validate_patch(noisy_patch, denoised_patch,
                            TRAINING_PATCH_SIZE, TRAINING_PATCH_SIZE))
            continue;

        training_data_submit_patch(
            noisy_patch, denoised_patch,
            TRAINING_PATCH_SIZE, TRAINING_PATCH_SIZE,
            width, height, px, py,
            frame_idx, noise_sigma, flow_mag,
            camera_model, iso);
    }

    free(noisy_patch);
    free(denoised_patch);
}

static void *cnn_thread_func(void *arg) {
    CnnThreadCtx *ctx = (CnnThreadCtx *)arg;
    PipelineSync *sync = ctx->sync;

    for (;;) {
        pthread_mutex_lock(&sync->mutex);
        while (!sync->cnn_has_work && !sync->should_exit)
            pthread_cond_wait(&sync->cnn_work_cond, &sync->mutex);
        if (sync->should_exit) { pthread_mutex_unlock(&sync->mutex); break; }
        sync->cnn_has_work = 0;
        pthread_mutex_unlock(&sync->mutex);

        /* Spatial denoise (CPU / NEON) */
        if (ctx->spatial_str > 0) {
            double t0 = timer_now();
            spatial_denoise_frame(ctx->bayer_buf, ctx->width, ctx->height,
                                  ctx->noise_sigma, ctx->spatial_str);
            ctx->t_accum_spatial += timer_now() - t0;
        }

        /* CNN post-filter (MPS GPU / CoreML) */
        if (ctx->use_cnn) {
            double t0 = timer_now();
            if (ctx->shared_buf_idx >= 0 && mps_postfilter_available()) {
                /* Zero-copy path: CNN reads directly from TF's shared MTLBuffer */
                postfilter_frame_mps_shared(ctx->shared_buf_idx, ctx->width, ctx->height, ctx->cnn_blend);
                if (ctx->cnn_two_pass)
                    postfilter_frame_mps_shared(ctx->shared_buf_idx, ctx->width, ctx->height, ctx->cnn_blend_pass2);
            } else if (mps_postfilter_available()) {
                postfilter_frame_mps(ctx->bayer_buf, ctx->width, ctx->height, ctx->cnn_blend);
                if (ctx->cnn_two_pass)
                    postfilter_frame_mps(ctx->bayer_buf, ctx->width, ctx->height, ctx->cnn_blend_pass2);
            } else if (cnn_postfilter_available()) {
                postfilter_frame_cnn(ctx->bayer_buf, ctx->bayer_buf, ctx->width, ctx->height);
            }
            ctx->t_accum_cnn += timer_now() - t0;
        }

        /* Unsharp mask: restore micro-contrast after temporal+CNN smoothing */
        if (ctx->unsharp_amount > 0) {
            double t0 = timer_now();
            unsharp_mask_bayer(ctx->bayer_buf, ctx->width, ctx->height,
                               ctx->unsharp_amount, ctx->noise_sigma);
            ctx->t_accum_unsharp += timer_now() - t0;
        }

        pthread_mutex_lock(&sync->mutex);
        sync->cnn_done = 1;
        pthread_cond_signal(&sync->cnn_done_cond);
        pthread_mutex_unlock(&sync->mutex);
    }
    return NULL;
}

/* Build a view of W frames from the ring buffer starting at base */
static void build_ring_view(uint16_t **ring, int base, int W_plus_1, int count,
                            const uint16_t **view) {
    for (int i = 0; i < count; i++)
        view[i] = ring[(base + i) % W_plus_1];
}

/* Build green view, substituting cached denoised greens for past neighbors */
static void build_green_view_with_cache(
    uint16_t **noisy_greens, uint16_t **denoised_greens, int *valid,
    int base, int W_plus_1, int count, int center_idx,
    const uint16_t **view_out)
{
    for (int i = 0; i < count; i++) {
        int slot = (base + i) % W_plus_1;
        if (i < center_idx && denoised_greens && valid && valid[slot]) {
            view_out[i] = denoised_greens[slot];  /* past neighbor: use cached denoised */
        } else {
            view_out[i] = noisy_greens[slot];      /* center or future: use noisy */
        }
    }
}

/* Build frame view, substituting cached denoised frames for past neighbors.
 * This is the key to recursive temporal filtering: past frames have already
 * been denoised, so they carry much less noise → higher NLM weights → deeper
 * noise reduction that compounds over time.
 * use_cache: set to 0 during motion to prevent ghost compounding. */
static void build_frame_view_with_cache(
    uint16_t **raw_frames, uint16_t **denoised_frames, int *valid,
    int base, int W_plus_1, int count, int center_idx,
    const uint16_t **view_out, int use_cache)
{
    for (int i = 0; i < count; i++) {
        int slot = (base + i) % W_plus_1;
        if (use_cache && i < center_idx && denoised_frames && valid && valid[slot]) {
            view_out[i] = denoised_frames[slot];  /* past neighbor: use denoised */
        } else {
            view_out[i] = raw_frames[slot];        /* center or future: use raw */
        }
    }
}

/* Build ring slot indices + denoised flags for the GPU ring path.
 * Called alongside build_frame_view_with_cache() to produce the parallel
 * arrays needed by temporal_filter_frame_gpu_ring(). */
static void build_ring_slot_info(
    int *valid, int base, int W_plus_1, int count, int center_idx,
    int *ring_slots_out, int *use_denoised_out, int use_cache)
{
    for (int i = 0; i < count; i++) {
        int slot = (base + i) % W_plus_1;
        ring_slots_out[i] = slot;
        use_denoised_out[i] = (use_cache && i < center_idx && valid && valid[slot]) ? 1 : 0;
    }
}

static void clear_flow_set(float **fx, float **fy, int n) {
    /* Just NULL pointers — buffers are in pre-allocated pool, not freed per-frame */
    for (int i = 0; i < n; i++) {
        fx[i] = NULL;
        fy[i] = NULL;
    }
}

/* Compute motion-adaptive center weight from optical flow arrays.
 * Averages flow magnitude across all neighbor flow fields for this frame.
 * Also returns the average flow magnitude via out_avg_mag (if non-NULL).
 * Returns 0.3 for static scenes (neighbors dominate).
 * For motion, ramps gently to 0.8 — the 5×5 NLM patches handle ghosting
 * rejection, so center weight stays low for maximum temporal averaging. */
static float compute_adaptive_center_weight(
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int green_w, int green_h,
    float *out_avg_mag)
{
    int n_flow = green_w * green_h;
    double total_mag = 0;
    int count = 0;

    for (int i = 0; i < num_frames; i++) {
        if (i == center_idx || !flows_x[i] || !flows_y[i]) continue;
        /* Sample every 16th pixel for speed (full resolution not needed) */
        double frame_mag = 0;
        int samples = 0;
        for (int j = 0; j < n_flow; j += 16) {
            float fx = flows_x[i][j];
            float fy = flows_y[i][j];
            frame_mag += sqrtf(fx * fx + fy * fy);
            samples++;
        }
        if (samples > 0) {
            total_mag += frame_mag / samples;
            count++;
        }
    }

    if (count == 0) {
        if (out_avg_mag) *out_avg_mag = 0.0f;
        return 0.3f;  /* no valid neighbors → default */
    }

    float avg_mag = (float)(total_mag / count);
    if (out_avg_mag) *out_avg_mag = avg_mag;

    /* Ramp: 0 px → 0.3, ≥3 px → 0.6
     * Wider ramp (3px instead of 2px) smooths the transition between
     * heavy temporal averaging (static) and motion-limited averaging.
     * Ghost suppression handled by adaptive dark-pixel boost in
     * normalize kernel (per-pixel, based on neighbor_sum). */
    float t = avg_mag / 3.0f;
    if (t > 1.0f) t = 1.0f;
    return 0.3f + 0.3f * t;
}

static int compute_center(int f, int half_w, int W, int num_frames, int frames_loaded) {
    int center;
    if (f < half_w)
        center = f;
    else if (f >= num_frames - half_w)
        center = W - (num_frames - f);
    else
        center = half_w;
    if (center < 0) center = 0;
    if (center >= frames_loaded) center = frames_loaded - 1;
    return center;
}

/* ---- RGB denoising pipeline (for R3D and other RGB-output formats) ---- */

/* Convert planar RGB to 16-bit luma for optical flow computation. */
static void rgb_to_luma_u16(const uint16_t *rgb_planar, int width, int height,
                             uint16_t *luma_out) {
    size_t n = (size_t)width * height;
    const uint16_t *r = rgb_planar;
    const uint16_t *g = rgb_planar + n;
    const uint16_t *b = rgb_planar + 2 * n;
    for (size_t i = 0; i < n; i++) {
        float l = 0.2126f * r[i] + 0.7152f * g[i] + 0.0722f * b[i];
        luma_out[i] = (uint16_t)(l > 65535.0f ? 65535 : (l + 0.5f));
    }
}

static int denoise_file_rgb(
    FrameReader         *reader,
    const char          *output_path,
    const DenoiseCConfig *cfg,
    DenoiseCProgressCB   progress_cb,
    void                *progress_ctx)
{
    int width      = reader->width;
    int height     = reader->height;
    int num_frames = reader->frame_count;
    size_t plane_pixels    = (size_t)width * height;
    size_t rgb_frame_bytes = plane_pixels * 3 * sizeof(uint16_t);
    size_t luma_bytes      = plane_pixels * sizeof(uint16_t);

    /* Trim range */
    int trim_start = cfg->start_frame > 0 ? cfg->start_frame : 0;
    int trim_end   = (cfg->end_frame > 0 && cfg->end_frame <= num_frames)
                     ? cfg->end_frame : num_frames;
    if (trim_start >= trim_end) { trim_start = 0; trim_end = num_frames; }
    int range_frames = trim_end - trim_start;

    float fps = reader->fps > 0 ? reader->fps : 24.0f;

    /* Filter config */
    RgbTemporalFilterConfig tcfg;
    rgb_temporal_filter_init(&tcfg);
    tcfg.window_size = cfg->window_size > 0 ? cfg->window_size : 15;
    tcfg.strength    = cfg->strength > 0 ? cfg->strength : 1.5f;
    tcfg.noise_sigma = cfg->noise_sigma;

    int W = tcfg.window_size;
    if (W > range_frames) W = range_frames;
    if (W > MAX_WINDOW)   W = MAX_WINDOW;
    int half_w = W / 2;

    /* Open output */
    int use_exr = (cfg->output_format == 4);
    int use_cfhd = (cfg->output_format == 5);
    CfMovWriter *rgb_cf_writer = NULL;
    uint16_t *rgb_yuv_y = NULL, *rgb_yuv_cb = NULL, *rgb_yuv_cr = NULL;

    if (use_exr) {
        mkdir(output_path, 0755);
        fprintf(stderr, "denoise_file_rgb: EXR sequence output → %s/\n", output_path);
    } else if (use_cfhd) {
        if (cf_mov_writer_open(&rgb_cf_writer, output_path, width, height, fps,
                                CF_QUALITY_FILM3P) != CF_ENC_OK) {
            fprintf(stderr, "denoise_file_rgb: cannot open CineForm output '%s'\n", output_path);
            return DENOISE_ERR_OUTPUT_OPEN;
        }
        rgb_yuv_y  = (uint16_t *)malloc(plane_pixels * sizeof(uint16_t));
        rgb_yuv_cb = (uint16_t *)malloc((size_t)(width / 2) * height * sizeof(uint16_t));
        rgb_yuv_cr = (uint16_t *)malloc((size_t)(width / 2) * height * sizeof(uint16_t));
        if (!rgb_yuv_y || !rgb_yuv_cb || !rgb_yuv_cr) {
            cf_mov_writer_close(rgb_cf_writer);
            free(rgb_yuv_y); free(rgb_yuv_cb); free(rgb_yuv_cr);
            return DENOISE_ERR_ALLOC;
        }
        fprintf(stderr, "denoise_file_rgb: CineForm output → %s\n", output_path);
    } else {
        if (prores444_writer_open(output_path, width, height, fps) != 0) {
            fprintf(stderr, "denoise_file_rgb: cannot open ProRes 4444 output '%s'\n",
                    output_path);
            return DENOISE_ERR_OUTPUT_OPEN;
        }
    }

    /* Allocate sliding window buffers */
    uint16_t **rgb_ring  = (uint16_t **)calloc(W, sizeof(uint16_t *));
    uint16_t **luma_ring = (uint16_t **)calloc(W, sizeof(uint16_t *));
    uint16_t *denoised   = (uint16_t *)malloc(rgb_frame_bytes);
    float **flow_x = (float **)calloc(W, sizeof(float *));
    float **flow_y = (float **)calloc(W, sizeof(float *));

    int ret = DENOISE_OK;

    if (!rgb_ring || !luma_ring || !denoised || !flow_x || !flow_y) {
        ret = DENOISE_ERR_ALLOC; goto rgb_cleanup;
    }

    for (int i = 0; i < W; i++) {
        rgb_ring[i]  = (uint16_t *)malloc(rgb_frame_bytes);
        luma_ring[i] = (uint16_t *)malloc(luma_bytes);
        flow_x[i]    = (float *)malloc(plane_pixels * sizeof(float));
        flow_y[i]    = (float *)malloc(plane_pixels * sizeof(float));
        if (!rgb_ring[i] || !luma_ring[i] || !flow_x[i] || !flow_y[i]) {
            ret = DENOISE_ERR_ALLOC; goto rgb_cleanup;
        }
    }

    /* Skip to trim start */
    if (trim_start > 0) {
        uint16_t *skip = (uint16_t *)malloc(rgb_frame_bytes);
        if (skip) {
            for (int i = 0; i < trim_start; i++)
                frame_reader_read_frame_rgb(reader, skip);
            free(skip);
            fprintf(stderr, "denoise_file_rgb: skipped %d frames to trim start\n", trim_start);
        }
    }

    /* Pre-load initial window */
    int frames_loaded = 0;
    for (int i = 0; i < W && i < range_frames; i++) {
        if (frame_reader_read_frame_rgb(reader, rgb_ring[i]) != 0) break;
        rgb_to_luma_u16(rgb_ring[i], width, height, luma_ring[i]);
        frames_loaded++;
    }

    fprintf(stderr, "denoise_file_rgb: loaded %d/%d frames, %dx%d, fps=%.2f\n",
            frames_loaded, W, width, height, fps);

    if (frames_loaded == 0) {
        ret = DENOISE_ERR_INPUT_OPEN; goto rgb_cleanup;
    }

    /* Auto-estimate noise from first frame if not provided */
    if (tcfg.noise_sigma == 0)
        tcfg.noise_sigma = rgb_temporal_filter_estimate_noise(rgb_ring[0], width, height);

    fprintf(stderr, "denoise_file_rgb: noise=%.1f, strength=%.2f, window=%d, "
            "format=%d → ProRes 4444 XQ\n",
            tcfg.noise_sigma, tcfg.strength, W, reader->format);

    /* Main processing loop */
    int next_frame_idx = trim_start + frames_loaded;
    double t_wall_start = timer_now();

    for (int f = 0; f < range_frames; f++) {
        if (progress_cb && progress_cb(f, range_frames, progress_ctx) != 0) {
            ret = DENOISE_ERR_CANCELLED; break;
        }

        /* Center index within the current window */
        int center = compute_center(f, half_w, W, range_frames, frames_loaded);

        /* Compute optical flow from center luma to each neighbor */
        double t0 = timer_now();
        for (int i = 0; i < frames_loaded; i++) {
            if (i == center) {
                flow_x[i] = flow_x[i]; /* keep pointer, but flow is unused */
                flow_y[i] = flow_y[i];
                continue;
            }
            compute_apple_flow(luma_ring[center], luma_ring[i],
                               width, height,
                               flow_x[i], flow_y[i]);
        }
        t_accum_flow += timer_now() - t0;

        /* Build frame pointer array with NULL flows for center */
        const uint16_t *frame_ptrs[MAX_WINDOW];
        const float *fx_ptrs[MAX_WINDOW];
        const float *fy_ptrs[MAX_WINDOW];
        for (int i = 0; i < frames_loaded; i++) {
            frame_ptrs[i] = rgb_ring[i];
            fx_ptrs[i] = (i == center) ? NULL : flow_x[i];
            fy_ptrs[i] = (i == center) ? NULL : flow_y[i];
        }

        /* Temporal denoise (VST + bilateral on RGB channels) */
        t0 = timer_now();
        rgb_temporal_filter_frame(
            denoised,
            frame_ptrs,
            fx_ptrs,
            fy_ptrs,
            frames_loaded, center,
            width, height,
            &tcfg);
        t_accum_temporal += timer_now() - t0;

        /* Write output frame */
        t0 = timer_now();
        if (use_exr) {
            char exr_path[4096];
            snprintf(exr_path, sizeof(exr_path), "%s/frame_%06d.exr",
                     output_path, trim_start + f);
            if (exr_write_frame(exr_path, denoised, width, height) != 0) {
                fprintf(stderr, "denoise_file_rgb: EXR write failed at frame %d\n", f);
                ret = DENOISE_ERR_ENCODE; break;
            }
        } else if (use_cfhd) {
            /* RGB 16-bit → YUV 4:2:2 10-bit → CineForm encode */
            uint16_t *r_in = denoised;
            uint16_t *g_in = denoised + plane_pixels;
            uint16_t *b_in = denoised + plane_pixels * 2;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col += 2) {
                    size_t idx0 = (size_t)row * width + col;
                    size_t idx1 = idx0 + 1;
                    /* BT.709 RGB→YCbCr (16-bit input → 10-bit output) */
                    int R0 = r_in[idx0], G0 = g_in[idx0], B0 = b_in[idx0];
                    int Y0 = (int)(0.2126f * R0 + 0.7152f * G0 + 0.0722f * B0);
                    int R1 = r_in[idx1], G1 = g_in[idx1], B1 = b_in[idx1];
                    int Y1 = (int)(0.2126f * R1 + 0.7152f * G1 + 0.0722f * B1);
                    int Cb = (int)((((B0 + B1) / 2) - (Y0 + Y1) / 2) * 0.5389f) + 32768;
                    int Cr = (int)((((R0 + R1) / 2) - (Y0 + Y1) / 2) * 0.6350f) + 32768;
                    /* Clamp and downshift 16→10 bit */
                    if (Y0 < 0) Y0 = 0; if (Y0 > 65535) Y0 = 65535;
                    if (Y1 < 0) Y1 = 0; if (Y1 > 65535) Y1 = 65535;
                    if (Cb < 0) Cb = 0; if (Cb > 65535) Cb = 65535;
                    if (Cr < 0) Cr = 0; if (Cr > 65535) Cr = 65535;
                    rgb_yuv_y[idx0] = (uint16_t)(Y0 >> 6);
                    rgb_yuv_y[idx1] = (uint16_t)(Y1 >> 6);
                    rgb_yuv_cb[row * (width / 2) + col / 2] = (uint16_t)(Cb >> 6);
                    rgb_yuv_cr[row * (width / 2) + col / 2] = (uint16_t)(Cr >> 6);
                }
            }
            const uint16_t *planes[3] = { rgb_yuv_y, rgb_yuv_cb, rgb_yuv_cr };
            int strides[3] = { width, width / 2, width / 2 };
            if (cf_mov_writer_add_frame(rgb_cf_writer, planes, strides) != CF_ENC_OK) {
                fprintf(stderr, "denoise_file_rgb: CineForm encode failed at frame %d\n", f);
                ret = DENOISE_ERR_ENCODE; break;
            }
        } else {
            if (prores444_writer_write_frame(denoised) != 0) {
                fprintf(stderr, "denoise_file_rgb: write failed at frame %d\n", f);
                ret = DENOISE_ERR_ENCODE; break;
            }
        }
        t_accum_encode += timer_now() - t0;

        t_frame_count++;

        /* Slide window: when center is at half_w, shift left by one frame */
        if (f >= half_w && next_frame_idx < trim_end) {
            /* Rotate buffer pointers: oldest (0) → becomes new last slot */
            uint16_t *old_rgb  = rgb_ring[0];
            uint16_t *old_luma = luma_ring[0];
            float    *old_fx   = flow_x[0];
            float    *old_fy   = flow_y[0];
            for (int i = 0; i < frames_loaded - 1; i++) {
                rgb_ring[i]  = rgb_ring[i + 1];
                luma_ring[i] = luma_ring[i + 1];
                flow_x[i]   = flow_x[i + 1];
                flow_y[i]   = flow_y[i + 1];
            }
            rgb_ring[frames_loaded - 1]  = old_rgb;
            luma_ring[frames_loaded - 1] = old_luma;
            flow_x[frames_loaded - 1]   = old_fx;
            flow_y[frames_loaded - 1]   = old_fy;

            /* Read next frame into the recycled slot */
            double tio = timer_now();
            if (frame_reader_read_frame_rgb(reader, rgb_ring[frames_loaded - 1]) == 0) {
                rgb_to_luma_u16(rgb_ring[frames_loaded - 1], width, height,
                                luma_ring[frames_loaded - 1]);
                next_frame_idx++;
            } else {
                frames_loaded--;
            }
            t_accum_io += timer_now() - tio;
        }

        if (f % 10 == 0 || f == range_frames - 1) {
            double elapsed = timer_now() - t_wall_start;
            double actual_fps = (f + 1) / elapsed;
            fprintf(stderr, "denoise_file_rgb: frame %d/%d (%.2f fps)\n",
                    f + 1, range_frames, actual_fps);
        }
    }

    /* Print timing summary */
    {
        double total = timer_now() - t_wall_start;
        fprintf(stderr, "\ndenoise_file_rgb: DONE — %d frames in %.1fs (%.2f fps)\n",
                t_frame_count, total, t_frame_count / total);
        fprintf(stderr, "  flow=%.1fs  temporal=%.1fs  encode=%.1fs  io=%.1fs\n",
                t_accum_flow, t_accum_temporal, t_accum_encode, t_accum_io);
    }

rgb_cleanup:
    if (use_cfhd) {
        cf_mov_writer_close(rgb_cf_writer);
        free(rgb_yuv_y); free(rgb_yuv_cb); free(rgb_yuv_cr);
    } else if (!use_exr) {
        prores444_writer_close();
    }
    free(denoised);
    if (rgb_ring) {
        for (int i = 0; i < W; i++) free(rgb_ring[i]);
        free(rgb_ring);
    }
    if (luma_ring) {
        for (int i = 0; i < W; i++) free(luma_ring[i]);
        free(luma_ring);
    }
    if (flow_x) {
        for (int i = 0; i < W; i++) free(flow_x[i]);
        free(flow_x);
    }
    if (flow_y) {
        for (int i = 0; i < W; i++) free(flow_y[i]);
        free(flow_y);
    }
    return ret;
}

/* ---- Denoise pipeline ---- */

int denoise_file(
    const char          *input_path,
    const char          *output_path,
    const DenoiseCConfig *cfg,
    DenoiseCProgressCB   progress_cb,
    void                *progress_ctx)
{
    /* ---- Open input ---- */
    FrameReader reader;
    if (frame_reader_open(&reader, input_path) != 0) {
        fprintf(stderr, "denoise_file: cannot open input '%s'\n", input_path);
        return DENOISE_ERR_INPUT_OPEN;
    }

    int width      = reader.width;
    int height     = reader.height;
    int num_frames = reader.frame_count;

    /* Validate dimensions */
    if (width < 2 || height < 2 || width > 16384 || height > 16384 ||
        width % 2 != 0 || height % 2 != 0) {
        fprintf(stderr, "denoise_file: invalid dimensions %dx%d (must be even, 2-16384)\n",
                width, height);
        frame_reader_close(&reader);
        return DENOISE_ERR_INPUT_OPEN;
    }

    size_t frame_pixels = (size_t)width * height;
    size_t frame_bytes  = frame_pixels * sizeof(uint16_t);

    /* ---- RGB format dispatch ---- */
    if (reader.is_rgb) {
        fprintf(stderr, "denoise_file: RGB input detected (format=%d) — "
                "routing to RGB pipeline (ProRes 4444 XQ output)\n",
                reader.format);
        int rgb_ret = denoise_file_rgb(&reader, output_path, cfg,
                                        progress_cb, progress_ctx);
        frame_reader_close(&reader);
        return rgb_ret;
    }

    /* ---- Resolve trim range ---- */
    int trim_start = cfg->start_frame > 0 ? cfg->start_frame : 0;
    int trim_end   = (cfg->end_frame > 0 && cfg->end_frame <= num_frames) ? cfg->end_frame : num_frames;
    if (trim_start >= trim_end) { trim_start = 0; trim_end = num_frames; }
    int range_frames = trim_end - trim_start;

    if (trim_start > 0 || trim_end < num_frames)
        fprintf(stderr, "denoise_core: trim range [%d, %d) — %d of %d frames\n",
                trim_start, trim_end, range_frames, num_frames);

    /* ---- Training data collection ---- */
    int collect_training = cfg->collect_training_data && training_data_enabled();
    char td_camera_model[256] = "";
    int td_iso = cfg->detected_iso;
    if (collect_training) {
        denoise_probe_camera(input_path, td_camera_model, sizeof(td_camera_model), &td_iso);
        srand((unsigned)time(NULL)); /* seed RNG for patch position selection */
        fprintf(stderr, "denoise_core: training data collection ENABLED (camera=%s, ISO=%d)\n",
                td_camera_model[0] ? td_camera_model : "unknown", td_iso);
    }

    /* ---- Resolve output format ---- */
    int output_is_dng = 0;
    int output_is_braw = 0;
    int output_is_cfhd = 0;
    if (cfg->output_format == 1)       { /* force MOV */ }
    else if (cfg->output_format == 2)  output_is_dng = 1;
    else if (cfg->output_format == 3)  output_is_braw = 1;
    else if (cfg->output_format == 5)  output_is_cfhd = 1;
    else {
        /* Auto: match input format */
        output_is_dng = (reader.format == FORMAT_CINEMADNG);
        output_is_braw = (reader.format == FORMAT_BRAW);
    }

    int dng_source_bps = 16, dng_source_cfa = 0;
    const char *dng_template_path = NULL;
    DngReader *dng_reader_impl = NULL;
    int dng_to_mov = 0; /* CinemaDNG input → MOV output (need bit-depth downshift) */
    if (reader.format == FORMAT_CINEMADNG) {
        frame_reader_get_dng_info(&reader, &dng_source_bps, &dng_source_cfa, &dng_template_path);
        dng_reader_impl = (DngReader *)reader.impl;
        if (!output_is_dng && !output_is_braw) dng_to_mov = 1;
    }

    /* ---- Open output ---- */
    MovWriter writer;
    DngWriter *dng_writer = NULL;
    BrawWriter braw_writer_obj;
    BrawEncContext braw_enc_ctx;
    CfMovWriter *cf_writer = NULL;
    uint16_t *cfhd_yuv_y = NULL, *cfhd_yuv_cb = NULL, *cfhd_yuv_cr = NULL;
    memset(&writer, 0, sizeof(writer));
    memset(&braw_writer_obj, 0, sizeof(braw_writer_obj));
    memset(&braw_enc_ctx, 0, sizeof(braw_enc_ctx));

    if (output_is_dng) {
        /* Create output directory */
        mkdir(output_path, 0755);
        dng_writer = dng_writer_open(dng_template_path);
        if (!dng_writer) {
            fprintf(stderr, "denoise_file: cannot open DNG writer from template '%s'\n",
                    dng_template_path ? dng_template_path : "(null)");
            frame_reader_close(&reader);
            return DENOISE_ERR_OUTPUT_OPEN;
        }
        fprintf(stderr, "denoise_core: DNG output → %s/ (%d-bit source, CFA=%d)\n",
                output_path, dng_source_bps, dng_source_cfa);
    } else if (output_is_braw) {
        /* BRAW output: configure encoder from source BRAW packet */
        BrawEncConfig enc_cfg;
        memset(&enc_cfg, 0, sizeof(enc_cfg));

        /* Read first frame packet to extract quant tables and tile layout */
        {
            BrawMovInfo bmov;
            if (braw_mov_parse(input_path, &bmov) != BRAW_OK) {
                fprintf(stderr, "denoise_file: cannot parse source BRAW for encoder config\n");
                frame_reader_close(&reader);
                return DENOISE_ERR_OUTPUT_OPEN;
            }
            uint8_t *pkt = NULL;
            size_t pkt_sz = 0;
            if (braw_mov_read_frame(input_path, &bmov, 0, &pkt, &pkt_sz) != BRAW_OK) {
                braw_mov_free(&bmov);
                frame_reader_close(&reader);
                return DENOISE_ERR_OUTPUT_OPEN;
            }
            if (braw_enc_config_from_packet(&enc_cfg, pkt, pkt_sz, width, height) != BRAW_ENC_OK) {
                free(pkt);
                braw_mov_free(&bmov);
                frame_reader_close(&reader);
                return DENOISE_ERR_OUTPUT_OPEN;
            }
            double braw_fps = bmov.fps > 0 ? bmov.fps : 25.0;
            free(pkt);
            braw_mov_free(&bmov);

            if (braw_enc_init(&braw_enc_ctx, &enc_cfg) != BRAW_ENC_OK) {
                fprintf(stderr, "denoise_file: BRAW encoder init failed\n");
                frame_reader_close(&reader);
                return DENOISE_ERR_OUTPUT_OPEN;
            }
            if (braw_writer_open(&braw_writer_obj, output_path, width, height, braw_fps) != 0) {
                braw_enc_free(&braw_enc_ctx);
                fprintf(stderr, "denoise_file: cannot open BRAW output '%s'\n", output_path);
                frame_reader_close(&reader);
                return DENOISE_ERR_OUTPUT_OPEN;
            }
            braw_writer_copy_metadata(&braw_writer_obj, input_path);
            fprintf(stderr, "denoise_core: BRAW output → %s (qscale=%d)\n",
                    output_path, enc_cfg.qscale[0]);
        }
    } else if (output_is_cfhd) {
        /* CineForm output: Bayer → YUV 4:2:2 10-bit → CFHD MOV */
        double fps = reader.fps > 0 ? reader.fps : 24.0;
        if (cf_mov_writer_open(&cf_writer, output_path, width, height, fps,
                                CF_QUALITY_FILM3P) != CF_ENC_OK) {
            fprintf(stderr, "denoise_file: cannot open CineForm output '%s'\n", output_path);
            frame_reader_close(&reader);
            return DENOISE_ERR_OUTPUT_OPEN;
        }
        /* Allocate YUV temp buffers */
        cfhd_yuv_y  = (uint16_t *)malloc((size_t)width * height * sizeof(uint16_t));
        cfhd_yuv_cb = (uint16_t *)malloc((size_t)(width / 2) * height * sizeof(uint16_t));
        cfhd_yuv_cr = (uint16_t *)malloc((size_t)(width / 2) * height * sizeof(uint16_t));
        if (!cfhd_yuv_y || !cfhd_yuv_cb || !cfhd_yuv_cr) {
            cf_mov_writer_close(cf_writer);
            free(cfhd_yuv_y); free(cfhd_yuv_cb); free(cfhd_yuv_cr);
            frame_reader_close(&reader);
            return DENOISE_ERR_ALLOC;
        }
        fprintf(stderr, "denoise_core: CineForm output → %s (%.3f fps)\n", output_path, fps);
    } else {
        if (mov_writer_open(&writer, output_path, width, height) != 0) {
            fprintf(stderr, "denoise_file: cannot open output '%s'\n", output_path);
            frame_reader_close(&reader);
            return DENOISE_ERR_OUTPUT_OPEN;
        }

        /* MOV-specific: copy metadata and frame header from source */
        if (reader.format == FORMAT_MOV) {
            mov_writer_copy_metadata(&writer, input_path);
            if (set_frame_header_from_mov(input_path) != 0)
                fprintf(stderr, "denoise_file: warning — could not read source frame header\n");
        }
        if (dng_to_mov) {
            fprintf(stderr, "denoise_core: DNG→MOV export (%d-bit → 12-bit ProRes RAW)\n",
                    dng_source_bps);
            /* Default frame_header_template is used (no source MOV to extract from) */
        }
    }

    /* ---- Allocations ---- */
    int encode_buf_size = 50 * 1024 * 1024;

    denoise_init(&g_denoise_config);

    TemporalFilterConfig tcfg;
    temporal_filter_init(&tcfg);
    tcfg.window_size    = cfg->window_size > 0 ? cfg->window_size : 15;
    tcfg.strength       = cfg->strength > 0 ? cfg->strength : 1.5f;
    tcfg.noise_sigma    = cfg->noise_sigma;
    float spatial_str   = cfg->spatial_strength;
    int tf_mode = cfg->temporal_filter_mode; /* 0=NLM, 2=VST+Bilateral */

    /* VST noise model — use calibrated values from profiler, fall back to S1M2 defaults */
    float vst_bl = cfg->black_level > 0 ? cfg->black_level : 6032.0f;
    float vst_sg = cfg->shot_gain   > 0 ? cfg->shot_gain   : 180.0f;
    float vst_rn = cfg->read_noise  > 0 ? cfg->read_noise  : 616.0f;

    /* When CNN post-filter is active, disable spatial denoise entirely.
     * Analysis shows temporal filter already reduces shadow noise below the texture
     * floor — any additional spatial smoothing destroys real detail. The CNN handles
     * residual noise in bright areas via its blend kernel. */
    if (cfg->use_cnn_postfilter && spatial_str > 0 &&
        (mps_postfilter_available() || cnn_postfilter_available())) {
        fprintf(stderr, "denoise_core: spatial OFF (CNN active, shadow texture protection)\n");
        spatial_str = 0;
    }

    /* Temporal filter tuning — the 5×5 NLM patches do the heavy lifting for
     * ghosting rejection, so flow params can be moderate rather than extreme. */
    float tf_chroma_boost    = 1.0f;
    {
        float wb_r = 1.0f, wb_b = 1.0f;
        if (frame_reader_probe_wb_gains(input_path, &wb_r, &wb_b) == 0 &&
            (wb_r > 1.0f || wb_b > 1.0f)) {
            /* Use sqrt scaling: WB gains amplify noise by gain×, but full
             * compensation over-smooths bright skin (pulling/waxy artifacts).
             * sqrt() gives ~1.3× for typical daylight (R/G=1.95, B/G=1.69)
             * — enough to help chroma noise without destroying texture. */
            tf_chroma_boost = 1.2f; /* auto-tune override */
            fprintf(stderr, "denoise_core: WB R/G=%.2f B/G=%.2f → chroma_boost=%.2f (sqrt scaling)\n",
                    wb_r, wb_b, tf_chroma_boost);
        }
    }
    float tf_dist_sigma      = 2.0f;  /* moderate: 3px flow → 32% confidence */
    float tf_flow_tightening = 1.5f;  /* ghost suppression now in normalize kernel
                                          * (dark-pixel center weight boost) */

    int W      = tcfg.window_size;

    /* Auto-clamp window based on motion (safety net — flow-adaptive bilateral
     * weights handle most ghosting, this catches extreme cases) */
    if (cfg->motion_avg > 0) {
        int max_w;
        if (cfg->motion_avg < 2.0f)       max_w = 15; /* low motion — full NLM window */
        else if (cfg->motion_avg < 5.0f)   max_w = 9;  /* moderate-high motion */
        else                               max_w = 5;  /* extreme motion */
        if (W > max_w) {
            fprintf(stderr, "denoise_core: motion=%.1f px → auto-clamped window %d → %d\n",
                    cfg->motion_avg, W, max_w);
            W = max_w;
        }
    }

    if (W > range_frames) W = range_frames;
    if (W > MAX_WINDOW) W = MAX_WINDOW;
    int half_w = W / 2;
    int green_w = width  / 2;
    int green_h = height / 2;
    int W1 = W + 1; /* ring buffer size: W+1 for lookahead */

    /* Ring buffer: W+1 slots for window frames + green channels */
    uint16_t **window_frames = (uint16_t **)calloc(W1, sizeof(uint16_t *));
    uint16_t **green_frames  = (uint16_t **)calloc(W1, sizeof(uint16_t *));

    /* Denoised green cache: after temporal filter, cache the denoised green channel
     * so future frames can use it for higher-quality optical flow (past neighbors). */
    uint16_t **denoised_greens      = (uint16_t **)calloc(W1, sizeof(uint16_t *));
    int       *denoised_green_valid = (int *)calloc(W1, sizeof(int));

    /* Denoised full-frame cache: recursive temporal filtering.
     * Past frames that have already been denoised are used as reference data
     * instead of raw noisy frames, dramatically reducing noise in the average. */
    uint16_t **denoised_cache       = (uint16_t **)calloc(W1, sizeof(uint16_t *));
    int       *denoised_cache_valid = (int *)calloc(W1, sizeof(int));

    /* Double-buffered denoised output and encode buffers */
    uint16_t *denoised_bufs[2] = {NULL, NULL};
    uint8_t  *encode_bufs[2]   = {NULL, NULL};
    uint16_t *downshift_bufs[2] = {NULL, NULL}; /* DNG→MOV: 16→12 bit temp buffers */

    /* Double-buffered optical flow arrays — pre-allocated pool to avoid
     * per-frame calloc/free of ~600MB (14 pairs × 22MB each).
     * of_fx_sets/of_fy_sets point into pool; center slot set to NULL. */
    float *of_fx_sets[2][MAX_WINDOW];
    float *of_fy_sets[2][MAX_WINDOW];
    float *flow_pool_fx[2][MAX_WINDOW];
    float *flow_pool_fy[2][MAX_WINDOW];
    memset(of_fx_sets, 0, sizeof(of_fx_sets));
    memset(of_fy_sets, 0, sizeof(of_fy_sets));
    memset(flow_pool_fx, 0, sizeof(flow_pool_fx));
    memset(flow_pool_fy, 0, sizeof(flow_pool_fy));

    uint16_t  *dark_frame    = NULL;
    HotPixelProfile *hp_profile = NULL;
    uint16_t *bootstrap_center_green = NULL; /* allocated later if bootstrap_flow enabled */
    int ret = DENOISE_OK;
    int win_base = 0; /* ring buffer base */
    int ping = 0;     /* double-buffer index */

    if (!window_frames || !green_frames || !denoised_greens || !denoised_green_valid ||
        !denoised_cache || !denoised_cache_valid) {
        ret = DENOISE_ERR_ALLOC; goto cleanup;
    }

    /* Allocate GPU-resident ring buffers for frames and denoised cache.
     * These are MTLBuffer-backed (storageModeShared on Apple Silicon) —
     * CPU writes directly, GPU reads without any memcpy. */
    gpu_ring_init(W1, width, height);
    for (int i = 0; i < W1; i++) {
        window_frames[i] = gpu_ring_frame_ptr(i);
        green_frames[i]  = (uint16_t *)malloc((size_t)green_w * green_h * sizeof(uint16_t));
        denoised_greens[i] = (uint16_t *)malloc((size_t)green_w * green_h * sizeof(uint16_t));
        denoised_cache[i] = gpu_ring_denoised_ptr(i);
        if (!window_frames[i] || !green_frames[i] || !denoised_greens[i] || !denoised_cache[i]) {
            ret = DENOISE_ERR_ALLOC; goto cleanup;
        }
    }

    /* Track whether denoised_bufs were malloc'd (vs shared MTLBuffer pointers).
     * When zero-copy TF→CNN is active, denoised_bufs[ping] gets reassigned to
     * a shared GPU buffer pointer — we must NOT free those at cleanup. */
    int denoised_bufs_owned[2] = {1, 1};
    for (int b = 0; b < 2; b++) {
        denoised_bufs[b] = (uint16_t *)malloc(frame_bytes);
        encode_bufs[b]   = (uint8_t *)malloc(encode_buf_size);
        if (!denoised_bufs[b] || !encode_bufs[b]) { ret = DENOISE_ERR_ALLOC; goto cleanup; }
        if (dng_to_mov || output_is_braw) {
            downshift_bufs[b] = (uint16_t *)malloc(frame_bytes);
            if (!downshift_bufs[b]) { ret = DENOISE_ERR_ALLOC; goto cleanup; }
        }
    }

    /* Pre-allocate optical flow buffer pool (2 ping-pong × MAX_WINDOW slots) */
    {
        size_t n_flow = (size_t)green_w * green_h;
        for (int p = 0; p < 2; p++)
            for (int i = 0; i < MAX_WINDOW; i++) {
                flow_pool_fx[p][i] = (float *)malloc(n_flow * sizeof(float));
                flow_pool_fy[p][i] = (float *)malloc(n_flow * sizeof(float));
                if (!flow_pool_fx[p][i] || !flow_pool_fy[p][i]) {
                    ret = DENOISE_ERR_ALLOC; goto cleanup;
                }
            }
    }

    /* ---- Load or auto-estimate dark frame ---- */
    dark_frame = NULL;
    if (cfg->dark_frame_path && cfg->dark_frame_path[0] != '\0') {
        dark_frame = load_dark_frame(cfg->dark_frame_path, width, height);
        if (!dark_frame)
            fprintf(stderr, "denoise_core: dark frame load failed — continuing without\n");
    } else if (cfg->auto_dark_frame) {
        dark_frame = (uint16_t *)calloc(frame_pixels, sizeof(uint16_t));
        if (dark_frame) {
            int hot = 0;
            if (estimate_dark_frame(input_path, dark_frame, &hot) != 0 || hot == 0) {
                free(dark_frame);
                dark_frame = NULL;
                if (hot == 0)
                    fprintf(stderr, "denoise_core: no hot pixels detected — skipping dark frame\n");
            }
        }
    }

    /* ---- Load calibrated hot pixel profile ---- */
    if (cfg->hotpixel_profile_path && cfg->hotpixel_profile_path[0] != '\0' && cfg->detected_iso > 0) {
        hp_profile = hotpixel_profile_load(cfg->hotpixel_profile_path, cfg->detected_iso);
    }

    /* ---- Skip to trim start ---- */
    if (trim_start > 0) {
        uint16_t *skip_buf = (uint16_t *)malloc(frame_bytes);
        if (!skip_buf) {
            fprintf(stderr, "denoise_core: failed to allocate skip buffer (%d bytes)\n", (int)frame_bytes);
            frame_reader_close(&reader);
            return DENOISE_ERR_ALLOC;
        }
        for (int i = 0; i < trim_start; i++)
            frame_reader_read_frame(&reader, skip_buf);
        free(skip_buf);
        fprintf(stderr, "denoise_core: skipped %d frames to trim start\n", trim_start);
    }

    /* ---- Pre-load initial window ---- */
    int frames_loaded = 0;
    int frames_remaining = num_frames - trim_start;  /* frames left in reader */
    for (int i = 0; i < W && i < frames_remaining; i++) {
        if (frame_reader_read_frame(&reader, window_frames[i]) != 0) break;
        /* Temporary: dump raw decoded frame 0 for decoder verification */
        if (i == 0) {
            FILE *dbgf = fopen("/tmp/native_decoded_f0.raw", "wb");
            if (dbgf) { fwrite(window_frames[0], sizeof(uint16_t), frame_pixels, dbgf); fclose(dbgf); }
        }
        if (dark_frame) subtract_dark_frame(window_frames[i], dark_frame, width, height);
        if (hp_profile) hotpixel_profile_apply(hp_profile, window_frames[i], width, height);
        extract_green_channel(window_frames[i], width, height, green_frames[i]);
        spatial_filter_green(green_frames[i], green_w, green_h, tcfg.noise_sigma);
        frames_loaded++;
    }

    fprintf(stderr, "denoise_core: loaded %d/%d frames into initial window (GPU: %s)\n",
            frames_loaded, W, metal_gpu_available() ? "Metal" : "CPU");

    if (frames_loaded == 0) {
        fprintf(stderr, "denoise_core: failed to read any frames — aborting\n");
        ret = DENOISE_ERR_INPUT_OPEN;
        goto cleanup;
    }

    if (tcfg.noise_sigma == 0)
        tcfg.noise_sigma = temporal_filter_estimate_noise(window_frames[0], width, height);

    fprintf(stderr, "DIAG: noise_sigma=%.1f, h=sigma*strength=%.1f, strength=%.2f, window=%d, spatial=%.2f, cnn=%d, mps=%d, coreml=%d\n",
            tcfg.noise_sigma, tcfg.noise_sigma * tcfg.strength, tcfg.strength,
            W, cfg->spatial_strength, cfg->use_cnn_postfilter,
            mps_postfilter_available(), cnn_postfilter_available());

    /* Subject protection: reduce CNN denoise on detected persons */
    if (mps_postfilter_available()) {
        mps_postfilter_set_protect_subjects(cfg->protect_subjects, 0.9f, cfg->invert_mask);
        if (cfg->protect_subjects)
            fprintf(stderr, "denoise_core: subject protection ENABLED%s\n",
                    cfg->invert_mask ? " (inverted — background boost)" : "");

        /* Set calibrated noise model for CNN input channel.
         * TODO: detect camera and use per-camera params.
         * For now: S1M2 defaults (BL=6032, RN=616, SG=180). */
        mps_postfilter_set_noise_model(6032.0f, 616.0f, 180.0f);
    }

    /* ---- Main encode loop (pipelined) ---- */
    int next_frame_to_read = trim_start + frames_loaded;
    int use_ml = cfg->use_ml && ml_denoiser_available();
    double t_wall_start = timer_now();

    /* CNN blend: UNet-Lite with SSIM loss preserves texture natively,
     * so we can use aggressive blend without destroying detail.
     * The old DnCNN (L1 loss) needed conservative 0.65 blend. */
    float cnn_blend = 0.9f;
    int cnn_two_pass = 0;
    float cnn_blend_pass2 = 0.4f;

    /* Noise-adaptive: ramp from 0.9 to 1.0 for extreme noise */
    if (tcfg.noise_sigma > 80.0f) {
        float nt = (tcfg.noise_sigma - 80.0f) / 100.0f;
        if (nt > 1.0f) nt = 1.0f;
        cnn_blend = 0.9f + 0.1f * nt;
        /* Two-pass for extreme noise (sigma > 120) */
        if (tcfg.noise_sigma > 120.0f) {
            cnn_two_pass = 1;
            float nt2 = (tcfg.noise_sigma - 120.0f) / 60.0f;
            if (nt2 > 1.0f) nt2 = 1.0f;
            cnn_blend_pass2 = 0.3f + 0.2f * nt2;
        }
    }
    fprintf(stderr, "denoise_core: noise=%.0f → CNN blend=%.2f%s\n",
            tcfg.noise_sigma, cnn_blend, cnn_two_pass ? " (2-pass)" : " (1-pass)");
    if (cnn_two_pass)
        fprintf(stderr, "denoise_core: CNN pass2 blend=%.2f\n", cnn_blend_pass2);

    /* Motion-adaptive: slight boost when temporal filter is less effective */
    if (cfg->motion_avg > 0) {
        /* Ramp from 0.65 to 0.75 for high motion — much gentler than before */
        float t = cfg->motion_avg / 5.0f;
        if (t > 1.0f) t = 1.0f;
        float motion_blend = 0.65f + 0.10f * t;
        if (motion_blend > cnn_blend) cnn_blend = motion_blend;
        /* Only enable 2-pass for extreme motion (>4px), not 1.5px */
        if (cfg->motion_avg > 4.0f && !cnn_two_pass) cnn_two_pass = 1;
        fprintf(stderr, "denoise_core: motion=%.1f px → CNN blend=%.2f%s\n",
                cfg->motion_avg, cnn_blend, cnn_two_pass ? " (2-pass)" : "");
    }

    /* Bootstrapped flow: re-compute OF from denoised center green for high-motion clips.
     * This breaks the chicken-and-egg problem: noisy frames → noisy flow → rejected neighbors.
     * After temporal pass 1 produces a cleaner center, we re-estimate flow and run pass 2. */
    int bootstrap_flow = 1;
    if (bootstrap_flow) {
        bootstrap_center_green = (uint16_t *)malloc((size_t)green_w * green_h * sizeof(uint16_t));
        if (!bootstrap_center_green) { ret = DENOISE_ERR_ALLOC; goto cleanup; }
        fprintf(stderr, "denoise_core: bootstrap flow ENABLED (motion=%.1f px)\n", cfg->motion_avg);
    }

    /* Helper macro: run GPU denoise stages on denoised_bufs[p] */
    #define GPU_SPATIAL_CNN(p) do { \
        if (spatial_str > 0) { \
            double _t0 = timer_now(); \
            spatial_denoise_frame(denoised_bufs[(p)], width, height, tcfg.noise_sigma, spatial_str); \
            t_accum_spatial += timer_now() - _t0; \
        } \
        if (cfg->use_cnn_postfilter) { \
            double _t0 = timer_now(); \
            if (mps_postfilter_available()) { \
                if (t_frame_count == 0) fprintf(stderr, "CNN: MPS postfilter ACTIVE (blend=%.2f, 2pass=%d)\n", cnn_blend, cnn_two_pass); \
                postfilter_frame_mps(denoised_bufs[(p)], width, height, cnn_blend); \
                if (cnn_two_pass) \
                    postfilter_frame_mps(denoised_bufs[(p)], width, height, cnn_blend_pass2); \
            } else if (cnn_postfilter_available()) { \
                if (t_frame_count == 0) fprintf(stderr, "CNN: CoreML postfilter ACTIVE\n"); \
                postfilter_frame_cnn(denoised_bufs[(p)], denoised_bufs[(p)], width, height); \
            } else { \
                if (t_frame_count == 0) fprintf(stderr, "CNN: NO postfilter available! mps=%d coreml=%d\n", mps_postfilter_available(), cnn_postfilter_available()); \
            } \
            t_accum_cnn += timer_now() - _t0; \
        } \
    } while(0)

    /* Helper: read next frame into a ring slot with corrections */
    #define READ_INTO_SLOT(slot) do { \
        double _t_dec = timer_now(); \
        int _rd_ok = frame_reader_read_frame(&reader, window_frames[(slot)]); \
        t_accum_decode += timer_now() - _t_dec; \
        if (_rd_ok == 0) { \
            double _t_pp = timer_now(); \
            if (dark_frame) subtract_dark_frame(window_frames[(slot)], dark_frame, width, height); \
            if (hp_profile) hotpixel_profile_apply(hp_profile, window_frames[(slot)], width, height); \
            extract_green_channel(window_frames[(slot)], width, height, green_frames[(slot)]); \
            spatial_filter_green(green_frames[(slot)], green_w, green_h, tcfg.noise_sigma); \
            t_accum_preproc += timer_now() - _t_pp; \
            denoised_green_valid[(slot)] = 0; /* invalidate cached denoised green */ \
            denoised_cache_valid[(slot)] = 0; /* invalidate cached denoised frame */ \
            next_frame_to_read++; \
        } else { \
            frames_loaded--; \
        } \
    } while(0)

    if (!use_ml) {
        /* ===== 4-STAGE PIPELINE: OF(Neural Engine) | Temporal(GPU) | Spatial+CNN(MPS) | Encode(CPU) ===== */

        PipelineSync sync;
        memset(&sync, 0, sizeof(sync));
        pthread_mutex_init(&sync.mutex, NULL);
        pthread_cond_init(&sync.of_done_cond, NULL);
        pthread_cond_init(&sync.enc_done_cond, NULL);
        pthread_cond_init(&sync.cnn_done_cond, NULL);
        pthread_cond_init(&sync.of_work_cond, NULL);
        pthread_cond_init(&sync.enc_work_cond, NULL);
        pthread_cond_init(&sync.cnn_work_cond, NULL);
        sync.cnn_done = 1;  /* no CNN work pending initially */
        sync.enc_done = 1;  /* no encode pending initially */

        OFThreadCtx of_ctx;
        memset(&of_ctx, 0, sizeof(of_ctx));
        of_ctx.sync = &sync;
        of_ctx.green_w = green_w;
        of_ctx.green_h = green_h;
        /* Pool pointers set per-kick (ping-dependent) */

        EncodeThreadCtx enc_ctx;
        memset(&enc_ctx, 0, sizeof(enc_ctx));
        enc_ctx.sync = &sync;
        enc_ctx.width = width;
        enc_ctx.height = height;
        enc_ctx.encode_buf_size = encode_buf_size;
        enc_ctx.writer = &writer;
        enc_ctx.dng_writer = dng_writer;
        enc_ctx.source_bps = dng_source_bps;
        enc_ctx.source_cfa = dng_source_cfa;
        enc_ctx.dng_source_shift = dng_to_mov ? (16 - dng_source_bps) : 0;
        if (output_is_dng)
            strncpy(enc_ctx.dng_output_dir, output_path, sizeof(enc_ctx.dng_output_dir) - 1);
        if (output_is_braw) {
            enc_ctx.braw_writer = &braw_writer_obj;
            enc_ctx.braw_enc = &braw_enc_ctx;
        }
        if (output_is_cfhd) {
            enc_ctx.cf_writer = cf_writer;
            enc_ctx.yuv_y  = cfhd_yuv_y;
            enc_ctx.yuv_cb = cfhd_yuv_cb;
            enc_ctx.yuv_cr = cfhd_yuv_cr;
        }

        CnnThreadCtx cnn_ctx;
        memset(&cnn_ctx, 0, sizeof(cnn_ctx));
        cnn_ctx.sync = &sync;
        cnn_ctx.width = width;
        cnn_ctx.height = height;
        cnn_ctx.spatial_str = spatial_str;
        cnn_ctx.noise_sigma = tcfg.noise_sigma;
        if (cfg->use_cnn_postfilter && mps_postfilter_available()) {
            cnn_ctx.cnn_blend = cnn_blend;
            cnn_ctx.cnn_two_pass = cnn_two_pass;
            cnn_ctx.cnn_blend_pass2 = cnn_blend_pass2;
            cnn_ctx.use_cnn = 1;
        } else {
            cnn_ctx.cnn_blend = 0.0f;
            cnn_ctx.cnn_two_pass = 0;
            cnn_ctx.cnn_blend_pass2 = 0.0f;
            cnn_ctx.use_cnn = 0;
        }
        cnn_ctx.unsharp_amount = 0.0f;

        pthread_t of_tid, enc_tid, cnn_tid;
        pthread_create(&of_tid, NULL, of_thread_func, &of_ctx);
        pthread_create(&enc_tid, NULL, encode_thread_func, &enc_ctx);
        pthread_create(&cnn_tid, NULL, cnn_thread_func, &cnn_ctx);

        /* Log CNN availability */
        if (cfg->use_cnn_postfilter) {
            if (mps_postfilter_available())
                fprintf(stderr, "CNN: MPS postfilter ACTIVE (blend=%.2f, 2pass=%d) — 4-stage pipeline\n", cnn_blend, cnn_two_pass);
            else if (cnn_postfilter_available())
                fprintf(stderr, "CNN: CoreML postfilter ACTIVE — 4-stage pipeline\n");
            else
                fprintf(stderr, "CNN: NO postfilter available! mps=%d coreml=%d\n", mps_postfilter_available(), cnn_postfilter_available());
        }

        const uint16_t *view_frames[MAX_WINDOW];
        int next_win_base = win_base;
        int next_fl = frames_loaded;
        float smoothed_flow = 2.0f;  /* IIR-smoothed flow magnitude — start high (assume motion)
                                       * to prevent cache/guided NLM from kicking in immediately.
                                       * alpha=0.1: spreads the transition over ~15 frames.
                                       * Cache enables at smoothed < 0.7, bootstrap at < 0.35,
                                       * staggering the two transitions for a smoother ramp. */

        for (int f = 0; f < range_frames; f++) {
            if (progress_cb && progress_cb(f, range_frames, progress_ctx) != 0) {
                ret = DENOISE_ERR_CANCELLED; break;
            }

            if (f == 0) {
                /* --- Ramp-up: frame 0, OF computed synchronously --- */
                int center = compute_center(0, half_w, W, range_frames, frames_loaded);
                build_frame_view_with_cache(window_frames, denoised_cache,
                                            denoised_cache_valid,
                                            win_base, W1, frames_loaded, center,
                                            view_frames, 1 /* first frame, always cache */);
                int ring_slots[MAX_WINDOW], use_denoised[MAX_WINDOW];
                build_ring_slot_info(denoised_cache_valid, win_base, W1,
                                     frames_loaded, center,
                                     ring_slots, use_denoised, 1);

                /* OF(0) synchronously — use green channels (not full Bayer) */
                double t0 = timer_now();
                const uint16_t *view_greens_0[MAX_WINDOW];
                build_ring_view(green_frames, win_base, W1, frames_loaded, view_greens_0);
                {
                    const uint16_t *b_nbrs[MAX_WINDOW];
                    float *b_fx[MAX_WINDOW], *b_fy[MAX_WINDOW];
                    int b_idx[MAX_WINDOW];
                    int b_n = 0;
                    for (int i = 0; i < frames_loaded; i++) {
                        if (i == center) { of_fx_sets[ping][i] = NULL; of_fy_sets[ping][i] = NULL; continue; }
                        int dist = abs(i - center);
                        if (dist > MAX_OF_RADIUS) { of_fx_sets[ping][i] = NULL; of_fy_sets[ping][i] = NULL; continue; }
                        of_fx_sets[ping][i] = flow_pool_fx[ping][i];
                        of_fy_sets[ping][i] = flow_pool_fy[ping][i];
                        if (dist <= OF_COMPUTE_RADIUS) {
                            b_nbrs[b_n] = view_greens_0[i];
                            b_fx[b_n] = of_fx_sets[ping][i];
                            b_fy[b_n] = of_fy_sets[ping][i];
                            b_idx[b_n] = i;
                            b_n++;
                        }
                    }
                    if (b_n > 0)
                        compute_apple_flow_batch(view_greens_0[center], b_nbrs, b_n,
                                                 green_w, green_h, b_fx, b_fy);
                    /* Scale flow for far neighbors */
                    size_t npix_sc = (size_t)green_w * green_h;
                    for (int i = 0; i < frames_loaded; i++) {
                        if (i == center || !of_fx_sets[ping][i]) continue;
                        int dist = abs(i - center);
                        if (dist <= OF_COMPUTE_RADIUS) continue;
                        int sign = (i > center) ? 1 : -1;
                        int ref_i = -1, best_dist = INT_MAX;
                        for (int b = 0; b < b_n; b++) {
                            int bj = b_idx[b];
                            if ((bj - center) * sign <= 0) continue;
                            int bd = abs(bj - center);
                            if (bd < best_dist) { best_dist = bd; ref_i = bj; }
                        }
                        if (ref_i < 0 || !of_fx_sets[ping][ref_i]) {
                            memset(of_fx_sets[ping][i], 0, npix_sc * sizeof(float));
                            memset(of_fy_sets[ping][i], 0, npix_sc * sizeof(float));
                        } else {
                            float scale = (float)dist / (float)best_dist;
                            for (size_t p = 0; p < npix_sc; p++) {
                                of_fx_sets[ping][i][p] = of_fx_sets[ping][ref_i][p] * scale;
                                of_fy_sets[ping][i][p] = of_fy_sets[ping][ref_i][p] * scale;
                            }
                        }
                    }
                }
                t_accum_flow += timer_now() - t0;
                if (ret != DENOISE_OK) break;

                /* GPU(0): temporal pass 1 (no guide — raw center) */
                float cw0 = compute_adaptive_center_weight(
                    (const float **)of_fx_sets[ping], (const float **)of_fy_sets[ping],
                    frames_loaded, center, green_w, green_h, NULL);
                int use_shared = (tf_mode == 2 && cnn_ctx.use_cnn);
                /* Motion-adaptive GPU window: far neighbors have negligible bilateral weight
                 * at high flow (exp(-flow²/8) → 0.006% at 5px). Reducing from 14 to 4
                 * neighbors cuts Metal dispatches ~71%, saving ~120ms at high flow. */
                int max_gpu_nbrs = (smoothed_flow < 2.0f) ? 14 : (smoothed_flow < 5.0f) ? 8 : 4;
                t0 = timer_now();
                if (tf_mode == 2) {
                    if (use_shared) {
                        if (denoised_bufs_owned[ping]) { free(denoised_bufs[ping]); denoised_bufs_owned[ping] = 0; }
                        denoised_bufs[ping] = temporal_filter_vst_bilateral_gpu_ring_shared(
                            ping, ring_slots, use_denoised,
                            (const float **)of_fx_sets[ping],
                            (const float **)of_fy_sets[ping],
                            frames_loaded, center,
                            width, height,
                            tcfg.noise_sigma, vst_bl, vst_sg, vst_rn);
                    } else {
                        temporal_filter_vst_bilateral_gpu_ring(
                            denoised_bufs[ping],
                            ring_slots, use_denoised,
                            (const float **)of_fx_sets[ping],
                            (const float **)of_fy_sets[ping],
                            frames_loaded, center,
                            width, height,
                            tcfg.noise_sigma, vst_bl, vst_sg, vst_rn,
                            max_gpu_nbrs);
                    }
                } else {
                    temporal_filter_frame_gpu_ring(denoised_bufs[ping],
                                             ring_slots, use_denoised,
                                             (const float **)of_fx_sets[ping],
                                             (const float **)of_fy_sets[ping],
                                             frames_loaded, center,
                                             width, height,
                                             tcfg.strength, tcfg.noise_sigma,
                                             tf_chroma_boost, tf_dist_sigma, tf_flow_tightening,
                                             NULL, cw0);
                }
                t_accum_temporal += timer_now() - t0;

                /* Cache denoised green for this frame's ring slot */
                {
                    int ring_slot = (win_base + center) % W1;
                    extract_green_channel(denoised_bufs[ping], width, height,
                                          denoised_greens[ring_slot]);
                    denoised_green_valid[ring_slot] = 1;
                }

                /* Bootstrap pass 2: re-compute flow from denoised center.
                 * Skip during motion to prevent ghost compounding.
                 * Skip entirely for VST+Bilateral (mode 2) — no guided NLM benefit. */
                if (tf_mode != 2 && bootstrap_flow && cw0 < 1.0f) {
                    extract_green_channel(denoised_bufs[ping], width, height,
                                          bootstrap_center_green);
                    clear_flow_set(of_fx_sets[ping], of_fy_sets[ping], frames_loaded);

                    /* Build green view with cached denoised greens for past neighbors */
                    const uint16_t *bs_greens[MAX_WINDOW];
                    build_green_view_with_cache(green_frames, denoised_greens,
                                                denoised_green_valid,
                                                win_base, W1, frames_loaded, center,
                                                bs_greens);

                    t0 = timer_now();
                    for (int i = 0; i < frames_loaded; i++) {
                        if (i == center) { of_fx_sets[ping][i] = NULL; of_fy_sets[ping][i] = NULL; continue; }
                        of_fx_sets[ping][i] = flow_pool_fx[ping][i];
                        of_fy_sets[ping][i] = flow_pool_fy[ping][i];
                        compute_apple_flow(bootstrap_center_green, bs_greens[i],
                                           green_w, green_h,
                                           of_fx_sets[ping][i], of_fy_sets[ping][i]);
                    }
                    t_accum_flow += timer_now() - t0;

                    /* Temporal pass 2: guided NLM — use pass-1 denoised as guide */
                    float cw2 = compute_adaptive_center_weight(
                        (const float **)of_fx_sets[ping], (const float **)of_fy_sets[ping],
                        frames_loaded, center, green_w, green_h, NULL);
                    t0 = timer_now();
                    temporal_filter_frame_gpu_ring(denoised_bufs[ping],
                                             ring_slots, use_denoised,
                                             (const float **)of_fx_sets[ping],
                                             (const float **)of_fy_sets[ping],
                                             frames_loaded, center,
                                             width, height,
                                             tcfg.strength, tcfg.noise_sigma,
                                             tf_chroma_boost, tf_dist_sigma, tf_flow_tightening,
                                             denoised_bufs[ping], cw2);
                    t_accum_temporal += timer_now() - t0;

                    /* Update cache with pass 2 result (even cleaner) */
                    {
                        int ring_slot = (win_base + center) % W1;
                        extract_green_channel(denoised_bufs[ping], width, height,
                                              denoised_greens[ring_slot]);
                    }
                }

                /* Cache the final denoised frame for recursive temporal filtering */
                {
                    int ring_slot = (win_base + center) % W1;
                    memcpy(denoised_cache[ring_slot], denoised_bufs[ping], frame_bytes);
                    denoised_cache_valid[ring_slot] = 1;
                }

                clear_flow_set(of_fx_sets[ping], of_fy_sets[ping], frames_loaded);

                t_frame_count++;

                /* Training data: extract patches from frame 0 */
                if (collect_training) {
                    int center0 = compute_center(0, half_w, W, range_frames, frames_loaded);
                    int noisy_slot = (win_base + center0) % W1;
                    extract_training_patches(
                        window_frames[noisy_slot], denoised_bufs[ping],
                        width, height, trim_start,
                        tcfg.noise_sigma, 0.0f,
                        td_camera_model, td_iso);
                }

                fprintf(stderr, "denoise_core: frame 0 (ramp-up, %s)\n",
                        tf_mode == 2 ? "VST+Bilateral" : "guided-NLM");

                if (range_frames == 1) {
                    /* Single frame clip: spatial+CNN+encode synchronously */
                    GPU_SPATIAL_CNN(ping);
                    double t0e = timer_now();
                    if (output_is_dng) {
                        const char *fn = dng_frame_filename(dng_reader_impl, trim_start);
                        if (write_dng_frame_inline(dng_writer, denoised_bufs[ping],
                                                   width, height, output_path, fn,
                                                   dng_source_bps, dng_source_cfa) != 0)
                            ret = DENOISE_ERR_ENCODE;
                    } else if (output_is_braw) {
                        /* BRAW inline encode */
                        if (downshift_bufs[ping]) {
                            for (size_t i = 0; i < frame_pixels; i++)
                                downshift_bufs[ping][i] = denoised_bufs[ping][i] >> 4;
                        }
                        int enc = braw_enc_encode_frame(&braw_enc_ctx,
                                    downshift_bufs[ping] ? downshift_bufs[ping] : denoised_bufs[ping],
                                    width, encode_bufs[ping], encode_buf_size);
                        if (enc > 0) braw_writer_add_frame(&braw_writer_obj, encode_bufs[ping], enc);
                        else ret = DENOISE_ERR_ENCODE;
                    } else {
                        const uint16_t *esrc = denoised_bufs[ping];
                        if (dng_to_mov && downshift_bufs[ping]) {
                            int shift = 16 - dng_source_bps;
                            for (size_t i = 0; i < frame_pixels; i++)
                                downshift_bufs[ping][i] = denoised_bufs[ping][i] >> shift;
                            esrc = downshift_bufs[ping];
                        }
                        int enc = encode_frame(esrc, width, height,
                                               encode_bufs[ping], encode_buf_size);
                        if (enc > 0) mov_writer_add_frame(&writer, encode_bufs[ping], enc);
                        else ret = DENOISE_ERR_ENCODE;
                    }
                    t_accum_encode += timer_now() - t0e;
                    break;
                }

                /* Kick CNN(0) in background — spatial + MPS post-filter */
                cnn_ctx.bayer_buf = denoised_bufs[ping];
                cnn_ctx.shared_buf_idx = use_shared ? ping : -1;
                pthread_mutex_lock(&sync.mutex);
                sync.cnn_done = 0; sync.cnn_has_work = 1;
                pthread_cond_signal(&sync.cnn_work_cond);
                pthread_mutex_unlock(&sync.mutex);

                /* Read lookahead frame for OF(1) */
                if (0 >= half_w && next_frame_to_read < num_frames) {
                    double tio = timer_now();
                    READ_INTO_SLOT((win_base + W) % W1);
                    next_win_base = (win_base + 1) % W1;
                    next_fl = frames_loaded;
                    t_accum_io += timer_now() - tio;
                } else {
                    next_win_base = win_base;
                    next_fl = frames_loaded;
                }

                /* Wait for CNN(0) to finish before kicking OF — serialize GPU work
                 * to avoid OOM when CNN (Restormer) + OF run concurrently */
                pthread_mutex_lock(&sync.mutex);
                while (!sync.cnn_done && !sync.error)
                    pthread_cond_wait(&sync.cnn_done_cond, &sync.mutex);
                if (sync.error && ret == DENOISE_OK) ret = sync.error;
                pthread_mutex_unlock(&sync.mutex);

                /* Kick OF(1) in background — use denoised greens for past neighbors */
                int np = 1 - ping;
                int nc = compute_center(1, half_w, W, range_frames, next_fl);
                build_green_view_with_cache(green_frames, denoised_greens,
                                            denoised_green_valid,
                                            next_win_base, W1, next_fl, nc,
                                            (const uint16_t **)of_ctx.view_green);
                of_ctx.frames_loaded = next_fl;
                of_ctx.center_idx = nc;
                of_ctx.fx_out = of_fx_sets[np];
                of_ctx.fy_out = of_fy_sets[np];
                of_ctx.fx_pool = flow_pool_fx[np];
                of_ctx.fy_pool = flow_pool_fy[np];

                pthread_mutex_lock(&sync.mutex);
                sync.of_done = 0; sync.of_has_work = 1;
                pthread_cond_signal(&sync.of_work_cond);
                pthread_mutex_unlock(&sync.mutex);

                ping = np;

            } else {
                /* --- Steady state: frames 1..range_frames-1 --- */
                /* Pipeline reorder: READ overlaps with OF(f) running in background.
                 * Original: Wait OF → Temporal → CNN kick → READ(0.6s) → OF kick
                 * Reorder:  CNN/Enc wait → Enc kick → Apply win → READ(0.6s) → Wait OF → Temporal → CNN kick → OF kick
                 * Since OF(f) was kicked at end of prev iteration, it runs during
                 * our READ. When we wait for OF(f), it's already done. */

                /* Wait for CNN(f-1) — spatial+CNN on previous frame's buffer */
                pthread_mutex_lock(&sync.mutex);
                while (!sync.cnn_done && !sync.error)
                    pthread_cond_wait(&sync.cnn_done_cond, &sync.mutex);
                if (sync.error && ret == DENOISE_OK) ret = sync.error;
                pthread_mutex_unlock(&sync.mutex);
                if (ret != DENOISE_OK) break;

                /* Wait for Encode to finish (so encode thread is free) */
                pthread_mutex_lock(&sync.mutex);
                while (!sync.enc_done && !sync.error)
                    pthread_cond_wait(&sync.enc_done_cond, &sync.mutex);
                if (sync.error && ret == DENOISE_OK) ret = sync.error;
                pthread_mutex_unlock(&sync.mutex);
                if (ret != DENOISE_OK) break;

                /* Kick Encode(f-1) — CNN just finished on buf[1-ping] */
                enc_ctx.denoised_in = denoised_bufs[1 - ping];
                enc_ctx.encode_out  = encode_bufs[1 - ping];
                enc_ctx.downshift_buf = downshift_bufs[1 - ping];
                if (output_is_dng)
                    enc_ctx.dng_frame_name = dng_frame_filename(dng_reader_impl, trim_start + f - 1);
                pthread_mutex_lock(&sync.mutex);
                sync.enc_done = 0; sync.enc_has_work = 1;
                pthread_cond_signal(&sync.enc_work_cond);
                pthread_mutex_unlock(&sync.mutex);

                /* Apply window state from previous iteration */
                win_base = next_win_base;
                frames_loaded = next_fl;

                /* READ next frame — overlaps with OF(f) running on background thread.
                 * OF(f) was kicked at end of prev iteration (~0.52s). READ takes ~0.6s.
                 * By the time READ finishes, OF(f) is done or nearly done. */
                if (f >= half_w && next_frame_to_read < num_frames) {
                    double tio = timer_now();
                    READ_INTO_SLOT((win_base + W) % W1);
                    next_win_base = (win_base + 1) % W1;
                    next_fl = frames_loaded;
                    t_accum_io += timer_now() - tio;
                } else {
                    next_win_base = win_base;
                    next_fl = frames_loaded;
                }

                /* Wait for OF(f) — should be done by now (ran during READ) */
                pthread_mutex_lock(&sync.mutex);
                while (!sync.of_done && !sync.error)
                    pthread_cond_wait(&sync.of_done_cond, &sync.mutex);
                if (sync.error && ret == DENOISE_OK) ret = sync.error;
                pthread_mutex_unlock(&sync.mutex);
                if (ret != DENOISE_OK) break;

                int center = compute_center(f, half_w, W, range_frames, frames_loaded);

                /* Compute motion level FIRST — controls cache and bootstrap */
                float avg_flow_mag = 0.0f;
                float cw1 = compute_adaptive_center_weight(
                    (const float **)of_fx_sets[ping], (const float **)of_fy_sets[ping],
                    frames_loaded, center, green_w, green_h, &avg_flow_mag);
                /* IIR-smooth the flow magnitude to spread cache/guided transitions
                 * over ~15 frames instead of a single-frame binary switch.
                 * alpha=0.1: reaches 83% in ~15 frames (was 0.2/~7 frames).
                 * Lowered thresholds (0.7/0.35 vs 1.0/0.5) compensate for
                 * half-res OF underestimating small motions. */
                smoothed_flow = 0.1f * avg_flow_mag + 0.9f * smoothed_flow;
                int low_motion = (smoothed_flow < 0.7f);

                build_frame_view_with_cache(window_frames, denoised_cache,
                                            denoised_cache_valid,
                                            win_base, W1, frames_loaded, center,
                                            view_frames, low_motion);
                int ring_slots[MAX_WINDOW], use_denoised[MAX_WINDOW];
                build_ring_slot_info(denoised_cache_valid, win_base, W1,
                                     frames_loaded, center,
                                     ring_slots, use_denoised, low_motion);

                /* Temporal pass 1 — no guide (raw center).
                 * VST+Bilateral uses async commit+wait to overlap GPU with ANE OF(f+1).
                 * On M-series, ANE and GPU are independent hardware — they run truly
                 * in parallel, turning the 0.21+0.17=0.38s serial into max(0.21,0.17)=0.21s. */
                int use_shared_ss = (tf_mode == 2 && cnn_ctx.use_cnn);
                /* Motion-adaptive GPU window (same logic as warm-up path) */
                int max_gpu_nbrs_ss = (smoothed_flow < 2.0f) ? 14 : (smoothed_flow < 5.0f) ? 8 : 4;
                double t0 = timer_now();
                int temporal_committed_async = 0;
                if (tf_mode == 2) {
                    if (use_shared_ss) {
                        if (denoised_bufs_owned[ping]) { free(denoised_bufs[ping]); denoised_bufs_owned[ping] = 0; }
                        denoised_bufs[ping] = temporal_filter_vst_bilateral_gpu_ring_shared(
                            ping, ring_slots, use_denoised,
                            (const float **)of_fx_sets[ping],
                            (const float **)of_fy_sets[ping],
                            frames_loaded, center,
                            width, height,
                            tcfg.noise_sigma, vst_bl, vst_sg, vst_rn);
                    } else if (f < range_frames - 1) {
                        /* Async path: commit GPU, then immediately kick OF(f+1) while GPU runs */
                        temporal_filter_vst_bilateral_gpu_ring_commit(
                            denoised_bufs[ping],
                            ring_slots, use_denoised,
                            (const float **)of_fx_sets[ping],
                            (const float **)of_fy_sets[ping],
                            frames_loaded, center,
                            width, height,
                            tcfg.noise_sigma, vst_bl, vst_sg, vst_rn,
                            max_gpu_nbrs_ss);
                        temporal_committed_async = 1;
                    } else {
                        /* Last frame: no OF(f+1) to kick, use sync path */
                        temporal_filter_vst_bilateral_gpu_ring(
                            denoised_bufs[ping],
                            ring_slots, use_denoised,
                            (const float **)of_fx_sets[ping],
                            (const float **)of_fy_sets[ping],
                            frames_loaded, center,
                            width, height,
                            tcfg.noise_sigma, vst_bl, vst_sg, vst_rn,
                            max_gpu_nbrs_ss);
                    }
                } else {
                    temporal_filter_frame_gpu_ring(denoised_bufs[ping],
                                             ring_slots, use_denoised,
                                             (const float **)of_fx_sets[ping],
                                             (const float **)of_fy_sets[ping],
                                             frames_loaded, center,
                                             width, height,
                                             tcfg.strength, tcfg.noise_sigma,
                                             tf_chroma_boost, tf_dist_sigma, tf_flow_tightening,
                                             NULL, cw1);
                }

                /* Kick OF(f+1) while GPU temporal runs (async path only).
                 * Uses currently available denoised greens — frame f's denoised green
                 * isn't cached yet, so OF sees raw green for that one slot. This is
                 * identical to the ramp-up behavior and has no perceptible quality impact. */
                if (temporal_committed_async && f < range_frames - 1) {
                    int np = 1 - ping;
                    int nc = compute_center(f + 1, half_w, W, range_frames, next_fl);
                    build_green_view_with_cache(green_frames, denoised_greens,
                                                denoised_green_valid,
                                                next_win_base, W1, next_fl, nc,
                                                (const uint16_t **)of_ctx.view_green);
                    of_ctx.frames_loaded = next_fl;
                    of_ctx.center_idx = nc;
                    of_ctx.fx_out = of_fx_sets[np];
                    of_ctx.fy_out = of_fy_sets[np];
                    of_ctx.fx_pool = flow_pool_fx[np];
                    of_ctx.fy_pool = flow_pool_fy[np];

                    pthread_mutex_lock(&sync.mutex);
                    sync.of_done = 0; sync.of_has_work = 1;
                    pthread_cond_signal(&sync.of_work_cond);
                    pthread_mutex_unlock(&sync.mutex);
                }

                /* Wait for GPU temporal to complete (async path: OF was kicked above) */
                if (temporal_committed_async) {
                    if (!temporal_filter_vst_bilateral_gpu_ring_wait()) {
                        /* GPU error fallback: already handled inside wait (center frame copied) */
                    }
                }
                t_accum_temporal += timer_now() - t0;

                /* Cache denoised green for this frame's ring slot */
                {
                    int ring_slot = (win_base + center) % W1;
                    extract_green_channel(denoised_bufs[ping], width, height,
                                          denoised_greens[ring_slot]);
                    denoised_green_valid[ring_slot] = 1;
                }

                /* Bootstrap pass 2: re-compute flow from denoised center.
                 * Stricter threshold than cache (0.5 vs 1.0) to stagger the
                 * two transitions — cache kicks in first (more temporal averaging),
                 * then bootstrap kicks in later (even more). This prevents the
                 * "everything snaps to denoised at once" artifact.
                 * Skip for VST+Bilateral (mode 2) — no guided NLM benefit. */
                int very_low_motion = (smoothed_flow < 0.35f);
                if (tf_mode != 2 && bootstrap_flow && very_low_motion) {
                    extract_green_channel(denoised_bufs[ping], width, height,
                                          bootstrap_center_green);
                    clear_flow_set(of_fx_sets[ping], of_fy_sets[ping], frames_loaded);

                    const uint16_t *bs_greens[MAX_WINDOW];
                    build_green_view_with_cache(green_frames, denoised_greens,
                                                denoised_green_valid,
                                                win_base, W1, frames_loaded, center,
                                                bs_greens);

                    t0 = timer_now();
                    for (int i = 0; i < frames_loaded; i++) {
                        if (i == center) { of_fx_sets[ping][i] = NULL; of_fy_sets[ping][i] = NULL; continue; }
                        of_fx_sets[ping][i] = flow_pool_fx[ping][i];
                        of_fy_sets[ping][i] = flow_pool_fy[ping][i];
                        compute_apple_flow(bootstrap_center_green, bs_greens[i],
                                           green_w, green_h,
                                           of_fx_sets[ping][i], of_fy_sets[ping][i]);
                    }
                    t_accum_flow += timer_now() - t0;

                    /* Temporal pass 2: guided NLM — pass-1 result as guide */
                    float cw2s = compute_adaptive_center_weight(
                        (const float **)of_fx_sets[ping], (const float **)of_fy_sets[ping],
                        frames_loaded, center, green_w, green_h, NULL);
                    t0 = timer_now();
                    temporal_filter_frame_gpu_ring(denoised_bufs[ping],
                                             ring_slots, use_denoised,
                                             (const float **)of_fx_sets[ping],
                                             (const float **)of_fy_sets[ping],
                                             frames_loaded, center,
                                             width, height,
                                             tcfg.strength, tcfg.noise_sigma,
                                             tf_chroma_boost, tf_dist_sigma, tf_flow_tightening,
                                             denoised_bufs[ping], cw2s);
                    t_accum_temporal += timer_now() - t0;

                    /* Update cache with pass 2 result */
                    {
                        int ring_slot = (win_base + center) % W1;
                        extract_green_channel(denoised_bufs[ping], width, height,
                                              denoised_greens[ring_slot]);
                    }
                }

                /* Cache the final denoised frame for recursive temporal filtering */
                {
                    int ring_slot = (win_base + center) % W1;
                    memcpy(denoised_cache[ring_slot], denoised_bufs[ping], frame_bytes);
                    denoised_cache_valid[ring_slot] = 1;
                }

                clear_flow_set(of_fx_sets[ping], of_fy_sets[ping], frames_loaded);
                t_frame_count++;

                /* Training data: extract patches from this frame */
                if (collect_training) {
                    int noisy_slot = (win_base + center) % W1;
                    extract_training_patches(
                        window_frames[noisy_slot], denoised_bufs[ping],
                        width, height, trim_start + f,
                        tcfg.noise_sigma, avg_flow_mag,
                        td_camera_model, td_iso);
                }

                fprintf(stderr, "denoise_core: frame %d (%s cw=%.2f flow=%.1fpx smooth=%.2fpx%s%s)\n", f,
                        tf_mode == 2 ? "BL" : "NLM", cw1, avg_flow_mag, smoothed_flow,
                        low_motion ? " +cache" : "",
                        (tf_mode != 2 && bootstrap_flow && very_low_motion) ? " +guided" : "");

                /* Kick CNN(f) — spatial + MPS post-filter */
                cnn_ctx.bayer_buf = denoised_bufs[ping];
                cnn_ctx.shared_buf_idx = use_shared_ss ? ping : -1;
                pthread_mutex_lock(&sync.mutex);
                sync.cnn_done = 0; sync.cnn_has_work = 1;
                pthread_cond_signal(&sync.cnn_work_cond);
                pthread_mutex_unlock(&sync.mutex);

                /* Wait for CNN(f) */
                pthread_mutex_lock(&sync.mutex);
                while (!sync.cnn_done && !sync.error)
                    pthread_cond_wait(&sync.cnn_done_cond, &sync.mutex);
                if (sync.error && ret == DENOISE_OK) ret = sync.error;
                pthread_mutex_unlock(&sync.mutex);
                if (ret != DENOISE_OK) break;

                /* Kick OF(f+1) — only for non-async path (async path kicked OF earlier) */
                if (!temporal_committed_async && f < range_frames - 1) {
                    int np = 1 - ping;
                    int nc = compute_center(f + 1, half_w, W, range_frames, next_fl);
                    build_green_view_with_cache(green_frames, denoised_greens,
                                                denoised_green_valid,
                                                next_win_base, W1, next_fl, nc,
                                                (const uint16_t **)of_ctx.view_green);
                    of_ctx.frames_loaded = next_fl;
                    of_ctx.center_idx = nc;
                    of_ctx.fx_out = of_fx_sets[np];
                    of_ctx.fy_out = of_fy_sets[np];
                    of_ctx.fx_pool = flow_pool_fx[np];
                    of_ctx.fy_pool = flow_pool_fy[np];

                    pthread_mutex_lock(&sync.mutex);
                    sync.of_done = 0; sync.of_has_work = 1;
                    pthread_cond_signal(&sync.of_work_cond);
                    pthread_mutex_unlock(&sync.mutex);
                }

                ping = 1 - ping;
            }
        }

        /* --- Drain pipeline: CNN(last) → Enc(last) --- */
        if (ret == DENOISE_OK && range_frames > 1) {
            /* Wait for CNN of last frame */
            pthread_mutex_lock(&sync.mutex);
            while (!sync.cnn_done && !sync.error)
                pthread_cond_wait(&sync.cnn_done_cond, &sync.mutex);
            if (sync.error && ret == DENOISE_OK) ret = sync.error;
            pthread_mutex_unlock(&sync.mutex);

            /* Wait for previous encode to finish */
            pthread_mutex_lock(&sync.mutex);
            while (!sync.enc_done && !sync.error)
                pthread_cond_wait(&sync.enc_done_cond, &sync.mutex);
            if (sync.error && ret == DENOISE_OK) ret = sync.error;
            pthread_mutex_unlock(&sync.mutex);

            /* Encode last frame synchronously (ping was flipped, last buf is 1-ping) */
            if (ret == DENOISE_OK) {
                double t0 = timer_now();
                int last_slot = 1 - ping;
                if (output_is_dng) {
                    const char *fn = dng_frame_filename(dng_reader_impl, trim_start + range_frames - 1);
                    if (write_dng_frame_inline(dng_writer, denoised_bufs[last_slot],
                                               width, height, output_path, fn,
                                               dng_source_bps, dng_source_cfa) != 0)
                        ret = DENOISE_ERR_ENCODE;
                } else if (output_is_braw) {
                    if (downshift_bufs[last_slot]) {
                        for (size_t i = 0; i < frame_pixels; i++)
                            downshift_bufs[last_slot][i] = denoised_bufs[last_slot][i] >> 4;
                    }
                    int enc = braw_enc_encode_frame(&braw_enc_ctx,
                                downshift_bufs[last_slot] ? downshift_bufs[last_slot] : denoised_bufs[last_slot],
                                width, encode_bufs[last_slot], encode_buf_size);
                    if (enc > 0) braw_writer_add_frame(&braw_writer_obj, encode_bufs[last_slot], enc);
                    else ret = DENOISE_ERR_ENCODE;
                } else {
                    const uint16_t *esrc = denoised_bufs[last_slot];
                    if (dng_to_mov && downshift_bufs[last_slot]) {
                        int shift = 16 - dng_source_bps;
                        for (size_t i = 0; i < frame_pixels; i++)
                            downshift_bufs[last_slot][i] = denoised_bufs[last_slot][i] >> shift;
                        esrc = downshift_bufs[last_slot];
                    }
                    int enc = encode_frame(esrc, width, height,
                                           encode_bufs[last_slot], encode_buf_size);
                    if (enc > 0) mov_writer_add_frame(&writer, encode_bufs[last_slot], enc);
                    else ret = DENOISE_ERR_ENCODE;
                }
                t_accum_encode += timer_now() - t0;
            }
        }

        /* Shut down pipeline threads */
        pthread_mutex_lock(&sync.mutex);
        sync.should_exit = 1;
        pthread_cond_signal(&sync.of_work_cond);
        pthread_cond_signal(&sync.enc_work_cond);
        pthread_cond_signal(&sync.cnn_work_cond);
        pthread_mutex_unlock(&sync.mutex);
        pthread_join(of_tid, NULL);
        pthread_join(enc_tid, NULL);
        pthread_join(cnn_tid, NULL);

        t_accum_flow   += of_ctx.t_accum;
        t_accum_encode += enc_ctx.t_accum;
        t_accum_spatial += cnn_ctx.t_accum_spatial;
        t_accum_cnn    += cnn_ctx.t_accum_cnn;
        t_accum_unsharp += cnn_ctx.t_accum_unsharp;

        pthread_mutex_destroy(&sync.mutex);
        pthread_cond_destroy(&sync.of_done_cond);
        pthread_cond_destroy(&sync.enc_done_cond);
        pthread_cond_destroy(&sync.cnn_done_cond);
        pthread_cond_destroy(&sync.of_work_cond);
        pthread_cond_destroy(&sync.enc_work_cond);
        pthread_cond_destroy(&sync.cnn_work_cond);

    } else {
        /* ===== 2-STAGE PIPELINE (ML): GPU | Encode(CPU) ===== */

        PipelineSync sync;
        memset(&sync, 0, sizeof(sync));
        pthread_mutex_init(&sync.mutex, NULL);
        pthread_cond_init(&sync.enc_done_cond, NULL);
        pthread_cond_init(&sync.enc_work_cond, NULL);
        sync.enc_done = 1; /* first iteration: no previous encode to wait for */

        EncodeThreadCtx enc_ctx;
        memset(&enc_ctx, 0, sizeof(enc_ctx));
        enc_ctx.sync = &sync;
        enc_ctx.width = width;
        enc_ctx.height = height;
        enc_ctx.encode_buf_size = encode_buf_size;
        enc_ctx.writer = &writer;
        enc_ctx.dng_writer = dng_writer;
        enc_ctx.source_bps = dng_source_bps;
        enc_ctx.source_cfa = dng_source_cfa;
        enc_ctx.dng_source_shift = dng_to_mov ? (16 - dng_source_bps) : 0;
        if (output_is_dng)
            strncpy(enc_ctx.dng_output_dir, output_path, sizeof(enc_ctx.dng_output_dir) - 1);
        if (output_is_braw) {
            enc_ctx.braw_writer = &braw_writer_obj;
            enc_ctx.braw_enc = &braw_enc_ctx;
        }
        if (output_is_cfhd) {
            enc_ctx.cf_writer = cf_writer;
            enc_ctx.yuv_y  = cfhd_yuv_y;
            enc_ctx.yuv_cb = cfhd_yuv_cb;
            enc_ctx.yuv_cr = cfhd_yuv_cr;
        }

        pthread_t enc_tid;
        pthread_create(&enc_tid, NULL, encode_thread_func, &enc_ctx);

        for (int f = 0; f < range_frames; f++) {
            if (progress_cb && progress_cb(f, range_frames, progress_ctx) != 0) {
                ret = DENOISE_ERR_CANCELLED; break;
            }

            /* Wait for previous encode */
            pthread_mutex_lock(&sync.mutex);
            while (!sync.enc_done && !sync.error)
                pthread_cond_wait(&sync.enc_done_cond, &sync.mutex);
            if (sync.error && ret == DENOISE_OK) ret = sync.error;
            pthread_mutex_unlock(&sync.mutex);
            if (ret != DENOISE_OK) break;

            int center = compute_center(f, half_w, W, range_frames, frames_loaded);
            const uint16_t *view_frames[MAX_WINDOW];
            build_frame_view_with_cache(window_frames, denoised_cache,
                                        denoised_cache_valid,
                                        win_base, W1, frames_loaded, center,
                                        view_frames, 1 /* ML path */);

            double t0 = timer_now();
            denoise_frame_ml(denoised_bufs[ping],
                             view_frames,
                             frames_loaded, center,
                             width, height,
                             tcfg.noise_sigma);
            t_accum_temporal += timer_now() - t0;
            GPU_SPATIAL_CNN(ping);

            t_frame_count++;

            if (f == range_frames - 1) {
                t0 = timer_now();
                if (output_is_dng) {
                    const char *fn = dng_frame_filename(dng_reader_impl, trim_start + f);
                    if (write_dng_frame_inline(dng_writer, denoised_bufs[ping],
                                               width, height, output_path, fn,
                                               dng_source_bps, dng_source_cfa) != 0)
                        ret = DENOISE_ERR_ENCODE;
                } else if (output_is_braw) {
                    if (downshift_bufs[ping]) {
                        for (size_t i = 0; i < frame_pixels; i++)
                            downshift_bufs[ping][i] = denoised_bufs[ping][i] >> 4;
                    }
                    int enc = braw_enc_encode_frame(&braw_enc_ctx,
                                downshift_bufs[ping] ? downshift_bufs[ping] : denoised_bufs[ping],
                                width, encode_bufs[ping], encode_buf_size);
                    if (enc > 0) braw_writer_add_frame(&braw_writer_obj, encode_bufs[ping], enc);
                    else ret = DENOISE_ERR_ENCODE;
                } else {
                    const uint16_t *esrc = denoised_bufs[ping];
                    if (dng_to_mov && downshift_bufs[ping]) {
                        int shift = 16 - dng_source_bps;
                        for (size_t i = 0; i < frame_pixels; i++)
                            downshift_bufs[ping][i] = denoised_bufs[ping][i] >> shift;
                        esrc = downshift_bufs[ping];
                    }
                    int enc = encode_frame(esrc, width, height,
                                           encode_bufs[ping], encode_buf_size);
                    if (enc > 0) mov_writer_add_frame(&writer, encode_bufs[ping], enc);
                    else ret = DENOISE_ERR_ENCODE;
                }
                t_accum_encode += timer_now() - t0;
            } else {
                enc_ctx.denoised_in = denoised_bufs[ping];
                enc_ctx.encode_out  = encode_bufs[ping];
                enc_ctx.downshift_buf = downshift_bufs[ping];
                if (output_is_dng)
                    enc_ctx.dng_frame_name = dng_frame_filename(dng_reader_impl, trim_start + f);
                pthread_mutex_lock(&sync.mutex);
                sync.enc_done = 0; sync.enc_has_work = 1;
                pthread_cond_signal(&sync.enc_work_cond);
                pthread_mutex_unlock(&sync.mutex);
            }

            ping = 1 - ping;

            /* Slide window */
            if (f >= half_w && next_frame_to_read < num_frames) {
                double tio = timer_now();
                win_base = (win_base + 1) % W1;
                READ_INTO_SLOT((win_base + W - 1) % W1);
                t_accum_io += timer_now() - tio;
            }

            fprintf(stderr, "denoise_core: frame %d (ML pipeline)\n", f);
        }

        pthread_mutex_lock(&sync.mutex);
        sync.should_exit = 1;
        pthread_cond_signal(&sync.enc_work_cond);
        pthread_mutex_unlock(&sync.mutex);
        pthread_join(enc_tid, NULL);

        t_accum_encode += enc_ctx.t_accum;

        pthread_mutex_destroy(&sync.mutex);
        pthread_cond_destroy(&sync.enc_done_cond);
        pthread_cond_destroy(&sync.enc_work_cond);
    }

    #undef GPU_SPATIAL_CNN
    #undef READ_INTO_SLOT

    /* Final 100% progress tick */
    if (progress_cb) progress_cb(range_frames, range_frames, progress_ctx);

    /* ---- Timing summary ---- */
    double t_wall_total = timer_now() - t_wall_start;
    if (t_frame_count > 0) {
        double t_stage_sum = t_accum_flow + t_accum_temporal + t_accum_spatial
                           + t_accum_cnn + t_accum_unsharp + t_accum_encode + t_accum_io;
        fprintf(stderr, "\n");
        fprintf(stderr, "===== PIPELINE TIMING SUMMARY (%d frames) =====\n", t_frame_count);
        fprintf(stderr, "  Optical flow:     %7.1fs  avg %.2fs/frame\n",
                t_accum_flow, t_accum_flow / t_frame_count);
        fprintf(stderr, "  Temporal filter:  %7.1fs  avg %.2fs/frame\n",
                t_accum_temporal, t_accum_temporal / t_frame_count);
        fprintf(stderr, "  Spatial denoise:  %7.1fs  avg %.3fs/frame\n",
                t_accum_spatial, t_accum_spatial / t_frame_count);
        fprintf(stderr, "  CNN post-filter:  %7.1fs  avg %.2fs/frame\n",
                t_accum_cnn, t_accum_cnn / t_frame_count);
        fprintf(stderr, "  Unsharp mask:     %7.1fs  avg %.3fs/frame\n",
                t_accum_unsharp, t_accum_unsharp / t_frame_count);
        fprintf(stderr, "  Encode + write:   %7.1fs  avg %.3fs/frame\n",
                t_accum_encode, t_accum_encode / t_frame_count);
        fprintf(stderr, "  Frame I/O:        %7.1fs  avg %.3fs/frame\n",
                t_accum_io, t_accum_io / t_frame_count);
        fprintf(stderr, "    decode:         %7.1fs  avg %.3fs/frame\n",
                t_accum_decode, t_accum_decode / t_frame_count);
        fprintf(stderr, "    preproc:        %7.1fs  avg %.3fs/frame\n",
                t_accum_preproc, t_accum_preproc / t_frame_count);
        fprintf(stderr, "  -------------------------------------------\n");
        fprintf(stderr, "  Stage sum:        %7.1fs  (overlapped)\n", t_stage_sum);
        fprintf(stderr, "  Wall clock:       %7.1fs  avg %.2fs/frame  (%.1f fps)\n",
                t_wall_total, t_wall_total / t_frame_count,
                t_frame_count / t_wall_total);
        fprintf(stderr, "================================================\n\n");

        t_accum_flow = t_accum_temporal = t_accum_spatial = 0;
        t_accum_cnn = t_accum_unsharp = t_accum_encode = t_accum_io = 0;
        t_accum_decode = t_accum_preproc = 0;
        t_frame_count = 0;
    }

cleanup:
    hotpixel_profile_free(hp_profile);
    free(dark_frame);
    for (int b = 0; b < 2; b++) {
        if (denoised_bufs_owned[b]) free(denoised_bufs[b]);
        free(encode_bufs[b]);
        free(downshift_bufs[b]);
    }
    for (int i = 0; i < W1; i++) {
        /* window_frames[i] and denoised_cache[i] are GPU-owned (MTLBuffer) — don't free */
        if (green_frames)  free(green_frames[i]);
        if (denoised_greens) free(denoised_greens[i]);
    }
    free(window_frames);    /* free the pointer array only (not the buffers) */
    free(green_frames);
    free(denoised_greens);
    free(denoised_green_valid);
    free(denoised_cache);   /* free the pointer array only */
    free(denoised_cache_valid);
    free(bootstrap_center_green);
    /* Free pre-allocated flow buffer pool */
    for (int p = 0; p < 2; p++)
        for (int i = 0; i < MAX_WINDOW; i++) {
            free(flow_pool_fx[p][i]);
            free(flow_pool_fy[p][i]);
        }
    frame_reader_close(&reader);
    if (dng_writer)
        dng_writer_close(dng_writer);
    else if (output_is_braw) {
        braw_writer_close(&braw_writer_obj);
        braw_enc_free(&braw_enc_ctx);
    } else if (output_is_cfhd) {
        cf_mov_writer_close(cf_writer);
        free(cfhd_yuv_y); free(cfhd_yuv_cb); free(cfhd_yuv_cr);
    } else
        mov_writer_close(&writer);
    return ret;
}

/* ---- Single-frame preview ---- */
/* Denoise one frame and encode it as a 1-frame ProRes RAW .mov.
 * Swift then uses AVAssetImageGenerator to decode it — giving the
 * exact same look as the final output (Apple's ProRes RAW decoder). */

int denoise_preview_frame(
    const char          *input_path,
    int                  frame_index,
    const DenoiseCConfig *cfg,
    const char          *temp_output_path)
{
    FrameReader reader;
    if (frame_reader_open(&reader, input_path) != 0)
        return DENOISE_ERR_INPUT_OPEN;

    int width      = reader.width;
    int height     = reader.height;
    int num_frames = reader.frame_count;

    size_t frame_bytes  = (size_t)width * height * sizeof(uint16_t);
    int green_w = width / 2;
    int green_h = height / 2;
    size_t green_bytes = (size_t)green_w * green_h * sizeof(uint16_t);
    int n_flow = green_w * green_h;

    int W = cfg->window_size;
    if (W > num_frames) W = num_frames;
    int half_w = W / 2;

    /* Determine which frames to read */
    int start = frame_index - half_w;
    if (start < 0) start = 0;
    if (start + W > num_frames) start = num_frames - W;
    if (start < 0) start = 0;
    int actual_w = (start + W <= num_frames) ? W : (num_frames - start);
    int center = frame_index - start;
    if (center >= actual_w) center = actual_w - 1;
    if (center < 0) center = 0;

    /* Allocate window */
    uint16_t **window_frames = (uint16_t **)calloc(actual_w, sizeof(uint16_t *));
    uint16_t **green_frames  = (uint16_t **)calloc(actual_w, sizeof(uint16_t *));
    uint16_t *denoised       = (uint16_t *)malloc(frame_bytes);
    uint16_t *dark_frame     = NULL;
    HotPixelProfile *hp_profile = NULL;
    int ret = DENOISE_OK;

    if (!window_frames || !green_frames || !denoised) {
        ret = DENOISE_ERR_ALLOC; goto prev_cleanup;
    }

    for (int i = 0; i < actual_w; i++) {
        window_frames[i] = (uint16_t *)malloc(frame_bytes);
        green_frames[i]  = (uint16_t *)malloc(green_bytes);
        if (!window_frames[i] || !green_frames[i]) {
            ret = DENOISE_ERR_ALLOC; goto prev_cleanup;
        }
    }

    /* Load or auto-estimate dark frame */
    if (cfg->dark_frame_path && cfg->dark_frame_path[0] != '\0') {
        dark_frame = load_dark_frame(cfg->dark_frame_path, width, height);
    } else if (cfg->auto_dark_frame) {
        size_t px = (size_t)width * height;
        dark_frame = (uint16_t *)calloc(px, sizeof(uint16_t));
        if (dark_frame) {
            int hot = 0;
            if (estimate_dark_frame(input_path, dark_frame, &hot) != 0 || hot == 0) {
                free(dark_frame);
                dark_frame = NULL;
            }
        }
    }

    /* Load calibrated hot pixel profile */
    if (cfg->hotpixel_profile_path && cfg->hotpixel_profile_path[0] != '\0' && cfg->detected_iso > 0) {
        hp_profile = hotpixel_profile_load(cfg->hotpixel_profile_path, cfg->detected_iso);
    }

    /* Skip to 'start' frame */
    for (int i = 0; i < start; i++) {
        if (frame_reader_read_frame(&reader, window_frames[0]) != 0) {
            ret = DENOISE_ERR_INPUT_OPEN; goto prev_cleanup;
        }
    }

    /* Read actual_w frames */
    int loaded = 0;
    for (int i = 0; i < actual_w; i++) {
        if (frame_reader_read_frame(&reader, window_frames[i]) != 0) break;
        if (dark_frame) subtract_dark_frame(window_frames[i], dark_frame, width, height);
        if (hp_profile) hotpixel_profile_apply(hp_profile, window_frames[i], width, height);
        extract_green_channel(window_frames[i], width, height, green_frames[i]);
        spatial_filter_green(green_frames[i], green_w, green_h, cfg->noise_sigma);
        loaded++;
    }

    if (loaded == 0) { ret = DENOISE_ERR_INPUT_OPEN; goto prev_cleanup; }
    if (center >= loaded) center = loaded - 1;

    /* Set up temporal filter config */
    if (reader.format == FORMAT_MOV) {
        if (set_frame_header_from_mov(input_path) != 0)
            fprintf(stderr, "preview: warning — could not read source frame header\n");
    }
    denoise_init(&g_denoise_config);

    TemporalFilterConfig tcfg;
    temporal_filter_init(&tcfg);
    tcfg.window_size = cfg->window_size;
    tcfg.strength    = cfg->strength;
    tcfg.noise_sigma = cfg->noise_sigma;
    if (tcfg.noise_sigma == 0)
        tcfg.noise_sigma = temporal_filter_estimate_noise(window_frames[center], width, height);

    float vst_bl = cfg->black_level > 0 ? cfg->black_level : 6032.0f;
    float vst_sg = cfg->shot_gain   > 0 ? cfg->shot_gain   : 180.0f;
    float vst_rn = cfg->read_noise  > 0 ? cfg->read_noise  : 616.0f;

    if (cfg->use_ml && ml_denoiser_available()) {
        /* ML path */
        denoise_frame_ml(denoised,
                         (const uint16_t **)window_frames,
                         loaded, center,
                         width, height,
                         tcfg.noise_sigma);
    } else {
        /* Temporal filter path */
        float **fx = (float **)calloc(loaded, sizeof(float *));
        float **fy = (float **)calloc(loaded, sizeof(float *));
        if (!fx || !fy) { free(fx); free(fy); ret = DENOISE_ERR_ALLOC; goto prev_cleanup; }

        /* Batch OF: compute real flow only for dist ≤ OF_COMPUTE_RADIUS; scale for farther */
        const uint16_t *batch_nbrs[MAX_WINDOW];
        float *batch_fx[MAX_WINDOW], *batch_fy[MAX_WINDOW];
        int batch_idx_p[MAX_WINDOW];
        int batch_n = 0;
        for (int i = 0; i < loaded; i++) {
            if (i == center) { fx[i] = NULL; fy[i] = NULL; continue; }
            int dist = abs(i - center);
            if (dist > MAX_OF_RADIUS) { fx[i] = NULL; fy[i] = NULL; continue; }
            fx[i] = (float *)calloc(n_flow, sizeof(float));
            fy[i] = (float *)calloc(n_flow, sizeof(float));
            if (!fx[i] || !fy[i]) { ret = DENOISE_ERR_ALLOC; break; }
            if (dist <= OF_COMPUTE_RADIUS) {
                batch_nbrs[batch_n] = green_frames[i];
                batch_fx[batch_n] = fx[i];
                batch_fy[batch_n] = fy[i];
                batch_idx_p[batch_n] = i;
                batch_n++;
            }
        }
        if (ret == DENOISE_OK && batch_n > 0)
            compute_apple_flow_batch(green_frames[center], batch_nbrs, batch_n,
                                     green_w, green_h, batch_fx, batch_fy);
        /* Scale flow for far neighbors */
        if (ret == DENOISE_OK) {
            for (int i = 0; i < loaded; i++) {
                if (i == center || !fx[i]) continue;
                int dist = abs(i - center);
                if (dist <= OF_COMPUTE_RADIUS) continue;
                int sign = (i > center) ? 1 : -1;
                int ref_i = -1, best_dist = INT_MAX;
                for (int b = 0; b < batch_n; b++) {
                    int bj = batch_idx_p[b];
                    if ((bj - center) * sign <= 0) continue;
                    int bd = abs(bj - center);
                    if (bd < best_dist) { best_dist = bd; ref_i = bj; }
                }
                if (ref_i < 0 || !fx[ref_i]) {
                    memset(fx[i], 0, n_flow * sizeof(float));
                    memset(fy[i], 0, n_flow * sizeof(float));
                } else {
                    float scale = (float)dist / (float)best_dist;
                    for (int p = 0; p < n_flow; p++) {
                        fx[i][p] = fx[ref_i][p] * scale;
                        fy[i][p] = fy[ref_i][p] * scale;
                    }
                }
            }
        }

        if (ret == DENOISE_OK) {
            int tf_mode = cfg->temporal_filter_mode;
            fprintf(stderr, "preview: tf_mode=%d, loaded=%d, center=%d, batch_n=%d, sigma=%.1f\n",
                    tf_mode, loaded, center, batch_n, tcfg.noise_sigma);
            if (tf_mode == 2) {
                /* VST+Bilateral: upload frames to GPU ring, then call ring function.
                 * Run 2 passes to simulate the recursive feedback of the full pipeline:
                 *  Pass 1: all raw neighbors → rough denoise
                 *  Pass 2: copy pass-1 result into center's denoised ring slot,
                 *           use_denoised[center]=1 → cleaner self-guided reference */
                gpu_ring_init(loaded, width, height);
                size_t fb = (size_t)width * height * sizeof(uint16_t);
                for (int i = 0; i < loaded; i++) {
                    uint16_t *dst = gpu_ring_frame_ptr(i);
                    if (dst) memcpy(dst, window_frames[i], fb);
                }
                int ring_slots[MAX_WINDOW];
                int use_denoised[MAX_WINDOW];
                for (int i = 0; i < loaded; i++) { ring_slots[i] = i; use_denoised[i] = 0; }

                /* Pass 1: all raw */
                temporal_filter_vst_bilateral_gpu_ring(
                    denoised, ring_slots, use_denoised,
                    (const float **)fx, (const float **)fy,
                    loaded, center, width, height,
                    tcfg.noise_sigma, vst_bl, vst_sg, vst_rn, 14);

                /* Pass 2: feed pass-1 result back as denoised center */
                uint16_t *dn_slot = gpu_ring_denoised_ptr(center);
                if (dn_slot) {
                    memcpy(dn_slot, denoised, fb);
                    use_denoised[center] = 1;
                    temporal_filter_vst_bilateral_gpu_ring(
                        denoised, ring_slots, use_denoised,
                        (const float **)fx, (const float **)fy,
                        loaded, center, width, height,
                        tcfg.noise_sigma, vst_bl, vst_sg, vst_rn, 14);
                }
            } else {
                float cw_simple = compute_adaptive_center_weight(
                    (const float **)fx, (const float **)fy,
                    loaded, center, green_w, green_h, NULL);
                temporal_filter_frame_gpu(denoised,
                                          (const uint16_t **)window_frames,
                                          (const float **)fx,
                                          (const float **)fy,
                                          loaded, center,
                                          width, height,
                                          tcfg.strength, tcfg.noise_sigma,
                                          1.0f, 1.5f, 3.0f, NULL, cw_simple);
            }
        }

        for (int i = 0; i < loaded; i++) { free(fx[i]); free(fy[i]); }
        free(fx); free(fy);
    }

    if (ret == DENOISE_OK) {
        if (cfg->spatial_strength > 0)
            spatial_denoise_frame(denoised, width, height, tcfg.noise_sigma, cfg->spatial_strength);

        /* CNN post-filter (after temporal + spatial) — prefer MPS GPU path. */
        if (cfg->use_cnn_postfilter) {
            if (mps_postfilter_available())
                postfilter_frame_mps(denoised, width, height, 0.65f);
            else if (cnn_postfilter_available())
                postfilter_frame_cnn(denoised, denoised, width, height);
        }

        /* Unsharp mask: restore micro-contrast for preview too */
        unsharp_mask_bayer(denoised, width, height, 0.45f, tcfg.noise_sigma);

        /* Encode denoised frame as a 1-frame ProRes RAW .mov */
        /* If DNG input, downshift 16-bit to 12-bit for ProRes RAW encoder */
        int preview_dng_bps = 16;
        if (reader.format == FORMAT_CINEMADNG) {
            frame_reader_get_dng_info(&reader, &preview_dng_bps, NULL, NULL);
            if (preview_dng_bps < 16) {
                int shift = 16 - preview_dng_bps;
                size_t npix = (size_t)width * height;
                for (size_t i = 0; i < npix; i++)
                    denoised[i] = denoised[i] >> shift;
            }
        }
        int encode_buf_size = 50 * 1024 * 1024;
        uint8_t *encode_buf = (uint8_t *)malloc(encode_buf_size);
        if (!encode_buf) { ret = DENOISE_ERR_ALLOC; }
        else {
            int encoded = encode_frame(denoised, width, height, encode_buf, encode_buf_size);
            if (encoded <= 0) {
                ret = DENOISE_ERR_ENCODE;
            } else {
                MovWriter writer;
                if (mov_writer_open(&writer, temp_output_path, width, height) != 0) {
                    ret = DENOISE_ERR_OUTPUT_OPEN;
                } else {
                    if (reader.format == FORMAT_MOV)
                        mov_writer_copy_metadata(&writer, input_path);
                    if (mov_writer_add_frame(&writer, encode_buf, encoded) != 0)
                        ret = DENOISE_ERR_ENCODE;
                    mov_writer_close(&writer);
                }
            }
            free(encode_buf);
        }
    }

prev_cleanup:
    hotpixel_profile_free(hp_profile);
    free(dark_frame);
    free(denoised);
    if (window_frames) {
        for (int i = 0; i < actual_w; i++) free(window_frames[i]);
    }
    if (green_frames) {
        for (int i = 0; i < actual_w; i++) free(green_frames[i]);
    }
    free(window_frames);
    free(green_frames);
    frame_reader_close(&reader);
    return ret;
}

/* ---- Auto dark frame estimation bridge ---- */

int denoise_estimate_dark_frame(const char *input_path,
                                const char *temp_output_path,
                                int *out_hot_count) {
    /* Probe dimensions */
    int width = 0, height = 0;
    if (frame_reader_probe_dimensions(input_path, &width, &height) != 0)
        return -1;

    size_t pixels = (size_t)width * height;
    uint16_t *dark = (uint16_t *)calloc(pixels, sizeof(uint16_t));
    if (!dark) return -1;

    int hot_count = 0;
    if (estimate_dark_frame(input_path, dark, &hot_count) != 0) {
        free(dark);
        return -1;
    }

    if (out_hot_count) *out_hot_count = hot_count;

    /* Write raw uint16 dark frame to temp file */
    FILE *f = fopen(temp_output_path, "wb");
    if (!f) { free(dark); return -1; }
    fwrite(dark, sizeof(uint16_t), pixels, f);
    fclose(f);
    free(dark);
    return 0;
}

/* ---- Probe wrappers (dispatch via frame_reader) ---- */

int denoise_probe_dimensions(const char *path, int *width, int *height) {
    return frame_reader_probe_dimensions(path, width, height);
}

int denoise_probe_frame_count(const char *path) {
    return frame_reader_probe_frame_count(path);
}

int denoise_probe_format(const char *path) {
    return (int)detect_input_format(path);
}

/* ---- Camera metadata probe ---- */

#define FFPROBE "/opt/homebrew/bin/ffprobe"

int denoise_probe_camera(const char *path, char *camera_model, int model_len,
                         int *iso) {
    camera_model[0] = '\0';
    *iso = 0;

    /* Use ffprobe to extract QuickTime metadata tags */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             FFPROBE " -v quiet -print_format flat -show_entries format_tags \"%s\"",
             path);

    FILE *p = popen(cmd, "r");
    if (!p) return -1;

    char make[256] = "";
    char model[256] = "";
    char line[1024];

    while (fgets(line, sizeof(line), p)) {
        /* Look for camera make/model tags (Apple QuickTime metadata) */
        char *val;

        if ((val = strstr(line, "com.apple.quicktime.make=\""))) {
            val += strlen("com.apple.quicktime.make=\"");
            char *end = strchr(val, '"');
            if (end) { *end = '\0'; strncpy(make, val, sizeof(make) - 1); }
        }
        else if ((val = strstr(line, "com.apple.quicktime.model=\""))) {
            val += strlen("com.apple.quicktime.model=\"");
            char *end = strchr(val, '"');
            if (end) { *end = '\0'; strncpy(model, val, sizeof(model) - 1); }
        }
        else if ((val = strstr(line, "make=\""))) {
            if (make[0] == '\0') {
                val += strlen("make=\"");
                char *end = strchr(val, '"');
                if (end) { *end = '\0'; strncpy(make, val, sizeof(make) - 1); }
            }
        }
        else if ((val = strstr(line, "model=\""))) {
            if (model[0] == '\0') {
                val += strlen("model=\"");
                char *end = strchr(val, '"');
                if (end) { *end = '\0'; strncpy(model, val, sizeof(model) - 1); }
            }
        }
        /* Panasonic/Atomos ProRes RAW metadata tags (ffprobe flat format uses underscores) */
        else if ((val = strstr(line, "proapps_manufacturer=\""))) {
            if (make[0] == '\0') {
                val += strlen("proapps_manufacturer=\"");
                char *end = strchr(val, '"');
                if (end) { *end = '\0'; strncpy(make, val, sizeof(make) - 1); }
            }
        }
        else if ((val = strstr(line, "proapps_modelname=\""))) {
            if (model[0] == '\0') {
                val += strlen("proapps_modelname=\"");
                char *end = strchr(val, '"');
                if (end) { *end = '\0'; strncpy(model, val, sizeof(model) - 1); }
            }
        }
        /* ISO — various tag names used by different cameras/recorders */
        else if ((val = strstr(line, "com.apple.quicktime.camera.iso=\""))) {
            val += strlen("com.apple.quicktime.camera.iso=\"");
            *iso = atoi(val);
        }
        else if ((val = strstr(line, "org.smpte.rdd18.camera.isosensitivity=\""))) {
            if (*iso == 0) {
                val += strlen("org.smpte.rdd18.camera.isosensitivity=\"");
                *iso = atoi(val);
            }
        }
        else if ((val = strstr(line, "ExposureIndex=\""))) {
            if (*iso == 0) {
                val += strlen("ExposureIndex=\"");
                *iso = (int)(atof(val) + 0.5);
            }
        }
        else if ((val = strstr(line, "iso=\""))) {
            if (*iso == 0) {
                val += strlen("iso=\"");
                *iso = atoi(val);
            }
        }
    }

    pclose(p);

    /* Build camera model string: "Make Model" */
    if (make[0] && model[0]) {
        snprintf(camera_model, model_len, "%s %s", make, model);
    } else if (model[0]) {
        strncpy(camera_model, model, model_len - 1);
        camera_model[model_len - 1] = '\0';
    } else if (make[0]) {
        strncpy(camera_model, make, model_len - 1);
        camera_model[model_len - 1] = '\0';
    }

    return 0;
}

/* ---- Parameter Sweep ---- */

typedef struct {
    const char *label;
    float chroma_boost;
    float dist_sigma;
    float flow_tightening;
    int   cnn_passes;      /* 1 or 2 */
    float cnn_blend2;      /* blend for 2nd pass */
} SweepConfig;

int denoise_parameter_sweep(
    const char *input_path,
    int start_frame,
    const DenoiseCConfig *base_cfg)
{
    /* Define parameter combos to test */
    SweepConfig configs[] = {
        /* ---- Blend2 fine sweep (top finding from round 1) ---- */
        /* label                     chroma  dist_s  flow_t  cnn_p  blend2 */
        { "bl2_0.5",                 1.0f,   1.5f,   3.0f,   2,     0.5f  },
        { "bl2_0.6",                 1.0f,   1.5f,   3.0f,   2,     0.6f  },
        { "bl2_0.7",                 1.0f,   1.5f,   3.0f,   2,     0.7f  },
        { "bl2_0.8",                 1.0f,   1.5f,   3.0f,   2,     0.8f  },
        { "bl2_0.9",                 1.0f,   1.5f,   3.0f,   2,     0.9f  },
        { "bl2_1.0",                 1.0f,   1.5f,   3.0f,   2,     1.0f  },

        /* ---- Flow tightening fine sweep (2nd best finding) ---- */
        { "flow_0.5",               1.0f,   1.5f,   0.5f,   2,     0.7f  },
        { "flow_1.0",               1.0f,   1.5f,   1.0f,   2,     0.7f  },
        { "flow_2.0",               1.0f,   1.5f,   2.0f,   2,     0.7f  },
        { "flow_3.0",               1.0f,   1.5f,   3.0f,   2,     0.7f  },
        { "flow_5.0",               1.0f,   1.5f,   5.0f,   2,     0.7f  },

        /* ---- dist_sigma fine sweep ---- */
        { "dist_1.0",               1.0f,   1.0f,   3.0f,   2,     0.7f  },
        { "dist_1.5",               1.0f,   1.5f,   3.0f,   2,     0.7f  },
        { "dist_2.0",               1.0f,   2.0f,   3.0f,   2,     0.7f  },
        { "dist_3.0",               1.0f,   3.0f,   3.0f,   2,     0.7f  },

        /* ---- Cross-combinations of top findings ---- */
        { "best_bl7_fl1",           1.0f,   1.5f,   1.0f,   2,     0.7f  },
        { "best_bl8_fl1",           1.0f,   1.5f,   1.0f,   2,     0.8f  },
        { "best_bl9_fl1",           1.0f,   1.5f,   1.0f,   2,     0.9f  },
        { "best_bl7_fl2",           1.0f,   1.5f,   2.0f,   2,     0.7f  },
        { "best_bl8_fl2",           1.0f,   1.5f,   2.0f,   2,     0.8f  },

        /* ---- CNN pass count ---- */
        { "cnn_1pass",              1.0f,   1.5f,   3.0f,   1,     0.0f  },
        { "cnn_3pass_bl5",          1.0f,   1.5f,   3.0f,   3,     0.5f  },
        { "cnn_3pass_bl7",          1.0f,   1.5f,   3.0f,   3,     0.7f  },
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    int SWEEP_FRAMES = 6;

    /* Probe dimensions */
    int width = 0, height = 0;
    if (frame_reader_probe_dimensions(input_path, &width, &height) != 0)
        return -1;

    size_t frame_pixels = (size_t)width * height;
    size_t frame_bytes  = frame_pixels * sizeof(uint16_t);
    int green_w = width / 2, green_h = height / 2;
    int n_flow = green_w * green_h;

    /* Read frames into window */
    int actual_w = base_cfg->window_size;
    if (actual_w < SWEEP_FRAMES) actual_w = SWEEP_FRAMES + 2;
    int half_w = actual_w / 2;

    FrameReader reader;
    if (frame_reader_open(&reader, input_path) != 0) return -1;
    int num_frames = frame_reader_probe_frame_count(input_path);
    if (num_frames < 0) num_frames = 9999;

    int center = half_w;
    int first_frame = start_frame - half_w;
    if (first_frame < 0) first_frame = 0;

    uint16_t **window_frames = (uint16_t **)calloc(actual_w, sizeof(uint16_t *));
    uint16_t **green_frames  = (uint16_t **)calloc(actual_w, sizeof(uint16_t *));
    if (!window_frames || !green_frames) {
        free(window_frames); free(green_frames);
        frame_reader_close(&reader);
        return -1;
    }

    for (int i = 0; i < actual_w; i++) {
        window_frames[i] = (uint16_t *)malloc(frame_bytes);
        green_frames[i]  = (uint16_t *)malloc(n_flow * sizeof(uint16_t));
    }

    /* Skip to first_frame by reading and discarding */
    uint16_t *skip_buf = (uint16_t *)malloc(frame_bytes);
    if (!skip_buf && first_frame > 0) {
        fprintf(stderr, "denoise_core: skip_buf alloc failed\n");
        goto sweep_cleanup;
    }
    for (int i = 0; i < first_frame; i++)
        frame_reader_read_frame(&reader, skip_buf);
    free(skip_buf); skip_buf = NULL;

    /* Read window frames */
    int loaded = 0;
    for (int i = 0; i < actual_w && (first_frame + i) < num_frames; i++) {
        if (frame_reader_read_frame(&reader, window_frames[i]) == 0) {
            extract_green_channel(window_frames[i], width, height, green_frames[i]);
            spatial_filter_green(green_frames[i], green_w, green_h, base_cfg->noise_sigma);
            loaded++;
        }
    }
    if (center >= loaded) center = loaded - 1;

    fprintf(stderr, "sweep: loaded %d frames (center=%d), %dx%d\n",
            loaded, center, width, height);

    /* Compute optical flow for all frames relative to center */
    float **fx = (float **)calloc(loaded, sizeof(float *));
    float **fy = (float **)calloc(loaded, sizeof(float *));
    if (!fx || !fy) { free(fx); free(fy); goto sweep_cleanup; }

    for (int i = 0; i < loaded; i++) {
        if (i == center) continue;
        fx[i] = (float *)calloc(n_flow, sizeof(float));
        fy[i] = (float *)calloc(n_flow, sizeof(float));
        if (fx[i] && fy[i])
            compute_apple_flow(green_frames[center], green_frames[i],
                               green_w, green_h, fx[i], fy[i]);
    }

    /* Estimate noise once */
    float noise_sigma = base_cfg->noise_sigma;
    if (noise_sigma == 0)
        noise_sigma = temporal_filter_estimate_noise(window_frames[center], width, height);
    if (noise_sigma < 1.0f) noise_sigma = 1.0f;
    fprintf(stderr, "sweep: noise_sigma=%.1f\n", noise_sigma);

    /* Motion-adaptive CNN blend (same logic as denoise_file) */
    float cnn_blend = 0.9f;
    if (base_cfg->motion_avg > 0) {
        float t = base_cfg->motion_avg / 3.0f;
        if (t > 1.0f) t = 1.0f;
        cnn_blend = 0.9f + 0.1f * t;
    }

    /* Create output directory */
    system("mkdir -p /tmp/bayerflow_sweep");

    uint16_t *denoised = (uint16_t *)malloc(frame_bytes);
    if (!denoised) goto sweep_flow_cleanup;

    /* Run each config */
    for (int c = 0; c < num_configs; c++) {
        const SweepConfig *sc = &configs[c];
        fprintf(stderr, "sweep: config %d/%d [%s] chroma=%.1f dist=%.1f flow=%.1f cnn=%d blend2=%.1f\n",
                c + 1, num_configs, sc->label,
                sc->chroma_boost, sc->dist_sigma, sc->flow_tightening,
                sc->cnn_passes, sc->cnn_blend2);

        /* Process center frame with this config */
        float cw_sweep = compute_adaptive_center_weight(
            (const float **)fx, (const float **)fy,
            loaded, center, green_w, green_h, NULL);
        temporal_filter_frame_gpu(denoised,
                                  (const uint16_t **)window_frames,
                                  (const float **)fx,
                                  (const float **)fy,
                                  loaded, center,
                                  width, height,
                                  base_cfg->strength, noise_sigma,
                                  sc->chroma_boost, sc->dist_sigma, sc->flow_tightening,
                                  NULL, cw_sweep);

        /* CNN post-filter */
        if (base_cfg->use_cnn_postfilter && mps_postfilter_available()) {
            postfilter_frame_mps(denoised, width, height, cnn_blend);
            for (int pass = 2; pass <= sc->cnn_passes; pass++)
                postfilter_frame_mps(denoised, width, height, sc->cnn_blend2);
        }

        /* Write denoised center frame */
        char path[512];
        snprintf(path, sizeof(path), "/tmp/bayerflow_sweep/%s_center.raw", sc->label);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(denoised, sizeof(uint16_t), frame_pixels, f); fclose(f); }

        /* Process SWEEP_FRAMES sequential frames for temporal noise metrics.
         * Re-compute flow for each frame-as-center (expensive but accurate). */
        for (int sf = 0; sf < SWEEP_FRAMES && (center - 2 + sf) >= 0 && (center - 2 + sf) < loaded; sf++) {
            int fc = center - 2 + sf;

            /* Recompute flow relative to this frame as center */
            float **sfx = (float **)calloc(loaded, sizeof(float *));
            float **sfy = (float **)calloc(loaded, sizeof(float *));
            if (!sfx || !sfy) { free(sfx); free(sfy); continue; }

            for (int i = 0; i < loaded; i++) {
                if (i == fc) continue;
                sfx[i] = (float *)calloc(n_flow, sizeof(float));
                sfy[i] = (float *)calloc(n_flow, sizeof(float));
                if (sfx[i] && sfy[i])
                    compute_apple_flow(green_frames[fc], green_frames[i],
                                       green_w, green_h, sfx[i], sfy[i]);
            }

            float cw_sf = compute_adaptive_center_weight(
                (const float **)sfx, (const float **)sfy,
                loaded, fc, green_w, green_h, NULL);
            temporal_filter_frame_gpu(denoised,
                                      (const uint16_t **)window_frames,
                                      (const float **)sfx,
                                      (const float **)sfy,
                                      loaded, fc,
                                      width, height,
                                      base_cfg->strength, noise_sigma,
                                      sc->chroma_boost, sc->dist_sigma, sc->flow_tightening,
                                      NULL, cw_sf);

            if (base_cfg->use_cnn_postfilter && mps_postfilter_available()) {
                postfilter_frame_mps(denoised, width, height, cnn_blend);
                if (sc->cnn_passes >= 2)
                    postfilter_frame_mps(denoised, width, height, sc->cnn_blend2);
            }

            snprintf(path, sizeof(path), "/tmp/bayerflow_sweep/%s_frame%d.raw", sc->label, sf);
            f = fopen(path, "wb");
            if (f) { fwrite(denoised, sizeof(uint16_t), frame_pixels, f); fclose(f); }

            for (int i = 0; i < loaded; i++) { free(sfx[i]); free(sfy[i]); }
            free(sfx); free(sfy);
        }
    }

    /* Also write 6 original (undenoised) frames for reference */
    for (int sf = 0; sf < SWEEP_FRAMES && (center - 2 + sf) >= 0 && (center - 2 + sf) < loaded; sf++) {
        int fc = center - 2 + sf;
        char path[512];
        snprintf(path, sizeof(path), "/tmp/bayerflow_sweep/original_frame%d.raw", sf);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(window_frames[fc], sizeof(uint16_t), frame_pixels, f); fclose(f); }
    }

    /* Write dimensions file for the Python script */
    {
        char path[512];
        snprintf(path, sizeof(path), "/tmp/bayerflow_sweep/dimensions.txt");
        FILE *f = fopen(path, "w");
        if (f) { fprintf(f, "%d %d %d\n", width, height, num_configs); fclose(f); }
    }

    fprintf(stderr, "sweep: done! %d configs × %d frames written to /tmp/bayerflow_sweep/\n",
            num_configs, SWEEP_FRAMES);

    free(denoised);

sweep_flow_cleanup:
    if (fx && fy) {
        for (int i = 0; i < loaded; i++) { free(fx[i]); free(fy[i]); }
    }
    free(fx); free(fy);

sweep_cleanup:
    for (int i = 0; i < actual_w; i++) {
        free(window_frames[i]);
        free(green_frames[i]);
    }
    free(window_frames);
    free(green_frames);
    frame_reader_close(&reader);
    return num_configs;
}


/* ================================================================== */
/*  Technique Sweep: compare 5 alt temporal techniques vs baseline NLM */
/* ================================================================== */

int denoise_technique_sweep(
    const char *input_path,
    int start_frame,
    const DenoiseCConfig *base_cfg)
{
    const int SWEEP_FRAMES = 6;

    /* Technique function table (0 = baseline NLM, handled separately) */
    TechniqueFn technique_fns[] = {
        NULL,                        /* 0: baseline NLM */
        technique_vst_wiener,        /* 1 */
        technique_inv_variance,      /* 2 */
        technique_vst_nlm,           /* 3 */
        technique_dct_shrinkage,     /* 4 */
        technique_sigma_clip,        /* 5 */
        technique_vst_guided,        /* 6: Guided ε=1.0 */
        technique_vst_bilateral,     /* 7: Bilateral h=1.0 */
        technique_vst_guided_e01,    /* 8: Guided ε=0.1 */
        technique_vst_guided_e03,    /* 9: Guided ε=0.3 */
        technique_vst_bilateral_h08, /* 10: Bilateral h=0.8 */
        technique_vst_bilateral_h12  /* 11: Bilateral h=1.2 */
    };

    /* Probe dimensions */
    int width = 0, height = 0;
    if (frame_reader_probe_dimensions(input_path, &width, &height) != 0)
        return -1;

    size_t frame_pixels = (size_t)width * height;
    size_t frame_bytes  = frame_pixels * sizeof(uint16_t);
    int green_w = width / 2, green_h = height / 2;
    int n_flow = green_w * green_h;

    /* Read frames into window */
    int actual_w = base_cfg->window_size;
    if (actual_w < SWEEP_FRAMES + 4) actual_w = SWEEP_FRAMES + 4;
    int half_w = actual_w / 2;

    FrameReader reader;
    if (frame_reader_open(&reader, input_path) != 0) return -1;
    int num_frames = frame_reader_probe_frame_count(input_path);
    if (num_frames < 0) num_frames = 9999;

    int first_frame = start_frame - half_w;
    if (first_frame < 0) first_frame = 0;

    uint16_t **window_frames = (uint16_t **)calloc(actual_w, sizeof(uint16_t *));
    uint16_t **green_frames  = (uint16_t **)calloc(actual_w, sizeof(uint16_t *));
    if (!window_frames || !green_frames) {
        free(window_frames); free(green_frames);
        frame_reader_close(&reader);
        return -1;
    }

    for (int i = 0; i < actual_w; i++) {
        window_frames[i] = (uint16_t *)malloc(frame_bytes);
        green_frames[i]  = (uint16_t *)malloc(n_flow * sizeof(uint16_t));
    }

    /* Skip to first_frame */
    uint16_t *skip_buf = (uint16_t *)malloc(frame_bytes);
    if (!skip_buf && first_frame > 0) {
        fprintf(stderr, "denoise_core: skip_buf alloc failed\n");
        goto tsweep_cleanup;
    }
    for (int i = 0; i < first_frame; i++)
        frame_reader_read_frame(&reader, skip_buf);
    free(skip_buf); skip_buf = NULL;

    /* Read and preprocess window frames */
    int loaded = 0;
    for (int i = 0; i < actual_w && (first_frame + i) < num_frames; i++) {
        if (frame_reader_read_frame(&reader, window_frames[i]) == 0) {
            extract_green_channel(window_frames[i], width, height, green_frames[i]);
            spatial_filter_green(green_frames[i], green_w, green_h, base_cfg->noise_sigma);
            loaded++;
        }
    }

    int center = half_w;
    if (center >= loaded) center = loaded - 1;

    fprintf(stderr, "technique_sweep: loaded %d frames (center=%d), %dx%d\n",
            loaded, center, width, height);

    /* Estimate noise */
    float noise_sigma = base_cfg->noise_sigma;
    if (noise_sigma == 0)
        noise_sigma = temporal_filter_estimate_noise(window_frames[center], width, height);
    if (noise_sigma < 1.0f) noise_sigma = 1.0f;
    fprintf(stderr, "technique_sweep: noise_sigma=%.1f\n", noise_sigma);

    /* Create output directory */
    system("rm -rf /tmp/bayerflow_technique_sweep");
    system("mkdir -p /tmp/bayerflow_technique_sweep");

    uint16_t *denoised = (uint16_t *)malloc(frame_bytes);
    if (!denoised) goto tsweep_cleanup;

    /* Process SWEEP_FRAMES sequential center frames */
    for (int sf = 0; sf < SWEEP_FRAMES; sf++) {
        int fc = center - SWEEP_FRAMES / 2 + sf;
        if (fc < 0 || fc >= loaded) continue;

        fprintf(stderr, "technique_sweep: center frame %d/%d (window idx=%d)\n",
                sf + 1, SWEEP_FRAMES, fc);

        /* Compute optical flow for all frames relative to this center */
        float **fx = (float **)calloc(loaded, sizeof(float *));
        float **fy = (float **)calloc(loaded, sizeof(float *));
        if (!fx || !fy) { free(fx); free(fy); continue; }

        double t_of = timer_now();
        for (int i = 0; i < loaded; i++) {
            if (i == fc) continue;
            fx[i] = (float *)calloc(n_flow, sizeof(float));
            fy[i] = (float *)calloc(n_flow, sizeof(float));
            if (fx[i] && fy[i])
                compute_apple_flow(green_frames[fc], green_frames[i],
                                   green_w, green_h, fx[i], fy[i]);
        }
        fprintf(stderr, "  OF computed in %.2fs\n", timer_now() - t_of);

        /* Run each technique on this center frame */
        for (int tech = 0; tech < NUM_TECHNIQUES; tech++) {
            double t_tech = timer_now();

            if (tech == 0) {
                /* Baseline NLM via temporal_filter_frame_cpu */
                TemporalFilterTuning tuning = {
                    .chroma_boost = 1.2f,
                    .dist_sigma = 2.0f,
                    .flow_tightening = 1.5f
                };
                temporal_filter_frame_cpu(
                    denoised,
                    (const uint16_t **)window_frames,
                    (const float **)fx, (const float **)fy,
                    loaded, fc,
                    width, height,
                    base_cfg->strength, noise_sigma,
                    &tuning);
            } else {
                /* Alternative technique */
                technique_fns[tech](
                    denoised,
                    (const uint16_t **)window_frames,
                    (const float **)fx, (const float **)fy,
                    loaded, fc,
                    width, height,
                    noise_sigma);
            }

            fprintf(stderr, "  tech%d [%s] %.2fs\n",
                    tech, technique_names[tech], timer_now() - t_tech);

            /* Write raw output */
            char path[512];
            snprintf(path, sizeof(path),
                     "/tmp/bayerflow_technique_sweep/tech%d_frame%d.raw",
                     tech, sf);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(denoised, sizeof(uint16_t), frame_pixels, f); fclose(f); }
        }

        /* Write original (undenoised) frame */
        {
            char path[512];
            snprintf(path, sizeof(path),
                     "/tmp/bayerflow_technique_sweep/original_frame%d.raw", sf);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(window_frames[fc], sizeof(uint16_t), frame_pixels, f); fclose(f); }
        }

        /* Free flow arrays for this center */
        for (int i = 0; i < loaded; i++) { free(fx[i]); free(fy[i]); }
        free(fx); free(fy);
    }

    /* Write metadata */
    {
        char path[512];
        snprintf(path, sizeof(path),
                 "/tmp/bayerflow_technique_sweep/metadata.json");
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "{\n");
            fprintf(f, "  \"width\": %d,\n", width);
            fprintf(f, "  \"height\": %d,\n", height);
            fprintf(f, "  \"num_techniques\": %d,\n", NUM_TECHNIQUES);
            fprintf(f, "  \"num_frames\": %d,\n", SWEEP_FRAMES);
            fprintf(f, "  \"noise_sigma\": %.1f,\n", noise_sigma);
            fprintf(f, "  \"techniques\": [\n");
            for (int t = 0; t < NUM_TECHNIQUES; t++)
                fprintf(f, "    \"%s\"%s\n", technique_names[t],
                        t < NUM_TECHNIQUES - 1 ? "," : "");
            fprintf(f, "  ]\n}\n");
            fclose(f);
        }
    }

    fprintf(stderr, "technique_sweep: done! %d techniques × %d frames → "
            "/tmp/bayerflow_technique_sweep/\n", NUM_TECHNIQUES, SWEEP_FRAMES);

    free(denoised);

tsweep_cleanup:
    for (int i = 0; i < actual_w; i++) {
        free(window_frames[i]);
        free(green_frames[i]);
    }
    free(window_frames);
    free(green_frames);
    frame_reader_close(&reader);
    return NUM_TECHNIQUES;
}
