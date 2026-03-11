#include "include/wiener_temporal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <dispatch/dispatch.h>

/* Noise estimation — reuse from temporal_filter.c */
extern float temporal_filter_estimate_noise(const uint16_t *bayer, int width, int height);

/* ---- Constants ---- */

#define PATCH_SIZE   8
#define PATCH_STEP   4    /* 50% overlap */
#define PATCH_COEFFS 64   /* 8 * 8 */
#define BAYER_COMPS  4
#define MIN_SIGNAL_VAR 1.0f
#define PEDESTAL     6080.0f   /* S1/S5 black level */

/* ---- Orthonormal DCT-II matrices ---- */

static float dct_mat[8][8];
static float idct_mat[8][8];   /* = transpose */
static int dct_initialized = 0;

static float blend_1d[PATCH_SIZE];
static float blend_2d[PATCH_COEFFS];
static int blend_initialized = 0;

static void init_dct_matrices(void) {
    if (dct_initialized) return;
    for (int k = 0; k < 8; k++) {
        float alpha = (k == 0) ? sqrtf(1.0f / 8.0f) : sqrtf(2.0f / 8.0f);
        for (int n = 0; n < 8; n++) {
            float val = alpha * cosf((float)M_PI * (2 * n + 1) * k / 16.0f);
            dct_mat[k][n] = val;
            idct_mat[n][k] = val;
        }
    }
    dct_initialized = 1;
}

static void init_blend_window(void) {
    if (blend_initialized) return;
    for (int i = 0; i < PATCH_SIZE; i++)
        blend_1d[i] = 0.5f - 0.5f * cosf((float)M_PI * (i + 0.5f) / PATCH_SIZE);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            blend_2d[y * 8 + x] = blend_1d[y] * blend_1d[x];
    blend_initialized = 1;
}

/* ---- Forward / Inverse 2D DCT (separable, in-place) ---- */

static void dct2d_forward(float patch[PATCH_COEFFS]) {
    float tmp[8];
    /* Column pass: for each column x, transform rows */
    for (int x = 0; x < 8; x++) {
        for (int k = 0; k < 8; k++) {
            float sum = 0;
            for (int n = 0; n < 8; n++)
                sum += dct_mat[k][n] * patch[n * 8 + x];
            tmp[k] = sum;
        }
        for (int k = 0; k < 8; k++)
            patch[k * 8 + x] = tmp[k];
    }
    /* Row pass: for each row y, transform columns */
    for (int y = 0; y < 8; y++) {
        float *row = &patch[y * 8];
        for (int k = 0; k < 8; k++) {
            float sum = 0;
            for (int n = 0; n < 8; n++)
                sum += dct_mat[k][n] * row[n];
            tmp[k] = sum;
        }
        memcpy(row, tmp, 8 * sizeof(float));
    }
}

static void dct2d_inverse(float patch[PATCH_COEFFS]) {
    float tmp[8];
    /* Row pass */
    for (int y = 0; y < 8; y++) {
        float *row = &patch[y * 8];
        for (int n = 0; n < 8; n++) {
            float sum = 0;
            for (int k = 0; k < 8; k++)
                sum += idct_mat[n][k] * row[k];
            tmp[n] = sum;
        }
        memcpy(row, tmp, 8 * sizeof(float));
    }
    /* Column pass */
    for (int x = 0; x < 8; x++) {
        for (int n = 0; n < 8; n++) {
            float sum = 0;
            for (int k = 0; k < 8; k++)
                sum += idct_mat[n][k] * patch[k * 8 + x];
            tmp[n] = sum;
        }
        for (int n = 0; n < 8; n++)
            patch[n * 8 + x] = tmp[n];
    }
}

/* ---- Noise model ---- */

typedef struct {
    float sigma_read_sq;
    float shot_gain;
} NoiseModel;

static NoiseModel build_noise_model(float sigma_total) {
    NoiseModel nm;
    float sigma_read = 0.4f * sigma_total;
    nm.sigma_read_sq = sigma_read * sigma_read;
    float shot_var = sigma_total * sigma_total - nm.sigma_read_sq;
    if (shot_var < 0) shot_var = 0;
    /* Derive shot gain from typical midtone signal above pedestal */
    nm.shot_gain = shot_var / 10000.0f;
    return nm;
}

/* Noise variance for a patch given its mean brightness */
static float patch_noise_var(const float *patch, int n, const NoiseModel *nm) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += patch[i];
    float mean = sum / n;
    float signal = mean - PEDESTAL;
    if (signal < 0) signal = 0;
    float var = nm->sigma_read_sq + nm->shot_gain * signal;
    if (var < 1.0f) var = 1.0f;
    return var;
}

/* ---- Bayer sub-channel extract/insert ---- */

static void extract_sub(const uint16_t *bayer, int w, int h,
                         float *out, int comp) {
    int dy = (comp >> 1) & 1;
    int dx = comp & 1;
    int sw = w / 2, sh = h / 2;
    for (int y = 0; y < sh; y++)
        for (int x = 0; x < sw; x++)
            out[y * sw + x] = (float)bayer[(y * 2 + dy) * w + (x * 2 + dx)];
}

static void insert_sub(uint16_t *bayer, int w, int h,
                        const float *in, int comp) {
    int dy = (comp >> 1) & 1;
    int dx = comp & 1;
    int sw = w / 2, sh = h / 2;
    for (int y = 0; y < sh; y++)
        for (int x = 0; x < sw; x++) {
            int v = (int)(in[y * sw + x] + 0.5f);
            if (v < 0) v = 0;
            if (v > 65535) v = 65535;
            bayer[(y * 2 + dy) * w + (x * 2 + dx)] = (uint16_t)v;
        }
}

/* ---- Warp neighbor sub-channel with bilinear interpolation ---- */

static void warp_sub(const uint16_t *nbr_bayer, int w, int h,
                      const float *flow_x, const float *flow_y,
                      int comp, float *warped) {
    int dy = (comp >> 1) & 1;
    int dx = comp & 1;
    int sw = w / 2, sh = h / 2;

    dispatch_apply((size_t)sh,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ^(size_t y_idx) {
        int y = (int)y_idx;
        for (int x = 0; x < sw; x++) {
            float fx = flow_x[y * sw + x];
            float fy = flow_y[y * sw + x];

            float sx = (float)x + fx;
            float sy = (float)y + fy;

            int ix = (int)floorf(sx);
            int iy = (int)floorf(sy);
            float frac_x = sx - (float)ix;
            float frac_y = sy - (float)iy;

            if (ix < 0 || ix + 1 >= sw || iy < 0 || iy + 1 >= sh) {
                warped[y * sw + x] = NAN;
                continue;
            }

            float s00 = (float)nbr_bayer[(iy * 2 + dy) * w + (ix * 2 + dx)];
            float s10 = (float)nbr_bayer[(iy * 2 + dy) * w + ((ix + 1) * 2 + dx)];
            float s01 = (float)nbr_bayer[((iy + 1) * 2 + dy) * w + (ix * 2 + dx)];
            float s11 = (float)nbr_bayer[((iy + 1) * 2 + dy) * w + ((ix + 1) * 2 + dx)];

            warped[y * sw + x] = (1.0f - frac_x) * (1.0f - frac_y) * s00
                               +         frac_x  * (1.0f - frac_y) * s10
                               + (1.0f - frac_x) *         frac_y  * s01
                               +         frac_x  *         frac_y  * s11;
        }
    });
}

/* ---- Core: fuse one neighbor into the running estimate ---- */

static void fuse_one_neighbor(float *estimate, const float *warped,
                               float *accum, float *weight,
                               int sw, int sh,
                               const NoiseModel *nm, float strength) {
    /* 50% overlapping 8x8 patches with raised-cosine blend window */
    int grid_x = (sw - PATCH_SIZE) / PATCH_STEP + 1;
    int grid_y = (sh - PATCH_SIZE) / PATCH_STEP + 1;

    if (grid_x <= 0 || grid_y <= 0) return;

    size_t sub_pixels = (size_t)sw * sh;
    memset(accum, 0, sub_pixels * sizeof(float));
    memset(weight, 0, sub_pixels * sizeof(float));

    /* Capture values for block */
    const float str = strength;
    const NoiseModel local_nm = *nm;

    /* Process patches — sequential to accumulate safely into shared buffers.
     * Each patch row can be parallelized since overlap is only horizontal
     * within a row, but vertical overlap between rows requires serialization.
     * For simplicity and correctness, we parallelize across rows of patches
     * that are 2 steps apart (even/odd pass). */
    for (int phase = 0; phase < 2; phase++) {
        dispatch_apply((size_t)((grid_y - phase + 1) / 2),
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^(size_t row_idx) {
            int py = (int)row_idx * 2 + phase;
            if (py >= grid_y) return;
            int oy = py * PATCH_STEP;

            /* Thread-local scratch buffers */
            float est_patch[PATCH_COEFFS], nbr_patch[PATCH_COEFFS];
            float est_dct[PATCH_COEFFS], nbr_dct[PATCH_COEFFS];
            float fused[PATCH_COEFFS];

            for (int px = 0; px < grid_x; px++) {
                int ox = px * PATCH_STEP;

                /* Extract 8x8 patches from estimate and warped neighbor */
                int valid = 1;
                for (int j = 0; j < 8 && valid; j++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = (oy + j) * sw + (ox + i);
                        est_patch[j * 8 + i] = estimate[idx];
                        float nv = warped[idx];
                        if (isnan(nv)) { valid = 0; break; }
                        nbr_patch[j * 8 + i] = nv;
                    }
                }

                if (!valid) continue;

                /* Motion rejection: skip patches where warped neighbor
                 * deviates too much from estimate (flow failure / occlusion).
                 * Threshold = 3× expected noise std — anything beyond that
                 * is motion, not noise. Use soft falloff for partial rejection. */
                float patch_mse = 0;
                for (int k = 0; k < PATCH_COEFFS; k++) {
                    float d = est_patch[k] - nbr_patch[k];
                    patch_mse += d * d;
                }
                patch_mse /= PATCH_COEFFS;
                float nvar_reject = patch_noise_var(est_patch, PATCH_COEFFS, &local_nm);
                /* Hard reject if MSE > 9× noise variance (3σ) */
                if (patch_mse > 9.0f * nvar_reject) continue;
                /* Soft scale: reduce fusion strength for borderline patches */
                float motion_scale = 1.0f;
                if (patch_mse > 2.0f * nvar_reject) {
                    motion_scale = (9.0f * nvar_reject - patch_mse) /
                                   (7.0f * nvar_reject);
                    if (motion_scale < 0.0f) motion_scale = 0.0f;
                }

                /* Forward DCT on both patches */
                memcpy(est_dct, est_patch, PATCH_COEFFS * sizeof(float));
                memcpy(nbr_dct, nbr_patch, PATCH_COEFFS * sizeof(float));
                dct2d_forward(est_dct);
                dct2d_forward(nbr_dct);

                /* Noise variance for this patch (signal-dependent) */
                float nvar = patch_noise_var(est_patch, PATCH_COEFFS, &local_nm);
                float adjusted_nvar = nvar / (str * str);

                /* Wiener fusion per DCT coefficient, scaled by motion confidence */
                for (int c = 0; c < PATCH_COEFFS; c++) {
                    float est_power = est_dct[c] * est_dct[c];
                    float sig_power = est_power - adjusted_nvar;
                    if (sig_power < MIN_SIGNAL_VAR) sig_power = MIN_SIGNAL_VAR;

                    float wiener_w = sig_power / (sig_power + adjusted_nvar);
                    wiener_w *= motion_scale;  /* reduce near motion boundaries */

                    fused[c] = est_dct[c] + wiener_w * (nbr_dct[c] - est_dct[c]);
                }

                /* Inverse DCT → overlap-add with blend window */
                dct2d_inverse(fused);

                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = (oy + j) * sw + (ox + i);
                        float w = blend_2d[j * 8 + i];
                        accum[idx]  += fused[j * 8 + i] * w;
                        weight[idx] += w;
                    }
                }
            }
        });
    }

    /* Normalize: where we have overlap-add coverage, use it; else keep estimate */
    for (size_t i = 0; i < sub_pixels; i++) {
        if (weight[i] > 0.0001f)
            estimate[i] = accum[i] / weight[i];
    }
}

/* ---- Nearest-first neighbor ordering ---- */

static void nearest_first_order(int num_frames, int center_idx,
                                 int *order, int *out_count) {
    int n = 0;
    /* Interleave: center-1, center+1, center-2, center+2, ... */
    for (int d = 1; d < num_frames; d++) {
        int lo = center_idx - d;
        int hi = center_idx + d;
        if (lo >= 0) order[n++] = lo;
        if (hi < num_frames) order[n++] = hi;
    }
    *out_count = n;
}

/* ---- Top-level Wiener temporal filter ---- */

void wiener_temporal_filter_frame(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x,
    const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma)
{
    init_dct_matrices();
    init_blend_window();

    float sigma = noise_sigma;
    if (sigma <= 0) {
        sigma = temporal_filter_estimate_noise(frames[center_idx], width, height);
        if (sigma < 1.0f) sigma = 1.0f;
    }

    if (strength <= 0) strength = 1.0f;

    NoiseModel nm = build_noise_model(sigma);

    int sw = width / 2;
    int sh = height / 2;
    size_t sub_pixels = (size_t)sw * sh;

    /* Neighbor ordering */
    int *nbr_order = (int *)malloc(num_frames * sizeof(int));
    int nbr_count = 0;
    if (!nbr_order) {
        memcpy(output, frames[center_idx], (size_t)width * height * sizeof(uint16_t));
        return;
    }
    nearest_first_order(num_frames, center_idx, nbr_order, &nbr_count);

    /* Working buffers (allocated once, reused across sub-channels and neighbors) */
    float *estimate = (float *)malloc(sub_pixels * sizeof(float));
    float *warped   = (float *)malloc(sub_pixels * sizeof(float));
    float *accum    = (float *)malloc(sub_pixels * sizeof(float));
    float *weight_buf = (float *)malloc(sub_pixels * sizeof(float));

    if (!estimate || !warped || !accum || !weight_buf) {
        memcpy(output, frames[center_idx], (size_t)width * height * sizeof(uint16_t));
        free(estimate); free(warped); free(accum); free(weight_buf); free(nbr_order);
        return;
    }

    /* Start with center frame in output (border pixels not covered by patches) */
    memcpy(output, frames[center_idx], (size_t)width * height * sizeof(uint16_t));

    /* Process each Bayer sub-channel independently */
    for (int comp = 0; comp < BAYER_COMPS; comp++) {
        /* Phase 1: Initialize estimate = center frame sub-channel */
        extract_sub(frames[center_idx], width, height, estimate, comp);

        /* Phase 2: Recursively fuse each neighbor (nearest-first) */
        for (int ni = 0; ni < nbr_count; ni++) {
            int nbr_idx = nbr_order[ni];
            if (!flows_x[nbr_idx] || !flows_y[nbr_idx]) continue;

            /* Warp neighbor sub-channel into center's coordinate space */
            warp_sub(frames[nbr_idx], width, height,
                     flows_x[nbr_idx], flows_y[nbr_idx],
                     comp, warped);

            /* Fuse into running estimate */
            fuse_one_neighbor(estimate, warped, accum, weight_buf, sw, sh, &nm, strength);
        }

        /* Phase 3: Write denoised sub-channel back to output */
        insert_sub(output, width, height, estimate, comp);
    }

    free(estimate);
    free(warped);
    free(accum);
    free(weight_buf);
    free(nbr_order);
}
