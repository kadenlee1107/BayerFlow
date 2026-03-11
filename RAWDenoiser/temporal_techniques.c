/*
 * temporal_techniques.c — 6 alternative temporal denoising techniques
 *
 * All share the same interface: receive a window of aligned Bayer frames
 * with per-frame optical flow, produce a denoised center frame.
 *
 * Calibrated for S5II: black_level=6032, shot_gain=180, read_noise_frac=0.55
 */

#include "../include/temporal_techniques.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ---- Calibrated noise model constants ---- */
#define BLACK_LEVEL     6032.0f
#define SHOT_GAIN       180.0f
#define READ_NOISE_FRAC 0.55f
#define MAX_SAMPLES     32   /* max frames in window */

/* ---- Shared Utility Functions ---- */

/* Per-pixel noise sigma from calibrated Poisson-Gaussian model. */
static inline float pixel_noise_sigma(float v, float global_sigma) {
    float read_var = READ_NOISE_FRAC * global_sigma * global_sigma;
    float signal = (v > BLACK_LEVEL) ? (v - BLACK_LEVEL) : 0.0f;
    return sqrtf(read_var + SHOT_GAIN * signal);
}

/* Look up optical flow for a raw Bayer pixel.
 * Flow arrays are green-pixel resolution (width/2 × height/2). */
static inline void flow_lookup(const float *fx, const float *fy,
                               int rx, int ry, int green_w,
                               float *out_fdx, float *out_fdy)
{
    int gx = rx >> 1;
    int gy = ry >> 1;
    *out_fdx = fx[gy * green_w + gx];
    *out_fdy = fy[gy * green_w + gx];
}

/* Bilinear warp of a same-color Bayer pixel via optical flow.
 * Flow is in green-pixel units; same-color stride is 2 in raw coords.
 * Returns interpolated value, or -1.0f if out of bounds. */
static inline float bilinear_warp_pixel(int rx, int ry,
                                        float fdx, float fdy,
                                        const uint16_t *ref,
                                        int width, int height)
{
    /* Flow is in green-pixel units. Raw displacement = flow * 2. */
    float raw_dx = fdx * 2.0f;
    float raw_dy = fdy * 2.0f;

    float sx = (float)rx + raw_dx;
    float sy = (float)ry + raw_dy;

    /* Same-color grid: snap to nearest same-color position */
    int bx0 = ((int)floorf(sx)) & ~1;  /* ensure even step for same-color */
    int by0 = ((int)floorf(sy)) & ~1;
    /* Adjust to match parity of (rx, ry) */
    if ((bx0 & 1) != (rx & 1)) bx0 += (bx0 < (int)sx) ? 1 : -1;
    if ((by0 & 1) != (ry & 1)) by0 += (by0 < (int)sy) ? 1 : -1;

    int bx1 = bx0 + 2;
    int by1 = by0 + 2;

    if (bx0 < 0 || bx1 >= width || by0 < 0 || by1 >= height)
        return -1.0f;

    float frac_x = (sx - (float)bx0) / 2.0f;
    float frac_y = (sy - (float)by0) / 2.0f;
    if (frac_x < 0.0f) frac_x = 0.0f;
    if (frac_x > 1.0f) frac_x = 1.0f;
    if (frac_y < 0.0f) frac_y = 0.0f;
    if (frac_y > 1.0f) frac_y = 1.0f;

    float s00 = (float)ref[by0 * width + bx0];
    float s10 = (float)ref[by0 * width + bx1];
    float s01 = (float)ref[by1 * width + bx0];
    float s11 = (float)ref[by1 * width + bx1];

    return (1.0f - frac_x) * (1.0f - frac_y) * s00
         +         frac_x  * (1.0f - frac_y) * s10
         + (1.0f - frac_x) *         frac_y  * s01
         +         frac_x  *         frac_y  * s11;
}

/* Forward Generalized Anscombe Transform.
 * Stabilizes Poisson-Gaussian noise to approximately unit variance. */
static inline float anscombe_forward(float v, float global_sigma) {
    float read_var = READ_NOISE_FRAC * global_sigma * global_sigma;
    float signal = (v > BLACK_LEVEL) ? (v - BLACK_LEVEL) : 0.0f;
    float x = signal / SHOT_GAIN + 3.0f / 8.0f
            + read_var / (SHOT_GAIN * SHOT_GAIN);
    return 2.0f * sqrtf(fmaxf(x, 0.0f));
}

/* Inverse Generalized Anscombe Transform (with bias correction). */
static inline float anscombe_inverse(float z, float global_sigma) {
    float read_var = READ_NOISE_FRAC * global_sigma * global_sigma;
    if (z < 0.5f) z = 0.5f;  /* clamp to avoid divergence */
    float z2 = z * z;
    /* Exact inverse: val = gain * (z²/4 - 3/8 - read²/gain²) + BL */
    float val = SHOT_GAIN * (z2 / 4.0f - 3.0f / 8.0f
                - read_var / (SHOT_GAIN * SHOT_GAIN))
                + BLACK_LEVEL;
    /* Makitalo-Foi asymptotic bias correction */
    if (z > 1.0f)
        val += SHOT_GAIN / (4.0f * z2);
    return val;
}

/* Collect warped temporal samples for a single pixel across all frames. */
static int collect_samples(int rx, int ry,
                           const uint16_t **frames,
                           const float **flows_x, const float **flows_y,
                           int num_frames, int center_idx,
                           int width, int height,
                           float *samples, int *sample_is_center)
{
    int green_w = width / 2;
    int n = 0;

    for (int f = 0; f < num_frames; f++) {
        if (f == center_idx) {
            samples[n] = (float)frames[f][ry * width + rx];
            if (sample_is_center) sample_is_center[n] = 1;
            n++;
        } else if (flows_x[f] && flows_y[f]) {
            float fdx, fdy;
            flow_lookup(flows_x[f], flows_y[f], rx, ry, green_w, &fdx, &fdy);
            float v = bilinear_warp_pixel(rx, ry, fdx, fdy,
                                          frames[f], width, height);
            if (v >= 0.0f) {
                samples[n] = v;
                if (sample_is_center) sample_is_center[n] = 0;
                n++;
            }
        }
    }
    return n;
}


/* ================================================================== */
/*  TECHNIQUE 1: VST + Temporal Wiener Shrinkage                      */
/* ================================================================== */

void technique_vst_wiener(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    for (int ry = 0; ry < height; ry++) {
        for (int rx = 0; rx < width; rx++) {
            float raw_samples[MAX_SAMPLES];
            int n = collect_samples(rx, ry, frames, flows_x, flows_y,
                                    num_frames, center_idx, width, height,
                                    raw_samples, NULL);

            float center_val = (float)frames[center_idx][ry * width + rx];
            if (n <= 1) {
                output[ry * width + rx] = (uint16_t)center_val;
                continue;
            }

            /* Forward Anscombe transform */
            float z_center = anscombe_forward(center_val, noise_sigma);
            float z_samples[MAX_SAMPLES];
            int valid = 0;

            for (int i = 0; i < n; i++) {
                float z = anscombe_forward(raw_samples[i], noise_sigma);
                /* Reject if > 3σ from center in VST domain (σ≈1) */
                if (fabsf(z - z_center) <= 3.0f) {
                    z_samples[valid++] = z;
                }
            }

            if (valid <= 1) {
                output[ry * width + rx] = (uint16_t)center_val;
                continue;
            }

            /* Compute temporal mean and variance */
            float z_sum = 0, z_sum2 = 0;
            for (int i = 0; i < valid; i++) {
                z_sum  += z_samples[i];
                z_sum2 += z_samples[i] * z_samples[i];
            }
            float z_mean = z_sum / (float)valid;
            float z_var  = z_sum2 / (float)valid - z_mean * z_mean;

            /* Wiener shrinkage: estimate = mean + max(0, var-1)/var * (center - mean)
             * In VST domain, noise variance ≈ 1.0 */
            float signal_var = fmaxf(z_var - 1.0f, 0.0f);
            float wiener_gain = (z_var > 0.01f) ? signal_var / z_var : 0.0f;
            float z_est = z_mean + wiener_gain * (z_center - z_mean);

            /* Inverse Anscombe */
            float result = anscombe_inverse(z_est, noise_sigma);
            if (result < 0.0f) result = 0.0f;
            if (result > 65535.0f) result = 65535.0f;
            output[ry * width + rx] = (uint16_t)(result + 0.5f);
        }
    }
}


/* ================================================================== */
/*  TECHNIQUE 2: Inverse-Variance Weighted Mean                       */
/* ================================================================== */

void technique_inv_variance(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    for (int ry = 0; ry < height; ry++) {
        for (int rx = 0; rx < width; rx++) {
            float raw_samples[MAX_SAMPLES];
            int n = collect_samples(rx, ry, frames, flows_x, flows_y,
                                    num_frames, center_idx, width, height,
                                    raw_samples, NULL);

            float center_val = (float)frames[center_idx][ry * width + rx];
            if (n <= 1) {
                output[ry * width + rx] = (uint16_t)center_val;
                continue;
            }

            /* 3σ rejection threshold based on center pixel noise */
            float thresh = 3.0f * pixel_noise_sigma(center_val, noise_sigma);

            /* Inverse-variance weighted mean */
            float w_sum = 0.0f, val_sum = 0.0f;
            for (int i = 0; i < n; i++) {
                if (fabsf(raw_samples[i] - center_val) > thresh)
                    continue;  /* reject outlier */

                float sig = pixel_noise_sigma(raw_samples[i], noise_sigma);
                float w = 1.0f / (sig * sig + 1.0f);  /* +1 for numerical stability */
                w_sum   += w;
                val_sum += w * raw_samples[i];
            }

            float result = (w_sum > 0.0f) ? val_sum / w_sum : center_val;
            if (result < 0.0f) result = 0.0f;
            if (result > 65535.0f) result = 65535.0f;
            output[ry * width + rx] = (uint16_t)(result + 0.5f);
        }
    }
}


/* ================================================================== */
/*  TECHNIQUE 3: VST + Uniform NLM (no signal-dependent heuristics)   */
/* ================================================================== */

void technique_vst_nlm(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    int green_w = width / 2;
    size_t frame_pixels = (size_t)width * height;

    /* Pre-transform all frames to Anscombe domain */
    float **z_frames = (float **)calloc(num_frames, sizeof(float *));
    for (int f = 0; f < num_frames; f++) {
        z_frames[f] = (float *)malloc(frame_pixels * sizeof(float));
        for (size_t p = 0; p < frame_pixels; p++)
            z_frames[f][p] = anscombe_forward((float)frames[f][p], noise_sigma);
    }

    const float h = 1.0f;          /* natural bandwidth in VST domain */
    const float h2 = h * h;
    const int PATCH_R = 2;         /* 5×5 patch radius in same-color coords */
    const float center_weight = 0.4f;  /* fixed center weight */

    for (int ry = 2; ry < height - 2; ry++) {
        for (int rx = 2; rx < width - 2; rx++) {
            float z_cv = z_frames[center_idx][ry * width + rx];
            float val_sum = center_weight * z_cv;
            float w_sum   = center_weight;

            for (int f = 0; f < num_frames; f++) {
                if (f == center_idx) continue;
                if (!flows_x[f] || !flows_y[f]) continue;

                float fdx, fdy;
                flow_lookup(flows_x[f], flows_y[f], rx, ry, green_w, &fdx, &fdy);

                /* Compute patch distance in Anscombe domain */
                float dist_sq = 0.0f;
                int   patch_count = 0;

                for (int py = -PATCH_R; py <= PATCH_R; py++) {
                    for (int px = -PATCH_R; px <= PATCH_R; px++) {
                        int cx = rx + px * 2;  /* same-color stride */
                        int cy = ry + py * 2;
                        if (cx < 0 || cx >= width || cy < 0 || cy >= height)
                            continue;

                        float cv = z_frames[center_idx][cy * width + cx];
                        float nv = bilinear_warp_pixel(cx, cy, fdx, fdy,
                                                       frames[f], width, height);
                        if (nv < 0.0f) continue;
                        float z_nv = anscombe_forward(nv, noise_sigma);

                        float d = cv - z_nv;
                        dist_sq += d * d;
                        patch_count++;
                    }
                }

                if (patch_count < 5) continue;
                float mean_dist_sq = dist_sq / (float)patch_count;

                /* Reject if patch distance > 3σ (where σ≈1 in VST domain) */
                if (mean_dist_sq > 9.0f) continue;

                /* NLM weight with fixed bandwidth */
                float w = expf(-mean_dist_sq / (2.0f * h2));

                /* Warp center pixel */
                float warped = bilinear_warp_pixel(rx, ry, fdx, fdy,
                                                    frames[f], width, height);
                if (warped < 0.0f) continue;
                float z_warped = anscombe_forward(warped, noise_sigma);

                val_sum += w * z_warped;
                w_sum   += w;
            }

            /* Inverse Anscombe */
            float z_result = val_sum / w_sum;
            float result = anscombe_inverse(z_result, noise_sigma);
            if (result < 0.0f) result = 0.0f;
            if (result > 65535.0f) result = 65535.0f;
            output[ry * width + rx] = (uint16_t)(result + 0.5f);
        }
    }

    /* Handle border pixels (copy from center frame) */
    for (int ry = 0; ry < height; ry++)
        for (int rx = 0; rx < width; rx++)
            if (ry < 2 || ry >= height - 2 || rx < 2 || rx >= width - 2)
                output[ry * width + rx] = frames[center_idx][ry * width + rx];

    for (int f = 0; f < num_frames; f++) free(z_frames[f]);
    free(z_frames);
}


/* ================================================================== */
/*  TECHNIQUE 4: Temporal DCT Shrinkage                               */
/* ================================================================== */

void technique_dct_shrinkage(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    /* Pre-compute DCT-II basis for each possible sample count (3..MAX_SAMPLES) */
    float dct_basis[MAX_SAMPLES][MAX_SAMPLES];

    for (int ry = 0; ry < height; ry++) {
        for (int rx = 0; rx < width; rx++) {
            float raw_samples[MAX_SAMPLES];
            int center_sample_idx = -1;
            int sample_is_center[MAX_SAMPLES];
            int n = collect_samples(rx, ry, frames, flows_x, flows_y,
                                    num_frames, center_idx, width, height,
                                    raw_samples, sample_is_center);

            float center_val = (float)frames[center_idx][ry * width + rx];
            if (n < 3) {
                output[ry * width + rx] = (uint16_t)center_val;
                continue;
            }

            /* Find center sample index */
            for (int i = 0; i < n; i++)
                if (sample_is_center[i]) { center_sample_idx = i; break; }

            /* Reject outliers > 3σ, replace with center value */
            float sig = pixel_noise_sigma(center_val, noise_sigma);
            float thresh = 3.0f * sig;
            for (int i = 0; i < n; i++) {
                if (!sample_is_center[i] &&
                    fabsf(raw_samples[i] - center_val) > thresh)
                    raw_samples[i] = center_val;
            }

            /* Compute DCT-II basis for this N */
            float scale_0 = sqrtf(1.0f / (float)n);
            float scale_k = sqrtf(2.0f / (float)n);
            for (int k = 0; k < n; k++)
                for (int i = 0; i < n; i++)
                    dct_basis[k][i] = ((k == 0) ? scale_0 : scale_k)
                        * cosf((float)M_PI * (float)k * (2.0f * (float)i + 1.0f)
                               / (2.0f * (float)n));

            /* Forward DCT */
            float dct_coeffs[MAX_SAMPLES];
            for (int k = 0; k < n; k++) {
                float sum = 0.0f;
                for (int i = 0; i < n; i++)
                    sum += dct_basis[k][i] * raw_samples[i];
                dct_coeffs[k] = sum;
            }

            /* Hard threshold: zero coefficients below 3σ/sqrt(N) */
            float dct_thresh = 3.0f * sig / sqrtf((float)n);
            for (int k = 1; k < n; k++) {  /* skip DC (k=0) */
                if (fabsf(dct_coeffs[k]) < dct_thresh)
                    dct_coeffs[k] = 0.0f;
            }

            /* Inverse DCT (transpose of forward for orthonormal basis) */
            float recon[MAX_SAMPLES];
            for (int i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++)
                    sum += dct_basis[k][i] * dct_coeffs[k];
                recon[i] = sum;
            }

            /* Output = reconstructed center sample */
            float result = (center_sample_idx >= 0) ?
                           recon[center_sample_idx] : recon[0];
            if (result < 0.0f) result = 0.0f;
            if (result > 65535.0f) result = 65535.0f;
            output[ry * width + rx] = (uint16_t)(result + 0.5f);
        }
    }
}


/* ================================================================== */
/*  TECHNIQUE 5: Sigma-Clipped Robust Mean                            */
/* ================================================================== */

void technique_sigma_clip(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    int green_w = width / 2;

    for (int ry = 0; ry < height; ry++) {
        for (int rx = 0; rx < width; rx++) {
            float raw_samples[MAX_SAMPLES];
            float flow_mags[MAX_SAMPLES];
            int n = 0;

            /* Collect samples with flow magnitudes */
            for (int f = 0; f < num_frames && f < MAX_SAMPLES; f++) {
                if (f == center_idx) {
                    raw_samples[n] = (float)frames[f][ry * width + rx];
                    flow_mags[n] = 0.0f;
                    n++;
                } else if (flows_x[f] && flows_y[f]) {
                    float fdx, fdy;
                    flow_lookup(flows_x[f], flows_y[f], rx, ry, green_w,
                                &fdx, &fdy);
                    float v = bilinear_warp_pixel(rx, ry, fdx, fdy,
                                                  frames[f], width, height);
                    if (v >= 0.0f) {
                        raw_samples[n] = v;
                        flow_mags[n] = sqrtf(fdx * fdx + fdy * fdy);
                        n++;
                    }
                }
            }

            float center_val = (float)frames[center_idx][ry * width + rx];
            if (n <= 1) {
                output[ry * width + rx] = (uint16_t)center_val;
                continue;
            }

            /* Initial sigma from calibrated noise model */
            float sig = pixel_noise_sigma(center_val, noise_sigma);
            float clip_thresh = 2.5f * sig;
            int valid[MAX_SAMPLES];
            for (int i = 0; i < n; i++) valid[i] = 1;

            /* 2 iterations of sigma-clipping */
            float mean = 0.0f;
            for (int iter = 0; iter < 2; iter++) {
                /* Compute mean of valid samples */
                float sum = 0.0f;
                int count = 0;
                for (int i = 0; i < n; i++) {
                    if (valid[i]) { sum += raw_samples[i]; count++; }
                }
                if (count == 0) break;
                mean = sum / (float)count;

                /* Reject samples > 2.5σ from mean */
                for (int i = 0; i < n; i++) {
                    if (valid[i] && fabsf(raw_samples[i] - mean) > clip_thresh)
                        valid[i] = 0;
                }
            }

            /* Weighted mean of survivors (weight by flow distance confidence) */
            float dist_sigma = 2.0f;
            float w_sum = 0.0f, val_sum = 0.0f;
            for (int i = 0; i < n; i++) {
                if (!valid[i]) continue;
                float mag = flow_mags[i];
                float w = expf(-(mag * mag) / (2.0f * dist_sigma * dist_sigma));
                w_sum   += w;
                val_sum += w * raw_samples[i];
            }

            float result = (w_sum > 0.0f) ? val_sum / w_sum : center_val;
            if (result < 0.0f) result = 0.0f;
            if (result > 65535.0f) result = 65535.0f;
            output[ry * width + rx] = (uint16_t)(result + 0.5f);
        }
    }
}


/* ================================================================== */
/*  TECHNIQUE 6: VST + Wiener + Guided Filter                          */
/*                                                                      */
/*  Phase 1: Per-pixel Wiener in Anscombe domain (max NR, blurs edges). */
/*  Phase 2: Guided filter using center VST frame as guide.             */
/*    - Flat regions (low guide var): output ≈ Wiener (max NR)          */
/*    - Edge regions (high guide var): output ≈ center (preserve edge)  */
/*    - One parameter: ε = noise variance (1.0 in VST domain)          */
/*    - O(1) per pixel with box filter / same-color windowing           */
/*                                                                      */
/*  The guided filter formula (He et al. 2013):                         */
/*    a_k = cov(guide, input)_k / (var(guide)_k + ε)                   */
/*    b_k = mean(input)_k - a_k * mean(guide)_k                        */
/*    output_i = mean(a_k) * guide_i + mean(b_k)                        */
/*                                                                      */
/*  When var(guide) >> ε: a→1, output≈guide (edge preserved).          */
/*  When var(guide) << ε: a→0, output≈mean(input) (max NR).            */
/* ================================================================== */

/* Guided filter radius in same-color pixels.
 * r=4 → 9x9 same-color window (18x18 raw pixels). */
#define GF_RADIUS 4

static void vst_guided_core(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height,
    float noise_sigma, float epsilon)
{
    size_t npix = (size_t)width * height;

    float *z_guide  = (float *)malloc(npix * sizeof(float));
    float *z_input  = (float *)malloc(npix * sizeof(float));
    float *z_output = (float *)malloc(npix * sizeof(float));
    if (!z_guide || !z_input || !z_output) {
        memcpy(output, frames[center_idx], npix * sizeof(uint16_t));
        free(z_guide); free(z_input); free(z_output);
        return;
    }

    /* ---- Phase 1: VST+Wiener → z_input (denoised), z_guide (center) ---- */

    for (int ry = 0; ry < height; ry++) {
        for (int rx = 0; rx < width; rx++) {
            size_t idx = (size_t)ry * width + rx;
            float center_val = (float)frames[center_idx][idx];
            float z_cv = anscombe_forward(center_val, noise_sigma);
            z_guide[idx] = z_cv;

            float raw_samples[MAX_SAMPLES];
            int n = collect_samples(rx, ry, frames, flows_x, flows_y,
                                    num_frames, center_idx, width, height,
                                    raw_samples, NULL);
            if (n <= 1) {
                z_input[idx] = z_cv;
                continue;
            }

            /* Forward Anscombe + 3σ outlier rejection */
            float z_samples[MAX_SAMPLES];
            int valid = 0;
            for (int i = 0; i < n; i++) {
                float z = anscombe_forward(raw_samples[i], noise_sigma);
                if (fabsf(z - z_cv) <= 3.0f)
                    z_samples[valid++] = z;
            }
            if (valid <= 1) {
                z_input[idx] = z_cv;
                continue;
            }

            /* Wiener shrinkage */
            float z_sum = 0, z_sum2 = 0;
            for (int i = 0; i < valid; i++) {
                z_sum  += z_samples[i];
                z_sum2 += z_samples[i] * z_samples[i];
            }
            float z_mean = z_sum / (float)valid;
            float z_var  = z_sum2 / (float)valid - z_mean * z_mean;
            float signal_var = fmaxf(z_var - 1.0f, 0.0f);
            float wiener_gain = (z_var > 0.01f) ? signal_var / z_var : 0.0f;
            z_input[idx] = z_mean + wiener_gain * (z_cv - z_mean);
        }
    }

    /* ---- Phase 2: Guided filter per Bayer component ----
     *
     * For each of 4 Bayer components, run the guided filter on the
     * sub-grid (stride 2). The guide is z_guide (noisy center, has edges),
     * the input is z_input (Wiener output, smooth but edge-blurred).
     *
     * ε = 1.0 in VST domain (noise variance ≈ 1.0). This is the
     * theoretically correct value — no tuning needed.
     *
     * Using direct local computation (no integral images for simplicity).
     */
    for (int comp = 0; comp < 4; comp++) {
        int y_off = comp / 2;
        int x_off = comp % 2;
        int sc_w = width / 2;
        int sc_h = height / 2;

        /* Allocate per-component a and b coefficient maps */
        float *a_map = (float *)calloc((size_t)sc_w * sc_h, sizeof(float));
        float *b_map = (float *)calloc((size_t)sc_w * sc_h, sizeof(float));
        if (!a_map || !b_map) {
            free(a_map); free(b_map);
            continue;
        }

        /* Pass 1: compute a_k, b_k for each window */
        for (int sy = 0; sy < sc_h; sy++) {
            for (int sx = 0; sx < sc_w; sx++) {
                /* Gather guide and input values in (2r+1)² window */
                float sum_g = 0, sum_p = 0, sum_gg = 0, sum_gp = 0;
                int count = 0;

                int y0 = sy - GF_RADIUS; if (y0 < 0) y0 = 0;
                int y1 = sy + GF_RADIUS; if (y1 >= sc_h) y1 = sc_h - 1;
                int x0 = sx - GF_RADIUS; if (x0 < 0) x0 = 0;
                int x1 = sx + GF_RADIUS; if (x1 >= sc_w) x1 = sc_w - 1;

                for (int wy = y0; wy <= y1; wy++) {
                    int ry = wy * 2 + y_off;
                    for (int wx = x0; wx <= x1; wx++) {
                        int rx = wx * 2 + x_off;
                        size_t idx = (size_t)ry * width + rx;
                        float g = z_guide[idx];
                        float p = z_input[idx];
                        sum_g  += g;
                        sum_p  += p;
                        sum_gg += g * g;
                        sum_gp += g * p;
                        count++;
                    }
                }

                float inv_n = 1.0f / (float)count;
                float mean_g = sum_g * inv_n;
                float mean_p = sum_p * inv_n;
                float var_g  = sum_gg * inv_n - mean_g * mean_g;
                float cov_gp = sum_gp * inv_n - mean_g * mean_p;

                float a = cov_gp / (var_g + epsilon);
                float b = mean_p - a * mean_g;

                a_map[sy * sc_w + sx] = a;
                b_map[sy * sc_w + sx] = b;
            }
        }

        /* Pass 2: average a and b over each window, then output */
        for (int sy = 0; sy < sc_h; sy++) {
            for (int sx = 0; sx < sc_w; sx++) {
                float sum_a = 0, sum_b = 0;
                int count = 0;

                int y0 = sy - GF_RADIUS; if (y0 < 0) y0 = 0;
                int y1 = sy + GF_RADIUS; if (y1 >= sc_h) y1 = sc_h - 1;
                int x0 = sx - GF_RADIUS; if (x0 < 0) x0 = 0;
                int x1 = sx + GF_RADIUS; if (x1 >= sc_w) x1 = sc_w - 1;

                for (int wy = y0; wy <= y1; wy++) {
                    for (int wx = x0; wx <= x1; wx++) {
                        sum_a += a_map[wy * sc_w + wx];
                        sum_b += b_map[wy * sc_w + wx];
                        count++;
                    }
                }

                float mean_a = sum_a / (float)count;
                float mean_b = sum_b / (float)count;

                int ry = sy * 2 + y_off;
                int rx = sx * 2 + x_off;
                size_t idx = (size_t)ry * width + rx;

                z_output[idx] = mean_a * z_guide[idx] + mean_b;
            }
        }

        free(a_map);
        free(b_map);
    }

    /* ---- Phase 3: Inverse Anscombe → uint16 output ---- */
    for (size_t i = 0; i < npix; i++) {
        float result = anscombe_inverse(z_output[i], noise_sigma);
        if (result < 0.0f) result = 0.0f;
        if (result > 65535.0f) result = 65535.0f;
        output[i] = (uint16_t)(result + 0.5f);
    }

    free(z_guide);
    free(z_input);
    free(z_output);
}


/* ================================================================== */
/*  TECHNIQUE 7: VST + Temporal Bilateral                               */
/*                                                                      */
/*  Bilateral range kernel in Anscombe domain.                          */
/*  In VST domain, noise σ ≈ 1.0, so the range kernel bandwidth        */
/*  h = 1.0 is analytically determined — no tuning.                     */
/*                                                                      */
/*  Weight for each temporal neighbor:                                   */
/*    w_i = exp(-|z_center - z_warped_i|² / (2 * h²))                  */
/*  Output = Σ(w_i * z_i) / Σ(w_i)                                     */
/*                                                                      */
/*  Edge preservation: large z-difference at edges → low weight →       */
/*  center frame dominates → edge preserved.                            */
/*  Flat regions: small z-difference → high weight → temporal average   */
/*  → maximum NR.                                                       */
/*                                                                      */
/*  Per-pixel, no patches. Fast. One analytically determined parameter. */
/* ================================================================== */

static void vst_bilateral_core(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height,
    float noise_sigma, float h)
{
    /* -1/(2h²) precomputed for range kernel */
    const float neg_inv_2h2 = -1.0f / (2.0f * h * h);
    const float z_reject = 3.0f;        /* Hard rejection in VST units */
    const float flow_sigma2 = 2.0f * 4.0f; /* Flow attenuation: 2*σ², σ=2px */
    int green_w = width / 2;

    for (int ry = 0; ry < height; ry++) {
        for (int rx = 0; rx < width; rx++) {
            size_t idx = (size_t)ry * width + rx;
            float center_val = (float)frames[center_idx][idx];
            float z_center = anscombe_forward(center_val, noise_sigma);

            /* Accumulate neighbor contributions with per-sample flow info */
            float nb_w_sum = 0.0f;
            float nb_wz_sum = 0.0f;
            int n_neighbors = 0;
            float max_flow = 0.0f;

            for (int f = 0; f < num_frames; f++) {
                if (f == center_idx) continue;
                if (!flows_x[f] || !flows_y[f]) continue;

                float fdx, fdy;
                flow_lookup(flows_x[f], flows_y[f], rx, ry, green_w, &fdx, &fdy);

                float v = bilinear_warp_pixel(rx, ry, fdx, fdy,
                                              frames[f], width, height);
                if (v < 0.0f) continue;

                float z = anscombe_forward(v, noise_sigma);
                float diff = z - z_center;

                /* 1. Hard rejection: skip outliers beyond 3σ in VST domain */
                if (fabsf(diff) > z_reject) continue;

                /* 2. Bilateral range kernel */
                float w = expf(diff * diff * neg_inv_2h2);

                /* 3. Flow-dependent attenuation: high flow → less trust */
                float flow_mag2 = fdx * fdx + fdy * fdy;
                w *= expf(-flow_mag2 / flow_sigma2);

                float fm = sqrtf(flow_mag2);
                if (fm > max_flow) max_flow = fm;

                nb_w_sum  += w;
                nb_wz_sum += w * z;
                n_neighbors++;
            }

            if (n_neighbors == 0) {
                output[idx] = (uint16_t)center_val;
                continue;
            }

            /* 4. Adaptive center weight floor:
             *    Low flow (static): center gets at least 30% → more averaging
             *    High flow (≥3px):  center gets at least 60% → less ghosting */
            float center_floor = 0.3f + 0.3f * fminf(max_flow / 3.0f, 1.0f);
            float center_w = 1.0f; /* bilateral self-weight: exp(0) = 1 */

            /* If neighbors would outweigh center beyond the floor, scale them down */
            float center_frac = center_w / (center_w + nb_w_sum);
            if (center_frac < center_floor && nb_w_sum > 0.0f) {
                float scale = center_w * (1.0f - center_floor)
                            / (center_floor * nb_w_sum);
                nb_w_sum  *= scale;
                nb_wz_sum *= scale;
            }

            float w_sum  = center_w + nb_w_sum;
            float wz_sum = center_w * z_center + nb_wz_sum;
            float z_est  = wz_sum / w_sum;

            /* Inverse Anscombe */
            float result = anscombe_inverse(z_est, noise_sigma);
            if (result < 0.0f) result = 0.0f;
            if (result > 65535.0f) result = 65535.0f;
            output[idx] = (uint16_t)(result + 0.5f);
        }
    }
}


/* ================================================================== */
/*  Public wrappers with specific parameter values                      */
/* ================================================================== */

/* Guided Filter variants */
void technique_vst_guided(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    vst_guided_core(output, frames, flows_x, flows_y,
                    num_frames, center_idx, width, height,
                    noise_sigma, 1.0f);
}

void technique_vst_guided_e01(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    vst_guided_core(output, frames, flows_x, flows_y,
                    num_frames, center_idx, width, height,
                    noise_sigma, 0.1f);
}

void technique_vst_guided_e03(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    vst_guided_core(output, frames, flows_x, flows_y,
                    num_frames, center_idx, width, height,
                    noise_sigma, 0.3f);
}

/* Bilateral variants */
void technique_vst_bilateral(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    vst_bilateral_core(output, frames, flows_x, flows_y,
                       num_frames, center_idx, width, height,
                       noise_sigma, 1.0f);
}

void technique_vst_bilateral_h08(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    vst_bilateral_core(output, frames, flows_x, flows_y,
                       num_frames, center_idx, width, height,
                       noise_sigma, 0.8f);
}

void technique_vst_bilateral_h12(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx, int width, int height, float noise_sigma)
{
    vst_bilateral_core(output, frames, flows_x, flows_y,
                       num_frames, center_idx, width, height,
                       noise_sigma, 1.2f);
}
