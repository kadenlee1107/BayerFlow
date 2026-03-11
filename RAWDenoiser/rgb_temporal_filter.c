/*
 * RGB Temporal Filter — VST + Bilateral for 3-channel RGB
 *
 * Adapts the proven Bayer VST+bilateral temporal denoiser for debayered RGB input.
 * Used by formats that only provide RGB (e.g., RED R3D via SDK).
 *
 * Algorithm per channel:
 *   1. Anscombe VST: f(x) = 2*sqrt(x + 3/8) — stabilizes Poisson noise
 *   2. Bilateral temporal averaging with flow-compensated warp
 *   3. Inverse Anscombe: f^-1(z) = (z/2)^2 - 3/8
 */

#include "../include/rgb_temporal_filter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ---- Config ---- */

void rgb_temporal_filter_init(RgbTemporalFilterConfig *cfg) {
    cfg->window_size = 15;
    cfg->strength    = 1.5f;
    cfg->noise_sigma = 0;
}

/* ---- Noise estimation ---- */

float rgb_temporal_filter_estimate_noise(const uint16_t *rgb_planar,
                                          int width, int height) {
    /* Use green channel (most accurate for noise estimation).
     * Laplacian-based MAD estimator (robust to edges). */
    const uint16_t *green = rgb_planar + (size_t)width * height; /* G plane */

    /* Compute Laplacian: L(x,y) = 4*G(x,y) - G(x-1,y) - G(x+1,y) - G(x,y-1) - G(x,y+1) */
    double sum_abs = 0;
    int count = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int c = green[y * width + x];
            int lap = 4 * c - green[(y-1)*width + x] - green[(y+1)*width + x]
                            - green[y*width + (x-1)] - green[y*width + (x+1)];
            sum_abs += abs(lap);
            count++;
        }
    }

    /* MAD of Laplacian → sigma estimate.
     * For Gaussian noise: sigma ≈ MAD * sqrt(pi/2) / (4 * sqrt(2))
     * ≈ MAD * 0.2215 */
    float mad = (float)(sum_abs / count);
    float sigma = mad * 0.2215f;

    fprintf(stderr, "rgb_temporal: noise estimate = %.1f (16-bit)\n", sigma);
    return sigma;
}

/* ---- Luma computation ---- */

void rgb_compute_luma(const uint16_t *rgb_planar, int width, int height,
                      float *luma_out) {
    size_t n = (size_t)width * height;
    const uint16_t *r = rgb_planar;
    const uint16_t *g = rgb_planar + n;
    const uint16_t *b = rgb_planar + 2 * n;

    for (size_t i = 0; i < n; i++) {
        luma_out[i] = 0.2126f * r[i] + 0.7152f * g[i] + 0.0722f * b[i];
    }
}

/* ---- Anscombe VST ---- */

static inline float anscombe_fwd(float x) {
    return 2.0f * sqrtf(fmaxf(x, 0.0f) + 0.375f);
}

static inline float anscombe_inv(float z) {
    float half_z = z * 0.5f;
    return half_z * half_z - 0.375f;
}

/* ---- Bilateral temporal filter (one channel) ---- */

static void filter_channel(
    uint16_t       *out,          /* output: w*h */
    const uint16_t *const *planes, /* input planes for this channel, per window frame */
    const float    **flows_x,
    const float    **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float h_param,                /* bilateral bandwidth in VST domain */
    float flow_sigma2)            /* flow confidence sigma² */
{
    size_t n = (size_t)width * height;
    const uint16_t *center = planes[center_idx];

    /* Pre-compute VST of center */
    float *z_center = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        z_center[i] = anscombe_fwd((float)center[i]);

    float *z_sum   = (float *)calloc(n, sizeof(float));
    float *w_sum   = (float *)calloc(n, sizeof(float));

    float h2 = h_param * h_param;
    float inv_2h2 = 1.0f / (2.0f * h2);
    float inv_2fs2 = 1.0f / (2.0f * flow_sigma2);

    /* Add center contribution */
    for (size_t i = 0; i < n; i++) {
        z_sum[i] = z_center[i];
        w_sum[i] = 1.0f;
    }

    /* Accumulate neighbor contributions */
    for (int f = 0; f < num_frames; f++) {
        if (f == center_idx) continue;
        if (!flows_x[f] || !flows_y[f]) continue;

        const uint16_t *nbr = planes[f];
        const float *fx = flows_x[f];
        const float *fy = flows_y[f];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t idx = (size_t)y * width + x;

                /* Warp neighbor position */
                float src_x = x + fx[idx];
                float src_y = y + fy[idx];

                /* Bounds check */
                if (src_x < 0 || src_x >= width - 1 ||
                    src_y < 0 || src_y >= height - 1)
                    continue;

                /* Bilinear interpolation */
                int sx = (int)src_x, sy = (int)src_y;
                float dx = src_x - sx, dy = src_y - sy;
                float w00 = (1-dx)*(1-dy), w10 = dx*(1-dy);
                float w01 = (1-dx)*dy,     w11 = dx*dy;

                float val = w00 * nbr[sy*width + sx]
                          + w10 * nbr[sy*width + sx+1]
                          + w01 * nbr[(sy+1)*width + sx]
                          + w11 * nbr[(sy+1)*width + sx+1];

                float z_nbr = anscombe_fwd(val);

                /* Photometric weight (bilateral) */
                float diff = z_nbr - z_center[idx];
                float w_photo = expf(-diff * diff * inv_2h2);

                /* Flow confidence weight */
                float flow_mag2 = fx[idx] * fx[idx] + fy[idx] * fy[idx];
                float w_flow = expf(-flow_mag2 * inv_2fs2);

                float w = w_photo * w_flow;
                z_sum[idx] += z_nbr * w;
                w_sum[idx] += w;
            }
        }
    }

    /* Inverse Anscombe and write output */
    for (size_t i = 0; i < n; i++) {
        float z = z_sum[i] / fmaxf(w_sum[i], 1e-6f);
        float val = anscombe_inv(z);
        if (val < 0) val = 0;
        if (val > 65535.0f) val = 65535.0f;
        out[i] = (uint16_t)(val + 0.5f);
    }

    free(z_center);
    free(z_sum);
    free(w_sum);
}

/* ---- Main entry point ---- */

void rgb_temporal_filter_frame(
    uint16_t       *output,
    const uint16_t **frames,
    const float    **flows_x,
    const float    **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    const RgbTemporalFilterConfig *cfg)
{
    size_t plane_size = (size_t)width * height;

    /* h parameter: in VST domain, h=1.0 is analytically optimal.
     * Scale by strength for user control. */
    float h_param = 1.0f * cfg->strength;
    float flow_sigma2 = 8.0f;  /* same as Bayer VST bilateral */

    /* Build per-channel plane pointers for each frame */
    const uint16_t **r_planes = (const uint16_t **)malloc(num_frames * sizeof(uint16_t *));
    const uint16_t **g_planes = (const uint16_t **)malloc(num_frames * sizeof(uint16_t *));
    const uint16_t **b_planes = (const uint16_t **)malloc(num_frames * sizeof(uint16_t *));

    for (int f = 0; f < num_frames; f++) {
        r_planes[f] = frames[f];                      /* R plane */
        g_planes[f] = frames[f] + plane_size;          /* G plane */
        b_planes[f] = frames[f] + 2 * plane_size;      /* B plane */
    }

    /* Filter each channel independently using shared flow field */
    uint16_t *out_r = output;
    uint16_t *out_g = output + plane_size;
    uint16_t *out_b = output + 2 * plane_size;

    filter_channel(out_r, r_planes, flows_x, flows_y,
                   num_frames, center_idx, width, height, h_param, flow_sigma2);
    filter_channel(out_g, g_planes, flows_x, flows_y,
                   num_frames, center_idx, width, height, h_param, flow_sigma2);
    filter_channel(out_b, b_planes, flows_x, flows_y,
                   num_frames, center_idx, width, height, h_param, flow_sigma2);

    free(r_planes);
    free(g_planes);
    free(b_planes);
}
