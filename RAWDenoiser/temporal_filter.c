#include "../include/temporal_filter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void temporal_filter_init(TemporalFilterConfig *cfg) {
    cfg->window_size = 9;
    cfg->strength = 1.0f;
    cfg->noise_sigma = 0.0f;
}

float temporal_filter_estimate_noise(const uint16_t *bayer, int width, int height) {
    double sum = 0, sum_sq = 0;
    int count = 0;

    /* Sample G1-G2 differences across the frame */
    for (int y = 0; y + 1 < height; y += 2) {
        for (int x = 0; x + 1 < width; x += 2) {
            uint16_t gr = bayer[y * width + x + 1];
            uint16_t gb = bayer[(y + 1) * width + x];
            double diff = (double)gr - (double)gb;
            sum += diff;
            sum_sq += diff * diff;
            count++;
        }
    }

    if (count < 2) return 0.0f;

    double mean = sum / count;
    double var = (sum_sq / count) - (mean * mean);
    if (var < 0) var = 0;

    return (float)sqrt(var / 2.0);
}

/* LUT for bilateral weights: w(diff) = exp(-diff^2 / (2h^2)) */
#define WEIGHT_LUT_SIZE 256

static void build_weight_lut(float *lut, float h) {
    float h2 = h * h;
    float max_diff = 3.0f * h;
    float step = max_diff / WEIGHT_LUT_SIZE;
    for (int i = 0; i < WEIGHT_LUT_SIZE; i++) {
        float diff = i * step;
        lut[i] = expf(-(diff * diff) / (2.0f * h2));
    }
}

/* CPU fallback callable from the Metal bridge (same signature as GPU path) */
void temporal_filter_frame_cpu(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x,
    const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma,
    const TemporalFilterTuning *tuning);

void temporal_filter_frame(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x,
    const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    const TemporalFilterConfig *cfg,
    const TemporalFilterTuning *tuning)
{
    /* Use tuning params or defaults */
    float chroma_boost_val   = tuning ? tuning->chroma_boost    : 1.0f;
    float dist_sigma_val     = tuning ? tuning->dist_sigma      : 1.5f;
    float flow_tightening_val = tuning ? tuning->flow_tightening : 3.0f;

    float sigma = cfg->noise_sigma;
    if (sigma <= 0) {
        sigma = temporal_filter_estimate_noise(frames[center_idx], width, height);
        if (sigma < 1.0f) sigma = 1.0f;
    }

    float h = sigma * cfg->strength;
    float base_reject = 3.0f * sigma;

    float chroma_boost = chroma_boost_val;
    float h_chroma = h * chroma_boost;

    /* Bilateral weight LUTs: separate for luma (Gr/Gb) and chroma (R/B) */
    float weight_lut_luma[WEIGHT_LUT_SIZE];
    float weight_lut_chroma[WEIGHT_LUT_SIZE];
    build_weight_lut(weight_lut_luma, h);
    build_weight_lut(weight_lut_chroma, h_chroma);
    /* Signal-dependent rejection threshold LUT: calibrated from S5II dark
     * frames + static scene temporal variance. Uses proper black level (6032)
     * and affine noise model: σ(v) = sqrt(read_noise² + shot_gain * max(0, v - BL)).
     * The old sqrt(1 + v/32768) over-estimated dark thresholds by ~1.5×.
     * Above ~17000 the dual-gain sensor reduces noise, so we cap with the
     * legacy model to avoid runaway thresholds at bright pixels. */
#define THRESH_LUT_BINS 256
    float thresh_lut[THRESH_LUT_BINS];
    float black_level = 6032.0f;
    float read_noise_frac = 0.55f;
    float read_noise_var = read_noise_frac * sigma * sigma;
    float shot_gain_val = 180.0f;
    for (int i = 0; i < THRESH_LUT_BINS; i++) {
        float v = ((float)i + 0.5f) * (65535.0f / THRESH_LUT_BINS);
        float signal = v > black_level ? v - black_level : 0.0f;
        float calibrated = 3.0f * sqrtf(read_noise_var + shot_gain_val * signal);
        float legacy = base_reject * sqrtf(1.0f + v / 32768.0f);
        thresh_lut[i] = calibrated < legacy ? calibrated : legacy;
    }

    int green_w = width / 2;
    const uint16_t *center = frames[center_idx];
    size_t frame_pixels = (size_t)width * height;

    /* Distance confidence LUT: downweight large flow displacements */
#define DIST_LUT_SIZE 128
    float dist_lut[DIST_LUT_SIZE];
    float dist_sigma = dist_sigma_val;
    float dist_max = 3.0f * dist_sigma;
    for (int i = 0; i < DIST_LUT_SIZE; i++) {
        float d = (float)i * dist_max / DIST_LUT_SIZE;
        dist_lut[i] = expf(-(d * d) / (2.0f * dist_sigma * dist_sigma));
    }
    float dist_lut_scale = (float)DIST_LUT_SIZE / dist_max;

    float *w_sum   = (float *)malloc(frame_pixels * sizeof(float));
    float *val_sum = (float *)malloc(frame_pixels * sizeof(float));
    if (!w_sum || !val_sum) {
        free(w_sum);
        free(val_sum);
        memcpy(output, center, frame_pixels * sizeof(uint16_t));
        return;
    }

    /* Center frame contributes with weight = 1 at every pixel */
    for (size_t i = 0; i < frame_pixels; i++) {
        w_sum[i]   = 1.0f;
        val_sum[i] = (float)center[i];
    }

    /* For each neighbor frame: look up per-pixel Vision OF flow, warp a
     * 3×3 same-color patch into alignment with center using bilinear
     * interpolation, compute NLM patch distance, accumulate weighted pixels.
     *
     * Patch-based matching (NLM-style) prevents ghosting: even if a single
     * misaligned pixel has matching brightness, its surrounding patch won't. */
    for (int f = 0; f < num_frames; f++) {
        if (f == center_idx || !flows_x[f]) continue;

        const float *fx  = flows_x[f];
        const float *fy  = flows_y[f];
        const uint16_t *ref = frames[f];

        for (int ry = 0; ry < height; ry++) {
            float *ws = w_sum   + ry * width;
            float *vs = val_sum + ry * width;

            /* Bayer row parity */
            int row_odd = ry & 1;

            for (int rx = 0; rx < width; rx++) {
                int col_odd = rx & 1;
                int is_chroma = (row_odd == col_odd);

                float h_param = is_chroma ? h_chroma : h;

                /* Flow lookup */
                int gx = rx >> 1;
                int gy = ry >> 1;
                float fdx = fx[gy * green_w + gx];
                float fdy = fy[gy * green_w + gx];

                /* Distance confidence */
                float mag = sqrtf(fdx * fdx + fdy * fdy);
                int di = (int)(mag * dist_lut_scale);
                float conf = (di >= DIST_LUT_SIZE) ? 0.0f : dist_lut[di];
                if (conf < 0.01f) continue;

                /* SSTO: Screen-Space Temporal Occlusion (matches Metal shader) */
                int green_h = height >> 1;
                float ssto_att = 1.0f;
                if (gx > 0 && gx + 1 < green_w && gy > 0 && gy + 1 < green_h) {
                    float fx_l = fx[gy * green_w + (gx - 1)];
                    float fx_r = fx[gy * green_w + (gx + 1)];
                    float fy_l = fy[gy * green_w + (gx - 1)];
                    float fy_r = fy[gy * green_w + (gx + 1)];
                    float fx_u = fx[(gy - 1) * green_w + gx];
                    float fx_d = fx[(gy + 1) * green_w + gx];
                    float fy_u = fy[(gy - 1) * green_w + gx];
                    float fy_d = fy[(gy + 1) * green_w + gx];
                    float gdx = (fx_r - fx_l) * (fx_r - fx_l) + (fy_r - fy_l) * (fy_r - fy_l);
                    float gdy = (fx_d - fx_u) * (fx_d - fx_u) + (fy_d - fy_u) * (fy_d - fy_u);
                    float flow_grad = sqrtf(gdx > gdy ? gdx : gdy) * 0.5f;

                    /* SSTO: directional occlusion from flow magnitude gradient */
                    if (flow_grad > 0.15f && mag > 0.3f) {
                        float mag_l = sqrtf(fx_l * fx_l + fy_l * fy_l);
                        float mag_r = sqrtf(fx_r * fx_r + fy_r * fy_r);
                        float mag_u = sqrtf(fx_u * fx_u + fy_u * fy_u);
                        float mag_d = sqrtf(fx_d * fx_d + fy_d * fy_d);
                        float gmx = mag_r - mag_l;
                        float gmy = mag_d - mag_u;
                        float gm_len = sqrtf(gmx * gmx + gmy * gmy);
                        float fd_len = sqrtf(fdx * fdx + fdy * fdy);
                        if (gm_len > 0.01f && fd_len > 0.01f) {
                            float cosine = (fdx * gmx + fdy * gmy) / (fd_len * gm_len);
                            float bnd_str = fminf(fmaxf((flow_grad - 0.15f) / 0.6f, 0.0f), 1.0f);
                            if (cosine < 0.0f) {
                                ssto_att = 1.0f - 0.95f * (-cosine) * bnd_str;
                            } else if (cosine < 0.3f) {
                                float perp = 1.0f - cosine / 0.3f;
                                ssto_att = 1.0f - 0.5f * perp * bnd_str;
                            }
                        }
                    }

                    /* Symmetric boundary attenuation */
                    if (flow_grad > 0.5f) {
                        float boundary_att = 1.0f - fminf(fmaxf((flow_grad - 0.5f) / 1.5f, 0.0f), 1.0f);
                        conf *= boundary_att;
                        if (conf < 0.01f) continue;
                    }
                }
                /* SSTO directional attenuation */
                conf *= ssto_att;
                if (conf < 0.01f) continue;

                /* Flow-adaptive tightening */
                float flow_tightening = flow_tightening_val;
                float flow_scale = 1.0f + mag * flow_tightening;

                /* Subpixel interpolation fractions (shared by entire patch) */
                int ix = (int)floorf(fdx);
                int iy = (int)floorf(fdy);
                float frac_x = fdx - (float)ix;
                float frac_y = fdy - (float)iy;

                /* Base warped position */
                int bx0_base = rx + ix * 2;
                int by0_base = ry + iy * 2;

                /* Bounds check for center pixel's bilinear region */
                if (bx0_base < 0 || bx0_base + 2 >= width ||
                    by0_base < 0 || by0_base + 2 >= height)
                    continue;

                /* --- Patch-based NLM matching (3×3 same-color patch) --- */
                float patch_dist_sq = 0.0f;
                float patch_count = 0.0f;
                float center_rval = 0.0f;

                for (int py = -1; py <= 1; py++) {
                    for (int px = -1; px <= 1; px++) {
                        /* Center patch pixel (same-color, stride 2) */
                        int cx = rx + px * 2;
                        int cy = ry + py * 2;
                        if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;

                        /* Warped neighbor patch pixel */
                        int nx0 = bx0_base + px * 2;
                        int ny0 = by0_base + py * 2;
                        int nx1 = nx0 + 2;
                        int ny1 = ny0 + 2;
                        if (nx0 < 0 || nx1 >= width || ny0 < 0 || ny1 >= height) continue;

                        float cv = (float)center[cy * width + cx];

                        float s00 = (float)ref[ny0 * width + nx0];
                        float s10 = (float)ref[ny0 * width + nx1];
                        float s01 = (float)ref[ny1 * width + nx0];
                        float s11 = (float)ref[ny1 * width + nx1];
                        float nv = (1.0f - frac_x) * (1.0f - frac_y) * s00
                                 +         frac_x  * (1.0f - frac_y) * s10
                                 + (1.0f - frac_x) *         frac_y  * s01
                                 +         frac_x  *         frac_y  * s11;

                        if (px == 0 && py == 0) center_rval = nv;

                        float d = cv - nv;
                        patch_dist_sq += d * d;
                        patch_count += 1.0f;
                    }
                }

                if (patch_count < 1.0f) continue;

                float mean_dist_sq = patch_dist_sq / patch_count;
                float mean_dist = sqrtf(mean_dist_sq);

                /* Signal-dependent rejection: floor at noise-limited patch dist */
                uint16_t cval = center[ry * width + rx];
                int tidx = cval >> 8;
                float base_thresh = thresh_lut[tidx];
                float noise_floor = base_thresh * 0.471f; /* sqrt(2)/3 */
                float eff_thresh = base_thresh / flow_scale;
                if (eff_thresh < noise_floor) eff_thresh = noise_floor;
                if (mean_dist > eff_thresh) continue;

                /* Signal-dependent NLM bandwidth: dark pixels get wider h */
                float cv_for_h = (float)center[ry * width + rx];
                float dark_h_boost = 1.0f + 0.5f * fminf(fmaxf(1.0f - cv_for_h / 8000.0f, 0.0f), 1.0f);
                float h_adj = h_param * dark_h_boost;
                float h_sq = h_adj * h_adj;
                float nlm_weight = expf(-mean_dist_sq / (2.0f * h_sq));

                float final_weight = nlm_weight * conf;

                /* Brightness-proportional weight attenuation for dark pixels */
                float raw_cv_cpu = (float)center[ry * width + rx];
                if (raw_cv_cpu < 10000.0f && center_rval > raw_cv_cpu) {
                    int pidx_cpu = (int)center[ry * width + rx] >> 8;
                    if (pidx_cpu > 255) pidx_cpu = 255;
                    float thr_cpu = thresh_lut[pidx_cpu];
                    float excess = (center_rval - raw_cv_cpu) / fmaxf(thr_cpu, 1.0f);
                    float bright_att = expf(-excess * excess * 2.0f);
                    final_weight *= bright_att;
                }

                vs[rx] += final_weight * center_rval;
                ws[rx] += final_weight;
            }
        }
    }

    /* Write output */
    for (size_t i = 0; i < frame_pixels; i++) {
        float result = val_sum[i] / w_sum[i];
        int r = (int)(result + 0.5f);
        if (r < 0) r = 0;
        if (r > 65535) r = 65535;
        output[i] = (uint16_t)r;
    }

    free(w_sum);
    free(val_sum);
}

/* CPU fallback with flat parameter signature (called from Metal bridge) */
void temporal_filter_frame_cpu(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x,
    const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma,
    const TemporalFilterTuning *tuning)
{
    TemporalFilterConfig cfg;
    temporal_filter_init(&cfg);
    cfg.strength    = strength;
    cfg.noise_sigma = noise_sigma;
    temporal_filter_frame(output, frames, flows_x, flows_y,
                          num_frames, center_idx, width, height, &cfg, tuning);
}
