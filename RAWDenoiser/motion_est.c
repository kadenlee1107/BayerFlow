#include "../include/motion_est.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

void extract_green_channel(const uint16_t *bayer, int width, int height,
                           uint16_t *green_out) {
    int gw = width / 2;
    int gh = height / 2;

    for (int y = 0; y < gh; y++) {
        const uint16_t *row_even = bayer + (y * 2) * width;
        const uint16_t *row_odd  = bayer + (y * 2 + 1) * width;
        uint16_t *out_row = green_out + y * gw;
        int x = 0;
#ifdef __ARM_NEON__
        for (; x + 8 <= gw; x += 8) {
            /* Deinterleave load: even row [R,Gr,R,Gr,...] → val[0]=R, val[1]=Gr */
            uint16x8x2_t even = vld2q_u16(row_even + x * 2);
            /* Deinterleave load: odd row [Gb,B,Gb,B,...] → val[0]=Gb, val[1]=B */
            uint16x8x2_t odd  = vld2q_u16(row_odd + x * 2);
            /* Rounding halving add: (Gr + Gb + 1) >> 1 */
            vst1q_u16(out_row + x, vrhaddq_u16(even.val[1], odd.val[0]));
        }
#endif
        for (; x < gw; x++) {
            uint16_t gr = row_even[x * 2 + 1];
            uint16_t gb = row_odd[x * 2];
            out_row[x] = (gr + gb) / 2;
        }
    }
}

/* Downsample image by 2x (box filter) */
static uint16_t *downsample_2x(const uint16_t *src, int w, int h,
                                int *out_w, int *out_h) {
    int nw = w / 2;
    int nh = h / 2;
    uint16_t *dst = (uint16_t *)malloc(nw * nh * sizeof(uint16_t));
    if (!dst) return NULL;

    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            int sy = y * 2, sx = x * 2;
            uint32_t sum = src[sy * w + sx] + src[sy * w + sx + 1]
                         + src[(sy+1) * w + sx] + src[(sy+1) * w + sx + 1];
            dst[y * nw + x] = (uint16_t)(sum / 4);
        }
    }
    *out_w = nw;
    *out_h = nh;
    return dst;
}

/* Compute SAD for a block at (bx,by) in cur vs (bx+dx,by+dy) in ref */
static uint64_t block_sad(const uint16_t *ref, const uint16_t *cur,
                          int w, int h, int bs,
                          int bx, int by, int dx, int dy) {
    uint64_t sad = 0;
    for (int y = 0; y < bs; y++) {
        int cy = by + y;
        int ry = by + y + dy;
        if (cy < 0 || cy >= h || ry < 0 || ry >= h) {
            sad += (uint64_t)bs * 32768;
            continue;
        }
        for (int x = 0; x < bs; x++) {
            int cx = bx + x;
            int rx = bx + x + dx;
            if (cx < 0 || cx >= w || rx < 0 || rx >= w) {
                sad += 32768;
                continue;
            }
            int diff = (int)cur[cy * w + cx] - (int)ref[ry * w + rx];
            sad += (diff < 0) ? -diff : diff;
        }
    }
    return sad;
}

/* Compute SAD with half-pixel precision using bilinear interpolation.
 * hdx, hdy are in half-pixel units. */
static uint64_t block_sad_halfpel(const uint16_t *ref, const uint16_t *cur,
                                   int w, int h, int bs,
                                   int bx, int by, int hdx, int hdy) {
    int idx = hdx >> 1;   /* integer part */
    int idy = hdy >> 1;
    int fx = hdx & 1;     /* fractional flag: 0=integer, 1=half */
    int fy = hdy & 1;

    /* If both are integer, use fast integer SAD */
    if (!fx && !fy) {
        return block_sad(ref, cur, w, h, bs, bx, by, idx, idy);
    }

    uint64_t sad = 0;
    for (int y = 0; y < bs; y++) {
        int cy = by + y;
        int ry = by + y + idy;
        if (cy < 0 || cy >= h || ry < 0 || ry + fy >= h) {
            sad += (uint64_t)bs * 32768;
            continue;
        }
        for (int x = 0; x < bs; x++) {
            int cx = bx + x;
            int rx = bx + x + idx;
            if (cx < 0 || cx >= w || rx < 0 || rx + fx >= w) {
                sad += 32768;
                continue;
            }

            /* Bilinear interpolation of ref */
            uint32_t val;
            if (!fx && fy) {
                val = ((uint32_t)ref[ry * w + rx] + ref[(ry + 1) * w + rx] + 1) >> 1;
            } else if (fx && !fy) {
                val = ((uint32_t)ref[ry * w + rx] + ref[ry * w + rx + 1] + 1) >> 1;
            } else {
                val = ((uint32_t)ref[ry * w + rx] + ref[ry * w + rx + 1]
                     + ref[(ry + 1) * w + rx] + ref[(ry + 1) * w + rx + 1] + 2) >> 2;
            }

            int diff = (int)cur[cy * w + cx] - (int)val;
            sad += (diff < 0) ? -diff : diff;
        }
    }
    return sad;
}

/* Full search within a range around (init_dx, init_dy) */
static void search_block(const uint16_t *ref, const uint16_t *cur,
                         int w, int h, int bs,
                         int bx, int by,
                         int init_dx, int init_dy, int range,
                         int *best_dx, int *best_dy) {
    uint64_t best_sad = UINT64_MAX;
    *best_dx = init_dx;
    *best_dy = init_dy;

    for (int dy = init_dy - range; dy <= init_dy + range; dy++) {
        for (int dx = init_dx - range; dx <= init_dx + range; dx++) {
            uint64_t sad = block_sad(ref, cur, w, h, bs, bx, by, dx, dy);
            if (sad < best_sad) {
                best_sad = sad;
                *best_dx = dx;
                *best_dy = dy;
            }
        }
    }
}

MotionVector *motion_estimate(const uint16_t *ref_green, const uint16_t *cur_green,
                              int green_w, int green_h, int block_size) {
    /* Build 3-level pyramid */
    int w0 = green_w, h0 = green_h;
    int w1, h1, w2, h2;

    uint16_t *ref1 = downsample_2x(ref_green, w0, h0, &w1, &h1);
    uint16_t *cur1 = downsample_2x(cur_green, w0, h0, &w1, &h1);
    if (!ref1 || !cur1) { free(ref1); free(cur1); return NULL; }
    uint16_t *ref2 = downsample_2x(ref1, w1, h1, &w2, &h2);
    uint16_t *cur2 = downsample_2x(cur1, w1, h1, &w2, &h2);
    if (!ref2 || !cur2) { free(ref1); free(cur1); free(ref2); free(cur2); return NULL; }

    int grid_w = green_w / block_size;
    int grid_h = green_h / block_size;
    int num_blocks = grid_w * grid_h;

    MotionVector *mvs = (MotionVector *)calloc(num_blocks, sizeof(MotionVector));
    if (!mvs) goto cleanup;

    int bs2 = block_size / 4;  /* block size at level 2 */
    int bs1 = block_size / 2;  /* block size at level 1 */
    if (bs2 < 2) bs2 = 2;
    if (bs1 < 2) bs1 = 2;

    for (int gy = 0; gy < grid_h; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            int dx, dy;

            /* Level 2: coarse search, range +-8 at 1/4 res (= +-32 at full) */
            int bx2 = gx * bs2;
            int by2 = gy * bs2;
            search_block(ref2, cur2, w2, h2, bs2, bx2, by2, 0, 0, 8, &dx, &dy);

            /* Level 1: refine, range +-4 at 1/2 res */
            int bx1 = gx * bs1;
            int by1 = gy * bs1;
            search_block(ref1, cur1, w1, h1, bs1, bx1, by1, dx*2, dy*2, 4, &dx, &dy);

            /* Level 0: refine, range +-2 at full green res */
            int bx0 = gx * block_size;
            int by0 = gy * block_size;
            search_block(ref_green, cur_green, w0, h0, block_size,
                         bx0, by0, dx*2, dy*2, 2, &dx, &dy);

            /* Half-pixel refinement: 3x3 search around best integer position.
             * Convert to half-pixel units (multiply by 2), then test 9 candidates
             * at {-1,0,+1} offsets in each direction. */
            int hdx = dx * 2;
            int hdy = dy * 2;
            uint64_t best_sad = block_sad_halfpel(ref_green, cur_green, w0, h0,
                                                   block_size, bx0, by0, hdx, hdy);
            int best_hdx = hdx;
            int best_hdy = hdy;

            for (int oy = -1; oy <= 1; oy++) {
                for (int ox = -1; ox <= 1; ox++) {
                    if (ox == 0 && oy == 0) continue;
                    uint64_t sad = block_sad_halfpel(ref_green, cur_green, w0, h0,
                                                      block_size, bx0, by0,
                                                      hdx + ox, hdy + oy);
                    if (sad < best_sad) {
                        best_sad = sad;
                        best_hdx = hdx + ox;
                        best_hdy = hdy + oy;
                    }
                }
            }

            /* Store MV in half-green-pixel units (= raw Bayer pixel units) */
            mvs[gy * grid_w + gx].dx = (int16_t)best_hdx;
            mvs[gy * grid_w + gx].dy = (int16_t)best_hdy;
        }
    }

cleanup:
    free(ref1);
    free(cur1);
    free(ref2);
    free(cur2);
    return mvs;
}

void motion_vectors_free(MotionVector *mvs) {
    free(mvs);
}

void spatial_filter_green(uint16_t *green, int gw, int gh, float noise_sigma) {
    if (noise_sigma <= 0 || gw < 5 || gh < 5) return;

    /* 5×5 bilateral filter: spatial σ=1.5px, range σ=2×noise_sigma.
     * Preserves edges while smoothing noise for better OF accuracy. */
    float range_sq_inv = 1.0f / (4.0f * noise_sigma * noise_sigma);  /* 1/(2σ_range)² */

    /* Precompute exp LUT: 1024 entries for exp(-x), x ∈ [0, 10].
     * Beyond x=10, exp(-x) < 0.00005 — negligible weight. */
    #define EXP_LUT_SIZE 1024
    #define EXP_LUT_MAX  10.0f
    float exp_lut[EXP_LUT_SIZE];
    float lut_scale = (float)(EXP_LUT_SIZE - 1) / EXP_LUT_MAX;
    for (int i = 0; i < EXP_LUT_SIZE; i++)
        exp_lut[i] = expf(-(float)i / lut_scale);

    /* Precompute spatial Gaussian weights for 5×5 kernel (σ=1.5) */
    static const float sp[5][5] = {
        {0.0183f, 0.0821f, 0.1353f, 0.0821f, 0.0183f},
        {0.0821f, 0.3679f, 0.6065f, 0.3679f, 0.0821f},
        {0.1353f, 0.6065f, 1.0000f, 0.6065f, 0.1353f},
        {0.0821f, 0.3679f, 0.6065f, 0.3679f, 0.0821f},
        {0.0183f, 0.0821f, 0.1353f, 0.0821f, 0.0183f},
    };

    uint16_t *out = (uint16_t *)malloc(gw * gh * sizeof(uint16_t));
    if (!out) return;

    for (int y = 0; y < gh; y++) {
        for (int x = 0; x < gw; x++) {
            float center = (float)green[y * gw + x];
            float wsum = 0.0f;
            float vsum = 0.0f;

            int y0 = (y < 2) ? 0 : y - 2;
            int y1 = (y + 2 >= gh) ? gh - 1 : y + 2;
            int x0 = (x < 2) ? 0 : x - 2;
            int x1 = (x + 2 >= gw) ? gw - 1 : x + 2;

            for (int ky = y0; ky <= y1; ky++) {
                int sy = ky - y + 2;
                for (int kx = x0; kx <= x1; kx++) {
                    int sx = kx - x + 2;
                    float val = (float)green[ky * gw + kx];
                    float diff = val - center;
                    float arg = diff * diff * range_sq_inv;
                    float range_w;
                    if (arg < EXP_LUT_MAX) {
                        int idx = (int)(arg * lut_scale);
                        range_w = exp_lut[idx];
                    } else {
                        range_w = 0.0f;
                    }
                    float w = sp[sy][sx] * range_w;
                    wsum += w;
                    vsum += w * val;
                }
            }

            int result = (int)(vsum / wsum + 0.5f);
            out[y * gw + x] = (uint16_t)(result < 0 ? 0 : result > 65535 ? 65535 : result);
        }
    }

    memcpy(green, out, gw * gh * sizeof(uint16_t));
    free(out);
    #undef EXP_LUT_SIZE
    #undef EXP_LUT_MAX
}
