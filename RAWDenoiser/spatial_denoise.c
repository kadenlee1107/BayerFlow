#include "../include/spatial_denoise.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WAVELET_LEVELS 3

/* Extract one Bayer sub-channel into a quarter-res buffer.
 * component: 0=R (0,0), 1=Gr (0,1), 2=Gb (1,0), 3=B (1,1) */
static void extract_subchannel(const uint16_t *bayer, int width, int height,
                               float *out, int comp) {
    int dy = (comp >> 1) & 1;  /* row offset: 0 for R/Gr, 1 for Gb/B */
    int dx = comp & 1;         /* col offset: 0 for R/Gb, 1 for Gr/B */
    int sw = width / 2;
    int sh = height / 2;

    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < sw; x++) {
            out[y * sw + x] = (float)bayer[(y * 2 + dy) * width + (x * 2 + dx)];
        }
    }
}

/* Write sub-channel back into Bayer frame */
static void insert_subchannel(uint16_t *bayer, int width, int height,
                              const float *in, int comp) {
    int dy = (comp >> 1) & 1;
    int dx = comp & 1;
    int sw = width / 2;
    int sh = height / 2;

    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < sw; x++) {
            float v = in[y * sw + x];
            int iv = (int)(v + 0.5f);
            if (iv < 0) iv = 0;
            if (iv > 65535) iv = 65535;
            bayer[(y * 2 + dy) * width + (x * 2 + dx)] = (uint16_t)iv;
        }
    }
}

/* In-place Haar wavelet forward transform (one level).
 * Operates on a w x h region within a buffer of stride `stride`.
 * After transform:
 *   top-left  (w/2 x h/2) = LL (approximation)
 *   top-right (w/2 x h/2) = LH (horizontal detail)
 *   bot-left  (w/2 x h/2) = HL (vertical detail)
 *   bot-right (w/2 x h/2) = HH (diagonal detail)
 */
static void haar_forward(float *buf, int w, int h, int stride, float *tmp) {
    int hw = w / 2;
    int hh = h / 2;

    /* Row transform */
    for (int y = 0; y < h; y++) {
        float *row = buf + y * stride;
        for (int x = 0; x < hw; x++) {
            float a = row[x * 2];
            float b = row[x * 2 + 1];
            tmp[x]      = (a + b) * 0.5f;
            tmp[hw + x]  = (a - b) * 0.5f;
        }
        memcpy(row, tmp, w * sizeof(float));
    }

    /* Column transform */
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < hh; y++) {
            float a = buf[y * 2 * stride + x];
            float b = buf[(y * 2 + 1) * stride + x];
            tmp[y]      = (a + b) * 0.5f;
            tmp[hh + y]  = (a - b) * 0.5f;
        }
        for (int y = 0; y < hh; y++) {
            buf[y * stride + x] = tmp[y];
            buf[(y + hh) * stride + x] = tmp[hh + y];
        }
    }
}

/* In-place Haar wavelet inverse transform (one level) */
static void haar_inverse(float *buf, int w, int h, int stride, float *tmp) {
    int hw = w / 2;
    int hh = h / 2;

    /* Column inverse */
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < hh; y++) {
            float s = buf[y * stride + x];
            float d = buf[(y + hh) * stride + x];
            tmp[y * 2]     = s + d;
            tmp[y * 2 + 1] = s - d;
        }
        for (int y = 0; y < h; y++) {
            buf[y * stride + x] = tmp[y];
        }
    }

    /* Row inverse */
    for (int y = 0; y < h; y++) {
        float *row = buf + y * stride;
        for (int x = 0; x < hw; x++) {
            float s = row[x];
            float d = row[hw + x];
            tmp[x * 2]     = s + d;
            tmp[x * 2 + 1] = s - d;
        }
        memcpy(row, tmp, w * sizeof(float));
    }
}

/* Soft thresholding: sign(x) * max(|x| - T, 0) */
static void soft_threshold(float *coeff, int count, float T) {
    for (int i = 0; i < count; i++) {
        float v = coeff[i];
        if (v > T) coeff[i] = v - T;
        else if (v < -T) coeff[i] = v + T;
        else coeff[i] = 0.0f;
    }
}

/* Estimate sigma of detail coefficients using MAD (median absolute deviation).
 * For efficiency, use a partial sort / histogram approach. */
static float estimate_sigma_mad(const float *coeff, int count) {
    if (count < 2) return 0.0f;

    /* Compute absolute values */
    float *abs_vals = (float *)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) {
        abs_vals[i] = fabsf(coeff[i]);
    }

    /* Find median using histogram (fast for wavelet coefficients).
     * Most values cluster near zero, so bucket into 1024 bins. */
    float max_val = 0;
    for (int i = 0; i < count; i++) {
        if (abs_vals[i] > max_val) max_val = abs_vals[i];
    }
    if (max_val == 0) { free(abs_vals); return 0.0f; }

    #define NBINS 4096
    int bins[NBINS];
    memset(bins, 0, sizeof(bins));
    float bin_scale = (NBINS - 1) / max_val;

    for (int i = 0; i < count; i++) {
        int b = (int)(abs_vals[i] * bin_scale);
        if (b >= NBINS) b = NBINS - 1;
        bins[b]++;
    }

    /* Find median bin */
    int target = count / 2;
    int cumulative = 0;
    float median = 0;
    for (int b = 0; b < NBINS; b++) {
        cumulative += bins[b];
        if (cumulative > target) {
            median = (b + 0.5f) / bin_scale;
            break;
        }
    }
    #undef NBINS

    free(abs_vals);

    /* MAD estimator: sigma = MAD / 0.6745 */
    return median / 0.6745f;
}

/* Denoise one sub-channel using Haar wavelet + BayesShrink */
static void denoise_subchannel(float *buf, int w, int h, float noise_sigma,
                               float strength) {
    int stride = w;
    int max_dim = (w > h) ? w : h;
    float *tmp = (float *)malloc(max_dim * sizeof(float));

    /* Forward wavelet transform (multi-level) */
    int cw = w, ch = h;
    for (int level = 0; level < WAVELET_LEVELS; level++) {
        if (cw < 4 || ch < 4) break;
        haar_forward(buf, cw, ch, stride, tmp);
        cw /= 2;
        ch /= 2;
    }

    /* Threshold detail coefficients at each level */
    int tw = w, th = h;
    for (int level = 0; level < WAVELET_LEVELS; level++) {
        if (tw < 4 || th < 4) break;
        int hw = tw / 2;
        int hh = th / 2;

        /* Three detail sub-bands: LH, HL, HH */
        for (int band = 0; band < 3; band++) {
            int bx, by;
            switch (band) {
                case 0: bx = hw; by = 0;  break;  /* LH */
                case 1: bx = 0;  by = hh; break;  /* HL */
                default: bx = hw; by = hh; break;  /* HH */
            }

            /* Collect detail coefficients for this sub-band */
            int band_size = hw * hh;
            float *band_data = (float *)malloc(band_size * sizeof(float));
            for (int y = 0; y < hh; y++) {
                memcpy(band_data + y * hw,
                       buf + (by + y) * stride + bx,
                       hw * sizeof(float));
            }

            /* BayesShrink threshold:
             * sigma_y = MAD estimate of this sub-band
             * sigma_x = sqrt(max(sigma_y^2 - sigma_n^2, 0))
             * T = sigma_n^2 / sigma_x
             * Multiply by strength factor */
            float sigma_y = estimate_sigma_mad(band_data, band_size);
            float sigma_n = noise_sigma;
            /* Scale noise sigma for wavelet level (each level halves, so
             * noise in wavelet domain is sigma/sqrt(2) per level due to
             * the 0.5 normalization in our Haar transform) */
            for (int l = 0; l < level; l++) sigma_n *= 0.5f;

            float sigma_x2 = sigma_y * sigma_y - sigma_n * sigma_n;
            float T;
            if (sigma_x2 <= 0) {
                T = sigma_y * 3.0f;  /* all noise, kill it */
            } else {
                T = (sigma_n * sigma_n) / sqrtf(sigma_x2);
            }
            T *= strength;

            /* Apply soft thresholding in-place on the buffer */
            for (int y = 0; y < hh; y++) {
                soft_threshold(buf + (by + y) * stride + bx, hw, T);
            }

            free(band_data);
        }

        tw /= 2;
        th /= 2;
    }

    /* Inverse wavelet transform */
    /* Reconstruct from deepest level back up */
    int levels_done = 0;
    {
        int test_w = w, test_h = h;
        for (int l = 0; l < WAVELET_LEVELS; l++) {
            if (test_w < 4 || test_h < 4) break;
            test_w /= 2;
            test_h /= 2;
            levels_done++;
        }
    }

    int rw = w, rh = h;
    for (int l = 0; l < levels_done; l++) {
        rw /= 2;
        rh /= 2;
    }
    for (int level = levels_done - 1; level >= 0; level--) {
        rw *= 2;
        rh *= 2;
        haar_inverse(buf, rw, rh, stride, tmp);
    }

    free(tmp);
}

/* Cycle-spinning (translation-invariant) wavelet denoise.
 * Denoise at 4 shifted positions and average to eliminate Haar block artifacts.
 * Shifts: (0,0), (1,0), (0,1), (1,1) */
static void denoise_subchannel_ti(float *result, const float *input,
                                   int w, int h, float noise_sigma,
                                   float strength) {
    int pixels = w * h;
    float *shifted = (float *)malloc(pixels * sizeof(float));
    float *accum = (float *)calloc(pixels, sizeof(float));

    /* 4 shifts for cycle spinning */
    int shifts[4][2] = {{0,0}, {1,0}, {0,1}, {1,1}};

    for (int s = 0; s < 4; s++) {
        int sx = shifts[s][0];
        int sy = shifts[s][1];

        /* Cropped dimensions (lose 1 pixel when shifting by 1) */
        int cw = w - sx;
        int ch = h - sy;
        /* Make even for Haar */
        cw &= ~1;
        ch &= ~1;

        /* Copy shifted region */
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                shifted[y * cw + x] = input[(y + sy) * w + (x + sx)];
            }
        }

        /* Denoise the shifted copy */
        denoise_subchannel(shifted, cw, ch, noise_sigma, strength);

        /* Accumulate back into original positions */
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                accum[(y + sy) * w + (x + sx)] += shifted[y * cw + x];
            }
        }
    }

    /* Build count map (how many shifts contributed to each pixel) */
    float *count = (float *)calloc(pixels, sizeof(float));
    for (int s = 0; s < 4; s++) {
        int sx = shifts[s][0];
        int sy = shifts[s][1];
        int cw = (w - sx) & ~1;
        int ch = (h - sy) & ~1;
        for (int y = 0; y < ch; y++) {
            for (int x = 0; x < cw; x++) {
                count[(y + sy) * w + (x + sx)] += 1.0f;
            }
        }
    }

    /* Average */
    for (int i = 0; i < pixels; i++) {
        if (count[i] > 0)
            result[i] = accum[i] / count[i];
        else
            result[i] = input[i];
    }

    free(shifted);
    free(accum);
    free(count);
}

void spatial_denoise_frame(uint16_t *bayer, int width, int height,
                           float noise_sigma, float strength) {
    int sw = width / 2;
    int sh = height / 2;
    int sub_pixels = sw * sh;

    float *sub_in = (float *)malloc(sub_pixels * sizeof(float));
    float *sub_out = (float *)malloc(sub_pixels * sizeof(float));
    if (!sub_in || !sub_out) { free(sub_in); free(sub_out); return; }

    /* Denoise each Bayer sub-channel independently with cycle spinning */
    for (int comp = 0; comp < 4; comp++) {
        extract_subchannel(bayer, width, height, sub_in, comp);
        denoise_subchannel_ti(sub_out, sub_in, sw, sh, noise_sigma, strength);
        insert_subchannel(bayer, width, height, sub_out, comp);
    }

    free(sub_in);
    free(sub_out);
}
