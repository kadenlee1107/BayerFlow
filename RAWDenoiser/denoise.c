#include "../include/denoise.h"
#include "../include/prores_raw_enc.h"
#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/*  Global config — disabled by default so the pipeline is unchanged  */
/* ------------------------------------------------------------------ */
DenoiseConfig g_denoise_config = {
    .enabled       = 0,
    .threshold_mul = 3.0f,
    .noise_sigma   = 0.0f
};

void denoise_init(DenoiseConfig *cfg)
{
    cfg->enabled       = 0;
    cfg->threshold_mul = 3.0f;
    cfg->noise_sigma   = 0.0f;
}

/* ------------------------------------------------------------------ */
/*  Noise estimation from G1 / G2 channel difference                  */
/*                                                                    */
/*  Bayer RGGB layout:                                                */
/*      G1 = (even_row, odd_col)   — component 1                     */
/*      G2 = (odd_row,  even_col)  — component 2                     */
/*                                                                    */
/*  For each 2×2 cell the two greens capture nearly the same signal   */
/*  but with independent noise.  The variance of the difference is    */
/*  twice the single-channel noise variance:                          */
/*      var(G1 − G2) = 2 · σ²_noise                                  */
/*      σ_noise = sqrt( var(G1 − G2) / 2 )                           */
/* ------------------------------------------------------------------ */
float denoise_estimate_tile_noise(const uint16_t *bayer_data, int width,
                                  int tile_x, int tile_y,
                                  int tile_w, int tile_h)
{
    double sum      = 0.0;
    double sum_sq   = 0.0;
    int    count    = 0;

    /* Step through each 2×2 Bayer cell inside the tile. */
    for (int y = 0; y + 1 < tile_h; y += 2) {
        for (int x = 0; x + 1 < tile_w; x += 2) {
            int px = tile_x + x;
            int py = tile_y + y;

            /* G1: even row, odd col  →  (py, px+1) */
            uint16_t g1 = bayer_data[py * width + (px + 1)];
            /* G2: odd row, even col  →  (py+1, px) */
            uint16_t g2 = bayer_data[(py + 1) * width + px];

            double diff = (double)g1 - (double)g2;
            sum    += diff;
            sum_sq += diff * diff;
            count++;
        }
    }

    if (count < 2) return 0.0f;

    double mean     = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    if (variance < 0.0) variance = 0.0;     /* numerical guard */

    /* noise_var = var(G1−G2) / 2 */
    double noise_sigma = sqrt(variance / 2.0);
    return (float)noise_sigma;
}

/* ------------------------------------------------------------------ */
/*  Soft-threshold DCT coefficients (AC only)                         */
/*                                                                    */
/*  DCT coefficients are scaled by DCT_SCALE_DIVISOR.                 */
/*  Noise sigma is in 16-bit pixel units, while DCT runs in 12-bit.   */
/*  Convert σ to 12-bit before thresholding:                          */
/*      T = threshold_mul × (σ_pixel/16)                              */
/*  Soft rule:  coeff → sign(coeff) × max(0, |coeff| − T)            */
/*                                                                    */
/*  DC (index 0) is never touched — it carries the block average.     */
/* ------------------------------------------------------------------ */
void denoise_dct_block(int32_t *block, float noise_sigma, float threshold_mul)
{
    /* Noise sigma is in 16-bit pixel units. Convert to 12-bit (/16) then DCT domain (/4).
     * So noise in DCT coefficient domain ≈ σ_pixel / 64. */
    float T = threshold_mul * (noise_sigma / 64.0f);
    int   Ti = (int)(T + 0.5f);

    if (Ti <= 0) return;  /* nothing to do */

    /* AC coefficients only: indices 1..63 */
    for (int i = 1; i < 64; i++) {
        int v = block[i];
        int mag = (v < 0) ? -v : v;

        if (mag <= Ti) {
            block[i] = 0;
        } else {
            /* Soft shrinkage: reduce magnitude by Ti */
            block[i] = (v > 0) ? (int32_t)(mag - Ti)
                                : (int32_t)(-(mag - Ti));
        }
    }
}
