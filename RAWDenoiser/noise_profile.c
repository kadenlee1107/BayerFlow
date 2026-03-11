#include "../include/noise_profile.h"
#include "../include/frame_reader.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* -----------------------------------------------------------------------
 * noise_profile_from_patch
 *
 * Algorithm:
 *   1. Collect all Gr-Gb pairs within the patch → noise variance = var(Gr-Gb)/2
 *   2. Subdivide the patch into 16x16-cell blocks, compute mean & variance
 *      per block to build (signal, noise_var) scatter
 *   3. Fit noise_var = read_noise^2 + shot_gain * max(0, mean - black_level)
 *      via simple least-squares on the scatter
 * ----------------------------------------------------------------------- */

#define BLOCK_SIZE 16   /* must be even (covers 2x2 Bayer cells each step) */
#define MAX_BLOCKS 4096

void noise_profile_from_patch(const uint16_t *bayer,
                               int bayer_w, int bayer_h,
                               int px, int py, int pw, int ph,
                               CNoiseProfile *out)
{
    memset(out, 0, sizeof(*out));

    /* Clamp patch to frame */
    if (px < 0) { pw += px; px = 0; }
    if (py < 0) { ph += py; py = 0; }
    if (px + pw > bayer_w) pw = bayer_w - px;
    if (py + ph > bayer_h) ph = bayer_h - py;

    /* Align to 2x2 Bayer grid */
    if (px & 1) { px++; pw--; }
    if (py & 1) { py++; ph--; }

    if (pw < 32 || ph < 32) {
        /* Patch too small for reliable analysis */
        out->valid = 0;
        return;
    }

    /* --- Pass 1: overall Gr-Gb sigma (noise at mean brightness) --- */
    double sum_diff = 0, sum_diff_sq = 0;
    double sum_signal = 0;
    long count_pairs = 0;

    for (int y = py; y + 1 < py + ph; y += 2) {
        for (int x = px + 1; x + 1 < px + pw; x += 2) {
            /* RGGB layout: row y = R, Gr; row y+1 = Gb, B
             * Gr at (y, x), Gb at (y+1, x-1) */
            double gr = (double)bayer[y * bayer_w + x];
            double gb = (double)bayer[(y + 1) * bayer_w + (x - 1)];
            double diff = gr - gb;
            sum_diff    += diff;
            sum_diff_sq += diff * diff;
            sum_signal  += (gr + gb) * 0.5;
            count_pairs++;
        }
    }

    if (count_pairs < 16) {
        out->valid = 0;
        return;
    }

    double mean_diff = sum_diff / count_pairs;
    double var_diff  = (sum_diff_sq / count_pairs) - (mean_diff * mean_diff);
    if (var_diff < 0) var_diff = 0;

    out->sigma       = (float)sqrt(var_diff / 2.0);
    out->mean_signal = (float)(sum_signal / count_pairs);

    /* --- Pass 2: per-block scatter to fit noise model --- */
    /* Collect (mean, variance) per BLOCK_SIZE x BLOCK_SIZE block */
    double block_mean[MAX_BLOCKS];
    double block_var[MAX_BLOCKS];
    int    nblocks = 0;

    int step = BLOCK_SIZE;  /* pixels per block (must be even) */
    if (step & 1) step++;

    for (int by = py; by + step <= py + ph; by += step) {
        for (int bx = px; bx + step <= px + pw; bx += step) {
            if (nblocks >= MAX_BLOCKS) break;

            /* Use Gr channel only within block */
            double bsum = 0, bsum_sq = 0;
            int bcnt = 0;
            for (int y = by; y + 1 < by + step; y += 2) {
                for (int x = bx + 1; x < bx + step; x += 2) {
                    double v = (double)bayer[y * bayer_w + x];
                    bsum    += v;
                    bsum_sq += v * v;
                    bcnt++;
                }
            }
            if (bcnt < 4) continue;
            double bmean = bsum / bcnt;
            double bvar  = (bsum_sq / bcnt) - (bmean * bmean);
            if (bvar < 0) bvar = 0;
            block_mean[nblocks] = bmean;
            block_var[nblocks]  = bvar;
            nblocks++;
        }
    }

    if (nblocks < 4) {
        /* Not enough blocks for a fit — just use global sigma */
        out->black_level = 0;
        out->read_noise  = out->sigma;
        out->shot_gain   = 0;
        out->valid       = 1;
        return;
    }

    /* --- Estimate black level from 5th percentile of block means --- */
    /* Simple: sort block means and take lowest 5% */
    double sorted_means[MAX_BLOCKS];
    memcpy(sorted_means, block_mean, nblocks * sizeof(double));
    /* Insertion sort (nblocks usually < 200) */
    for (int i = 1; i < nblocks; i++) {
        double key = sorted_means[i];
        int j = i - 1;
        while (j >= 0 && sorted_means[j] > key) {
            sorted_means[j + 1] = sorted_means[j];
            j--;
        }
        sorted_means[j + 1] = key;
    }
    int pct5_idx = nblocks / 20;
    double bl = sorted_means[pct5_idx];

    /* --- Least-squares fit: var = a + b * max(0, mean - bl) --- */
    /* x_i = max(0, block_mean[i] - bl), y_i = block_var[i]
     * Minimize sum((y_i - a - b*x_i)^2) */
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (int i = 0; i < nblocks; i++) {
        double x = block_mean[i] - bl;
        if (x < 0) x = 0;
        double y = block_var[i];
        sx  += x;
        sy  += y;
        sxx += x * x;
        sxy += x * y;
        n++;
    }
    double denom = (double)n * sxx - sx * sx;
    double a, b;
    if (fabs(denom) < 1e-6) {
        /* Degenerate: all blocks at same brightness — can't separate components */
        a = sy / n;
        b = 0;
    } else {
        a = (sy * sxx - sx * sxy) / denom;   /* read_noise^2 */
        b = ((double)n * sxy - sx * sy) / denom;  /* shot_gain */
    }

    if (a < 0) a = 0;
    if (b < 0) b = 0;

    out->black_level = (float)bl;
    out->read_noise  = (float)sqrt(a);
    out->shot_gain   = (float)b;
    out->valid       = 1;

    /* Re-compute sigma at measured mean_signal using fitted model */
    double signal = out->mean_signal - bl;
    if (signal < 0) signal = 0;
    double model_var = a + b * signal;
    if (model_var > 0)
        out->sigma = (float)sqrt(model_var);
}

/* -----------------------------------------------------------------------
 * noise_profile_read_frame
 *
 * Opens the video file, decodes one frame (without denoising), returns
 * raw Bayer buffer. Caller must free().
 * ----------------------------------------------------------------------- */
uint16_t *noise_profile_read_frame(const char *path, int frame_index,
                                   int *out_width, int *out_height)
{
    FrameReader reader;
    if (frame_reader_open(&reader, path) != 0)
        return NULL;

    int width  = reader.width;
    int height = reader.height;
    size_t nbytes = (size_t)width * height * sizeof(uint16_t);

    uint16_t *buf = (uint16_t *)malloc(nbytes);
    if (!buf) {
        frame_reader_close(&reader);
        return NULL;
    }

    /* Clamp frame index */
    if (frame_index < 0) frame_index = 0;
    if (frame_index >= reader.frame_count) frame_index = reader.frame_count - 1;

    /* Skip frames to reach target index */
    for (int i = 0; i < frame_index; i++) {
        if (frame_reader_read_frame(&reader, buf) != 0) {
            free(buf);
            frame_reader_close(&reader);
            return NULL;
        }
    }

    int ret = frame_reader_read_frame(&reader, buf);
    frame_reader_close(&reader);

    if (ret != 0) {
        free(buf);
        return NULL;
    }

    *out_width  = width;
    *out_height = height;
    return buf;
}
