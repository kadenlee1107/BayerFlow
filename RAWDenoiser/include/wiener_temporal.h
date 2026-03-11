#ifndef WIENER_TEMPORAL_H
#define WIENER_TEMPORAL_H

#include <stdint.h>

/* Patch-Recursive Temporal Wiener Fusion.
 *
 * Replaces the bilateral temporal filter with frequency-domain optimal
 * denoising. For each Bayer sub-channel:
 *   1. Initialize estimate = center frame
 *   2. For each neighbor (nearest-first):
 *      a. Warp neighbor via optical flow (bilinear interpolation)
 *      b. Extract overlapping 8x8 patches
 *      c. Forward DCT, Wiener shrinkage fusion, inverse DCT
 *      d. Overlap-add with raised-cosine blend window
 *   3. Write denoised sub-channel to output
 *
 * The Wiener weights compound: each fused neighbor cleans the estimate,
 * improving subsequent fusions. */
void wiener_temporal_filter_frame(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x,
    const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma);

#endif /* WIENER_TEMPORAL_H */
