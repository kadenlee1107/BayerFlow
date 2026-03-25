#include <metal_stdlib>
using namespace metal;

// ============================================================
//  Hierarchical Block Matching Optical Flow (Metal GPU)
//
//  Replaces Apple Vision ANE optical flow for temporal denoising.
//  5-level Gaussian pyramid + coarse-to-fine block matching.
//
//  Performance target: <1ms per frame pair on M-series GPU.
//  Quality: ~0.5-1.0px accuracy, sufficient for bilateral denoiser
//  which has multi-hypothesis (M=4) and photometric rejection.
// ============================================================

struct BlockMatchParams {
    uint32_t width;          // current level width
    uint32_t height;         // current level height
    uint32_t block_size;     // 8
    uint32_t search_range;   // 16 at coarsest, 3 at finer levels
    uint32_t blocks_w;       // ceil(width / block_size)
    uint32_t blocks_h;       // ceil(height / block_size)
    uint32_t prev_blocks_w;  // blocks_w at coarser level (for upscale)
    uint32_t prev_blocks_h;
};

// ---- Kernel 1: Gaussian downsample 2x (build pyramid) ----
// Input: level N, Output: level N+1 at half resolution.
// 2x2 box filter (fast, matches the downsample in of_apple.m).
kernel void pyramid_downsample(
    device const uint16_t *src    [[buffer(0)]],
    device uint16_t       *dst    [[buffer(1)]],
    constant uint2        &src_dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint dw = src_dims.x / 2;
    uint dh = src_dims.y / 2;
    if (gid.x >= dw || gid.y >= dh) return;

    uint sx = gid.x * 2;
    uint sy = gid.y * 2;
    uint sw = src_dims.x;

    uint32_t v = (uint32_t)src[sy * sw + sx]
               + (uint32_t)src[sy * sw + sx + 1]
               + (uint32_t)src[(sy + 1) * sw + sx]
               + (uint32_t)src[(sy + 1) * sw + sx + 1];
    dst[gid.y * dw + gid.x] = (uint16_t)((v + 2) >> 2);
}

// ---- Kernel 2: Block matching at one pyramid level ----
// Each thread computes the best motion vector for one 8x8 block.
// At coarsest level: exhaustive search in search_range x search_range window.
// At finer levels: refine around upscaled coarser estimate (small search_range).
kernel void block_match(
    device const uint16_t *center   [[buffer(0)]],
    device const uint16_t *neighbor [[buffer(1)]],
    device const int2     *prev_mvs [[buffer(2)]],  // coarser level MVs (NULL at coarsest)
    device int2           *out_mvs  [[buffer(3)]],
    constant BlockMatchParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint bx = gid.x;
    uint by = gid.y;
    if (bx >= params.blocks_w || by >= params.blocks_h) return;

    uint bs = params.block_size;
    uint w = params.width;
    uint h = params.height;

    // Block top-left in pixel coordinates
    uint px = bx * bs;
    uint py = by * bs;

    // Initial estimate from coarser level (upscaled 2x)
    int2 init_mv = int2(0, 0);
    if (prev_mvs) {
        // Map this block to the coarser level block
        uint cbx = bx / 2;
        uint cby = by / 2;
        if (cbx >= params.prev_blocks_w) cbx = params.prev_blocks_w - 1;
        if (cby >= params.prev_blocks_h) cby = params.prev_blocks_h - 1;
        init_mv = prev_mvs[cby * params.prev_blocks_w + cbx] * 2;  // scale up
    }

    int sr = (int)params.search_range;
    int best_dx = init_mv.x;
    int best_dy = init_mv.y;
    uint best_sad = 0xFFFFFFFF;

    // Compute block extents (clamp to image bounds)
    uint bw = min(bs, w - px);
    uint bh = min(bs, h - py);

    for (int dy = -sr; dy <= sr; dy++) {
        for (int dx = -sr; dx <= sr; dx++) {
            int test_dx = init_mv.x + dx;
            int test_dy = init_mv.y + dy;

            // Check if neighbor block is in bounds
            int nx = (int)px + test_dx;
            int ny = (int)py + test_dy;
            if (nx < 0 || ny < 0 || nx + (int)bw > (int)w || ny + (int)bh > (int)h)
                continue;

            // Compute SAD
            uint sad = 0;
            for (uint y = 0; y < bh; y++) {
                for (uint x = 0; x < bw; x++) {
                    int cv = (int)center[(py + y) * w + (px + x)];
                    int nv = (int)neighbor[(ny + (int)y) * w + (nx + (int)x)];
                    sad += (uint)abs(cv - nv);
                }
            }

            if (sad < best_sad) {
                best_sad = sad;
                best_dx = test_dx;
                best_dy = test_dy;
            }
        }
    }

    out_mvs[by * params.blocks_w + bx] = int2(best_dx, best_dy);
}

// ---- Kernel 3: Interpolate block MVs to dense per-pixel flow ----
// Bilinear interpolation of block-level motion vectors to per-pixel.
// Output is float2 (dx, dy) per pixel.
kernel void interpolate_flow(
    device const int2  *block_mvs   [[buffer(0)]],
    device float       *flow_x      [[buffer(1)]],
    device float       *flow_y      [[buffer(2)]],
    constant uint4     &dims         [[buffer(3)]],  // x=width, y=height, z=blocks_w, w=block_size
    uint2 gid [[thread_position_in_grid]])
{
    uint w = dims.x;
    uint h = dims.y;
    if (gid.x >= w || gid.y >= h) return;

    uint blocks_w = dims.z;
    uint bs = dims.w;

    // Pixel center in block coordinates
    float fbx = ((float)gid.x + 0.5f) / (float)bs - 0.5f;
    float fby = ((float)gid.y + 0.5f) / (float)bs - 0.5f;

    int bx0 = (int)floor(fbx);
    int by0 = (int)floor(fby);
    float fx = fbx - (float)bx0;
    float fy = fby - (float)by0;

    int bx1 = bx0 + 1;
    int by1 = by0 + 1;

    // Clamp to block grid
    uint blocks_h = (h + bs - 1) / bs;
    if (bx0 < 0) { bx0 = 0; fx = 0; }
    if (by0 < 0) { by0 = 0; fy = 0; }
    if ((uint)bx1 >= blocks_w) bx1 = (int)blocks_w - 1;
    if ((uint)by1 >= blocks_h) by1 = (int)blocks_h - 1;

    // Bilinear interpolation of MVs
    int2 mv00 = block_mvs[(uint)by0 * blocks_w + (uint)bx0];
    int2 mv10 = block_mvs[(uint)by0 * blocks_w + (uint)bx1];
    int2 mv01 = block_mvs[(uint)by1 * blocks_w + (uint)bx0];
    int2 mv11 = block_mvs[(uint)by1 * blocks_w + (uint)bx1];

    float dx = (1.0f - fx) * (1.0f - fy) * (float)mv00.x
             +         fx  * (1.0f - fy) * (float)mv10.x
             + (1.0f - fx) *         fy  * (float)mv01.x
             +         fx  *         fy  * (float)mv11.x;

    float dy = (1.0f - fx) * (1.0f - fy) * (float)mv00.y
             +         fx  * (1.0f - fy) * (float)mv10.y
             + (1.0f - fx) *         fy  * (float)mv01.y
             +         fx  *         fy  * (float)mv11.y;

    uint idx = gid.y * w + gid.x;
    flow_x[idx] = dx;
    flow_y[idx] = dy;
}

// ---- Kernel 4: Warp/shift existing flow field ----
// For flow reuse: when window slides by 1 frame, warp existing
// flow using inter-frame flow to predict new flow.
// new_flow(x,y) = old_flow(x + inter_dx, y + inter_dy) + inter_flow(x,y)
kernel void warp_flow(
    device const float *old_flow_x   [[buffer(0)]],
    device const float *old_flow_y   [[buffer(1)]],
    device const float *inter_flow_x [[buffer(2)]],
    device const float *inter_flow_y [[buffer(3)]],
    device float       *new_flow_x   [[buffer(4)]],
    device float       *new_flow_y   [[buffer(5)]],
    constant uint2     &dims         [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = dims.x;
    uint h = dims.y;
    if (gid.x >= w || gid.y >= h) return;

    uint idx = gid.y * w + gid.x;

    // Where does this pixel come from in the old frame?
    float sx = (float)gid.x + inter_flow_x[idx];
    float sy = (float)gid.y + inter_flow_y[idx];

    // Bilinear sample old flow
    int x0 = (int)floor(sx);
    int y0 = (int)floor(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp
    if (x0 < 0) { x0 = 0; fx = 0; }
    if (y0 < 0) { y0 = 0; fy = 0; }
    if (x1 >= (int)w) x1 = (int)w - 1;
    if (y1 >= (int)h) y1 = (int)h - 1;

    float ofx_00 = old_flow_x[(uint)y0 * w + (uint)x0];
    float ofx_10 = old_flow_x[(uint)y0 * w + (uint)x1];
    float ofx_01 = old_flow_x[(uint)y1 * w + (uint)x0];
    float ofx_11 = old_flow_x[(uint)y1 * w + (uint)x1];

    float ofy_00 = old_flow_y[(uint)y0 * w + (uint)x0];
    float ofy_10 = old_flow_y[(uint)y0 * w + (uint)x1];
    float ofy_01 = old_flow_y[(uint)y1 * w + (uint)x0];
    float ofy_11 = old_flow_y[(uint)y1 * w + (uint)x1];

    float old_fx = (1-fx)*(1-fy)*ofx_00 + fx*(1-fy)*ofx_10
                 + (1-fx)*fy*ofx_01 + fx*fy*ofx_11;
    float old_fy = (1-fx)*(1-fy)*ofy_00 + fx*(1-fy)*ofy_10
                 + (1-fx)*fy*ofy_01 + fx*fy*ofy_11;

    // Compose: new flow = warped old flow + inter-frame flow
    new_flow_x[idx] = old_fx + inter_flow_x[idx];
    new_flow_y[idx] = old_fy + inter_flow_y[idx];
}
