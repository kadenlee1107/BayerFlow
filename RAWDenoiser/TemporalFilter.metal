#include <metal_stdlib>
using namespace metal;

struct FilterParams {
    uint32_t width;
    uint32_t height;
    float    lut_scale_luma;
    float    lut_scale_chroma;
    float    dist_lut_scale;
    float    flow_tightening;
    float    h_luma;        // bilateral bandwidth = noise_sigma * strength
    float    h_chroma;      // bilateral bandwidth for R/B channels
};

// Guided patch-based NLM temporal filter kernel.
// Compares 3×3 patches of same-color Bayer pixels (stride 2) for robust
// ghosting rejection. Uses a separate "guide" frame for patch distance
// computation — in bootstrap pass 2, the guide is the pass-1 denoised
// result, giving much cleaner patch matching and higher NLM weights.
//
// guide_frame: used for patch distance computation (can be denoised)
// center_frame: used for val_sum initialization (always raw, in Swift)
// The guide allows "oracle NLM" — clean matching with unbiased averaging.
kernel void temporal_filter_kernel(
    device const uint16_t *center_frame   [[buffer(0)]],
    device const uint16_t *neighbor_frame [[buffer(1)]],
    device const float    *flow_x         [[buffer(2)]],
    device const float    *flow_y         [[buffer(3)]],
    device float          *val_sum        [[buffer(4)]],
    device float          *w_sum          [[buffer(5)]],
    constant FilterParams &params         [[buffer(6)]],
    constant float        *weight_lut_luma   [[buffer(7)]],
    constant float        *thresh_lut        [[buffer(8)]],
    constant float        *dist_lut          [[buffer(9)]],
    constant float        *weight_lut_chroma [[buffer(10)]],
    device const uint16_t *guide_frame    [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint rx = gid.x;
    uint ry = gid.y;
    uint width  = params.width;
    uint height = params.height;

    if (rx >= width || ry >= height) return;

    // Determine if chroma (R or B) or luma (Gr or Gb) pixel.
    // RGGB: R=(even,even), Gr=(even,odd), Gb=(odd,even), B=(odd,odd)
    bool is_chroma = ((ry & 1) == (rx & 1));
    float h_param = is_chroma ? params.h_chroma : params.h_luma;

    // Flow is at green-pixel resolution; 4 raw pixels share one vector
    uint gx = rx >> 1;
    uint gy = ry >> 1;
    uint green_w = width >> 1;
    float fdx = flow_x[gy * green_w + gx];
    float fdy = flow_y[gy * green_w + gx];

    // Distance confidence: downweight large displacements
    float mag = sqrt(fdx * fdx + fdy * fdy);
    int di = int(mag * params.dist_lut_scale);
    float conf = (di >= 128) ? 0.0f : dist_lut[di];
    if (conf < 0.01f) return;

    // SSTO: Screen-Space Temporal Occlusion (inspired by SSAO in games).
    // At motion boundaries, the flow field has sharp discontinuities — like
    // depth edges in SSAO. Standard approach attenuates BOTH sides equally,
    // which kills temporal averaging on dark subject edges (good pixels rejected).
    //
    // SSTO computes the gradient of flow MAGNITUDE to find the boundary "normal"
    // (points from low-motion BG toward high-motion subject). Then:
    //   dot(flow_dir, boundary_normal) < 0  →  HALO SIDE
    //     Flow opposes gradient = warp crosses boundary into other region.
    //     Warped neighbor comes from bright BG → strong attenuation.
    //   dot(flow_dir, boundary_normal) >= 0  →  SAFE SIDE
    //     Flow aligns with gradient = warp stays within same region.
    //     Warped neighbor comes from same object → normal processing.
    uint green_h = height >> 1;
    float flow_grad = 0.0f;
    float ssto_att = 1.0f;  // directional attenuation (< 1.0 on halo side)
    if (gx > 0 && gx + 1 < green_w && gy > 0 && gy + 1 < green_h) {
        float fx_l = flow_x[gy * green_w + (gx - 1)];
        float fx_r = flow_x[gy * green_w + (gx + 1)];
        float fy_l = flow_y[gy * green_w + (gx - 1)];
        float fy_r = flow_y[gy * green_w + (gx + 1)];
        float fx_u = flow_x[(gy - 1) * green_w + gx];
        float fx_d = flow_x[(gy + 1) * green_w + gx];
        float fy_u = flow_y[(gy - 1) * green_w + gx];
        float fy_d = flow_y[(gy + 1) * green_w + gx];

        // Flow field gradient magnitude
        float gdx = (fx_r - fx_l) * (fx_r - fx_l) + (fy_r - fy_l) * (fy_r - fy_l);
        float gdy = (fx_d - fx_u) * (fx_d - fx_u) + (fy_d - fy_u) * (fy_d - fy_u);
        flow_grad = sqrt(max(gdx, gdy)) * 0.5f;

        // SSTO: compute boundary normal from flow magnitude gradient
        if (flow_grad > 0.15f && mag > 0.3f) {
            float mag_l = sqrt(fx_l * fx_l + fy_l * fy_l);
            float mag_r = sqrt(fx_r * fx_r + fy_r * fy_r);
            float mag_u = sqrt(fx_u * fx_u + fy_u * fy_u);
            float mag_d = sqrt(fx_d * fx_d + fy_d * fy_d);
            // Gradient of flow magnitude: points from BG (low) → subject (high)
            float2 grad_mag = float2(mag_r - mag_l, mag_d - mag_u);
            float2 flow_dir = float2(fdx, fdy);
            float gm_len = length(grad_mag);
            float fd_len = length(flow_dir);
            if (gm_len > 0.01f && fd_len > 0.01f) {
                float cosine = dot(flow_dir, grad_mag) / (fd_len * gm_len);
                // Steeper ramp: reach full strength at flow_grad=0.75 instead of 1.5
                float bnd_str = clamp((flow_grad - 0.15f) / 0.6f, 0.0f, 1.0f);
                if (cosine < 0.0f) {
                    // Halo side: flow opposes boundary normal → warp crosses
                    // into other region. Strong attenuation.
                    ssto_att = 1.0f - 0.95f * (-cosine) * bnd_str;
                } else if (cosine < 0.3f) {
                    // Perpendicular approach: flow nearly tangent to boundary.
                    // These neighbors can still carry bright bias from the edge.
                    // Moderate attenuation proportional to how perpendicular.
                    float perp = 1.0f - cosine / 0.3f;  // 1.0 at cosine=0, 0.0 at cosine=0.3
                    ssto_att = 1.0f - 0.5f * perp * bnd_str;
                }
            }
        }
    }

    // Symmetric boundary attenuation (applies to both sides of boundary)
    if (flow_grad > 0.5f) {
        float boundary_att = 1.0f - clamp((flow_grad - 0.5f) / 1.5f, 0.0f, 1.0f);
        conf *= boundary_att;
        if (conf < 0.01f) return;
    }
    // SSTO: additional directional attenuation on halo side
    conf *= ssto_att;
    if (conf < 0.01f) return;

    // Flow-adaptive tightening: high motion → stricter matching
    float flow_scale = 1.0f + mag * params.flow_tightening;

    // Subpixel bilinear interpolation fractions (shared by entire patch)
    int ix = int(floor(fdx));
    int iy = int(floor(fdy));
    float frac_x = fdx - float(ix);
    float frac_y = fdy - float(iy);

    // Base warped position for center pixel
    int bx0_base = int(rx) + ix * 2;
    int by0_base = int(ry) + iy * 2;

    // Check that the center pixel's bilinear interpolation region is in bounds
    if (bx0_base < 0 || bx0_base + 2 >= int(width) ||
        by0_base < 0 || by0_base + 2 >= int(height))
        return;

    // --- Patch-based NLM matching ---
    // Compare 5×5 patch of same-color Bayer pixels (stride 2 in raw grid).
    // 5×5 covers a 10×10 raw pixel area — large enough to reject occlusion
    // boundary ghosts where small patches of uniform background can fool 3×3.
    float patch_dist_sq = 0.0f;
    float patch_count = 0.0f;
    float center_rval = 0.0f;  // warped center pixel value (for accumulation later)

    for (int py = -2; py <= 2; py++) {
        for (int px = -2; px <= 2; px++) {
            // Center patch pixel (same-color, stride 2)
            int cx = int(rx) + px * 2;
            int cy = int(ry) + py * 2;
            if (cx < 0 || cx >= int(width) || cy < 0 || cy >= int(height)) continue;

            // Warped neighbor patch pixel (same offset from flow-aligned position)
            int nx0 = bx0_base + px * 2;
            int ny0 = by0_base + py * 2;
            int nx1 = nx0 + 2;
            int ny1 = ny0 + 2;
            if (nx0 < 0 || nx1 >= int(width) || ny0 < 0 || ny1 >= int(height)) continue;

            float cv = float(guide_frame[cy * width + cx]);

            // Bilinear interpolation of neighbor patch pixel
            float s00 = float(neighbor_frame[ny0 * width + nx0]);
            float s10 = float(neighbor_frame[ny0 * width + nx1]);
            float s01 = float(neighbor_frame[ny1 * width + nx0]);
            float s11 = float(neighbor_frame[ny1 * width + nx1]);
            float nv = (1.0f - frac_x) * (1.0f - frac_y) * s00
                     +         frac_x  * (1.0f - frac_y) * s10
                     + (1.0f - frac_x) *         frac_y  * s01
                     +         frac_x  *         frac_y  * s11;

            // Save center pixel's warped value for later accumulation
            if (px == 0 && py == 0) center_rval = nv;

            float d = cv - nv;
            patch_dist_sq += d * d;
            patch_count += 1.0f;
        }
    }

    if (patch_count < 1.0f) return;

    // Mean squared distance across the patch
    float mean_dist_sq = patch_dist_sq / patch_count;
    float mean_dist = sqrt(mean_dist_sq);

    // Signal-dependent rejection using mean patch distance (guide values).
    // The threshold is tightened by flow_scale during motion, BUT we floor it
    // at the noise-limited patch distance: for raw-vs-raw matching, well-aligned
    // patches have mean_dist ≈ sqrt(2) * σ_local = thresh_lut / 3 * sqrt(2).
    // Without this floor, flow_tightening pushes the threshold below the noise
    // floor at >1px flow, causing ALL dark neighbors to be rejected regardless
    // of alignment — killing temporal averaging in dark areas during motion.
    uint16_t cval = guide_frame[ry * width + rx];
    int tidx = cval >> 8;
    float base_thresh = thresh_lut[tidx];
    float noise_floor = base_thresh * 0.471f;  // sqrt(2)/3 ≈ expected noise-limited patch dist
    float effective_thresh = max(base_thresh / flow_scale, noise_floor);
    if (mean_dist > effective_thresh) return;

    // Asymmetric per-pixel luminance guard: prevent bright→dark leakage (halo)
    // while preserving dark→dark temporal averaging (denoising).
    // When a warped neighbor is BRIGHTER than a dark center pixel, tighten the
    // threshold — this is the halo failure mode. When the neighbor is the same
    // brightness or darker, use the full threshold to allow good denoising.
    float raw_cv = float(center_frame[ry * width + rx]);
    float pixel_diff = abs(center_rval - raw_cv);
    int pidx = int(center_frame[ry * width + rx]) >> 8;
    if (pidx > 255) pidx = 255;
    float threshold = thresh_lut[pidx];
    if (center_rval > raw_cv && raw_cv < 12000.0f) {
        // Neighbor is brighter than dark center: tighten to block halo.
        // Flat 0.4× up to 8000, then ramp to 1.0× at 12000.
        float tighten = (raw_cv < 8000.0f)
            ? 0.4f
            : 0.4f + 0.6f * clamp((raw_cv - 8000.0f) / 4000.0f, 0.0f, 1.0f);
        threshold *= tighten;
    }
    if (pixel_diff > threshold) return;

    // Signal-dependent NLM bandwidth:
    // Dark pixels get up to 1.5× wider h (genuine dark-on-dark matches need wider acceptance).
    // Bright pixels get up to 50% narrower h — they have high SNR and don't need heavy
    // denoising. Narrower h also rejects misaligned patches from flow errors on textureless
    // bright surfaces (skin), preventing the directional smear ("pulling") artifact.
    float cv_for_h = float(guide_frame[ry * width + rx]);
    float dark_h_boost = 1.0f + 0.5f * clamp(1.0f - cv_for_h / 8000.0f, 0.0f, 1.0f);
    float bright_h_reduce = 1.0f - 0.9f * clamp((cv_for_h - 10000.0f) / 20000.0f, 0.0f, 1.0f);
    float h_adj = h_param * dark_h_boost * bright_h_reduce;
    float h_sq = h_adj * h_adj;
    float nlm_weight = exp(-mean_dist_sq / (2.0f * h_sq));

    // Combine with distance confidence
    float final_weight = nlm_weight * conf;

    // Texture confidence for bright surfaces during motion:
    // On textureless bright skin, NLM can't distinguish aligned from misaligned
    // patches (all smooth patches match similarly) → causes directional smear
    // ("pulling") during motion. Measure local raw texture: if the center patch
    // contains only noise (no edges/texture for flow to track), reduce weight.
    if (cv_for_h > 10000.0f && mag > 0.3f) {
        // Sample 5 same-color raw pixels to estimate local texture
        float tsum = raw_cv, tsq = raw_cv * raw_cv, tn = 1.0f;
        int offsets[4] = {-2, 2, -2, 2};  // dx, dx, dy, dy
        for (int i = 0; i < 2; i++) {
            int tx = int(rx) + offsets[i];
            if (tx >= 0 && tx < int(width)) {
                float tv = float(center_frame[ry * width + tx]);
                tsum += tv; tsq += tv * tv; tn += 1.0f;
            }
        }
        for (int i = 2; i < 4; i++) {
            int ty = int(ry) + offsets[i];
            if (ty >= 0 && ty < int(height)) {
                float tv = float(center_frame[ty * width + rx]);
                tsum += tv; tsq += tv * tv; tn += 1.0f;
            }
        }
        float tvar = tsq / tn - (tsum / tn) * (tsum / tn);
        float tstd = sqrt(max(tvar, 0.0f));

        // Expected noise std at this signal level (thresh ≈ 3σ)
        float noise_std = thresh_lut[pidx] / 3.0f;

        // Texture ratio: tstd/noise_std ≈ 1.0 if textureless, > 2 if textured
        float tratio = tstd / max(noise_std, 1.0f);

        // tex_conf: 0.15 for pure noise (textureless), 1.0 for textured
        float tex_conf = clamp((tratio - 0.8f) / 1.5f, 0.1f, 1.0f);

        // Scale by motion: stronger penalty at higher flow
        float motion_scale = clamp(mag / 2.0f, 0.0f, 1.0f);
        tex_conf = 1.0f - (1.0f - tex_conf) * motion_scale;

        final_weight *= tex_conf;
    }

    // Brightness-proportional weight attenuation: when a warped neighbor is
    // brighter than a dark center, reduce its weight proportional to how much
    // of the per-pixel guard threshold it uses up. Many neighbors that individually
    // pass the guard but carry small bright biases collectively create halo —
    // this soft attenuation reduces their cumulative contribution.
    if (raw_cv < 10000.0f && center_rval > raw_cv) {
        float excess = (center_rval - raw_cv) / max(threshold, 1.0f);
        // Gaussian falloff: at 50% of threshold, weight × 0.61; at 80%, × 0.20
        float bright_att = exp(-excess * excess * 2.0f);
        final_weight *= bright_att;
    }

    uint idx = ry * width + rx;
    val_sum[idx] += final_weight * center_rval;
    w_sum[idx]   += final_weight;
}

kernel void temporal_filter_normalize(
    device const float    *val_sum      [[buffer(0)]],
    device const float    *w_sum        [[buffer(1)]],
    device uint16_t       *output       [[buffer(2)]],
    constant uint2        &dims         [[buffer(3)]],
    device const uint16_t *center_frame [[buffer(4)]],
    constant float        &center_weight [[buffer(5)]],
    constant float        &h_luma       [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y) return;

    uint idx = gid.y * dims.x + gid.x;
    float cv = float(center_frame[idx]);

    // Dark-pixel ghost suppression — ONLY at occlusion boundaries.
    // Ghost = bright data from neighbors leaking into dark background where
    // few neighbors pass NLM (occlusion → low w_sum).
    // Well-tracked dark pixels (hair, shadows) have many contributing neighbors
    // (high w_sum) and need full temporal averaging — DON'T boost those.
    float neighbor_sum = w_sum[idx] - center_weight;
    float neighbor_ratio = neighbor_sum / max(center_weight, 0.01f);
    // neighbor_ratio < 1: few neighbors → ghost-prone → boost
    // neighbor_ratio > 3: well-tracked → no boost
    float ghost_risk = 1.0f - clamp(neighbor_ratio / 3.0f, 0.0f, 1.0f);

    float dark_boost = 1.0f;
    if (cv < 7000.0f) {
        dark_boost = 1.0f + 2.0f * ghost_risk;  // up to 3x at ghost locations
    } else if (cv < 10000.0f) {
        float ramp = (10000.0f - cv) / 3000.0f;
        dark_boost = 1.0f + 2.0f * ghost_risk * ramp;
    }

    // Adjust val_sum and w_sum with the additional center weight
    float boost_delta = center_weight * (dark_boost - 1.0f);
    float result = (val_sum[idx] + boost_delta * cv) / (w_sum[idx] + boost_delta);

    // Symmetric brightness clamp: in dark regions, limit how much the temporal
    // average can deviate from the center pixel in EITHER direction.
    // Bright clamp prevents glow (bright background leaking into dark subject).
    // Dark clamp prevents the systematic darkening bias that causes the
    // "snap-back" artifact: during static periods, the asymmetric per-pixel
    // guard and brightness attenuation preferentially suppress bright neighbors,
    // creating a net darkening. When motion starts and fewer neighbors
    // contribute, this bias vanishes → visible brightness shift.
    // Making the clamp symmetric ensures the temporal filter's brightness
    // impact is bounded regardless of how many neighbors contribute.
    if (cv < 10000.0f) {
        float base_max = (cv < 5000.0f) ? h_luma * 1.0f : h_luma * 1.5f;
        result = clamp(result, cv - base_max, cv + base_max);
    }

    int r = int(result + 0.5f);
    if (r < 0) r = 0;
    if (r > 65535) r = 65535;
    output[idx] = uint16_t(r);
}


// ============================================================
//  VST + Temporal Bilateral (GPU)
//  4-pass pipeline with 3 research improvements:
//    1. Structural term (gradient comparison for anti-ghosting)
//    2. Self-guided reference (robust pre-estimate vs noisy center)
//    3. Multi-hypothesis sampling (M=4, handles subpixel flow errors)
// ============================================================

struct VSTBilateralParams {
    uint32_t width;
    uint32_t height;
    float    noise_sigma;   // global noise sigma
    float    h;             // bilateral bandwidth in VST domain (1.0)
    float    z_reject;      // hard rejection threshold (3.0 in VST units)
    float    flow_sigma2;   // 2*sigma_flow^2 for distance attenuation (8.0)
    float    sigma_g2;      // 2*sigma_g^2 for structural term (0.5)
    float    black_level;   // sensor black level (default 6032)
    float    shot_gain;     // shot noise gain (default 180)
    float    read_noise;    // read noise floor (default 616)
};

// Multi-hypothesis offsets (green-pixel units)
constant float2 VST_HYPS[4] = {
    float2(0.0f, 0.0f),
    float2(0.5f, 0.0f),
    float2(0.0f, 0.5f),
    float2(-0.5f, -0.5f)
};

// Forward Generalized Anscombe Transform
// bl=black_level, sg=shot_gain, rn=read_noise (all calibrated per-camera)
inline float vst_fwd(float v, float bl, float sg, float rn) {
    float rv = rn * rn;
    float sig = max(v - bl, 0.0f);
    float x = sig / sg + 0.375f + rv / (sg * sg);
    return 2.0f * sqrt(max(x, 0.0f));
}

// Inverse Generalized Anscombe Transform (Makitalo-Foi bias correction)
inline float vst_inv(float z, float bl, float sg, float rn) {
    float rv = rn * rn;
    float zc = max(z, 0.5f);
    float z2 = zc * zc;
    float val = sg * (z2 * 0.25f - 0.375f - rv / (sg * sg)) + bl;
    if (zc > 1.0f) val += sg / (4.0f * z2);
    return val;
}

// Bilinear warp of same-color Bayer pixel via optical flow.
// Flow in green-pixel units; same-color stride = 2 in raw coords.
inline float warp_bayer(uint rx, uint ry, float fdx, float fdy,
                        device const uint16_t *frame, uint w, uint h) {
    int ix = int(floor(fdx));
    int iy = int(floor(fdy));
    float fx = fdx - float(ix);
    float fy = fdy - float(iy);

    int bx0 = int(rx) + ix * 2;
    int by0 = int(ry) + iy * 2;
    int bx1 = bx0 + 2;
    int by1 = by0 + 2;

    if (bx0 < 0 || bx1 >= int(w) || by0 < 0 || by1 >= int(h))
        return -1.0f;

    float s00 = float(frame[by0 * int(w) + bx0]);
    float s10 = float(frame[by0 * int(w) + bx1]);
    float s01 = float(frame[by1 * int(w) + bx0]);
    float s11 = float(frame[by1 * int(w) + bx1]);

    return (1.0f - fx) * (1.0f - fy) * s00
         +         fx  * (1.0f - fy) * s10
         + (1.0f - fx) *         fy  * s01
         +         fx  *         fy  * s11;
}

// ---- Pass 1a: Collect z-values from non-rejected neighbors ----
// Dispatched once per neighbor. Accumulates z_sum, z_count, max_flow.
kernel void vst_bilateral_collect(
    device const uint16_t *center_frame   [[buffer(0)]],
    device const uint16_t *neighbor_frame [[buffer(1)]],
    device const float    *flow_x         [[buffer(2)]],
    device const float    *flow_y         [[buffer(3)]],
    device float          *z_sum          [[buffer(4)]],
    device float          *z_count        [[buffer(5)]],
    device float          *max_flow_buf   [[buffer(6)]],
    constant VSTBilateralParams &params   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint rx = gid.x, ry = gid.y;
    uint w = params.width, h = params.height;
    if (rx >= w || ry >= h) return;

    uint gx = rx >> 1, gy = ry >> 1, gw = w >> 1;
    float fdx = flow_x[gy * gw + gx];
    float fdy = flow_y[gy * gw + gx];

    float v = warp_bayer(rx, ry, fdx, fdy, neighbor_frame, w, h);
    if (v < 0.0f) return;

    uint idx = ry * w + rx;
    float z_c = vst_fwd(float(center_frame[idx]), params.black_level, params.shot_gain, params.read_noise);
    float z_n = vst_fwd(v, params.black_level, params.shot_gain, params.read_noise);

    if (abs(z_n - z_c) > params.z_reject) return;

    // Bilateral-weight the Phase-1 accumulation using z_center as reference.
    // Unweighted accumulation carries Poisson noise skew upward (low-signal
    // channels like R/B have right-skewed z distributions), producing a biased
    // z_preest that Phase-2 then uses to preferentially accept brighter neighbors
    // — causing the systematic R/B uplift (magenta cast). Weighting by proximity
    // to z_center keeps z_preest unbiased while still averaging multiple frames.
    float diff1 = z_n - z_c;
    float w1 = exp(-diff1 * diff1 / (2.0f * params.h * params.h));
    z_sum[idx]   += w1 * z_n;
    z_count[idx] += w1;

    float fm = sqrt(fdx * fdx + fdy * fdy);
    max_flow_buf[idx] = max(max_flow_buf[idx], fm);
}

// ---- Pass 1b: Compute self-guided pre-estimate ----
// z_preest = (z_sum + z_center) / (z_count + 1)
// Also zeros z_sum/z_count (aliased val_sum/w_sum) for Phase 2.
kernel void vst_bilateral_preestimate(
    device const uint16_t *center_frame  [[buffer(0)]],
    device float          *z_sum         [[buffer(1)]],
    device float          *z_count       [[buffer(2)]],
    device float          *z_preest      [[buffer(3)]],
    constant VSTBilateralParams &params  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint rx = gid.x, ry = gid.y;
    uint w = params.width, h = params.height;
    if (rx >= w || ry >= h) return;

    uint idx = ry * w + rx;
    float zs = z_sum[idx];
    float zc = z_count[idx];
    float z_center = vst_fwd(float(center_frame[idx]), params.black_level, params.shot_gain, params.read_noise);

    z_preest[idx] = (zs + z_center) / (zc + 1.0f);

    // Zero accumulators for Phase 2 reuse
    z_sum[idx]   = 0.0f;
    z_count[idx] = 0.0f;
}

// ---- Pass 2: Full bilateral fuse with research improvements ----
// Dispatched once per neighbor. Uses z_preest as self-guided reference.
// Structural term compares Bayer gradients. Multi-hypothesis picks best warp.
kernel void vst_bilateral_fuse(
    device const uint16_t *center_frame   [[buffer(0)]],
    device const uint16_t *neighbor_frame [[buffer(1)]],
    device const float    *flow_x         [[buffer(2)]],
    device const float    *flow_y         [[buffer(3)]],
    device float          *val_sum        [[buffer(4)]],
    device float          *w_sum          [[buffer(5)]],
    device const float    *z_preest       [[buffer(6)]],
    constant VSTBilateralParams &params   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint rx = gid.x, ry = gid.y;
    uint w = params.width, h = params.height;
    if (rx >= w || ry >= h) return;

    uint gx = rx >> 1, gy = ry >> 1, gw = w >> 1;
    float fdx = flow_x[gy * gw + gx];
    float fdy = flow_y[gy * gw + gx];

    uint idx = ry * w + rx;
    float cv_raw = float(center_frame[idx]);
    // z_preest is now a bilateral-weighted average from Phase 1 (weighted by
    // proximity to z_center), so it is unbiased and still smooth — safe to use
    // as reference for all channels including chroma.
    float z_ref = z_preest[idx];

    // Bright-surface h reduction: tighter bilateral acceptance for high-SNR
    // pixels where flow errors cause smearing on textureless skin.
    float bright_reduce = 1.0f - 0.5f * clamp((cv_raw - 10000.0f) / 20000.0f, 0.0f, 1.0f);
    float h_adj = params.h * bright_reduce;
    float neg_inv_2h2 = -1.0f / (2.0f * h_adj * h_adj);

    // Flow attenuation (shared across hypotheses)
    float flow_mag2 = fdx * fdx + fdy * fdy;
    float w_flow = exp(-flow_mag2 / params.flow_sigma2);

    // ---- Structural term (once per neighbor, not per hypothesis) ----
    // Gradients approximated via first-order VST linearization:
    //   dz/dv ≈ 2 / (z_ref * sg)  (Jacobian of z = 2*sqrt((v-bl)/sg))
    // This avoids 8 sqrt() calls per dispatch with <1% quality impact on the
    // structural weight, which is a heuristic edge-preserving term anyway.
    float w_struct = 1.0f;
    {
        float sg = params.shot_gain;
        // Jacobian: converts raw pixel differences to approximate VST units
        float vst_jac = 2.0f / max(z_ref * sg, 0.01f);

        // Center gradient (raw difference × Jacobian)
        float grad_cx = 0.0f, grad_cy = 0.0f;
        if (rx >= 2 && rx + 2 < w) {
            grad_cx = (float(center_frame[ry * w + rx + 2])
                     - float(center_frame[ry * w + rx - 2])) * 0.5f * vst_jac;
        }
        if (ry >= 2 && ry + 2 < h) {
            grad_cy = (float(center_frame[(ry + 2) * w + rx])
                     - float(center_frame[(ry - 2) * w + rx])) * 0.5f * vst_jac;
        }

        // Neighbor gradient at integer warp position
        int wx = int(rx) + int(round(fdx)) * 2;
        int wy = int(ry) + int(round(fdy)) * 2;
        float grad_nx = 0.0f, grad_ny = 0.0f;

        if (wx >= 2 && wx + 2 < int(w) && wy >= 0 && wy < int(h)) {
            grad_nx = (float(neighbor_frame[wy * int(w) + wx + 2])
                     - float(neighbor_frame[wy * int(w) + wx - 2])) * 0.5f * vst_jac;
        }
        if (wy >= 2 && wy + 2 < int(h) && wx >= 0 && wx < int(w)) {
            grad_ny = (float(neighbor_frame[(wy + 2) * int(w) + wx])
                     - float(neighbor_frame[(wy - 2) * int(w) + wx])) * 0.5f * vst_jac;
        }

        float gd_x = grad_nx - grad_cx;
        float gd_y = grad_ny - grad_cy;
        float grad_diff_sq = gd_x * gd_x + gd_y * gd_y;
        w_struct = exp(-grad_diff_sq / params.sigma_g2);
    }

    // ---- Multi-hypothesis sampling (M=4) ----
    // Try 4 candidates around warped position, pick highest composite weight
    float best_w = -1.0f;
    float best_z = 0.0f;

    for (int m = 0; m < 4; m++) {
        float hdx = fdx + VST_HYPS[m].x;
        float hdy = fdy + VST_HYPS[m].y;

        float v = warp_bayer(rx, ry, hdx, hdy, neighbor_frame, w, h);
        if (v < 0.0f) continue;

        float z = vst_fwd(v, params.black_level, params.shot_gain, params.read_noise);
        float diff = z - z_ref;

        if (abs(diff) > params.z_reject) continue;

        float w_photo = exp(diff * diff * neg_inv_2h2);
        float total = w_photo * w_struct * w_flow;

        if (total > best_w) {
            best_w = total;
            best_z = z;
        }
    }

    if (best_w <= 0.0f) return;

    // Texture confidence for bright surfaces during motion:
    // On textureless bright skin, OF can't track reliably → warped pixels
    // from wrong locations pass the bilateral (similar intensity) → smear.
    // Measure local same-color variance; if pure noise, reduce weight.
    float flow_mag = sqrt(flow_mag2);
    if (cv_raw > 10000.0f && flow_mag > 0.3f) {
        float tsum = cv_raw, tsq = cv_raw * cv_raw, tn = 1.0f;
        if (rx >= 2) { float tv = float(center_frame[ry * w + rx - 2]); tsum += tv; tsq += tv * tv; tn += 1.0f; }
        if (rx + 2 < w) { float tv = float(center_frame[ry * w + rx + 2]); tsum += tv; tsq += tv * tv; tn += 1.0f; }
        if (ry >= 2) { float tv = float(center_frame[(ry - 2) * w + rx]); tsum += tv; tsq += tv * tv; tn += 1.0f; }
        if (ry + 2 < h) { float tv = float(center_frame[(ry + 2) * w + rx]); tsum += tv; tsq += tv * tv; tn += 1.0f; }

        float tvar = tsq / tn - (tsum / tn) * (tsum / tn);
        float tstd = sqrt(max(tvar, 0.0f));

        // Expected noise std from calibrated model
        float noise_std = sqrt(params.read_noise * params.read_noise
                             + params.shot_gain * max(cv_raw - params.black_level, 0.0f));
        float tratio = tstd / max(noise_std, 1.0f);

        // tex_conf: 0.1 for pure noise (textureless), 1.0 for textured
        float tex_conf = clamp((tratio - 0.8f) / 1.5f, 0.1f, 1.0f);
        float motion_scale = clamp(flow_mag / 2.0f, 0.0f, 1.0f);
        tex_conf = 1.0f - (1.0f - tex_conf) * motion_scale;

        best_w *= tex_conf;
    }

    val_sum[idx] += best_w * best_z;
    w_sum[idx]   += best_w;
}

// ---- Pass 3: Center weight floor + inverse Anscombe → uint16 output ----
kernel void vst_bilateral_finalize(
    device const uint16_t *center_frame  [[buffer(0)]],
    device const float    *val_sum       [[buffer(1)]],
    device const float    *w_sum         [[buffer(2)]],
    device const float    *max_flow_buf  [[buffer(3)]],
    device uint16_t       *output        [[buffer(4)]],
    constant VSTBilateralParams &params  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint rx = gid.x, ry = gid.y;
    uint w = params.width, h = params.height;
    if (rx >= w || ry >= h) return;

    uint idx = ry * w + rx;
    float cv = float(center_frame[idx]);
    float z_center = vst_fwd(cv, params.black_level, params.shot_gain, params.read_noise);
    float nb_wsum = w_sum[idx];
    float nb_wzsum = val_sum[idx];

    if (nb_wsum <= 0.0f) {
        output[idx] = uint16_t(cv);
        return;
    }

    // Adaptive center weight floor
    float mf = max_flow_buf[idx];
    float center_floor = 0.3f + 0.3f * min(mf / 3.0f, 1.0f);

    // Chroma pixels (R and B: (ry&1)==(rx&1)) get a higher center-weight floor
    // to prevent the biased neighbor pre-estimate from pulling B/R values away
    // from the original, which otherwise causes a systematic B/G color shift.
    bool is_chroma = ((ry & 1) == (rx & 1));
    if (is_chroma) center_floor = min(center_floor + 0.25f, 0.85f);

    float center_w = 1.0f;

    float center_frac = center_w / (center_w + nb_wsum);
    if (center_frac < center_floor) {
        float scale = center_w * (1.0f - center_floor) / (center_floor * nb_wsum);
        nb_wsum  *= scale;
        nb_wzsum *= scale;
    }

    float total_w  = center_w + nb_wsum;
    float total_wz = center_w * z_center + nb_wzsum;
    float z_est = total_wz / total_w;

    float result = vst_inv(z_est, params.black_level, params.shot_gain, params.read_noise);
    result = clamp(result, 0.0f, 65535.0f);
    output[idx] = uint16_t(result + 0.5f);
}
