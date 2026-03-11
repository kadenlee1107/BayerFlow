#include <metal_stdlib>
using namespace metal;

// Camera noise model parameters for calibrated noise sigma map.
// σ(v) = sqrt(read_noise² + shot_gain * max(0, v - black_level)) / 65535
struct NoiseModelParams {
    float black_level;  // normalized [0,1] (raw_bl / 65535)
    float read_noise;   // normalized [0,1] (raw_rn / 65535)
    float shot_gain;    // raw scale (e.g. 180.0)
};

// Bayer extract: deinterleave uint16 Bayer into 5 float planes at full sub-channel resolution.
// 4 Bayer channels (R, Gr, Gb, B) + 1 calibrated noise sigma map.
// CNN resolution = (W/2, H/2) = full sub-channel resolution (no downsampling).
//
// Dispatch: (cnnW, cnnH, 1) threads — one thread per sub-channel pixel.
kernel void bayer_extract_downsample(
    device const uint16_t *bayer      [[buffer(0)]],  // W * H uint16
    device float          *cnn_input  [[buffer(1)]],  // 5 * cnnH * cnnW float32 (NCHW planar)
    constant uint2        &full_dims  [[buffer(2)]],  // (W, H)
    constant uint2        &cnn_dims   [[buffer(3)]],  // (cnnW, cnnH) = (W/2, H/2)
    constant NoiseModelParams &noise_model [[buffer(4)]],  // camera noise model
    uint2 gid [[thread_position_in_grid]])
{
    uint cnnW = cnn_dims.x;
    uint cnnH = cnn_dims.y;
    if (gid.x >= cnnW || gid.y >= cnnH) return;

    uint W = full_dims.x;
    uint cx = gid.x;
    uint cy = gid.y;

    // Extract 4 Bayer channels: comp 0=R(0,0), 1=Gr(0,1), 2=Gb(1,0), 3=B(1,1)
    float avg_val = 0.0f;
    for (uint comp = 0; comp < 4; comp++) {
        uint dy = comp >> 1;
        uint dx = comp & 1;

        uint raw_x = cx * 2 + dx;
        uint raw_y = cy * 2 + dy;
        float val = float(bayer[raw_y * W + raw_x]) / 65535.0f;

        cnn_input[comp * cnnH * cnnW + cy * cnnW + cx] = val;
        avg_val += val;
    }

    // 5th plane: calibrated noise sigma from camera noise model
    // σ(v) = sqrt(read_noise² + shot_gain/65535 * max(0, v - black_level))
    // All values in [0,1] normalized space — matches training exactly
    avg_val *= 0.25f;
    float bl = noise_model.black_level;
    float rn = noise_model.read_noise;
    float sg = noise_model.shot_gain;
    float noise_map = sqrt(max(rn * rn + (sg / 65535.0f) * max(avg_val - bl, 0.0f), 1e-6f));
    cnn_input[4 * cnnH * cnnW + cy * cnnW + cx] = noise_map;
}

// Noise subtract + blend + interleave: apply CNN-predicted noise at full Bayer resolution.
// CNN output is the predicted noise residual at sub-channel resolution (W/2, H/2) per channel.
// For each Bayer pixel: look up the noise prediction, subtract blended amount, write uint16.
//
// Dispatch: (W, H, 1) threads — one thread per Bayer pixel.
kernel void noise_subtract_blend_interleave(
    device const float    *cnn_noise    [[buffer(0)]],  // 4 * cnnH * cnnW float32 (predicted noise)
    device const uint16_t *bayer_in     [[buffer(1)]],  // W * H uint16 (original/bilateral Bayer)
    device uint16_t       *bayer_out    [[buffer(2)]],  // W * H uint16 (denoised output)
    constant uint2        &full_dims    [[buffer(3)]],  // (W, H)
    constant uint2        &cnn_dims     [[buffer(4)]],  // (cnnW, cnnH) = (W/2, H/2)
    constant float        &blend_factor [[buffer(5)]],  // 0.9
    uint2 gid [[thread_position_in_grid]])
{
    uint W = full_dims.x;
    uint H = full_dims.y;
    if (gid.x >= W || gid.y >= H) return;

    uint rx = gid.x;
    uint ry = gid.y;

    // Determine which Bayer channel: comp = (row_parity * 2) + col_parity
    uint comp = (ry & 1) * 2 + (rx & 1);

    // CNN coordinates: direct sub-channel mapping (no downsample factor)
    uint cnnW = cnn_dims.x;
    uint cnnH = cnn_dims.y;
    uint cx = min(rx >> 1, cnnW - 1);
    uint cy = min(ry >> 1, cnnH - 1);

    // Look up predicted noise
    float noise = cnn_noise[comp * cnnH * cnnW + cy * cnnW + cx];

    // Read original pixel, normalize
    float orig = float(bayer_in[ry * W + rx]) / 65535.0f;

    // Subtract blended noise
    float denoised = orig - blend_factor * noise;

    // Clamp and convert back to uint16
    denoised = clamp(denoised, 0.0f, 1.0f);
    bayer_out[ry * W + rx] = uint16_t(denoised * 65535.0f + 0.5f);
}

// Masked variant: person segmentation mask reduces blend in subject regions.
// mask_buf is at CNN resolution (cnnW × cnnH), 0.0=background, 1.0=person.
// subject_protection controls how much to reduce denoise on subjects (0.0=none, 0.7=heavy).
//
// Dispatch: (W, H, 1) threads — one thread per Bayer pixel.
kernel void noise_subtract_blend_masked(
    device const float    *cnn_noise          [[buffer(0)]],
    device const uint16_t *bayer_in           [[buffer(1)]],
    device uint16_t       *bayer_out          [[buffer(2)]],
    constant uint2        &full_dims          [[buffer(3)]],
    constant uint2        &cnn_dims           [[buffer(4)]],
    constant float        &blend_factor       [[buffer(5)]],
    device const float    *mask_buf           [[buffer(6)]],  // cnnW * cnnH float
    constant float        &subject_protection [[buffer(7)]],  // 0.0-1.0 boost on subjects
    uint2 gid [[thread_position_in_grid]])
{
    uint W = full_dims.x;
    uint H = full_dims.y;
    if (gid.x >= W || gid.y >= H) return;

    uint rx = gid.x;
    uint ry = gid.y;

    uint comp = (ry & 1) * 2 + (rx & 1);

    uint cnnW = cnn_dims.x;
    uint cnnH = cnn_dims.y;
    uint cx = min(rx >> 1, cnnW - 1);
    uint cy = min(ry >> 1, cnnH - 1);

    float noise = cnn_noise[comp * cnnH * cnnW + cy * cnnW + cx];
    float orig = float(bayer_in[ry * W + rx]) / 65535.0f;

    // Read person mask — boost denoising on detected subjects
    float mask_val = mask_buf[cy * cnnW + cx];
    float effective_blend = min(1.0f, blend_factor * (1.0f + mask_val * subject_protection));

    float denoised = orig - effective_blend * noise;
    denoised = clamp(denoised, 0.0f, 1.0f);
    bayer_out[ry * W + rx] = uint16_t(denoised * 65535.0f + 0.5f);
}
