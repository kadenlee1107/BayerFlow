#include "../include/prores_raw_enc.h"
#include <math.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

// ============================================================================
// Forward DCT matched to FFmpeg's 12-bit integer iDCT
//
// FFmpeg's iDCT (simple_idct_template.c, BIT_DEPTH=12):
//   Row:  output[k] = (sum_j M[k][j] * coeff[j] + round) >> ROW_SHIFT
//   Col:  output[k] = (sum_j M[k][j] * input[j] + round) >> COL_SHIFT
//   Bias: block[i] += 8192 before column pass (adds 2048 DC offset)
//
// W constants: W1=45451, W2=42813, W3=38531, W4=32767,
//              W5=25746, W6=17734, W7=9041
// Shifts:      ROW_SHIFT=16, COL_SHIFT=17
//
// The forward DCT inverts this by computing:
//   coeff = row_fwd(col_fwd(centered_pixels))
// where col_fwd and row_fwd use M^(-1) scaled by 2^shift.
// ============================================================================

#define FW1 45451
#define FW2 42813
#define FW3 38531
#define FW4 32767
#define FW5 25746
#define FW6 17734
#define FW7  9041
#define F_ROW_SHIFT 16
#define F_COL_SHIFT 17

// Precomputed forward DCT matrices (8x8)
// fwd_col[coeff][pixel] for column-direction (undoes COL_SHIFT=17)
// fwd_row[coeff][pixel] for row-direction (undoes ROW_SHIFT=16)
static double fwd_col[8][8];
static double fwd_row[8][8];
static float fwd_col_f[8][8];
static float fwd_row_f[8][8];
static int fwd_initialized = 0;

void init_fwd_matrices(void) {
    if (fwd_initialized) return;

    // Build the 8x8 iDCT butterfly matrix M[output][input].
    // The butterfly structure maps frequency inputs to spatial outputs.
    double M[8][8];

    // Even part: maps inputs {0,2,4,6} to {a0,a1,a2,a3}
    double even_coeffs[4][4] = {
        { FW4,  FW2,  FW4,  FW6},  // a0
        { FW4,  FW6, -FW4, -FW2},  // a1
        { FW4, -FW6, -FW4,  FW2},  // a2
        { FW4, -FW2,  FW4, -FW6},  // a3
    };

    // Odd part: maps inputs {1,3,5,7} to {b0,b1,b2,b3}
    double odd_coeffs[4][4] = {
        { FW1,  FW3,  FW5,  FW7},  // b0
        { FW3, -FW7, -FW1, -FW5},  // b1
        { FW5, -FW1,  FW7,  FW3},  // b2
        { FW7, -FW5,  FW3, -FW1},  // b3
    };

    // Output: out[0]=a0+b0, [1]=a1+b1, [2]=a2+b2, [3]=a3+b3,
    //         [4]=a3-b3, [5]=a2-b2, [6]=a1-b1, [7]=a0-b0
    int ei[] = {0, 1, 2, 3, 3, 2, 1, 0};
    int sign[] = {1, 1, 1, 1, -1, -1, -1, -1};

    for (int k = 0; k < 8; k++) {
        // Even inputs at positions 0, 2, 4, 6
        M[k][0] = even_coeffs[ei[k]][0];
        M[k][2] = even_coeffs[ei[k]][1];
        M[k][4] = even_coeffs[ei[k]][2];
        M[k][6] = even_coeffs[ei[k]][3];

        // Odd inputs at positions 1, 3, 5, 7
        M[k][1] = sign[k] * odd_coeffs[ei[k]][0];
        M[k][3] = sign[k] * odd_coeffs[ei[k]][1];
        M[k][5] = sign[k] * odd_coeffs[ei[k]][2];
        M[k][7] = sign[k] * odd_coeffs[ei[k]][3];
    }

    // M is orthogonal (up to scaling): M^T * M = D (diagonal)
    // D[j] = sum_k M[k][j]^2 = column norm squared
    double D[8];
    for (int j = 0; j < 8; j++) {
        D[j] = 0;
        for (int k = 0; k < 8; k++)
            D[j] += M[k][j] * M[k][j];
    }

    // Forward matrix = M^(-1) * 2^shift = (M^T / D) * 2^shift
    // fwd[j][k] = M[k][j] / D[j] * 2^shift
    //
    // Forward column matrix uses COL_SHIFT (undoes column iDCT)
    // Forward row matrix uses ROW_SHIFT (undoes row iDCT)
    double col_scale = (double)(1 << F_COL_SHIFT);  // 2^17
    double row_scale = (double)(1 << F_ROW_SHIFT);  // 2^16

    for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
            fwd_col[j][k] = M[k][j] / D[j] * col_scale;
            fwd_row[j][k] = M[k][j] / D[j] * row_scale;
        }
    }

    // Populate float32 copies for NEON path
    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++) {
            fwd_col_f[j][k] = (float)fwd_col[j][k];
            fwd_row_f[j][k] = (float)fwd_row[j][k];
        }

    fwd_initialized = 1;
}

#ifndef __ARM_NEON__
// Apply 1D forward DCT for column direction
static void fdct_col(double *data, int stride) {
    double input[8];
    for (int i = 0; i < 8; i++)
        input[i] = data[i * stride];

    double output[8];
    for (int j = 0; j < 8; j++) {
        output[j] = 0;
        for (int k = 0; k < 8; k++)
            output[j] += fwd_col[j][k] * input[k];
    }

    for (int i = 0; i < 8; i++)
        data[i * stride] = output[i];
}

// Apply 1D forward DCT for row direction
static void fdct_row(double *data, int stride) {
    double input[8];
    for (int i = 0; i < 8; i++)
        input[i] = data[i * stride];

    double output[8];
    for (int j = 0; j < 8; j++) {
        output[j] = 0;
        for (int k = 0; k < 8; k++)
            output[j] += fwd_row[j][k] * input[k];
    }

    for (int i = 0; i < 8; i++)
        data[i * stride] = output[i];
}
#endif

// Extract sample block from interleaved Bayer data.
// component: 0=R, 1=G(odd cols), 2=G(odd rows), 3=B, 4=full-resolution CFA
static void extract_bayer_block(uint16_t *block, const uint16_t *src,
                                int src_stride, int component) {
    if (component == 4) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                block[y * 8 + x] = src[y * src_stride + x];
            }
        }
        return;
    }

    // Bayer pattern: RGGB
    // component 0 (R): even row, even col
    // component 1 (G): even row, odd col
    // component 2 (G): odd row, even col
    // component 3 (B): odd row, odd col

    int row_offset = (component >> 1) & 1;  // 0 for comp 0,1; 1 for comp 2,3
    int col_offset = component & 1;         // 0 for comp 0,2; 1 for comp 1,3

    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int src_y = (y * 2) + row_offset;
            int src_x = (x * 2) + col_offset;
            block[y * 8 + x] = src[src_y * src_stride + src_x];
        }
    }
}

#ifdef __ARM_NEON__
// NEON-optimized 1D DCT pass: 8 dot products of 8 floats each
static void fdct_1d_neon(const float mat[8][8], const float *input, float *output) {
    float32x4_t inp_lo = vld1q_f32(input);
    float32x4_t inp_hi = vld1q_f32(input + 4);

    for (int j = 0; j < 8; j += 4) {
        float32x4_t acc0 = vmulq_f32(vld1q_f32(mat[j+0]), inp_lo);
        acc0 = vfmaq_f32(acc0, vld1q_f32(mat[j+0] + 4), inp_hi);

        float32x4_t acc1 = vmulq_f32(vld1q_f32(mat[j+1]), inp_lo);
        acc1 = vfmaq_f32(acc1, vld1q_f32(mat[j+1] + 4), inp_hi);

        float32x4_t acc2 = vmulq_f32(vld1q_f32(mat[j+2]), inp_lo);
        acc2 = vfmaq_f32(acc2, vld1q_f32(mat[j+2] + 4), inp_hi);

        float32x4_t acc3 = vmulq_f32(vld1q_f32(mat[j+3]), inp_lo);
        acc3 = vfmaq_f32(acc3, vld1q_f32(mat[j+3] + 4), inp_hi);

        output[j+0] = vaddvq_f32(acc0);
        output[j+1] = vaddvq_f32(acc1);
        output[j+2] = vaddvq_f32(acc2);
        output[j+3] = vaddvq_f32(acc3);
    }
}
#endif

// Forward DCT 8x8 with Bayer extraction.
// Uses forward transform matrices that are the exact inverse of FFmpeg's
// 12-bit integer iDCT, ensuring near-perfect round-trip reconstruction.
void forward_dct_8x8(int32_t *block, const uint16_t *src, int stride, int component) {
    uint16_t extracted[64];

    // Extract Bayer component
    extract_bayer_block(extracted, src, stride, component);

#ifdef __ARM_NEON__
    // NEON path: float32 throughout
    float tempf[64];
    for (int i = 0; i < 64; i++)
        tempf[i] = (float)(extracted[i] >> 4) - 2048.0f;

    for (int x = 0; x < 8; x++) {
        float col_in[8], col_out[8];
        for (int i = 0; i < 8; i++) col_in[i] = tempf[i * 8 + x];
        fdct_1d_neon(fwd_col_f, col_in, col_out);
        for (int i = 0; i < 8; i++) tempf[i * 8 + x] = col_out[i];
    }

    for (int y = 0; y < 8; y++) {
        float row_out[8];
        fdct_1d_neon(fwd_row_f, &tempf[y * 8], row_out);
        for (int i = 0; i < 8; i++) tempf[y * 8 + i] = row_out[i];
    }

    for (int i = 0; i < 64; i++)
        block[i] = (int32_t)roundf(tempf[i]);
#else
    // Scalar double path (fallback)
    double temp[64];
    for (int i = 0; i < 64; i++)
        temp[i] = (double)(extracted[i] >> 4) - 2048.0;

    for (int x = 0; x < 8; x++)
        fdct_col(&temp[x], 8);

    for (int y = 0; y < 8; y++)
        fdct_row(&temp[y * 8], 1);

    for (int i = 0; i < 64; i++)
        block[i] = (int32_t)round(temp[i]);
#endif
}

// Quantize DCT coefficients
void quantize_block(int32_t *block, const int32_t *qmat) {
    for (int i = 0; i < 64; i++) {
        if (qmat[i] > 0) {
            int32_t v = block[i];
            int32_t half = qmat[i] / 2;
            if (v >= 0)
                block[i] = (v + half) / qmat[i];
            else
                block[i] = -(((-v) + half) / qmat[i]);
        }
    }
}

// ============================================================================
// Post-quantization overshoot control
//
// DCT quantization can create Gibbs ringing at extreme contrast edges,
// producing pixel values below 0 or above 4095 after decode. This is
// especially visible with denoised content (cleaner edges = more ringing).
// The camera encoder doesn't show this because original sensor noise
// acts as natural dither.
//
// Fix: test-decode each quantized block using the exact integer iDCT.
// If any output pixel is out of range [0, 4095], compute a scale factor
// to reduce AC coefficients proportionally. This preserves block structure
// while preventing decoder-side clamping artifacts (colored dots in Bayer).
// ============================================================================

// Integer iDCT matching decoder exactly (W constants, shifts, DC bias)
#define OC_W1 45451
#define OC_W2 42813
#define OC_W3 38531
#define OC_W4 32767
#define OC_W5 25746
#define OC_W6 17734
#define OC_W7  9041
#define OC_ROW_SHIFT 16
#define OC_COL_SHIFT 17

static void oc_idct_row(int32_t *row) {
    if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
        int val = (int)(((unsigned)(row[0] * OC_W4) + (1u << (OC_ROW_SHIFT - 1))) >> OC_ROW_SHIFT);
        val <<= 1;
        row[0] = row[1] = row[2] = row[3] =
        row[4] = row[5] = row[6] = row[7] = val;
        return;
    }
    unsigned a0 = (unsigned)(row[0] * OC_W4) + (1u << (OC_ROW_SHIFT - 1));
    unsigned a1 = a0, a2 = a0, a3 = a0;
    a0 += (unsigned)(row[2] * OC_W2); a1 += (unsigned)(row[2] * OC_W6);
    a2 -= (unsigned)(row[2] * OC_W6); a3 -= (unsigned)(row[2] * OC_W2);
    unsigned t4 = (unsigned)(row[4] * OC_W4);
    a0 += t4; a1 -= t4; a2 -= t4; a3 += t4;
    a0 += (unsigned)(row[6] * OC_W6); a1 -= (unsigned)(row[6] * OC_W2);
    a2 += (unsigned)(row[6] * OC_W2); a3 -= (unsigned)(row[6] * OC_W6);
    unsigned b0 = (unsigned)(row[1]*OC_W1) + (unsigned)(row[3]*OC_W3) + (unsigned)(row[5]*OC_W5) + (unsigned)(row[7]*OC_W7);
    unsigned b1 = (unsigned)(row[1]*OC_W3) - (unsigned)(row[3]*OC_W7) - (unsigned)(row[5]*OC_W1) - (unsigned)(row[7]*OC_W5);
    unsigned b2 = (unsigned)(row[1]*OC_W5) - (unsigned)(row[3]*OC_W1) + (unsigned)(row[5]*OC_W7) + (unsigned)(row[7]*OC_W3);
    unsigned b3 = (unsigned)(row[1]*OC_W7) - (unsigned)(row[3]*OC_W5) + (unsigned)(row[5]*OC_W3) - (unsigned)(row[7]*OC_W1);
    row[0] = (int)(a0+b0) >> OC_ROW_SHIFT; row[1] = (int)(a1+b1) >> OC_ROW_SHIFT;
    row[2] = (int)(a2+b2) >> OC_ROW_SHIFT; row[3] = (int)(a3+b3) >> OC_ROW_SHIFT;
    row[4] = (int)(a3-b3) >> OC_ROW_SHIFT; row[5] = (int)(a2-b2) >> OC_ROW_SHIFT;
    row[6] = (int)(a1-b1) >> OC_ROW_SHIFT; row[7] = (int)(a0-b0) >> OC_ROW_SHIFT;
}

static void oc_idct_col(int32_t *block, int col) {
    int32_t *c = block + col;
    if (!(c[1*8] | c[2*8] | c[3*8] | c[4*8] | c[5*8] | c[6*8] | c[7*8])) {
        int val = (int)(((unsigned)(c[0*8] * OC_W4) + (1u << (OC_COL_SHIFT - 1))) >> OC_COL_SHIFT);
        c[0*8]=c[1*8]=c[2*8]=c[3*8]=c[4*8]=c[5*8]=c[6*8]=c[7*8] = val;
        return;
    }
    unsigned a0 = (unsigned)((c[0*8] + ((1 << (OC_COL_SHIFT-1)) / OC_W4)) * OC_W4);
    unsigned a1 = a0, a2 = a0, a3 = a0;
    a0 += (unsigned)(c[2*8]*OC_W2); a1 += (unsigned)(c[2*8]*OC_W6);
    a2 -= (unsigned)(c[2*8]*OC_W6); a3 -= (unsigned)(c[2*8]*OC_W2);
    unsigned t4 = (unsigned)(c[4*8]*OC_W4);
    a0 += t4; a1 -= t4; a2 -= t4; a3 += t4;
    a0 += (unsigned)(c[6*8]*OC_W6); a1 -= (unsigned)(c[6*8]*OC_W2);
    a2 += (unsigned)(c[6*8]*OC_W2); a3 -= (unsigned)(c[6*8]*OC_W6);
    unsigned b0 = (unsigned)(c[1*8]*OC_W1) + (unsigned)(c[3*8]*OC_W3) + (unsigned)(c[5*8]*OC_W5) + (unsigned)(c[7*8]*OC_W7);
    unsigned b1 = (unsigned)(c[1*8]*OC_W3) - (unsigned)(c[3*8]*OC_W7) - (unsigned)(c[5*8]*OC_W1) - (unsigned)(c[7*8]*OC_W5);
    unsigned b2 = (unsigned)(c[1*8]*OC_W5) - (unsigned)(c[3*8]*OC_W1) + (unsigned)(c[5*8]*OC_W7) + (unsigned)(c[7*8]*OC_W3);
    unsigned b3 = (unsigned)(c[1*8]*OC_W7) - (unsigned)(c[3*8]*OC_W5) + (unsigned)(c[5*8]*OC_W3) - (unsigned)(c[7*8]*OC_W1);
    c[0*8] = (int)(a0+b0) >> OC_COL_SHIFT; c[1*8] = (int)(a1+b1) >> OC_COL_SHIFT;
    c[2*8] = (int)(a2+b2) >> OC_COL_SHIFT; c[3*8] = (int)(a3+b3) >> OC_COL_SHIFT;
    c[4*8] = (int)(a3-b3) >> OC_COL_SHIFT; c[5*8] = (int)(a2-b2) >> OC_COL_SHIFT;
    c[6*8] = (int)(a1-b1) >> OC_COL_SHIFT; c[7*8] = (int)(a0-b0) >> OC_COL_SHIFT;
}

static void oc_test_decode(const int32_t *quantized, const int32_t *qmat, int32_t *out) {
    for (int i = 0; i < 64; i++) out[i] = quantized[i] * qmat[i];
    for (int y = 0; y < 8; y++) oc_idct_row(&out[y * 8]);
    for (int x = 0; x < 8; x++) out[x] += 8192;
    for (int x = 0; x < 8; x++) oc_idct_col(out, x);
}

void clamp_quantized_block(int32_t *block, const int32_t *qmat,
                           int32_t input_min, int32_t input_max) {
    int32_t test[64];
    oc_test_decode(block, qmat, test);

    // Use input pixel range as the valid output range (12-bit).
    // DCT ringing should not push values beyond the block's actual input range.
    int32_t floor_val = input_min;
    int32_t ceil_val  = input_max;
    if (floor_val < 0)    floor_val = 0;
    if (ceil_val > 4095)  ceil_val = 4095;

    int32_t min_val = test[0], max_val = test[0];
    for (int i = 1; i < 64; i++) {
        if (test[i] < min_val) min_val = test[i];
        if (test[i] > max_val) max_val = test[i];
    }
    if (min_val >= floor_val && max_val <= ceil_val) return;

    // Compute average (DC pixel value) — AC cosines sum to zero
    int32_t sum = 0;
    for (int i = 0; i < 64; i++) sum += test[i];
    float dc_avg = (float)sum / 64.0f;

    // Scale factor to keep all pixels in [floor_val, ceil_val]:
    // pixel ≈ dc_avg + scale * (pixel - dc_avg)
    float scale = 1.0f;
    if (min_val < floor_val && dc_avg > (float)floor_val + 0.5f) {
        float s = (dc_avg - (float)floor_val) / (dc_avg - (float)min_val);
        if (s < scale) scale = s;
    }
    if (max_val > ceil_val && dc_avg < (float)ceil_val - 0.5f) {
        float s = ((float)ceil_val - dc_avg) / ((float)max_val - dc_avg);
        if (s < scale) scale = s;
    }
    if (scale < 0.0f) scale = 0.0f;

    // Apply scale to AC coefficients (index 1..63, preserve DC at index 0)
    for (int i = 1; i < 64; i++)
        block[i] = (int32_t)roundf((float)block[i] * scale);

    // Verify — if still out of range after rounding, do per-coefficient fixup
    oc_test_decode(block, qmat, test);
    for (int iter = 0; iter < 10; iter++) {
        min_val = test[0]; max_val = test[0];
        for (int i = 1; i < 64; i++) {
            if (test[i] < min_val) min_val = test[i];
            if (test[i] > max_val) max_val = test[i];
        }
        if (min_val >= floor_val && max_val <= ceil_val) break;

        // Reduce the largest-magnitude AC coefficient by 1
        int best = -1;
        int32_t best_mag = 0;
        for (int i = 1; i < 64; i++) {
            int32_t m = block[i] < 0 ? -block[i] : block[i];
            if (m > best_mag) { best_mag = m; best = i; }
        }
        if (best < 0 || best_mag == 0) break;
        block[best] += (block[best] > 0) ? -1 : 1;

        oc_test_decode(block, qmat, test);
    }
}
