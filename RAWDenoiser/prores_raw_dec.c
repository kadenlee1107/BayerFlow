/*
 * prores_raw_dec.c — Native ProRes RAW frame decoder
 *
 * Exact inverse of the encoder in tile_encoder.c / rice_golomb.c / dct.c.
 * Decodes compressed ProRes RAW bitstream → raw Bayer RGGB16 output.
 *
 * Pipeline: Rice-Golomb decode → dequantize → iDCT → Bayer reconstruct
 */

#include "prores_raw_dec.h"
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dispatch/dispatch.h>

/* ---- Constants (must match encoder exactly) ---- */

#define TILE_WIDTH  256
#define TILE_HEIGHT 16
#define BLOCK_SIZE  8

#define DC_CB_MAX 12
#define AC_CB_MAX 94
#define RN_CB_MAX 27
#define LN_CB_MAX 14

#define TODCCODEBOOK(x) (((x) + 1) >> 1)
#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* Codebook tables — identical to encoder (rice_golomb.c) */
static const int16_t dc_cb[DC_CB_MAX + 1] = {
    0x010, 0x021, 0x032, 0x033, 0x033, 0x033, 0x044, 0x044,
    0x044, 0x044, 0x044, 0x044, 0x076
};

static const int16_t ac_cb[AC_CB_MAX + 1] = {
    0x000, 0x211, 0x111, 0x111, 0x222, 0x222, 0x222, 0x122, 0x122, 0x122,
    0x233, 0x233, 0x233, 0x233, 0x233, 0x233, 0x233, 0x233, 0x133, 0x133,
    0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244,
    0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244, 0x244,
    0x244, 0x244, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355,
    0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355,
    0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355,
    0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355,
    0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355, 0x355,
    0x355, 0x355, 0x355, 0x355, 0x166,
};

static const int16_t rn_cb[RN_CB_MAX + 1] = {
    0x200, 0x100, 0x000, 0x000, 0x211, 0x211, 0x111, 0x111, 0x011, 0x011,
    0x021, 0x021, 0x222, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022,
    0x022, 0x022, 0x022, 0x022, 0x022, 0x032, 0x032, 0x044
};

static const int16_t ln_cb[LN_CB_MAX + 1] = {
    0x100, 0x111, 0x222, 0x222, 0x122, 0x122, 0x433, 0x433,
    0x233, 0x233, 0x233, 0x233, 0x233, 0x233, 0x033,
};

static const uint8_t zigzag_scan[64] = {
     0,  8,  1,  9, 16, 24, 17, 25,
     2, 10,  3, 11, 18, 26, 19, 27,
    32, 40, 33, 34, 41, 48, 56, 49,
    42, 35, 43, 50, 57, 58, 51, 59,
     4, 12,  5,  6, 13, 20, 28, 21,
    14,  7, 15, 22, 29, 36, 44, 37,
    30, 23, 31, 38, 45, 52, 60, 53,
    46, 39, 47, 54, 61, 62, 55, 63
};

/* ---- BitReader (inverse of BitWriter) ---- */

typedef struct {
    const uint8_t *data;
    int size;
    uint64_t acc;
    int acc_bits;
    int byte_pos;
} BitReader;

static void br_init(BitReader *br, const uint8_t *data, int size) {
    br->data = data;
    br->size = size;
    br->acc = 0;
    br->acc_bits = 0;
    br->byte_pos = 0;
}

static void br_refill(BitReader *br) {
    while (br->acc_bits <= 56 && br->byte_pos < br->size) {
        br->acc = (br->acc << 8) | br->data[br->byte_pos++];
        br->acc_bits += 8;
    }
}

static uint32_t br_get_bits(BitReader *br, int n) {
    if (n <= 0) return 0;
    br_refill(br);
    if (br->acc_bits < n) return 0; /* not enough bits */
    br->acc_bits -= n;
    return (uint32_t)((br->acc >> br->acc_bits) & ((1ULL << n) - 1));
}

static int br_bits_left(BitReader *br) {
    return br->acc_bits + (br->size - br->byte_pos) * 8;
}

/* ---- Rice-Golomb Decoder (inverse of put_value) ---- */

static uint32_t get_value(BitReader *br, int16_t codebook) {
    int switch_bits = codebook >> 8;
    int rice_order  = codebook & 0xf;
    int exp_order   = (codebook >> 4) & 0xf;

    int q = 0;
    int found_stop = 0;
    while (br_bits_left(br) > 0 && q < 31) {
        if (br_get_bits(br, 1) == 1) { found_stop = 1; break; }
        q++;
    }

    if (!found_stop) return 0;

    if (q <= switch_bits) {
        uint32_t r = (rice_order > 0) ? br_get_bits(br, rice_order) : 0;
        return ((uint32_t)q << rice_order) | r;
    } else {
        int data_bits = exp_order + q - switch_bits - 1;
        if (data_bits < 0) data_bits = 0;
        if (data_bits > 31) return 0; /* safety: avoid UB in shift */
        uint32_t data = (data_bits > 0) ? br_get_bits(br, data_bits) : 0;
        uint32_t diff = (1U << data_bits) | data;
        uint32_t threshold = ((uint32_t)(switch_bits + 1) << rice_order) - (1U << exp_order);
        return threshold + diff;
    }
}

/* ---- DC Coefficient Decoder ---- */

static void decode_dc_coeffs(BitReader *br, int32_t blocks[][64], int nb_blocks) {
    if (nb_blocks <= 0) return;

    /* First block DC */
    uint32_t code = get_value(br, 700);
    int32_t dc = (int32_t)((code >> 1) ^ (-(int32_t)(code & 1)));
    blocks[0][0] = dc + 1; /* encoder subtracts 1 before encoding */

    int prev_dc = dc;
    uint32_t prev_dc_code = code;
    int sign = 0;

    for (int b = 1; b < nb_blocks; b++) {
        int16_t cb;
        if ((b & 15) == 1)
            cb = 100;
        else
            cb = dc_cb[MIN(TODCCODEBOOK(prev_dc_code), DC_CB_MAX)];

        uint32_t dc_code = get_value(br, cb);

        /* Mirror encoder state machine exactly */
        sign ^= dc_code & 1;
        int dc_add = ((-sign) ^ TODCCODEBOOK(dc_code)) + sign;
        sign = dc_add < 0;
        prev_dc += dc_add;

        blocks[b][0] = prev_dc + 1; /* undo the -1 offset */
        prev_dc_code = dc_code;
    }
}

/* ---- AC Coefficient Decoder ---- */

static void decode_ac_coeffs(BitReader *br, int32_t blocks[][64], int nb_blocks) {
    int total_ac = nb_blocks * 63;
    int32_t *ac_seq = (int32_t *)calloc(total_ac, sizeof(int32_t));
    if (!ac_seq) return;

    int16_t ac_codebook = 49;
    int16_t ln_codebook = 66;
    int16_t rn_codebook = 0;

    int n = nb_blocks; /* start after DC coefficients */
    int nb_codes = nb_blocks * 64;

    while (n < nb_codes && br_bits_left(br) > 0) {
        int seq_pos = n - nb_blocks;

        /* Read ln (count of non-zero values) */
        uint32_t ln_val = get_value(br, ln_codebook);
        int ln = (int)ln_val;

        /* Read ln non-zero AC values */
        for (int i = 0; i < ln && (seq_pos + i) < total_ac; i++) {
            uint32_t ac_mag = get_value(br, ac_codebook);
            uint32_t sign_bit = br_get_bits(br, 1);
            int32_t val = (int32_t)(ac_mag + 1);
            if (sign_bit) val = -val;
            ac_seq[seq_pos + i] = val;
            ac_codebook = ac_cb[MIN((int)ac_mag, AC_CB_MAX)];
        }
        n += ln;

        if (n >= nb_codes || br_bits_left(br) <= 0) break;

        /* Read rn (zero run length - 1) */
        uint32_t rn_val = get_value(br, rn_codebook);
        int rn = (int)rn_val;
        rn_codebook = rn_cb[MIN(rn, RN_CB_MAX)];
        n += rn + 1;

        if (n >= nb_codes || br_bits_left(br) <= 0) break;

        /* Read the AC after the zero run */
        {
            int pos = n - nb_blocks;
            if (pos >= 0 && pos < total_ac) {
                uint32_t ac_mag = get_value(br, ac_codebook);
                uint32_t sign_bit = br_get_bits(br, 1);
                int32_t val = (int32_t)(ac_mag + 1);
                if (sign_bit) val = -val;
                ac_seq[pos] = val;
                ac_codebook = ac_cb[MIN((int)ac_mag, AC_CB_MAX)];
                ln_codebook = ln_cb[MIN((int)ac_mag, LN_CB_MAX)];
            }
            n++;
        }
    }

    /* De-interleave: ac_seq is in coefficient-interleaved order
     * [block0_coeff1, block1_coeff1, ..., block0_coeff2, ...] */
    int idx = 0;
    for (int coef = 1; coef < 64; coef++) {
        for (int b = 0; b < nb_blocks; b++) {
            blocks[b][zigzag_scan[coef]] = ac_seq[idx++];
        }
    }

    free(ac_seq);
}

/* ---- iDCT (FFmpeg 12-bit simple_idct) ---- */
/* Uses unsigned intermediates (SUINT) matching FFmpeg to handle overflow
 * via well-defined modular arithmetic instead of signed overflow UB. */

#define W1 45451
#define W2 42813
#define W3 38531
#define W4 32767
#define W5 25746
#define W6 17734
#define W7  9041
#define ROW_SHIFT 16
#define COL_SHIFT 17

static void idct_row(int32_t *row) {
    /* DC-only shortcut: FFmpeg idctRowCondDC for 12-bit (DC_SHIFT=-1).
     * Signed shift: (row[0] + 1) >> 1 — divides by 2 with rounding.
     * FFmpeg uses int16_t blocks with (unsigned) cast + & 0xffff, but since
     * our blocks are int32_t, we use direct signed arithmetic right shift. */
    if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
        int val = (row[0] + 1) >> 1;
        row[0] = row[1] = row[2] = row[3] =
        row[4] = row[5] = row[6] = row[7] = val;
        return;
    }

    /* FFmpeg-compatible factored butterfly with unsigned intermediates (SUINT).
     * The rounding constant is folded into a0 first, then a1-a3 inherit it.
     * This matches FFmpeg's exact accumulation order for unsigned wrapping. */

    /* Even part: start with DC * W4 + rounding */
    unsigned a0 = (unsigned)(row[0] * W4) + (1u << (ROW_SHIFT - 1));
    unsigned a1 = a0;
    unsigned a2 = a0;
    unsigned a3 = a0;

    /* Add row[2] contributions */
    a0 += (unsigned)(row[2] * W2);
    a1 += (unsigned)(row[2] * W6);
    a2 -= (unsigned)(row[2] * W6);
    a3 -= (unsigned)(row[2] * W2);

    /* Add row[4] contributions */
    unsigned temp = (unsigned)(row[4] * W4);
    a0 += temp;
    a1 -= temp;
    a2 -= temp;
    a3 += temp;

    /* Add row[6] contributions */
    a0 += (unsigned)(row[6] * W6);
    a1 -= (unsigned)(row[6] * W2);
    a2 += (unsigned)(row[6] * W2);
    a3 -= (unsigned)(row[6] * W6);

    /* Odd part */
    unsigned b0 = (unsigned)(row[1] * W1);
    unsigned b1 = (unsigned)(row[1] * W3);
    unsigned b2 = (unsigned)(row[1] * W5);
    unsigned b3 = (unsigned)(row[1] * W7);

    b0 += (unsigned)(row[3] * W3);
    b1 -= (unsigned)(row[3] * W7);
    b2 -= (unsigned)(row[3] * W1);
    b3 -= (unsigned)(row[3] * W5);

    b0 += (unsigned)(row[5] * W5);
    b1 -= (unsigned)(row[5] * W1);
    b2 += (unsigned)(row[5] * W7);
    b3 += (unsigned)(row[5] * W3);

    b0 += (unsigned)(row[7] * W7);
    b1 -= (unsigned)(row[7] * W5);
    b2 += (unsigned)(row[7] * W3);
    b3 -= (unsigned)(row[7] * W1);

    /* Output — rounding already folded into a0-a3 */
    row[0] = (int)(a0 + b0) >> ROW_SHIFT;
    row[1] = (int)(a1 + b1) >> ROW_SHIFT;
    row[2] = (int)(a2 + b2) >> ROW_SHIFT;
    row[3] = (int)(a3 + b3) >> ROW_SHIFT;
    row[4] = (int)(a3 - b3) >> ROW_SHIFT;
    row[5] = (int)(a2 - b2) >> ROW_SHIFT;
    row[6] = (int)(a1 - b1) >> ROW_SHIFT;
    row[7] = (int)(a0 - b0) >> ROW_SHIFT;
}

static void idct_col(int32_t *block, int col) {
    int32_t *c = block + col; /* stride = 8 */

    /* No DC-only shortcut — FFmpeg's idctSparseCol always runs full butterfly. */

    /* FFmpeg-compatible factored column butterfly.
     * Column rounding: bias +2 on DC input before W4 multiply.
     * (1 << (COL_SHIFT-1)) / W4 = 65536 / 32767 = 2 */

    /* Even part: DC with rounding bias */
    unsigned a0 = (unsigned)((c[0*8] + ((1 << (COL_SHIFT - 1)) / W4)) * W4);
    unsigned a1 = a0;
    unsigned a2 = a0;
    unsigned a3 = a0;

    /* Add col[2] contributions */
    a0 += (unsigned)(c[2*8] * W2);
    a1 += (unsigned)(c[2*8] * W6);
    a2 -= (unsigned)(c[2*8] * W6);
    a3 -= (unsigned)(c[2*8] * W2);

    /* Add col[4] contributions */
    unsigned temp = (unsigned)(c[4*8] * W4);
    a0 += temp;
    a1 -= temp;
    a2 -= temp;
    a3 += temp;

    /* Add col[6] contributions */
    a0 += (unsigned)(c[6*8] * W6);
    a1 -= (unsigned)(c[6*8] * W2);
    a2 += (unsigned)(c[6*8] * W2);
    a3 -= (unsigned)(c[6*8] * W6);

    /* Odd part */
    unsigned b0 = (unsigned)(c[1*8] * W1);
    unsigned b1 = (unsigned)(c[1*8] * W3);
    unsigned b2 = (unsigned)(c[1*8] * W5);
    unsigned b3 = (unsigned)(c[1*8] * W7);

    b0 += (unsigned)(c[3*8] * W3);
    b1 -= (unsigned)(c[3*8] * W7);
    b2 -= (unsigned)(c[3*8] * W1);
    b3 -= (unsigned)(c[3*8] * W5);

    b0 += (unsigned)(c[5*8] * W5);
    b1 -= (unsigned)(c[5*8] * W1);
    b2 += (unsigned)(c[5*8] * W7);
    b3 += (unsigned)(c[5*8] * W3);

    b0 += (unsigned)(c[7*8] * W7);
    b1 -= (unsigned)(c[7*8] * W5);
    b2 += (unsigned)(c[7*8] * W3);
    b3 -= (unsigned)(c[7*8] * W1);

    /* Output — rounding already in DC bias */
    c[0*8] = (int)(a0 + b0) >> COL_SHIFT;
    c[1*8] = (int)(a1 + b1) >> COL_SHIFT;
    c[2*8] = (int)(a2 + b2) >> COL_SHIFT;
    c[3*8] = (int)(a3 + b3) >> COL_SHIFT;
    c[4*8] = (int)(a3 - b3) >> COL_SHIFT;
    c[5*8] = (int)(a2 - b2) >> COL_SHIFT;
    c[6*8] = (int)(a1 - b1) >> COL_SHIFT;
    c[7*8] = (int)(a0 - b0) >> COL_SHIFT;
}

static void idct_8x8(int32_t *block) {
    /* Row pass */
    for (int y = 0; y < 8; y++)
        idct_row(&block[y * 8]);

    /* DC bias: add 8192 to row 0 (DC coefficient row) before column pass.
     * Matches FFmpeg's ff_prores_idct_12 which injects the DC offset into
     * the column butterfly for matching rounding behavior.
     * 8192 * W4 / 2^17 ≈ 2048 per output pixel. */
    for (int x = 0; x < 8; x++)
        block[x] += 8192;

    /* Column pass */
    for (int x = 0; x < 8; x++)
        idct_col(block, x);
}

/* ---- Dequantization ---- */

static void dequantize_block(int32_t *block, const int32_t *qmat) {
#ifdef __ARM_NEON__
    /* 64 multiplies in 16 NEON ops (4 elements per op) */
    for (int i = 0; i < 64; i += 4) {
        int32x4_t v = vld1q_s32(&block[i]);
        int32x4_t q = vld1q_s32(&qmat[i]);
        vst1q_s32(&block[i], vmulq_s32(v, q));
    }
#else
    for (int i = 0; i < 64; i++)
        block[i] *= qmat[i];
#endif
}

/* ---- Bayer Reconstruction (inverse of extract_bayer_block) ---- */

static void place_bayer_block(const int32_t *block, uint16_t *dst,
                              int dst_stride, int component) {
    int row_offset = (component >> 1) & 1;
    int col_offset = component & 1;

#ifdef __ARM_NEON__
    int32x4_t vmin = vdupq_n_s32(4);
    int32x4_t vmax = vdupq_n_s32(4091);
    for (int y = 0; y < 8; y++) {
        int dst_y = (y * 2) + row_offset;
        /* Process 8 pixels in 2 NEON ops */
        int32x4_t v0 = vld1q_s32(&block[y * 8]);
        int32x4_t v1 = vld1q_s32(&block[y * 8 + 4]);
        v0 = vmaxq_s32(vminq_s32(v0, vmax), vmin);
        v1 = vmaxq_s32(vminq_s32(v1, vmax), vmin);
        /* Shift left by 4 (12-bit → 16-bit) */
        v0 = vshlq_n_s32(v0, 4);
        v1 = vshlq_n_s32(v1, 4);
        /* Narrow to uint16 and scatter to Bayer positions */
        uint16x4_t h0 = vmovn_u32(vreinterpretq_u32_s32(v0));
        uint16x4_t h1 = vmovn_u32(vreinterpretq_u32_s32(v1));
        /* Scatter with stride 2 to Bayer grid */
        uint16_t *row_dst = &dst[dst_y * dst_stride + col_offset];
        row_dst[0*2] = vget_lane_u16(h0, 0);
        row_dst[1*2] = vget_lane_u16(h0, 1);
        row_dst[2*2] = vget_lane_u16(h0, 2);
        row_dst[3*2] = vget_lane_u16(h0, 3);
        row_dst[4*2] = vget_lane_u16(h1, 0);
        row_dst[5*2] = vget_lane_u16(h1, 1);
        row_dst[6*2] = vget_lane_u16(h1, 2);
        row_dst[7*2] = vget_lane_u16(h1, 3);
    }
#else
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int dst_y = (y * 2) + row_offset;
            int dst_x = (x * 2) + col_offset;
            /* iDCT output is 12-bit. Clamp to [4, 4091] matching FFmpeg's
             * CLIP_MIN/CLIP_MAX_12 for ProRes RAW Bayer output. */
            int32_t val = block[y * 8 + x];
            if (val < 4) val = 4;
            if (val > 4091) val = 4091;
            dst[dst_y * dst_stride + dst_x] = (uint16_t)(val << 4);
        }
    }
#endif
}

/* ---- Component Decoder ---- */

static int decode_component(const uint8_t *data, int size,
                            uint16_t *bayer_out, int bayer_stride,
                            int tile_x, int tile_y,
                            int full_width, int full_height,
                            int component, const int32_t *qmat) {
    int comp_tile_w = MIN(TILE_WIDTH, full_width - tile_x);
    int comp_tile_h = MIN(TILE_HEIGHT, full_height - tile_y);
    int tile_width = comp_tile_w / 2;
    int tile_height = comp_tile_h / 2;
    int blocks_per_row = tile_width / 8;
    int rows = tile_height / 8;
    int nb_blocks = blocks_per_row * rows;

    if (nb_blocks <= 0) return 0;

    int32_t blocks[16][64];
    memset(blocks, 0, sizeof(blocks));

    /* Entropy decode */
    BitReader br;
    br_init(&br, data, size);

    decode_dc_coeffs(&br, blocks, nb_blocks);
    decode_ac_coeffs(&br, blocks, nb_blocks);

    /* Dequantize + iDCT + place into Bayer output */
    for (int b = 0; b < nb_blocks && b < 16; b++) {
        dequantize_block(blocks[b], qmat);
        idct_8x8(blocks[b]);

        int br_idx = b / blocks_per_row;
        int bc = b % blocks_per_row;
        int block_x = tile_x + (bc * 16);
        int block_y = tile_y + (br_idx * 16);

        if (block_x + 16 <= full_width && block_y + 16 <= full_height)
            place_bayer_block(blocks[b], bayer_out + block_y * bayer_stride + block_x,
                              bayer_stride, component);
    }

    return 0;
}

/* ---- Tile Decoder ---- */

static uint16_t read_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | (uint16_t)p[1];
}

static int decode_tile(const uint8_t *tile_data, int tile_size,
                       uint16_t *bayer_out, int bayer_stride,
                       int tile_x, int tile_y,
                       int full_width, int full_height,
                       const uint8_t *frame_qmat) {
    if (tile_size < 8) return -1;

    /* Tile header: 8 bytes */
    /* [0] = header_len marker (0x40 = 8 bytes) */
    /* [1] = quantizer scale */
    /* [2-3] = component 2 size (BE16) */
    /* [4-5] = component 1 size (BE16) */
    /* [6-7] = component 3 size (BE16) */
    int scale = tile_data[1];

    int comp2_size = read_be16(tile_data + 2);
    int comp1_size = read_be16(tile_data + 4);
    int comp3_size = read_be16(tile_data + 6);
    int comp0_size = tile_size - 8 - comp2_size - comp1_size - comp3_size;

    if (comp0_size < 0) return -1;

    /* Build tile quantization matrix:
     * tile_qmat[i] = (frame_qmat[i] * scale) >> 1 */
    int32_t qmat[64];
    for (int i = 0; i < 64; i++)
        qmat[i] = ((int32_t)frame_qmat[i] * scale) >> 1;

    /* Component data follows header in order: 2, 1, 3, 0 */
    const uint8_t *comp_data[4];
    int comp_sizes[4];

    const uint8_t *ptr = tile_data + 8;
    comp_data[2] = ptr; comp_sizes[2] = comp2_size; ptr += comp2_size;
    comp_data[1] = ptr; comp_sizes[1] = comp1_size; ptr += comp1_size;
    comp_data[3] = ptr; comp_sizes[3] = comp3_size; ptr += comp3_size;
    comp_data[0] = ptr; comp_sizes[0] = comp0_size;

    /* Decode each component */
    for (int c = 0; c < 4; c++) {
        if (comp_sizes[c] > 0) {
            decode_component(comp_data[c], comp_sizes[c],
                             bayer_out, bayer_stride,
                             tile_x, tile_y,
                             full_width, full_height,
                             c, qmat);
        }
    }

    return 0;
}

/* ---- Frame Decoder ---- */

int prores_raw_decode_frame(const uint8_t *frame_data, int frame_size,
                            uint16_t *bayer_out, int width, int height) {
    if (frame_size < 96) return -1;
    if (width < 2 || height < 2 || width > 16384 || height > 16384) return -1;
    if (width % 2 != 0 || height % 2 != 0) return -1;

    /* Frame header: size(4) + 'prrf'(4) + header_len(2) + payload(86) = 96 bytes */
    if (memcmp(frame_data + 4, "prrf", 4) != 0) {
        fprintf(stderr, "prores_raw_dec: not a prrf frame\n");
        return -1;
    }

    int header_len = read_be16(frame_data + 8);
    int payload_offset = 10; /* after size(4) + marker(4) + header_len(2) */

    /* Extract quantization matrix from frame header if present.
     * flags byte at payload[0]: bit 0x02 = qmat present.
     * qmat location depends on header content — for simplicity,
     * use all-1s default (matches encoder with default_qmat). */
    uint8_t frame_qmat[64];
    memset(frame_qmat, 1, 64);

    /* Check if custom qmat is embedded in header */
    uint8_t flags = frame_data[payload_offset];
    if (flags & 0x02) {
        /* qmat present at end of header payload (last 64 bytes before tile data) */
        int qmat_offset = 10 + header_len - 86 + 86 - 64;
        if (qmat_offset >= 10 && qmat_offset + 64 <= 10 + header_len - 86 + 86) {
            /* Copy from header — each byte is one qmat entry */
            /* For now, keep default. Camera files typically don't embed qmat. */
        }
    }

    /* Tile layout */
    int nb_tw = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int nb_th = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int nb_tiles = nb_tw * nb_th;

    /* Tile size table: nb_tiles × 2 bytes (BE16), right after frame header.
     * Actually, header_len field value is the total header size including the 2-byte field itself.
     * Frame layout: [4:frame_size][4:'prrf'][2:header_len][header_len-2 bytes payload]
     * So data after header starts at offset 8 + header_len */
    int data_start = 8 + header_len;
    if (data_start + nb_tiles * 2 > frame_size) {
        fprintf(stderr, "prores_raw_dec: frame too small for tile table (%d tiles, data_start=%d, frame_size=%d)\n",
                nb_tiles, data_start, frame_size);
        return -1;
    }

    /* Read tile sizes */
    const uint8_t *tile_table = frame_data + data_start;
    int tile_data_offset = data_start + nb_tiles * 2;

    /* Precompute per-tile offsets (serial scan — offsets are cumulative) */
    int *tile_offsets = (int *)malloc(nb_tiles * sizeof(int));
    int *tile_sizes   = (int *)malloc(nb_tiles * sizeof(int));
    if (!tile_offsets || !tile_sizes) {
        free(tile_offsets); free(tile_sizes);
        return -1;
    }

    int64_t offset = tile_data_offset;
    for (int t = 0; t < nb_tiles; t++) {
        tile_sizes[t] = read_be16(tile_table + t * 2);
        tile_offsets[t] = (int)offset;
        if (offset + tile_sizes[t] > frame_size) {
            fprintf(stderr, "prores_raw_dec: tile %d overflows frame (offset=%lld, size=%d, frame=%d)\n",
                    t, (long long)offset, tile_sizes[t], frame_size);
            free(tile_offsets); free(tile_sizes);
            return -1;
        }
        offset += tile_sizes[t];
    }

    /* Decode tiles in parallel using GCD */
    __block int decode_error = 0;
    const uint8_t *qmat_ptr = frame_qmat;  /* block-capturable pointer */
    dispatch_apply((size_t)nb_tiles, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t t) {
        if (decode_error) return;  /* early out if another tile failed */

        int ty = (int)t / nb_tw;
        int tx = (int)t % nb_tw;
        int tile_x = tx * TILE_WIDTH;
        int tile_y = ty * TILE_HEIGHT;

        if (decode_tile(frame_data + tile_offsets[t], tile_sizes[t],
                        bayer_out, width,
                        tile_x, tile_y,
                        width, height,
                        qmat_ptr) != 0) {
            decode_error = 1;
        }
    });

    free(tile_offsets);
    free(tile_sizes);

    if (decode_error) {
        fprintf(stderr, "prores_raw_dec: parallel tile decode failed\n");
        return -1;
    }

    return 0;
}
