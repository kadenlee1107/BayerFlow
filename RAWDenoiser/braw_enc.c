/*
 * Blackmagic RAW (BRAW) Encoder
 *
 * Encodes 12-bit Bayer RGGB frames to BRAW-compressed packets.
 * Exact inverse of braw_dec.c — matched DCT, VLC tables, quant.
 */

#include "../include/braw_enc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ========================================================================
 * Constants — same as braw_dec.c
 * ======================================================================== */

/* 12-bit iDCT W constants */
#define W1  45451
#define W2  42813
#define W3  38531
#define W4  32767
#define W5  25746
#define W6  17734
#define W7   9041

#define ROW_SHIFT 16
#define COL_SHIFT 17

/* DC bias added during iDCT */
#define LUMA_DC_ADD   16384
#define CHROMA_DC_ADD  8192

/* DC Huffman table: 16 symbols */
static const uint8_t dc_bits[16] = {
    2, 3, 3, 3, 3, 3, 5, 5, 6, 12, 12, 12, 12, 5, 12, 13,
};
static const uint16_t dc_codes[16] = {
    0x0, 0x2, 0x3, 0x4, 0x5, 0x6, 0x1C, 0x1D, 0x3E,
    0xFF4, 0xFF5, 0xFF7, 0xFED, 0x1E, 0xFFE, 0x1FFE,
};
static const uint8_t dc_table[16][3] = {
    {  0, 0, 15 }, {  0, 1,  0 }, {  1, 1,  1 }, {  2, 1,  2 },
    {  3, 1,  3 }, {  4, 1,  4 }, {  5, 1,  5 }, {  6, 1,  6 },
    {  7, 1,  7 }, {  8, 1,  8 }, {  9, 1,  9 }, { 10, 1, 10 },
    { 11, 1, 11 }, { 12, 1, 12 }, { 13, 1, 13 }, { 14, 1, 14 },
};
static const int16_t dcval_tab[2][16] = {
    {    -1,     -3,     -7,    -15,
        -31,    -63,   -127,   -255,
       -511,  -1023,  -2047,  -4095,
      -8191, -16383, -32767,      0 },
    {     1,      2,      4,      8,
         16,     32,     64,    128,
        256,    512,   1024,   2048,
       4096,   8192,  16384,      0 },
};

/* AC Huffman table: 194 symbols */
static const uint8_t ac_bits_table[194] = {
    2, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10,
    10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 15, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
};
static const uint32_t ac_codes_table[194] = {
    0x000000, 0x000001, 0x000004, 0x00000B, 0x00000C, 0x00000A,
    0x00001A, 0x00001B, 0x00001C, 0x00003A, 0x00003B, 0x000078,
    0x000079, 0x00007A, 0x00007B, 0x0000F8, 0x0000F9, 0x0000FA,
    0x0001F6, 0x0001F7, 0x0001F8, 0x0001F9, 0x0001FA, 0x0003F6,
    0x0003F7, 0x0003F8, 0x0003F9, 0x0003FA, 0x0007F8, 0x0007F9,
    0x000FED, 0x000FF4, 0x000FF5, 0x000FF7, 0x000FEC, 0x000FF6,
    0x001FDC, 0x001FDD, 0x001FDE, 0x001FDF, 0x007FC0, 0x00FF84,
    0x00FF85, 0x00FF86, 0x00FF87, 0x00FF88, 0x00FF89, 0x00FF8A,
    0x00FF8B, 0x00FF8C, 0x00FF8D, 0x00FF8E, 0x00FF8F, 0x00FF90,
    0x00FF91, 0x00FF92, 0x00FF93, 0x00FF94, 0x00FF95, 0x00FF96,
    0x00FF97, 0x00FF98, 0x00FF99, 0x00FF9A, 0x00FF9B, 0x00FF9C,
    0x00FF9D, 0x00FF9E, 0x00FF9F, 0x00FFA0, 0x00FFA1, 0x00FFA2,
    0x00FFA3, 0x00FFA4, 0x00FFA5, 0x00FFA6, 0x00FFA7, 0x00FFA8,
    0x00FFA9, 0x00FFAA, 0x00FFAB, 0x00FFAC, 0x00FFAE, 0x00FFAF,
    0x00FFB0, 0x00FFB1, 0x00FFB2, 0x00FFB3, 0x00FFB4, 0x00FFB6,
    0x00FFB7, 0x00FFB8, 0x00FFB9, 0x00FFBA, 0x00FFBB, 0x00FFBC,
    0x00FFBE, 0x00FFBF, 0x00FFC0, 0x00FFC1, 0x00FFC2, 0x00FFC3,
    0x00FFC4, 0x00FFC5, 0x00FFC7, 0x00FFC8, 0x00FFC9, 0x00FFCA,
    0x00FFCB, 0x00FFCC, 0x00FFCD, 0x00FFCE, 0x00FFD0, 0x00FFD1,
    0x00FFD2, 0x00FFD3, 0x00FFD4, 0x00FFD5, 0x00FFD6, 0x00FFD7,
    0x00FFD9, 0x00FFDA, 0x00FFDB, 0x00FFDC, 0x00FFDD, 0x00FFDE,
    0x00FFDF, 0x00FFE0, 0x00FFE2, 0x00FFE3, 0x00FFE4, 0x00FFE5,
    0x00FFE6, 0x00FFE7, 0x00FFE8, 0x00FFE9, 0x00FFEB, 0x00FFEC,
    0x00FFED, 0x00FFEE, 0x00FFEF, 0x00FFF0, 0x00FFF1, 0x00FFF2,
    0x00FFF3, 0x00FFF5, 0x00FFF6, 0x00FFF7, 0x00FFF8, 0x00FFF9,
    0x00FFFA, 0x00FFFB, 0x00FFFC, 0x00FFFD, 0x03FEB5, 0x03FEB6,
    0x03FEB7, 0x03FED5, 0x03FED6, 0x03FED7, 0x03FEF5, 0x03FEF6,
    0x03FEF7, 0x03FF19, 0x03FEB4, 0x03FF1A, 0x03FF1B, 0x03FED4,
    0x03FF3D, 0x03FF3E, 0x03FEF4, 0x03FF3F, 0x03FF61, 0x03FF18,
    0x03FF62, 0x03FF63, 0x03FF3C, 0x03FF85, 0x03FF86, 0x03FF60,
    0x03FF87, 0x03FFA9, 0x03FF84, 0x03FFAA, 0x03FFAB, 0x03FFA8,
    0x03FFD1, 0x03FFD2, 0x03FFD0, 0x03FFD3, 0x03FFF9, 0x03FFF8,
    0x03FFFA, 0x03FFFB,
};
static const uint8_t ac_table[194][2] = {
    { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 }, { 1, 1 }, { 255, 0 },
    { 0, 5 }, { 1, 2 }, { 2, 1 }, { 3, 1 }, { 4, 1 }, { 0, 6 },
    { 1, 3 }, { 5, 1 }, { 6, 1 }, { 0, 7 }, { 2, 2 }, { 7, 1 },
    { 1, 4 }, { 3, 2 }, { 8, 1 }, { 9, 1 }, { 10, 1 }, { 0, 8 },
    { 2, 3 }, { 4, 2 }, { 11, 1 }, { 12, 1 }, { 13, 1 }, { 15, 0 },
    { 0, 12 },{ 0, 9 }, { 0, 10 }, { 0, 11 }, { 1, 5 }, { 6, 2 },
    { 2, 4 }, { 3, 3 }, { 5, 2 }, { 7, 2 }, { 8, 2 }, { 1, 6 },
    { 1, 7 }, { 1, 8 }, { 1, 9 }, { 1, 10 }, { 2, 5 }, { 2, 6 },
    { 2, 7 }, { 2, 8 }, { 2, 9 }, { 2, 10 }, { 3, 4 }, { 3, 5 },
    { 3, 6 }, { 3, 7 }, { 3, 8 }, { 3, 9 }, { 3, 10 }, { 4, 3 },
    { 4, 4 }, { 4, 5 }, { 4, 6 }, { 4, 7 }, { 4, 8 }, { 4, 9 },
    { 4, 10 }, { 5, 3 }, { 5, 4 }, { 5, 5 }, { 5, 6 }, { 5, 7 },
    { 5, 8 }, { 5, 9 }, { 5, 10 }, { 6, 3 }, { 6, 4 }, { 6, 5 },
    { 6, 6 }, { 6, 7 }, { 6, 8 }, { 6, 9 }, { 7, 3 }, { 7, 4 },
    { 7, 5 }, { 7, 6 }, { 7, 7 }, { 7, 8 }, { 7, 9 }, { 8, 3 },
    { 8, 4 }, { 8, 5 }, { 8, 6 }, { 8, 7 }, { 8, 8 }, { 8, 9 },
    { 9, 2 }, { 9, 3 }, { 9, 4 }, { 9, 5 }, { 9, 6 }, { 9, 7 },
    { 9, 8 }, { 9, 9 }, { 10, 2 }, { 10, 3 }, { 10, 4 }, { 10, 5 },
    { 10, 6 }, { 10, 7 }, { 10, 8 }, { 10, 9 }, { 11, 2 }, { 11, 3 },
    { 11, 4 }, { 11, 5 }, { 11, 6 }, { 11, 7 }, { 11, 8 }, { 11, 9 },
    { 12, 2 }, { 12, 3 }, { 12, 4 }, { 12, 5 }, { 12, 6 }, { 12, 7 },
    { 12, 8 }, { 12, 9 }, { 13, 2 }, { 13, 3 }, { 13, 4 }, { 13, 5 },
    { 13, 6 }, { 13, 7 }, { 13, 8 }, { 13, 9 }, { 14, 1 }, { 14, 2 },
    { 14, 3 }, { 14, 4 }, { 14, 5 }, { 14, 6 }, { 14, 7 }, { 14, 8 },
    { 14, 9 }, { 15, 1 }, { 15, 2 }, { 15, 3 }, { 15, 4 }, { 15, 5 },
    { 15, 6 }, { 15, 7 }, { 15, 8 }, { 15, 9 }, { 1, 11 }, { 1, 12 },
    { 2, 11 }, { 2, 12 }, { 3, 11 }, { 3, 12 }, { 4, 11 }, { 4, 12 },
    { 5, 11 }, { 5, 12 }, { 6, 10 }, { 6, 11 }, { 6, 12 }, { 7, 10 },
    { 7, 11 }, { 7, 12 }, { 8, 10 }, { 8, 11 }, { 8, 12 }, { 9, 10 },
    { 9, 11 }, { 9, 12 }, { 10, 10 }, { 10, 11 }, { 10, 12 }, { 11, 10 },
    { 11, 11 }, { 11, 12 }, { 12, 10 }, { 12, 11 }, { 12, 12 }, { 13, 10 },
    { 13, 11 }, { 13, 12 }, { 14, 10 }, { 14, 11 }, { 14, 12 }, { 15, 10 },
    { 15, 11 }, { 15, 12 },
};

/* 8x4 chroma scan order */
static const uint8_t half_scan[32] = {
     0,  1,  2,  3,  8,  9, 16, 17, 10, 11,  4,  5,  6,  7, 12, 13,
    18, 19, 24, 25, 26, 27, 20, 21, 14, 15, 22, 23, 28, 29, 30, 31,
};

/* Standard 8x8 zigzag scan */
static const uint8_t zigzag_direct[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

/* Reverse lookup: ac_reverse[run][level_bits] = symbol index, -1 if none */
static int ac_reverse[16][13];
static int ac_eob_sym = -1;   /* EOB symbol index */
static int ac_zrl_sym = -1;   /* ZRL symbol index */
static int ac_reverse_built = 0;

static void build_ac_reverse(void) {
    if (ac_reverse_built) return;
    memset(ac_reverse, -1, sizeof(ac_reverse));
    for (int sym = 0; sym < 194; sym++) {
        int run = ac_table[sym][0];
        int lbits = ac_table[sym][1];
        if (run == 255 && lbits == 0) { ac_eob_sym = sym; continue; }
        if (run == 15 && lbits == 0)  { ac_zrl_sym = sym; continue; }
        if (run < 16 && lbits < 13 && ac_reverse[run][lbits] < 0)
            ac_reverse[run][lbits] = sym;  /* use FIRST occurrence (primary codes) */
    }
    ac_reverse_built = 1;
}

/* ========================================================================
 * MSB-first BitWriter
 * ======================================================================== */

typedef struct {
    uint8_t *data;
    int capacity;      /* bytes */
    int byte_pos;
    int bit_pos;       /* bits used in current byte (0..7) */
    uint8_t cur_byte;
} BitWriter;

static inline void bw_init(BitWriter *bw, uint8_t *buf, int capacity) {
    bw->data = buf;
    bw->capacity = capacity;
    bw->byte_pos = 0;
    bw->bit_pos = 0;
    bw->cur_byte = 0;
}

static inline void bw_write_bits(BitWriter *bw, uint32_t val, int nbits) {
    for (int i = nbits - 1; i >= 0; i--) {
        bw->cur_byte = (bw->cur_byte << 1) | ((val >> i) & 1);
        bw->bit_pos++;
        if (bw->bit_pos == 8) {
            if (bw->byte_pos < bw->capacity)
                bw->data[bw->byte_pos++] = bw->cur_byte;
            bw->cur_byte = 0;
            bw->bit_pos = 0;
        }
    }
}

/* Flush partial byte (pad with 1s to match original BRAW) */
static inline void bw_flush(BitWriter *bw) {
    if (bw->bit_pos > 0) {
        int pad = 8 - bw->bit_pos;
        bw->cur_byte = (bw->cur_byte << pad) | ((1 << pad) - 1);
        if (bw->byte_pos < bw->capacity)
            bw->data[bw->byte_pos++] = bw->cur_byte;
        bw->cur_byte = 0;
        bw->bit_pos = 0;
    }
}

/* Write exactly 8 one-bits of padding (original BRAW uses 0xFF between block groups) */
static inline void bw_pad8(BitWriter *bw) {
    bw_write_bits(bw, 0xFF, 8);
}

static inline int bw_bytes_written(const BitWriter *bw) {
    return bw->byte_pos + (bw->bit_pos > 0 ? 1 : 0);
}

/* ========================================================================
 * Forward DCT — matched to braw_dec.c iDCT
 * ======================================================================== */

static void build_fwd_8x8_matrices(double col[8][8], double row[8][8]) {
    /* Build the 8x8 iDCT butterfly matrix M[output][input] */
    double M[8][8];

    double even_coeffs[4][4] = {
        { W4,  W2,  W4,  W6},
        { W4,  W6, -W4, -W2},
        { W4, -W6, -W4,  W2},
        { W4, -W2,  W4, -W6},
    };
    double odd_coeffs[4][4] = {
        { W1,  W3,  W5,  W7},
        { W3, -W7, -W1, -W5},
        { W5, -W1,  W7,  W3},
        { W7, -W5,  W3, -W1},
    };

    int ei[] = {0, 1, 2, 3, 3, 2, 1, 0};
    int sign[] = {1, 1, 1, 1, -1, -1, -1, -1};

    for (int k = 0; k < 8; k++) {
        M[k][0] = even_coeffs[ei[k]][0];
        M[k][2] = even_coeffs[ei[k]][1];
        M[k][4] = even_coeffs[ei[k]][2];
        M[k][6] = even_coeffs[ei[k]][3];
        M[k][1] = sign[k] * odd_coeffs[ei[k]][0];
        M[k][3] = sign[k] * odd_coeffs[ei[k]][1];
        M[k][5] = sign[k] * odd_coeffs[ei[k]][2];
        M[k][7] = sign[k] * odd_coeffs[ei[k]][3];
    }

    /* Column norms D[j] = sum_k M[k][j]^2 */
    double D[8];
    for (int j = 0; j < 8; j++) {
        D[j] = 0;
        for (int k = 0; k < 8; k++)
            D[j] += M[k][j] * M[k][j];
    }

    /* Forward = M^T / D * 2^shift */
    double col_scale = (double)(1 << COL_SHIFT);
    double row_scale = (double)(1 << ROW_SHIFT);

    for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
            col[j][k] = M[k][j] / D[j] * col_scale;
            row[j][k] = M[k][j] / D[j] * row_scale;
        }
    }
}

static void build_fwd_4pt_col_matrix(double fwd4[4][4]) {
    /*
     * The 4-point iDCT from braw_dec.c (idct4_col_put_12bit):
     *   a0 = in[0] * X2       (X2 = 32768)
     *   a2 = in[2] * X2
     *   a1 = in[1]            (NOT multiplied by X2)
     *   a3 = in[3]
     *   c0 = a0 + a2
     *   c1 = a0 - a2
     *   c2 = a1*X0 - a3*X1   (X0=17734, X1=42813)
     *   c3 = a3*X0 + a1*X1
     *   out[k] = (ck +/- ...) >> 16
     *
     * Matrix M4[out][in]:
     *   M4[0] = [X2,  X1,  X2,  X0]
     *   M4[1] = [X2,  X0, -X2, -X1]
     *   M4[2] = [X2, -X0, -X2,  X1]
     *   M4[3] = [X2, -X1,  X2, -X0]
     */
    const double X0 = 17734.0;
    const double X1 = 42813.0;
    const double X2 = 32768.0;

    double M4[4][4] = {
        { X2,  X1,  X2,  X0},
        { X2,  X0, -X2, -X1},
        { X2, -X0, -X2,  X1},
        { X2, -X1,  X2, -X0},
    };

    /* Column norms */
    double D4[4];
    for (int j = 0; j < 4; j++) {
        D4[j] = 0;
        for (int k = 0; k < 4; k++)
            D4[j] += M4[k][j] * M4[k][j];
    }

    /* Forward = M4^T / D4 * 2^16 (the column shift is 16 for 4-point) */
    double scale = 65536.0;
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++)
            fwd4[j][k] = M4[k][j] / D4[j] * scale;
}

/* Forward 8x8 DCT: pixel block → DCT coefficients (no DC bias subtracted yet) */
static void forward_dct_8x8(const uint16_t *pixels, int stride,
                             int16_t *coeffs, const BrawEncContext *ctx) {
    double temp[64];

    /* Load pixels, column-first forward DCT */
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            temp[y * 8 + x] = (double)pixels[y * stride + x];

    /* Column pass */
    for (int x = 0; x < 8; x++) {
        double inp[8], out[8];
        for (int i = 0; i < 8; i++) inp[i] = temp[i * 8 + x];
        for (int j = 0; j < 8; j++) {
            out[j] = 0;
            for (int k = 0; k < 8; k++)
                out[j] += ctx->fwd_col_8[j][k] * inp[k];
        }
        for (int i = 0; i < 8; i++) temp[i * 8 + x] = out[i];
    }

    /* Row pass */
    for (int y = 0; y < 8; y++) {
        double inp[8], out[8];
        for (int i = 0; i < 8; i++) inp[i] = temp[y * 8 + i];
        for (int j = 0; j < 8; j++) {
            out[j] = 0;
            for (int k = 0; k < 8; k++)
                out[j] += ctx->fwd_row_8[j][k] * inp[k];
        }
        for (int i = 0; i < 8; i++) temp[y * 8 + i] = out[i];
    }

    for (int i = 0; i < 64; i++)
        coeffs[i] = (int16_t)lrint(temp[i]);
}

/* Forward 8x4 DCT: 8 columns × 4 rows → 32 coefficients */
static void forward_dct_8x4(const uint16_t *pixels, int stride,
                             int16_t *coeffs, const BrawEncContext *ctx) {
    double temp[32]; /* 4 rows × 8 cols */

    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 8; x++)
            temp[y * 8 + x] = (double)pixels[y * stride + x];

    /* Column pass (4-point) */
    for (int x = 0; x < 8; x++) {
        double inp[4], out[4];
        for (int i = 0; i < 4; i++) inp[i] = temp[i * 8 + x];
        for (int j = 0; j < 4; j++) {
            out[j] = 0;
            for (int k = 0; k < 4; k++)
                out[j] += ctx->fwd_col_4[j][k] * inp[k];
        }
        for (int i = 0; i < 4; i++) temp[i * 8 + x] = out[i];
    }

    /* Row pass (8-point, same as luma) */
    for (int y = 0; y < 4; y++) {
        double inp[8], out[8];
        for (int i = 0; i < 8; i++) inp[i] = temp[y * 8 + i];
        for (int j = 0; j < 8; j++) {
            out[j] = 0;
            for (int k = 0; k < 8; k++)
                out[j] += ctx->fwd_row_8[j][k] * inp[k];
        }
        for (int i = 0; i < 8; i++) temp[y * 8 + i] = out[i];
    }

    for (int i = 0; i < 32; i++)
        coeffs[i] = (int16_t)lrint(temp[i]);
}

/* ========================================================================
 * iDCT for reconstruction (needed for correlation)
 * Copy from braw_dec.c to reconstruct luma after quantization
 * ======================================================================== */

static inline int clip_12bit(int v) {
    if (v < 0) return 0;
    if (v > 4095) return 4095;
    return v;
}

static void idct_row_12bit(int16_t *row) {
    int a0, a1, a2, a3, b0, b1, b2, b3;

    if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
        int val = ((int)W4 * row[0] + (1 << (ROW_SHIFT - 1))) >> ROW_SHIFT;
        row[0] = row[1] = row[2] = row[3] = val;
        row[4] = row[5] = row[6] = row[7] = val;
        return;
    }

    a0 = (int)W4 * row[0] + (1 << (ROW_SHIFT - 1));
    a1 = a0; a2 = a0; a3 = a0;
    a0 += (int)W2 * row[2]; a1 += (int)W6 * row[2];
    a2 -= (int)W6 * row[2]; a3 -= (int)W2 * row[2];
    b0 = (int)W1 * row[1] + (int)W3 * row[3];
    b1 = (int)W3 * row[1] - (int)W7 * row[3];
    b2 = (int)W5 * row[1] - (int)W1 * row[3];
    b3 = (int)W7 * row[1] - (int)W5 * row[3];
    if (row[4] | row[5] | row[6] | row[7]) {
        a0 += (int)W4 * row[4] + (int)W6 * row[6];
        a1 -= (int)W4 * row[4] + (int)W2 * row[6];
        a2 -= (int)W4 * row[4] - (int)W2 * row[6];
        a3 += (int)W4 * row[4] - (int)W6 * row[6];
        b0 += (int)W5 * row[5] + (int)W7 * row[7];
        b1 -= (int)W1 * row[5] + (int)W5 * row[7];
        b2 += (int)W7 * row[5] + (int)W3 * row[7];
        b3 += (int)W3 * row[5] - (int)W1 * row[7];
    }
    row[0] = (a0 + b0) >> ROW_SHIFT; row[7] = (a0 - b0) >> ROW_SHIFT;
    row[1] = (a1 + b1) >> ROW_SHIFT; row[6] = (a1 - b1) >> ROW_SHIFT;
    row[2] = (a2 + b2) >> ROW_SHIFT; row[5] = (a2 - b2) >> ROW_SHIFT;
    row[3] = (a3 + b3) >> ROW_SHIFT; row[4] = (a3 - b3) >> ROW_SHIFT;
}

static void idct_col_put_12bit(uint16_t *dest, int line_stride, const int16_t *in) {
    int a0, a1, a2, a3, b0, b1, b2, b3;
    a0 = (int)W4 * in[8*0] + (1 << (COL_SHIFT - 1));
    a1 = a0; a2 = a0; a3 = a0;
    a0 += (int)W2 * in[8*2]; a1 += (int)W6 * in[8*2];
    a2 -= (int)W6 * in[8*2]; a3 -= (int)W2 * in[8*2];
    b0 = (int)W1 * in[8*1] + (int)W3 * in[8*3];
    b1 = (int)W3 * in[8*1] - (int)W7 * in[8*3];
    b2 = (int)W5 * in[8*1] - (int)W1 * in[8*3];
    b3 = (int)W7 * in[8*1] - (int)W5 * in[8*3];
    if (in[8*4] | in[8*5] | in[8*6] | in[8*7]) {
        a0 += (int)W4 * in[8*4] + (int)W6 * in[8*6];
        a1 -= (int)W4 * in[8*4] + (int)W2 * in[8*6];
        a2 -= (int)W4 * in[8*4] - (int)W2 * in[8*6];
        a3 += (int)W4 * in[8*4] - (int)W6 * in[8*6];
        b0 += (int)W5 * in[8*5] + (int)W7 * in[8*7];
        b1 -= (int)W1 * in[8*5] + (int)W5 * in[8*7];
        b2 += (int)W7 * in[8*5] + (int)W3 * in[8*7];
        b3 += (int)W3 * in[8*5] - (int)W1 * in[8*7];
    }
    dest[0 * line_stride] = clip_12bit((a0 + b0) >> COL_SHIFT);
    dest[1 * line_stride] = clip_12bit((a1 + b1) >> COL_SHIFT);
    dest[2 * line_stride] = clip_12bit((a2 + b2) >> COL_SHIFT);
    dest[3 * line_stride] = clip_12bit((a3 + b3) >> COL_SHIFT);
    dest[4 * line_stride] = clip_12bit((a3 - b3) >> COL_SHIFT);
    dest[5 * line_stride] = clip_12bit((a2 - b2) >> COL_SHIFT);
    dest[6 * line_stride] = clip_12bit((a1 - b1) >> COL_SHIFT);
    dest[7 * line_stride] = clip_12bit((a0 - b0) >> COL_SHIFT);
}

static void recon_idct_8x8(int16_t *block, uint16_t *dest, int stride) {
    /* Row pass (in-place on block) */
    for (int i = 0; i < 8; i++)
        idct_row_12bit(block + i * 8);
    /* Column pass → dest */
    for (int i = 0; i < 8; i++)
        idct_col_put_12bit(dest + i, stride, block + i);
}

/* ========================================================================
 * Quantization
 * ======================================================================== */

/* Forward quantize: coeff → quantized value (truncate towards zero = dead zone) */
static inline int16_t fwd_quant(int16_t coeff, int quant) {
    if (quant == 0) return coeff;
    /* Decoder dequant: val = (coded * Q + 0x8000) >> 16
     * Forward quant: coded = trunc(coeff * 65536 / Q) (dead-zone, matches camera) */
    double qf = (double)coeff * 65536.0 / (double)quant;
    return (int16_t)(int)qf;  /* C truncates towards zero */
}

/* Dequantize: quantized value → reconstructed coefficient */
static inline int16_t dequant(int16_t coded, int quant) {
    return (int16_t)(((int64_t)coded * quant + 0x8000) >> 16);
}

/* ========================================================================
 * VLC encoding
 * ======================================================================== */

/* Encode DC coefficient (differential) */
static void encode_dc(BitWriter *bw, int dc_val, int *prev_dc) {
    int delta = dc_val - *prev_dc;
    *prev_dc = dc_val;

    if (delta == 0) {
        bw_write_bits(bw, dc_codes[0], dc_bits[0]);
        return;
    }

    /* dc_table[sym]: [0]=extra_len, [1]=sign_bits, [2]=dcval_tab_col
     * Decoder: sgnbit = br_read(sign_bits); val = code + prev_dc + dcval_tab[sgnbit][col]
     * dcval_tab[0] = negative bases, dcval_tab[1] = positive bases
     * So sgnbit=1 for positive delta, sgnbit=0 for negative delta */
    int sgnbit = (delta > 0) ? 1 : 0;

    for (int sym = 1; sym < 16; sym++) {
        int extra_len = dc_table[sym][0];
        int col = dc_table[sym][2];
        int base = dcval_tab[sgnbit][col];
        int code = delta - base;
        int max_code = (1 << extra_len) - 1;

        if (code >= 0 && code <= max_code) {
            bw_write_bits(bw, dc_codes[sym], dc_bits[sym]);
            bw_write_bits(bw, sgnbit, 1);
            if (extra_len > 0)
                bw_write_bits(bw, (uint32_t)code, extra_len);
            return;
        }
    }

    /* Shouldn't reach here for valid 12-bit data */
    bw_write_bits(bw, dc_codes[0], dc_bits[0]);
}

/* Number of bits needed to represent abs(val) */
static inline int bit_length(int val) {
    if (val == 0) return 0;
    int a = val < 0 ? -val : val;
    int bits = 0;
    while (a > 0) { a >>= 1; bits++; }
    return bits;
}

/* Encode AC coefficients in zigzag/half_scan order */
static void encode_ac(BitWriter *bw, const int16_t *block, int max_coeff,
                       const uint8_t *scan) {
    /* Find last non-zero coefficient */
    int last_nz = -1;
    for (int i = max_coeff - 1; i >= 1; i--) {
        if (block[scan[i]] != 0) { last_nz = i; break; }
    }

    if (last_nz < 0) {
        /* All AC = 0, write EOB */
        bw_write_bits(bw, ac_codes_table[ac_eob_sym], ac_bits_table[ac_eob_sym]);
        return;
    }

    int run = 0;
    for (int i = 1; i <= last_nz; i++) {
        int coeff = block[scan[i]];
        if (coeff == 0) {
            run++;
            continue;
        }

        /* Emit ZRL symbols for runs >= 16 */
        while (run >= 16) {
            bw_write_bits(bw, ac_codes_table[ac_zrl_sym], ac_bits_table[ac_zrl_sym]);
            run -= 16;
        }

        int abs_coeff = coeff < 0 ? -coeff : coeff;
        int lbits = bit_length(abs_coeff);

        /* Find AC symbol for (run, lbits) */
        int sym = -1;
        if (run < 16 && lbits < 13)
            sym = ac_reverse[run][lbits];

        if (sym < 0) {
            /* Fallback: encode as 12-bit with ZRL if needed */
            fprintf(stderr, "braw_enc: AC (run=%d, lbits=%d) not in table\n", run, lbits);
            bw_write_bits(bw, ac_codes_table[ac_eob_sym], ac_bits_table[ac_eob_sym]);
            return;
        }

        /* Write Huffman code */
        bw_write_bits(bw, ac_codes_table[sym], ac_bits_table[sym]);

        /* Write value with JPEG sign extension */
        if (lbits > 0) {
            uint32_t val_bits;
            if (coeff > 0)
                val_bits = (uint32_t)coeff;
            else
                val_bits = (uint32_t)(coeff + (1 << lbits) - 1);
            bw_write_bits(bw, val_bits, lbits);
        }

        run = 0;
    }

    /* Write EOB */
    bw_write_bits(bw, ac_codes_table[ac_eob_sym], ac_bits_table[ac_eob_sym]);
}

/* Encode one block: forward quant + DC encode + AC encode */
static void encode_block(BitWriter *bw, int16_t *coeffs, const int *quant,
                         int *prev_dc, int max_coeff, int dc_add,
                         const uint8_t *scan) {
    int16_t qblock[64];
    memset(qblock, 0, sizeof(qblock));

    /* DC: subtract bias, quantize */
    int dc_raw = coeffs[0] - dc_add;
    qblock[0] = fwd_quant((int16_t)dc_raw, quant[0]);

    /* AC: quantize in scan order positions */
    for (int i = 1; i < max_coeff; i++) {
        int pos = scan[i];
        if (coeffs[pos] != 0)
            qblock[pos] = fwd_quant(coeffs[pos], quant[pos]);
    }

    /* Encode DC (differential) */
    encode_dc(bw, qblock[0], prev_dc);

    /* Encode AC */
    encode_ac(bw, qblock, max_coeff, scan);
}

/* Encode block AND reconstruct via dequant+iDCT (for correlation).
 * Returns the quantized DC value written to bitstream. */
static void encode_block_with_recon(BitWriter *bw, int16_t *coeffs,
                                    const int *quant, int *prev_dc,
                                    int max_coeff, int dc_add,
                                    const uint8_t *scan,
                                    uint16_t *recon_dest, int recon_stride,
                                    int is_8x8) {
    int16_t qblock[64];
    memset(qblock, 0, sizeof(qblock));

    /* DC: subtract bias, quantize */
    int dc_raw = coeffs[0] - dc_add;
    qblock[0] = fwd_quant((int16_t)dc_raw, quant[0]);

    /* AC: quantize */
    for (int i = 1; i < max_coeff; i++) {
        int pos = scan[i];
        if (coeffs[pos] != 0)
            qblock[pos] = fwd_quant(coeffs[pos], quant[pos]);
    }

    /* Encode to bitstream */
    encode_dc(bw, qblock[0], prev_dc);
    encode_ac(bw, qblock, max_coeff, scan);

    /* Reconstruct: dequant + add DC bias + iDCT */
    int16_t recon_block[64];
    memset(recon_block, 0, sizeof(recon_block));
    recon_block[0] = dequant(qblock[0], quant[0]) + dc_add;
    for (int i = 1; i < max_coeff; i++) {
        int pos = scan[i];
        if (qblock[pos] != 0)
            recon_block[pos] = dequant(qblock[pos], quant[pos]);
    }

    if (is_8x8) {
        recon_idct_8x8(recon_block, recon_dest, recon_stride);
    }
    /* 8x4 reconstruction not needed (chroma doesn't need reconstruction) */
}

/* ========================================================================
 * Tile encoder
 * ======================================================================== */

/* Build green-interpolated 8x8 block for luma DCT.
 * Replaces R (even row, even col) and B (odd row, odd col) positions
 * with bilinear-interpolated green from neighboring Gr/Gb positions.
 * Gr and Gb positions keep their original values. */
static void fill_green_luma(uint16_t *dst, const uint16_t *frame,
                            int frame_stride, int frame_w, int frame_h,
                            int pos_y, int pos_x) {
    for (int r = 0; r < 8; r++) {
        int fy = pos_y + r;
        for (int c = 0; c < 8; c++) {
            int fx = pos_x + c;
            if ((r % 2 == 0 && c % 2 == 0) || (r % 2 == 1 && c % 2 == 1)) {
                /* R or B position: interpolate green from neighbors */
                int sum = 0, cnt = 0;
                if (fx > 0)          { sum += frame[fy * frame_stride + fx - 1]; cnt++; }
                if (fx < frame_w-1)  { sum += frame[fy * frame_stride + fx + 1]; cnt++; }
                if (fy > 0)          { sum += frame[(fy-1) * frame_stride + fx]; cnt++; }
                if (fy < frame_h-1)  { sum += frame[(fy+1) * frame_stride + fx]; cnt++; }
                dst[r * 8 + c] = (uint16_t)(cnt > 0 ? sum / cnt : frame[fy * frame_stride + fx]);
            } else {
                /* Gr or Gb position: keep original */
                dst[r * 8 + c] = frame[fy * frame_stride + fx];
            }
        }
    }
}

static int encode_tile(BrawEncContext *ctx, BitWriter *bw,
                       const uint16_t *frame, int frame_stride,
                       int tile_x, int tile_y,
                       int blocks_w, int blocks_h) {
    int prev_dc[3] = {0, 0, 0};
    int frame_w = ctx->config.width;
    int frame_h = ctx->config.height;

    for (int y = 0; y < blocks_h; y++) {
        int pos_y = y * 8 + tile_y * ctx->tile_size_h;

        for (int x = 0; x < blocks_w; x++) {
            int pos_x = ctx->tile_offsets_w[tile_x] + x * 16;

            /* ---- Luma block 0 (left 8 columns) ---- */
            /* Fill with green-interpolated values at R/B positions */
            fill_green_luma(ctx->luma_green, frame, frame_stride,
                           frame_w, frame_h, pos_y, pos_x);
            forward_dct_8x8(ctx->luma_green, 8, ctx->block[0], ctx);

            /* Encode luma block 0 with reconstruction */
            encode_block_with_recon(bw, ctx->block[0], ctx->quant_luma,
                                    &prev_dc[0], 64, LUMA_DC_ADD,
                                    zigzag_direct,
                                    ctx->recon_luma, 16, 1);

            /* ---- Luma block 1 (right 8 columns) ---- */
            fill_green_luma(ctx->luma_green, frame, frame_stride,
                           frame_w, frame_h, pos_y, pos_x + 8);
            forward_dct_8x8(ctx->luma_green, 8, ctx->block[1], ctx);

            encode_block_with_recon(bw, ctx->block[1], ctx->quant_luma,
                                    &prev_dc[0], 64, LUMA_DC_ADD,
                                    zigzag_direct,
                                    ctx->recon_luma + 8, 16, 1);

            /* ---- Compute chroma from correlation ---- */
            /* chroma1[y][x] = R_pixel - recon_luma_at_R_pos + 2048
             * chroma0[y][x] = B_pixel - recon_luma_at_B_pos + 2048
             *
             * R positions: even row, even col (dst0 in decoder)
             * B positions: odd row, odd col (dst1 in decoder)
             */
            for (int cy = 0; cy < 4; cy++) {
                for (int cx = 0; cx < 8; cx++) {
                    /* R position: row = cy*2, col = cx*2 */
                    int r_row = cy * 2;
                    int r_col = cx * 2;
                    int r_pixel = (int)frame[(pos_y + r_row) * frame_stride + pos_x + r_col];
                    int r_recon = (int)ctx->recon_luma[r_row * 16 + r_col];
                    int chroma1_val = r_pixel - r_recon + 2048;
                    if (chroma1_val < 0) chroma1_val = 0;
                    if (chroma1_val > 4095) chroma1_val = 4095;
                    ctx->chroma_buf[1][cy * 8 + cx] = (uint16_t)chroma1_val;

                    /* B position: row = cy*2+1, col = cx*2+1 */
                    int b_row = cy * 2 + 1;
                    int b_col = cx * 2 + 1;
                    int b_pixel = (int)frame[(pos_y + b_row) * frame_stride + pos_x + b_col];
                    int b_recon = (int)ctx->recon_luma[b_row * 16 + b_col];
                    int chroma0_val = b_pixel - b_recon + 2048;
                    if (chroma0_val < 0) chroma0_val = 0;
                    if (chroma0_val > 4095) chroma0_val = 4095;
                    ctx->chroma_buf[0][cy * 8 + cx] = (uint16_t)chroma0_val;
                }
            }

            /* ---- Chroma block 0 (B channel residual) ---- */
            forward_dct_8x4(ctx->chroma_buf[0], 8, ctx->block[2], ctx);
            encode_block(bw, ctx->block[2], ctx->quant_chroma,
                        &prev_dc[1], 32, CHROMA_DC_ADD, half_scan);

            /* ---- Chroma block 1 (R channel residual) ---- */
            forward_dct_8x4(ctx->chroma_buf[1], 8, ctx->block[3], ctx);
            encode_block(bw, ctx->block[3], ctx->quant_chroma,
                        &prev_dc[2], 32, CHROMA_DC_ADD, half_scan);

            /* 8-bit padding between block groups (decoder does br_skip(br, 8)) */
            bw_pad8(bw);
        }
    }

    return BRAW_ENC_OK;
}

/* ========================================================================
 * Byte-stream helpers
 * ======================================================================== */

static inline void put_be16(uint8_t *p, uint16_t v) {
    p[0] = (v >> 8) & 0xFF; p[1] = v & 0xFF;
}
static inline void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (v >> 24) & 0xFF; p[1] = (v >> 16) & 0xFF;
    p[2] = (v >> 8) & 0xFF;  p[3] = v & 0xFF;
}
static inline uint32_t get_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}
static inline uint16_t get_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}
static inline uint32_t get_le32(const uint8_t *p) {
    return ((uint32_t)p[3] << 24) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[1] << 8) | p[0];
}

/* ========================================================================
 * Packet assembly
 * ======================================================================== */

/* Header layout:
 * [0x0000] bmdf metadata header (256 bytes, copied from source)
 * [0x0100] braw block: 'braw'(4) + block_size(4) + picture_header(120)
 * [0x0180] tile offset table (nb_tiles * 4 bytes, BE32 each)
 * [0x1180] quant tables: 64 BE16 luma + 32 BE16 chroma = 192 bytes
 * [0x1240] (end of quant tables)
 * [0x1240..0x1300] zero padding (192 bytes)
 * [0x1300] tile data starts here
 * hdr_size = bmdf_size (0x100 typically) → braw starts at hdr_size
 * tile data starts at hdr_size + tile_offset[i]
 */

#define PACKET_HDR_SIZE 0x1300  /* bmdf(0x100) + braw header + quant tables + padding */

int braw_enc_encode_frame(BrawEncContext *ctx,
                          const uint16_t *input, int in_stride,
                          uint8_t *output, int output_size) {
    if (!ctx || !ctx->initialized) return BRAW_ENC_ERR_INVALID;
    if (output_size < PACKET_HDR_SIZE + 1024) return BRAW_ENC_ERR_OVERFLOW;

    int nb_tiles = ctx->nb_tiles_w * ctx->nb_tiles_h;

    /* ---- Write bmdf metadata header at offset 0x0000 ---- */
    memset(output, 0, PACKET_HDR_SIZE);
    if (ctx->config.bmdf_blob_size > 0) {
        memcpy(output, ctx->config.bmdf_blob,
               ctx->config.bmdf_blob_size < 256 ? ctx->config.bmdf_blob_size : 256);
    } else {
        /* Minimal bmdf: size=0x100, tag='bmdf' */
        put_be32(output, 0x100);
        output[4] = 'b'; output[5] = 'm'; output[6] = 'd'; output[7] = 'f';
    }

    /* ---- Write braw block header at offset 0x0100 ---- */
    uint8_t *braw_start = output + 0x100;
    /* 'braw' tag (little-endian) */
    braw_start[0] = 'b'; braw_start[1] = 'r'; braw_start[2] = 'a'; braw_start[3] = 'w';
    /* block size placeholder (will patch later) */
    put_be32(braw_start + 4, 0);

    /* Picture header at braw_start + 8 */
    uint8_t *pic = braw_start + 8;
    if (ctx->config.pic_header_size > 0) {
        /* Copy the original picture header from source */
        memcpy(pic, ctx->config.pic_header,
               ctx->config.pic_header_size < 120 ? ctx->config.pic_header_size : 120);
    } else {
        /* Build minimal picture header */
        memset(pic, 0, 120);
        pic[0] = (uint8_t)ctx->config.version;
        pic[1] = (uint8_t)ctx->config.qscale[0];
        pic[2] = (uint8_t)ctx->nb_tiles_w;
        pic[3] = (uint8_t)ctx->nb_tiles_h;
        put_be16(pic + 4, (uint16_t)ctx->config.width);
        put_be16(pic + 6, (uint16_t)ctx->config.height);
        put_be16(pic + 8, (uint16_t)ctx->tile_size_h);
        put_be16(pic + 24, (uint16_t)ctx->config.qscale[1]);
        put_be16(pic + 26, (uint16_t)ctx->config.qscale[2]);
        for (int i = 0; i < ctx->nb_tiles_w; i++)
            put_be16(pic + 56 + i * 2, (uint16_t)ctx->tile_widths[i]);
    }

    /* ---- Tile offset table at offset 0x0180 ---- */
    /* Will be filled after encoding tiles */

    /* ---- Quant tables at offset 0x1180 ---- */
    uint8_t *qtab = output + 0x1180;
    for (int i = 0; i < 64; i++)
        put_be16(qtab + i * 2, ctx->config.base_quant_luma[i]);
    for (int i = 0; i < 32; i++)
        put_be16(qtab + 128 + i * 2, ctx->config.base_quant_chroma[i]);

    /* ---- Encode tiles ---- */
    /* Tile data starts right after the header */
    int tile_data_start = PACKET_HDR_SIZE;
    int tile_buf_size = output_size - tile_data_start;
    if (tile_buf_size <= 0) return BRAW_ENC_ERR_OVERFLOW;

    /* Temp buffer for all tile bitstreams */
    uint8_t *tile_data = output + tile_data_start;

    /* The tile offsets are relative to the start of tile data
     * (i.e., offset from packet[hdr_size] where hdr_size = bmdf_size = 0x100) */
    /* Actually looking at the decoder:
     * bs_offset = hdr_size + tile_offset
     * where hdr_size = read_be32(packet) = bmdf header size = 0x100
     * So tile_offset is relative to packet[0x100] = braw_start
     * And tile data lives at packet[0x100 + tile_offset]
     * Since PACKET_HDR_SIZE = 0x1200 and hdr_size = 0x100:
     * tile_offset = 0x1200 - 0x100 = 0x1100 for the first tile */

    int current_offset = PACKET_HDR_SIZE - 0x100;  /* = 0x1100 */
    int bytes_used = 0;

    /* Column-major order (x outer, y inner) */
    for (int tx = 0; tx < ctx->nb_tiles_w; tx++) {
        int blocks_w = ctx->tile_widths[tx] / 16;

        for (int ty = 0; ty < ctx->nb_tiles_h; ty++) {
            int tile_idx = tx * ctx->nb_tiles_h + ty;

            /* Write tile offset */
            put_be32(output + 0x180 + tile_idx * 4, (uint32_t)current_offset);

            /* Encode tile */
            BitWriter bw;
            bw_init(&bw, tile_data + bytes_used, tile_buf_size - bytes_used);

            int blocks_h;
            if (ty == ctx->nb_tiles_h - 1)
                blocks_h = (ctx->config.height - ty * ctx->tile_size_h) / 8;
            else
                blocks_h = ctx->tile_size_h / 8;

            int ret = encode_tile(ctx, &bw, input, in_stride,
                                  tx, ty, blocks_w, blocks_h);
            if (ret < 0) return ret;

            bw_flush(&bw);
            int tile_bytes = bw.byte_pos;

            /* Pad tile to 8-byte alignment (camera encoder does this,
             * DaVinci expects tiles at 8-byte aligned offsets) */
            int pad = (8 - (tile_bytes & 7)) & 7;
            if (pad > 0) {
                memset(tile_data + bytes_used + tile_bytes, 0xFF, pad);
                tile_bytes += pad;
            }

            bytes_used += tile_bytes;
            current_offset += tile_bytes;
        }
    }

    /* Patch braw block size */
    int total_packet_size = tile_data_start + bytes_used;
    int braw_block_size = total_packet_size - 0x100;
    put_be32(braw_start + 4, (uint32_t)braw_block_size);

    return total_packet_size;
}

/* ========================================================================
 * Config extraction from source packet
 * ======================================================================== */

int braw_enc_config_from_packet(BrawEncConfig *cfg,
                                const uint8_t *packet, size_t packet_size,
                                int width, int height) {
    if (!cfg || !packet || packet_size < 0x1240) return BRAW_ENC_ERR_INVALID;

    memset(cfg, 0, sizeof(*cfg));
    cfg->width = width;
    cfg->height = height;

    /* Copy bmdf metadata header */
    uint32_t hdr_size = get_be32(packet);
    if (hdr_size < 8 || hdr_size > 0x200) return BRAW_ENC_ERR_INVALID;
    cfg->bmdf_blob_size = hdr_size < 256 ? (int)hdr_size : 256;
    memcpy(cfg->bmdf_blob, packet, cfg->bmdf_blob_size);

    /* Check braw tag */
    const uint8_t *braw = packet + hdr_size;
    if (get_le32(braw) != 0x77617262) return BRAW_ENC_ERR_INVALID; /* 'braw' */

    /* Parse picture header */
    const uint8_t *p = braw + 8;
    cfg->version = p[0];
    cfg->qscale[0] = p[1];
    cfg->nb_tiles_w = p[2];
    cfg->nb_tiles_h = p[3];
    /* width/height already known from probe */
    cfg->tile_size_h = get_be16(p + 8);
    cfg->qscale[1] = get_be16(p + 24);
    cfg->qscale[2] = get_be16(p + 26);

    for (int i = 0; i < cfg->nb_tiles_w && i < 256; i++)
        cfg->tile_widths[i] = get_be16(p + 56 + i * 2);

    /* Copy raw picture header bytes */
    cfg->pic_header_size = 120;
    memcpy(cfg->pic_header, p, 120);

    /* Copy raw quant tables from offset 0x1180 */
    const uint8_t *qtab = packet + 0x1180;
    for (int i = 0; i < 64; i++)
        cfg->base_quant_luma[i] = get_be16(qtab + i * 2);
    for (int i = 0; i < 32; i++)
        cfg->base_quant_chroma[i] = get_be16(qtab + 128 + i * 2);

    return BRAW_ENC_OK;
}

/* ========================================================================
 * Init / Free
 * ======================================================================== */

int braw_enc_init(BrawEncContext *ctx, const BrawEncConfig *cfg) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->config = *cfg;

    /* Build forward DCT matrices */
    build_fwd_8x8_matrices(ctx->fwd_col_8, ctx->fwd_row_8);
    build_fwd_4pt_col_matrix(ctx->fwd_col_4);

    /* Build AC reverse lookup */
    build_ac_reverse();

    /* Set up tile grid */
    ctx->nb_tiles_w = cfg->nb_tiles_w;
    ctx->nb_tiles_h = cfg->nb_tiles_h;
    ctx->tile_size_h = cfg->tile_size_h;
    ctx->tile_offsets_w[0] = 0;
    for (int i = 0; i < ctx->nb_tiles_w; i++) {
        ctx->tile_widths[i] = cfg->tile_widths[i];
        ctx->tile_offsets_w[i + 1] = ctx->tile_offsets_w[i] + ctx->tile_widths[i];
    }

    /* Compute scaled quant tables (matching decoder's parse_braw_header) */
    int scale = (cfg->qscale[0] + 4) * 2048;
    for (int i = 0; i < 64; i++)
        ctx->quant_luma[i] = cfg->base_quant_luma[i] * scale;
    if (ctx->quant_luma[0] <= cfg->qscale[1])
        ctx->quant_luma[0] = cfg->qscale[1];

    int cscale = (int)lrint(fmax(4.0, round((cfg->qscale[0] + 4) * 0.5 + 0.5)) * 2048.0);
    for (int i = 0; i < 32; i++)
        ctx->quant_chroma[i] = cfg->base_quant_chroma[i] * cscale;
    if (ctx->quant_chroma[0] <= cfg->qscale[2])
        ctx->quant_chroma[0] = cfg->qscale[2];

    ctx->initialized = 1;
    return BRAW_ENC_OK;
}

void braw_enc_free(BrawEncContext *ctx) {
    if (!ctx) return;
    ctx->initialized = 0;
}
