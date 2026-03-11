/*
 * Blackmagic RAW (BRAW) Decoder
 *
 * Decodes BRAW-compressed video frames to 12-bit Bayer RGGB.
 * Based on reverse-engineering by Paul B Mahol (FFmpeg braw.c, 2019).
 */

#include "../include/braw_dec.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ========================================================================
 * Constants and tables from Paul B Mahol's FFmpeg BRAW decoder
 * ======================================================================== */

/* DC Huffman table: 16 symbols, max 13 bits */
static const uint8_t dc_bits[16] = {
    2, 3, 3, 3, 3, 3, 5, 5, 6, 12, 12, 12, 12, 5, 12, 13,
};

static const uint16_t dc_codes[16] = {
    0x0, 0x2, 0x3, 0x4, 0x5, 0x6, 0x1C, 0x1D, 0x3E,
    0xFF4, 0xFF5, 0xFF7, 0xFED, 0x1E, 0xFFE, 0x1FFE,
};

/* DC table: [symbol][0]=extra_len, [1]=sign_bits, [2]=dcval_tab_col */
static const uint8_t dc_table[16][3] = {
    {  0, 0, 15 }, {  0, 1,  0 }, {  1, 1,  1 }, {  2, 1,  2 },
    {  3, 1,  3 }, {  4, 1,  4 }, {  5, 1,  5 }, {  6, 1,  6 },
    {  7, 1,  7 }, {  8, 1,  8 }, {  9, 1,  9 }, { 10, 1, 10 },
    { 11, 1, 11 }, { 12, 1, 12 }, { 13, 1, 13 }, { 14, 1, 14 },
};

/* DC value lookup: dcval_tab[sign_bit][column] */
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

/* AC Huffman table: 194 symbols, max 18 bits */
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

/* AC table: [symbol][0]=run (255=EOB), [1]=level_bits */
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

/* 8x4 scan order for chroma blocks */
static const uint8_t half_scan[32] = {
     0,  1,  2,  3,  8,  9, 16, 17, 10, 11,  4,  5,  6,  7, 12, 13,
    18, 19, 24, 25, 26, 27, 20, 21, 14, 15, 22, 23, 28, 29, 30, 31,
};

/* Standard 8x8 zigzag scan order */
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

/* ========================================================================
 * 12-bit iDCT constants (same as FFmpeg ff_prores_idct_12)
 * ======================================================================== */

#define W1  45451
#define W2  42813
#define W3  38531
#define W4  32767
#define W5  25746
#define W6  17734
#define W7   9041

#define ROW_SHIFT 16
#define COL_SHIFT 17

static inline int clip_12bit(int v) {
    if (v < 0) return 0;
    if (v > 4095) return 4095;
    return v;
}

/* ========================================================================
 * MSB-first BitReader
 * ======================================================================== */

typedef struct {
    const uint8_t *data;
    size_t size_bytes;
    int bit_pos;    /* current bit position from start */
    int total_bits;
} BitReader;

static inline void br_init(BitReader *br, const uint8_t *data, size_t size) {
    br->data = data;
    br->size_bytes = size;
    br->bit_pos = 0;
    br->total_bits = (int)(size * 8);
}

static inline int br_bits_left(const BitReader *br) {
    return br->total_bits - br->bit_pos;
}

/* Peek up to 25 bits without advancing (MSB-first) */
static inline uint32_t br_peek(const BitReader *br, int n) {
    if (n <= 0) return 0;
    int byte_pos = br->bit_pos >> 3;
    int bit_off = br->bit_pos & 7;

    /* Read up to 4 bytes starting from byte_pos */
    uint32_t val = 0;
    int bytes_avail = (int)br->size_bytes - byte_pos;
    if (bytes_avail > 4) bytes_avail = 4;
    for (int i = 0; i < bytes_avail; i++)
        val = (val << 8) | br->data[byte_pos + i];
    /* Pad remaining with zeros */
    for (int i = bytes_avail; i < 4; i++)
        val <<= 8;

    /* Shift out consumed bits at the top */
    val <<= bit_off;

    /* Right-align the n bits we want */
    return val >> (32 - n);
}

static inline void br_skip(BitReader *br, int n) {
    br->bit_pos += n;
}

static inline uint32_t br_read(BitReader *br, int n) {
    if (n <= 0) return 0;
    uint32_t val = br_peek(br, n);
    br->bit_pos += n;
    return val;
}

/* ========================================================================
 * VLC lookup table construction
 * ======================================================================== */

#define DC_VLC_BITS 13
#define DC_VLC_SIZE (1 << DC_VLC_BITS)  /* 8192 */
#define AC_VLC_BITS 18
#define AC_VLC_SIZE (1 << AC_VLC_BITS)  /* 262144 */

static int build_vlc_table(BrawVlcEntry *table, int table_bits,
                           const void *codes_ptr, int code_elem_size,
                           const uint8_t *bits, int num_symbols) {
    int table_size = 1 << table_bits;

    /* Initialize all entries as invalid */
    memset(table, 0, table_size * sizeof(BrawVlcEntry));

    for (int sym = 0; sym < num_symbols; sym++) {
        int code_len = bits[sym];
        if (code_len <= 0 || code_len > table_bits) continue;

        uint32_t code;
        if (code_elem_size == 2)
            code = ((const uint16_t *)codes_ptr)[sym];
        else
            code = ((const uint32_t *)codes_ptr)[sym];

        /* Fill all table entries that share this prefix */
        int pad_bits = table_bits - code_len;
        uint32_t base = code << pad_bits;
        int count = 1 << pad_bits;

        for (int j = 0; j < count; j++) {
            uint32_t idx = base | j;
            if (idx < (uint32_t)table_size) {
                table[idx].symbol = (uint8_t)sym;
                table[idx].bits = (uint8_t)code_len;
            }
        }
    }

    return 0;
}

/* For AC codes longer than 16 bits, we need a two-level approach.
 * Level 1: 16-bit lookup handles codes up to 16 bits.
 * Level 2: for entries that need more bits, peek additional bits. */

#define AC_VLC_L1_BITS 16
#define AC_VLC_L1_SIZE (1 << AC_VLC_L1_BITS)  /* 65536 */

/* Two-level AC VLC: first try 16 bits, then try 18 bits for unresolved */
typedef struct {
    BrawVlcEntry *level1;      /* 65536 entries, 16-bit lookup */
    BrawVlcEntry *level2;      /* 262144 entries, 18-bit lookup (for long codes) */
    int has_long_codes;
} AcVlcTable;

static int build_ac_vlc(AcVlcTable *ac) {
    ac->level1 = calloc(AC_VLC_L1_SIZE, sizeof(BrawVlcEntry));
    if (!ac->level1) return BRAW_ERR_ALLOC;

    ac->has_long_codes = 0;

    /* Build level 1 (16-bit) for codes <= 16 bits */
    for (int sym = 0; sym < 194; sym++) {
        int code_len = ac_bits_table[sym];
        if (code_len > 16) {
            ac->has_long_codes = 1;
            continue;
        }
        uint32_t code = ac_codes_table[sym];
        int pad = 16 - code_len;
        uint32_t base = code << pad;
        int count = 1 << pad;
        for (int j = 0; j < count; j++) {
            uint32_t idx = base | j;
            if (idx < AC_VLC_L1_SIZE) {
                ac->level1[idx].symbol = (uint8_t)sym;
                ac->level1[idx].bits = (uint8_t)code_len;
            }
        }
    }

    /* Build level 2 (18-bit) only for long codes */
    if (ac->has_long_codes) {
        ac->level2 = calloc(AC_VLC_SIZE, sizeof(BrawVlcEntry));
        if (!ac->level2) return BRAW_ERR_ALLOC;

        /* Fill with ALL codes (short and long) */
        for (int sym = 0; sym < 194; sym++) {
            int code_len = ac_bits_table[sym];
            uint32_t code = ac_codes_table[sym];
            int pad = 18 - code_len;
            uint32_t base = code << pad;
            int count = 1 << pad;
            for (int j = 0; j < count; j++) {
                uint32_t idx = base | j;
                if (idx < AC_VLC_SIZE) {
                    ac->level2[idx].symbol = (uint8_t)sym;
                    ac->level2[idx].bits = (uint8_t)code_len;
                }
            }
        }
    } else {
        ac->level2 = NULL;
    }

    return BRAW_OK;
}

/* Decode one DC VLC symbol */
static inline int vlc_decode_dc(const BrawVlcEntry *dc_vlc, BitReader *br) {
    if (br_bits_left(br) < DC_VLC_BITS) return -1;
    uint32_t peek = br_peek(br, DC_VLC_BITS);
    BrawVlcEntry e = dc_vlc[peek];
    if (e.bits == 0) return -1;
    br_skip(br, e.bits);
    return e.symbol;
}

/* Decode one AC VLC symbol */
static inline int vlc_decode_ac(const AcVlcTable *ac, BitReader *br) {
    if (br_bits_left(br) < 2) return -1;

    /* Try level 1 (16-bit) first */
    int avail = br_bits_left(br);
    int peek_bits = avail < 16 ? avail : 16;
    uint32_t peek = br_peek(br, peek_bits);
    if (peek_bits < 16) peek <<= (16 - peek_bits);

    BrawVlcEntry e = ac->level1[peek];
    if (e.bits > 0 && e.bits <= peek_bits) {
        br_skip(br, e.bits);
        return e.symbol;
    }

    /* Try level 2 (18-bit) for long codes */
    if (ac->has_long_codes && ac->level2 && avail >= 17) {
        peek_bits = avail < 18 ? avail : 18;
        peek = br_peek(br, peek_bits);
        if (peek_bits < 18) peek <<= (18 - peek_bits);
        e = ac->level2[peek];
        if (e.bits > 0 && e.bits <= peek_bits) {
            br_skip(br, e.bits);
            return e.symbol;
        }
    }

    return -1;
}

/* ========================================================================
 * 8x8 iDCT (12-bit, outputs to uint16_t buffer)
 * ======================================================================== */

static void idct_row_12bit(int16_t *row) {
    int a0, a1, a2, a3, b0, b1, b2, b3;

    /* DC-only shortcut */
    if (!(row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7])) {
        int val = ((int)W4 * row[0] + (1 << (ROW_SHIFT - 1))) >> ROW_SHIFT;
        row[0] = row[1] = row[2] = row[3] = val;
        row[4] = row[5] = row[6] = row[7] = val;
        return;
    }

    a0 = (int)W4 * row[0] + (1 << (ROW_SHIFT - 1));
    a1 = a0;
    a2 = a0;
    a3 = a0;

    a0 += (int)W2 * row[2];
    a1 += (int)W6 * row[2];
    a2 -= (int)W6 * row[2];
    a3 -= (int)W2 * row[2];

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

    row[0] = (a0 + b0) >> ROW_SHIFT;
    row[7] = (a0 - b0) >> ROW_SHIFT;
    row[1] = (a1 + b1) >> ROW_SHIFT;
    row[6] = (a1 - b1) >> ROW_SHIFT;
    row[2] = (a2 + b2) >> ROW_SHIFT;
    row[5] = (a2 - b2) >> ROW_SHIFT;
    row[3] = (a3 + b3) >> ROW_SHIFT;
    row[4] = (a3 - b3) >> ROW_SHIFT;
}

/* 8-point column iDCT: reads int16_t column (stride 8), writes uint16_t column */
static void idct_col_put_12bit(uint16_t *dest, int line_stride, const int16_t *in) {
    int a0, a1, a2, a3, b0, b1, b2, b3;

    a0 = (int)W4 * in[8*0] + (1 << (COL_SHIFT - 1));
    a1 = a0;
    a2 = a0;
    a3 = a0;

    a0 += (int)W2 * in[8*2];
    a1 += (int)W6 * in[8*2];
    a2 -= (int)W6 * in[8*2];
    a3 -= (int)W2 * in[8*2];

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

/* Full 8x8 iDCT: row pass on int16_t block, then column pass writing uint16_t */
static void idct_put_8x8_12bit(uint16_t *dest, int line_stride, int16_t *block) {
    /* Row pass */
    for (int i = 0; i < 8; i++)
        idct_row_12bit(block + i * 8);

    /* Column pass */
    for (int i = 0; i < 8; i++)
        idct_col_put_12bit(dest + i, line_stride, block + i);
}

/* ========================================================================
 * 8x4 iDCT (12-bit): 8 columns, 4 rows. Row pass is 8-point, column is 4-point.
 * ======================================================================== */

/* 4-point column iDCT using W2/W6 (matches FFmpeg's idct4ColPut_int16_12bit) */
static void idct4_col_put_12bit(uint16_t *dest, int line_stride, const int16_t *in) {
    /* Constants from the FFmpeg patch */
    const int X0 = 17734;   /* W6 */
    const int X1 = 42813;   /* W2 */
    const int X2 = 32768;   /* ~W4, exactly 2^15 */

    int a0 = (int)in[8*0] * X2;
    int a2 = (int)in[8*2] * X2;
    int a1 = in[8*1];
    int a3 = in[8*3];

    int c0 = a0 + a2;
    int c1 = a0 - a2;
    int c2 = a1 * X0 - a3 * X1;
    int c3 = a3 * X0 + a1 * X1;

    dest[0 * line_stride] = clip_12bit((c0 + c3) >> 16);
    dest[1 * line_stride] = clip_12bit((c1 + c2) >> 16);
    dest[2 * line_stride] = clip_12bit((c1 - c2) >> 16);
    dest[3 * line_stride] = clip_12bit((c0 - c3) >> 16);
}

/* 8x4 iDCT: 8-point row pass on 4 rows, then 4-point column pass */
static void idct_put_8x4_12bit(uint16_t *dest, int line_stride, int16_t *block) {
    /* Row pass on 4 rows */
    for (int i = 0; i < 4; i++)
        idct_row_12bit(block + i * 8);

    /* Column pass: 4-point on each of 8 columns */
    for (int i = 0; i < 8; i++)
        idct4_col_put_12bit(dest + i, line_stride, block + i);
}

/* ========================================================================
 * Block decoder
 * ======================================================================== */

/* Global VLC tables (set during init, reference counted) */
static BrawVlcEntry *g_dc_vlc = NULL;
static AcVlcTable g_ac_vlc = {0};
static int g_vlc_refcount = 0;

static int decode_block(BitReader *br, int16_t *dst, const int *quant,
                        int *prev_dc, int max_coeff,
                        int dc_add, const uint8_t *scan) {
    int dc_idx, sgnbit, sign, len, val, code;

    memset(dst, 0, 64 * sizeof(int16_t));

    /* Decode DC coefficient */
    dc_idx = vlc_decode_dc(g_dc_vlc, br);
    if (dc_idx < 0) return BRAW_ERR_BITSTREAM;

    sign = dc_table[dc_idx][1];
    sgnbit = (sign > 0) ? (int)br_read(br, sign) : 0;
    len = dc_table[dc_idx][0];
    code = (len > 0) ? (int)br_read(br, len) : 0;
    val = code + *prev_dc + dcval_tab[sgnbit][dc_table[dc_idx][2]];
    *prev_dc = val;

    /* Dequantize DC and add bias */
    int dc_dequant = (int)(((int64_t)val * quant[0] + 0x8000) >> 16) + dc_add;
    if (dc_dequant > 32767) dc_dequant = 32767;
    dst[0] = (int16_t)dc_dequant;

    /* Decode AC coefficients */
    for (int i = 0;;) {
        int ac_idx = vlc_decode_ac(&g_ac_vlc, br);
        if (ac_idx < 0) return BRAW_ERR_BITSTREAM;

        int skip = ac_table[ac_idx][0];
        if (skip == 255) break;  /* EOB */

        int ac_len = ac_table[ac_idx][1];
        int ac_val;
        if (ac_len > 0) {
            ac_val = (int)br_read(br, ac_len);
            /* Sign extension (JPEG-style) */
            if (ac_val < (1 << (ac_len - 1)))
                ac_val -= (1 << ac_len) - 1;
        } else {
            ac_val = 0;
        }

        i = i + 1 + skip;
        if (i >= max_coeff) return BRAW_ERR_BITSTREAM;

        /* Dequantize AC */
        int pos = scan[i];
        dst[pos] = (int16_t)(((int64_t)ac_val * quant[pos] + 0x8000) >> 16);
    }

    return BRAW_OK;
}

/* ========================================================================
 * Decorrelation: chroma → Bayer reconstruction
 * ======================================================================== */

static void decorrelate(uint16_t *frame, int frame_stride,
                        int pos_x, int pos_y,
                        const uint16_t *chroma0, const uint16_t *chroma1) {
    /*
     * After luma iDCT, the frame has the "luma" component at each pixel.
     * The two chroma blocks (8x4 each) contain delta values centered at 2048.
     * Decorrelation adds the chroma delta to alternate Bayer positions:
     *   - chroma1 (block[2]) → adds to even-row, even-col (e.g., R or Gr)
     *   - chroma0 (block[3]) → adds to odd-row, odd-col (e.g., B or Gb)
     *
     * Note: dst0 = even row, even col positions
     *        dst1 = odd row, odd col positions
     */
    uint16_t *dst0 = frame + pos_y * frame_stride + pos_x;
    uint16_t *dst1 = frame + (pos_y + 1) * frame_stride + (pos_x + 1);

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 8; x++) {
            int v0 = (int)dst0[x * 2] + (int)chroma1[y * 8 + x] - 2048;
            int v1 = (int)dst1[x * 2] + (int)chroma0[y * 8 + x] - 2048;
            dst0[x * 2] = (uint16_t)clip_12bit(v0);
            dst1[x * 2] = (uint16_t)clip_12bit(v1);
        }
        /* Advance by 2 rows: chroma row y covers frame rows (2y, 2y+1).
         * FFmpeg does dst0 += linesize (bytes) on uint16_t* = 2 pixel-rows. */
        dst0 += frame_stride * 2;
        dst1 += frame_stride * 2;
    }

    /* Expand 12-bit to ~16-bit: val = (val << 4) | (val & 15) */
    /* This gives a pseudo-linear expansion from [0,4095] to [0,65535] */
    /* Actually: max = 4095<<4 | 15 = 65535. min = 0. */
    uint16_t *row = frame + pos_y * frame_stride + pos_x;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 16; x++) {
            row[x] = (row[x] << 4) | (row[x] & 15);
        }
        row += frame_stride;
    }
}

/* ========================================================================
 * Tile decoder
 * ======================================================================== */

static int decode_tile(BrawDecContext *ctx, BitReader *br,
                       uint16_t *frame, int frame_stride,
                       int tile_x, int tile_y,
                       int blocks_w, int blocks_h) {
    int prev_dc[3] = {0, 0, 0};
    int ret;

    for (int y = 0; y < blocks_h; y++) {
        int pos_y = y * 8 + tile_y * ctx->info.tile_size_h;

        for (int x = 0; x < blocks_w; x++) {
            int pos_x = ctx->info.tile_offset_w[tile_x] + x * 16;

            /* Decode 2 luma blocks (8x8 each, side by side = 16x8 pixels) */
            ret = decode_block(br, ctx->block[0], ctx->info.quant_luma,
                             &prev_dc[0], 64, 16384, zigzag_direct);
            if (ret < 0) return ret;

            ret = decode_block(br, ctx->block[1], ctx->info.quant_luma,
                             &prev_dc[0], 64, 16384, zigzag_direct);
            if (ret < 0) return ret;

            /* Decode 2 chroma blocks (8x4 each) */
            ret = decode_block(br, ctx->block[2], ctx->info.quant_chroma,
                             &prev_dc[1], 32, 8192, half_scan);
            if (ret < 0) return ret;

            ret = decode_block(br, ctx->block[3], ctx->info.quant_chroma,
                             &prev_dc[2], 32, 8192, half_scan);
            if (ret < 0) return ret;

            /* iDCT luma blocks → write directly to frame */
            idct_put_8x8_12bit(frame + pos_y * frame_stride + pos_x,
                              frame_stride, ctx->block[0]);
            idct_put_8x8_12bit(frame + pos_y * frame_stride + pos_x + 8,
                              frame_stride, ctx->block[1]);

            /* iDCT chroma blocks → temp buffers */
            idct_put_8x4_12bit(ctx->chroma_out[0], 8, ctx->block[2]);
            idct_put_8x4_12bit(ctx->chroma_out[1], 8, ctx->block[3]);

            /* Decorrelate: add chroma to luma → Bayer pattern */
            decorrelate(frame, frame_stride, pos_x, pos_y,
                       ctx->chroma_out[0], ctx->chroma_out[1]);

            /* Skip 8-bit alignment/padding between block groups */
            br_skip(br, 8);
            if (br_bits_left(br) < 0) return BRAW_ERR_BITSTREAM;
        }
    }

    return BRAW_OK;
}

/* ========================================================================
 * Byte-stream reader helpers
 * ======================================================================== */

static inline uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}

static inline uint32_t read_le32(const uint8_t *p) {
    return ((uint32_t)p[3] << 24) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[1] << 8) | p[0];
}

static inline uint16_t read_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

/* ========================================================================
 * Frame-level decoder
 * ======================================================================== */

static int parse_braw_header(BrawDecContext *ctx, const uint8_t *data, size_t size) {
    BrawFrameInfo *info = &ctx->info;

    /* Parse metadata header */
    if (size < 8) return BRAW_ERR_INVALID;
    uint32_t hdr_size = read_be32(data);
    if (hdr_size < 8 || hdr_size > size) return BRAW_ERR_INVALID;
    if (read_le32(data + 4) != 0x66646D62) return BRAW_ERR_INVALID; /* 'bmdf' */

    const uint8_t *braw = data + hdr_size;
    size_t braw_avail = size - hdr_size;

    /* Check 'braw' tag */
    if (braw_avail < 8) return BRAW_ERR_INVALID;
    if (read_le32(braw) != 0x77617262) return BRAW_ERR_INVALID; /* 'braw' */
    uint32_t braw_size = read_be32(braw + 4);
    if (braw_size < 4352 || braw_size - 8 > braw_avail - 8) return BRAW_ERR_INVALID;

    const uint8_t *p = braw + 8;  /* picture header start */

    info->version = p[0];
    if (info->version != 1 && info->version != 2) return BRAW_ERR_INVALID;

    info->qscale[0] = p[1];
    info->nb_tiles_w = p[2];
    info->nb_tiles_h = p[3];
    if (!info->nb_tiles_w || !info->nb_tiles_h ||
        info->nb_tiles_w * info->nb_tiles_h > BRAW_MAX_TILES)
        return BRAW_ERR_INVALID;

    info->width = read_be16(p + 4);
    info->height = read_be16(p + 6);
    info->tile_size_h = read_be16(p + 8);
    if (info->tile_size_h & 7) return BRAW_ERR_INVALID;

    /* Skip 2 bytes + 8 bytes (offsets) + 4 bytes skip = 14 bytes after tile_size_h */
    /* p+10: skip 2 */
    /* p+12: 4 * 2-byte offsets = 8 bytes */
    /* p+20: skip 4 bytes */
    info->qscale[1] = read_be16(p + 24);
    info->qscale[2] = read_be16(p + 26);

    /* p+28: skip 28 bytes */
    /* p+56: tile widths */
    info->tile_offset_w[0] = 0;
    for (int x = 0; x < info->nb_tiles_w; x++) {
        info->tile_size_w[x] = read_be16(p + 56 + x * 2);
        info->tile_offset_w[x + 1] = info->tile_offset_w[x] + info->tile_size_w[x];
        if (info->tile_offset_w[x + 1] > info->width)
            return BRAW_ERR_INVALID;
    }

    /* Parse quantization tables at offset 0x1180 from packet start */
    /* = hdr_size + 0x1080 from braw start, but offset is from data[0] */
    size_t qoffset = 0x1180;
    if (qoffset + 64 * 2 + 32 * 2 > size) return BRAW_ERR_INVALID;

    /* Luma quant table: 64 entries × BE16 */
    int scale = (info->qscale[0] + 4) * 2048;
    for (int n = 0; n < 64; n++) {
        int raw = read_be16(data + qoffset + n * 2);
        info->quant_luma[n] = raw * scale;
    }

    /* Ensure DC quant >= qscale[1] */
    if (info->quant_luma[0] <= info->qscale[1])
        info->quant_luma[0] = info->qscale[1];

    /* Chroma quant table: 32 entries × BE16 */
    int cscale = (int)lrint(fmax(4.0, round((info->qscale[0] + 4) * 0.5 + 0.5)) * 2048.0);
    for (int n = 0; n < 32; n++) {
        int raw = read_be16(data + qoffset + 128 + n * 2);
        info->quant_chroma[n] = raw * cscale;
    }

    if (info->quant_chroma[0] <= info->qscale[2])
        info->quant_chroma[0] = info->qscale[2];

    return BRAW_OK;
}

int braw_dec_decode_frame(BrawDecContext *ctx,
                          const uint8_t *packet, size_t packet_size,
                          uint16_t *output, int out_stride,
                          BrawFrameInfo *info_out) {
    if (!ctx || !ctx->initialized) return BRAW_ERR_INVALID;
    if (!packet || packet_size < 4608) return BRAW_ERR_INVALID;

    int ret = parse_braw_header(ctx, packet, packet_size);
    if (ret < 0) return ret;

    BrawFrameInfo *info = &ctx->info;
    if (info_out) *info_out = *info;

    /* Parse metadata header size */
    uint32_t hdr_size = read_be32(packet);

    /* Tile offset table starts at packet offset 0x180 */
    size_t tile_table_offset = 0x180;

    /* Decode tiles: column-major order (x outer, y inner) */
    for (int x = 0; x < info->nb_tiles_w; x++) {
        int blocks_w = info->tile_size_w[x] / 16;

        for (int y = 0; y < info->nb_tiles_h; y++) {
            int tile_idx = x * info->nb_tiles_h + y;
            size_t toff = tile_table_offset + tile_idx * 4;
            if (toff + 4 > packet_size) return BRAW_ERR_INVALID;

            uint32_t tile_offset = read_be32(packet + toff);

            /* Calculate tile size from next offset or end of data */
            int last_tile = (x == info->nb_tiles_w - 1) && (y == info->nb_tiles_h - 1);
            uint32_t tile_size;
            if (last_tile) {
                tile_size = (uint32_t)packet_size - tile_offset - hdr_size;
            } else {
                uint32_t next_offset = read_be32(packet + toff + 4);
                tile_size = next_offset - tile_offset;
            }

            /* Tile bitstream starts at packet[hdr_size + tile_offset] */
            size_t bs_offset = hdr_size + tile_offset;
            if (bs_offset + tile_size > packet_size) return BRAW_ERR_INVALID;

            BitReader br;
            br_init(&br, packet + bs_offset, tile_size);

            int blocks_h;
            if (y == info->nb_tiles_h - 1)
                blocks_h = (info->height - y * info->tile_size_h) / 8;
            else
                blocks_h = info->tile_size_h / 8;

            ret = decode_tile(ctx, &br, output, out_stride, x, y, blocks_w, blocks_h);
            if (ret < 0) {
                fprintf(stderr, "BRAW: decode_tile(%d,%d) failed: %d (bits_left=%d)\n",
                        x, y, ret, br_bits_left(&br));
                return ret;
            }
        }
    }

    return BRAW_OK;
}

int braw_dec_probe(const uint8_t *packet, size_t packet_size,
                   int *width, int *height) {
    BrawDecContext tmp = {0};
    tmp.initialized = 1;  /* skip VLC check for probe */
    int ret = parse_braw_header(&tmp, packet, packet_size);
    if (ret < 0) return ret;
    if (width) *width = tmp.info.width;
    if (height) *height = tmp.info.height;
    return BRAW_OK;
}

/* ========================================================================
 * Init / Free
 * ======================================================================== */

int braw_dec_init(BrawDecContext *ctx) {
    memset(ctx, 0, sizeof(*ctx));

    if (g_vlc_refcount == 0) {
        /* Build DC VLC lookup table (13-bit) */
        BrawVlcEntry *dc = calloc(DC_VLC_SIZE, sizeof(BrawVlcEntry));
        if (!dc) return BRAW_ERR_ALLOC;

        build_vlc_table(dc, DC_VLC_BITS,
                        dc_codes, 2, dc_bits, 16);
        g_dc_vlc = dc;

        /* Build AC VLC lookup table (two-level, 16+18 bit) */
        int ret = build_ac_vlc(&g_ac_vlc);
        if (ret < 0) {
            free(dc);
            g_dc_vlc = NULL;
            return ret;
        }
    }

    ctx->dc_vlc = g_dc_vlc;
    g_vlc_refcount++;
    ctx->initialized = 1;
    return BRAW_OK;
}

void braw_dec_free(BrawDecContext *ctx) {
    if (!ctx) return;
    ctx->dc_vlc = NULL;

    g_vlc_refcount--;
    if (g_vlc_refcount <= 0) {
        free(g_dc_vlc);
        g_dc_vlc = NULL;

        free(g_ac_vlc.level1);
        free(g_ac_vlc.level2);
        memset(&g_ac_vlc, 0, sizeof(g_ac_vlc));
        g_vlc_refcount = 0;
    }

    ctx->initialized = 0;
}

/* ========================================================================
 * Simple MOV parser for BRAW files
 * ======================================================================== */

/* Find an atom within a range. Returns offset of atom data, or -1. */
static int64_t find_atom(FILE *f, int64_t start, int64_t end, uint32_t tag, uint64_t *out_size) {
    int64_t pos = start;
    uint8_t hdr[8];

    while (pos + 8 <= end) {
        if (fseeko(f, pos, SEEK_SET) != 0) return -1;
        if (fread(hdr, 1, 8, f) != 8) return -1;

        uint32_t atom_size = read_be32(hdr);
        uint32_t atom_tag = read_be32(hdr + 4);

        if (atom_size < 8) {
            if (atom_size == 1) {
                /* 64-bit extended size */
                uint8_t ext[8];
                if (fread(ext, 1, 8, f) != 8) return -1;
                uint64_t ext_size = ((uint64_t)read_be32(ext) << 32) | read_be32(ext + 4);
                if (ext_size < 16) return -1;
                if (atom_tag == tag) {
                    if (out_size) *out_size = ext_size - 16;
                    return pos + 16;
                }
                pos += (int64_t)ext_size;
            } else {
                return -1;
            }
        } else {
            if (atom_tag == tag) {
                if (out_size) *out_size = atom_size - 8;
                return pos + 8;
            }
            pos += atom_size;
        }
    }
    return -1;
}

/* Tags as BE32 */
#define TAG(a,b,c,d) (((uint32_t)(a)<<24)|((uint32_t)(b)<<16)|((uint32_t)(c)<<8)|(d))

int braw_mov_parse(const char *path, BrawMovInfo *mov) {
    memset(mov, 0, sizeof(*mov));

    FILE *f = fopen(path, "rb");
    if (!f) return BRAW_ERR_IO;

    /* Get file size */
    fseeko(f, 0, SEEK_END);
    int64_t file_size = ftello(f);

    /* Find moov atom at top level */
    uint64_t moov_size;
    int64_t moov = find_atom(f, 0, file_size, TAG('m','o','o','v'), &moov_size);
    if (moov < 0) { fclose(f); return BRAW_ERR_INVALID; }

    /* Find trak → mdia → minf → stbl */
    uint64_t trak_size;
    int64_t trak = find_atom(f, moov, moov + moov_size, TAG('t','r','a','k'), &trak_size);
    if (trak < 0) { fclose(f); return BRAW_ERR_INVALID; }

    uint64_t mdia_size;
    int64_t mdia = find_atom(f, trak, trak + trak_size, TAG('m','d','i','a'), &mdia_size);
    if (mdia < 0) { fclose(f); return BRAW_ERR_INVALID; }

    /* Get timescale from mdhd for fps */
    uint64_t mdhd_size;
    int64_t mdhd = find_atom(f, mdia, mdia + mdia_size, TAG('m','d','h','d'), &mdhd_size);
    if (mdhd >= 0 && mdhd_size >= 24) {
        uint8_t mdhd_data[32];
        fseeko(f, mdhd, SEEK_SET);
        fread(mdhd_data, 1, 24, f);
        int version = mdhd_data[0];
        if (version == 0) {
            uint32_t timescale = read_be32(mdhd_data + 12);
            uint32_t duration = read_be32(mdhd_data + 16);
            if (timescale > 0 && duration > 0)
                mov->fps = (double)timescale; /* will correct later with sample count */
        }
    }

    uint64_t minf_size;
    int64_t minf = find_atom(f, mdia, mdia + mdia_size, TAG('m','i','n','f'), &minf_size);
    if (minf < 0) { fclose(f); return BRAW_ERR_INVALID; }

    uint64_t stbl_size;
    int64_t stbl = find_atom(f, minf, minf + minf_size, TAG('s','t','b','l'), &stbl_size);
    if (stbl < 0) { fclose(f); return BRAW_ERR_INVALID; }

    /* Parse stsd for dimensions */
    uint64_t stsd_size;
    int64_t stsd = find_atom(f, stbl, stbl + stbl_size, TAG('s','t','s','d'), &stsd_size);
    if (stsd >= 0 && stsd_size >= 86) {
        uint8_t stsd_data[96];
        fseeko(f, stsd, SEEK_SET);
        size_t rd = fread(stsd_data, 1, 86, f);
        if (rd >= 86) {
            /* version(4) + entry_count(4) + entry_size(4) + format(4) + reserved(6) + refidx(2) + ... */
            /* width at offset 32, height at offset 34 (from stsd data start, after version) */
            mov->width = read_be16(stsd_data + 4 + 4 + 4 + 4 + 6 + 2 + 2 + 2 + 4 + 4 + 4);
            mov->height = read_be16(stsd_data + 4 + 4 + 4 + 4 + 6 + 2 + 2 + 2 + 4 + 4 + 4 + 2);
        }
    }

    /* Parse stsz for sample sizes */
    uint64_t stsz_size;
    int64_t stsz = find_atom(f, stbl, stbl + stbl_size, TAG('s','t','s','z'), &stsz_size);
    if (stsz < 0) { fclose(f); return BRAW_ERR_INVALID; }
    if (stsz_size < 12) { fclose(f); return BRAW_ERR_INVALID; }

    uint8_t stsz_hdr[12];
    fseeko(f, stsz, SEEK_SET);
    if (fread(stsz_hdr, 1, 12, f) != 12) { fclose(f); return BRAW_ERR_IO; }

    /* uint32_t version_flags = read_be32(stsz_hdr); */
    uint32_t uniform_size = read_be32(stsz_hdr + 4);
    uint32_t sample_count = read_be32(stsz_hdr + 8);

    mov->frame_count = (int)sample_count;
    mov->frame_sizes = calloc(sample_count, sizeof(int32_t));
    if (!mov->frame_sizes) { fclose(f); return BRAW_ERR_ALLOC; }

    if (uniform_size > 0) {
        for (uint32_t i = 0; i < sample_count; i++)
            mov->frame_sizes[i] = (int32_t)uniform_size;
    } else {
        uint8_t *size_data = malloc(sample_count * 4);
        if (!size_data) { fclose(f); braw_mov_free(mov); return BRAW_ERR_ALLOC; }
        if (fread(size_data, 4, sample_count, f) != sample_count) {
            free(size_data); fclose(f); braw_mov_free(mov); return BRAW_ERR_IO;
        }
        for (uint32_t i = 0; i < sample_count; i++)
            mov->frame_sizes[i] = (int32_t)read_be32(size_data + i * 4);
        free(size_data);
    }

    /* Parse stco/co64 for chunk offsets */
    uint64_t stco_size;
    int64_t stco = find_atom(f, stbl, stbl + stbl_size, TAG('s','t','c','o'), &stco_size);
    int use_co64 = 0;

    if (stco < 0) {
        stco = find_atom(f, stbl, stbl + stbl_size, TAG('c','o','6','4'), &stco_size);
        if (stco < 0) { fclose(f); braw_mov_free(mov); return BRAW_ERR_INVALID; }
        use_co64 = 1;
    }

    uint8_t co_hdr[8];
    fseeko(f, stco, SEEK_SET);
    if (fread(co_hdr, 1, 8, f) != 8) { fclose(f); braw_mov_free(mov); return BRAW_ERR_IO; }

    uint32_t chunk_count = read_be32(co_hdr + 4);

    int64_t *chunk_offsets = malloc(chunk_count * sizeof(int64_t));
    if (!chunk_offsets) { fclose(f); braw_mov_free(mov); return BRAW_ERR_ALLOC; }

    for (uint32_t i = 0; i < chunk_count; i++) {
        if (use_co64) {
            uint8_t buf[8];
            if (fread(buf, 1, 8, f) != 8) { free(chunk_offsets); fclose(f); braw_mov_free(mov); return BRAW_ERR_IO; }
            chunk_offsets[i] = ((int64_t)read_be32(buf) << 32) | read_be32(buf + 4);
        } else {
            uint8_t buf[4];
            if (fread(buf, 1, 4, f) != 4) { free(chunk_offsets); fclose(f); braw_mov_free(mov); return BRAW_ERR_IO; }
            chunk_offsets[i] = (int64_t)read_be32(buf);
        }
    }

    /* Parse stsc (sample-to-chunk) to map samples to chunk offsets */
    uint64_t stsc_size;
    int64_t stsc = find_atom(f, stbl, stbl + stbl_size, TAG('s','t','s','c'), &stsc_size);

    mov->frame_offsets = calloc(sample_count, sizeof(int64_t));
    if (!mov->frame_offsets) { free(chunk_offsets); fclose(f); braw_mov_free(mov); return BRAW_ERR_ALLOC; }

    if (stsc >= 0 && stsc_size >= 8) {
        uint8_t stsc_hdr[8];
        fseeko(f, stsc, SEEK_SET);
        if (fread(stsc_hdr, 1, 8, f) != 8) { free(chunk_offsets); fclose(f); braw_mov_free(mov); return BRAW_ERR_IO; }
        uint32_t stsc_count = read_be32(stsc_hdr + 4);

        /* Read stsc entries */
        typedef struct { uint32_t first_chunk; uint32_t samples_per_chunk; uint32_t desc_idx; } StscEntry;
        StscEntry *stsc_entries = malloc(stsc_count * sizeof(StscEntry));
        if (!stsc_entries) { free(chunk_offsets); fclose(f); braw_mov_free(mov); return BRAW_ERR_ALLOC; }

        for (uint32_t i = 0; i < stsc_count; i++) {
            uint8_t buf[12];
            if (fread(buf, 1, 12, f) != 12) {
                free(stsc_entries); free(chunk_offsets); fclose(f); braw_mov_free(mov); return BRAW_ERR_IO;
            }
            stsc_entries[i].first_chunk = read_be32(buf);
            stsc_entries[i].samples_per_chunk = read_be32(buf + 4);
            stsc_entries[i].desc_idx = read_be32(buf + 8);
        }

        /* Map each sample to its file offset */
        uint32_t sample_idx = 0;
        for (uint32_t chunk = 0; chunk < chunk_count && sample_idx < sample_count; chunk++) {
            /* Find how many samples in this chunk */
            uint32_t spc = 1;
            for (uint32_t s = 0; s < stsc_count; s++) {
                if (chunk + 1 >= stsc_entries[s].first_chunk)
                    spc = stsc_entries[s].samples_per_chunk;
            }

            int64_t offset = chunk_offsets[chunk];
            for (uint32_t s = 0; s < spc && sample_idx < sample_count; s++) {
                mov->frame_offsets[sample_idx] = offset;
                offset += mov->frame_sizes[sample_idx];
                sample_idx++;
            }
        }
        free(stsc_entries);
    } else {
        /* No stsc: assume 1 sample per chunk */
        for (uint32_t i = 0; i < sample_count && i < chunk_count; i++)
            mov->frame_offsets[i] = chunk_offsets[i];
    }

    free(chunk_offsets);

    /* Compute fps from stts sample delta + mdhd timescale.
     * mov->fps currently holds the raw timescale from mdhd. */
    if (mov->fps > 0 && mov->frame_count > 0) {
        double timescale = mov->fps;
        uint64_t stts_size;
        int64_t stts = find_atom(f, stbl, stbl + stbl_size, TAG('s','t','t','s'), &stts_size);
        if (stts >= 0 && stts_size >= 12) {
            uint8_t stts_hdr[16];
            fseeko(f, stts, SEEK_SET);
            if (fread(stts_hdr, 1, 16, f) == 16) {
                uint32_t stts_entries = read_be32(stts_hdr + 4);
                if (stts_entries >= 1) {
                    /* uint32_t count = read_be32(stts_hdr + 8); */
                    uint32_t delta = read_be32(stts_hdr + 12);
                    if (delta > 0)
                        mov->fps = timescale / (double)delta;
                }
            }
        }
        /* If stts parsing failed, timescale=fps is only correct for delta=1 */
    }

    fclose(f);
    return BRAW_OK;
}

void braw_mov_free(BrawMovInfo *mov) {
    if (!mov) return;
    free(mov->frame_offsets);
    free(mov->frame_sizes);
    memset(mov, 0, sizeof(*mov));
}

int braw_mov_read_frame(const char *path, const BrawMovInfo *mov,
                        int frame_idx, uint8_t **packet_out, size_t *size_out) {
    if (frame_idx < 0 || frame_idx >= mov->frame_count)
        return BRAW_ERR_INVALID;

    FILE *f = fopen(path, "rb");
    if (!f) return BRAW_ERR_IO;

    size_t pkt_size = (size_t)mov->frame_sizes[frame_idx];
    int64_t pkt_offset = mov->frame_offsets[frame_idx];

    uint8_t *pkt = malloc(pkt_size);
    if (!pkt) { fclose(f); return BRAW_ERR_ALLOC; }

    fseeko(f, pkt_offset, SEEK_SET);
    if (fread(pkt, 1, pkt_size, f) != pkt_size) {
        free(pkt);
        fclose(f);
        return BRAW_ERR_IO;
    }

    fclose(f);
    *packet_out = pkt;
    *size_out = pkt_size;
    return BRAW_OK;
}
