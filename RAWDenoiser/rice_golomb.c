#include "../include/prores_raw_enc.h"

// Codebooks from FFmpeg ProRes RAW decoder
// Extracted from actual decoder implementation
const int16_t prores_raw_dc_cb[DC_CB_MAX + 1] = {
    0x010, 0x021, 0x032, 0x033, 0x033, 0x033, 0x044, 0x044,
    0x044, 0x044, 0x044, 0x044, 0x076
};

const int16_t prores_raw_ac_cb[AC_CB_MAX + 1] = {
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

const int16_t prores_raw_rn_cb[RN_CB_MAX + 1] = {
    0x200, 0x100, 0x000, 0x000, 0x211, 0x211, 0x111, 0x111, 0x011, 0x011,
    0x021, 0x021, 0x222, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022, 0x022,
    0x022, 0x022, 0x022, 0x022, 0x022, 0x032, 0x032, 0x044
};

const int16_t prores_raw_ln_cb[LN_CB_MAX + 1] = {
    0x100, 0x111, 0x222, 0x222, 0x122, 0x122, 0x433, 0x433,
    0x233, 0x233, 0x233, 0x233, 0x233, 0x233, 0x033,
};

// ProRes interlaced scan table (from FFmpeg ff_prores_interlaced_scan)
const uint8_t zigzag_scan[64] = {
     0,  8,  1,  9, 16, 24, 17, 25,
     2, 10,  3, 11, 18, 26, 19, 27,
    32, 40, 33, 34, 41, 48, 56, 49,
    42, 35, 43, 50, 57, 58, 51, 59,
     4, 12,  5,  6, 13, 20, 28, 21,
    14,  7, 15, 22, 29, 36, 44, 37,
    30, 23, 31, 38, 45, 52, 60, 53,
    46, 39, 47, 54, 61, 62, 55, 63
};

// Minimal quantization matrix for maximum quality
const uint8_t default_qmat[64] = {
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1
};

/* Accumulator-based BitWriter — bits accumulate in a 64-bit register
 * and flush as complete bytes.  No memset needed. */

void init_bitwriter(BitWriter *bw, uint8_t *buffer, int size) {
    bw->buffer = buffer;
    bw->buffer_size = size;
    bw->acc = 0;
    bw->acc_bits = 0;
    bw->byte_pos = 0;
}

void put_bits(BitWriter *bw, int n, uint32_t value) {
    if (n <= 0 || n > 32) return;

    bw->acc = (bw->acc << n) | (uint64_t)(value & ((1U << n) - 1));
    bw->acc_bits += n;

    while (bw->acc_bits >= 8) {
        bw->acc_bits -= 8;
        if (bw->byte_pos < bw->buffer_size)
            bw->buffer[bw->byte_pos++] = (uint8_t)(bw->acc >> bw->acc_bits);
    }
}

void flush_bitwriter(BitWriter *bw) {
    if (bw->acc_bits > 0) {
        if (bw->byte_pos < bw->buffer_size)
            bw->buffer[bw->byte_pos++] =
                (uint8_t)(bw->acc << (8 - bw->acc_bits));
        bw->acc_bits = 0;
        bw->acc = 0;
    }
}

int bitwriter_tell(BitWriter *bw) {
    return bw->byte_pos;
}

// Encode value using Rice/Golomb as the inverse of FFmpeg's get_value().
void put_value(BitWriter *bw, uint32_t value, int16_t codebook) {
    const int16_t switch_bits = codebook >> 8;
    const int16_t rice_order  = codebook & 0xf;
    const int16_t exp_order   = (codebook >> 4) & 0xf;

    uint32_t rice_threshold = (1U << rice_order);
    uint32_t rice_max = (uint32_t)(switch_bits + 1) << rice_order;
    uint32_t switch_threshold = ((uint32_t)(switch_bits + 1) << rice_order) - (1U << exp_order);

    // PATH 1: fast path (first bit = 1)
    if (value < rice_threshold) {
        put_bits(bw, 1, 1);
        put_bits(bw, rice_order, value);
        return;
    }

    // PATH 2: Rice (unary quotient + remainder)
    // Write q zeros + 1 stop bit as a single value of 1 in (q+1) bits
    if (value < rice_max) {
        int q = value >> rice_order;
        int r = value & ((1 << rice_order) - 1);
        put_bits(bw, q + 1, 1);
        if (rice_order > 0)
            put_bits(bw, rice_order, r);
        return;
    }

    // PATH 3: exponential branch
    uint32_t diff = value - switch_threshold;

    for (int q = switch_bits + 1; q < 32; q++) {
        int bits = exp_order + (q << 1) - switch_bits;
        if (bits <= 0 || bits > 32) continue;

        uint32_t max_bits_val = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        if (diff > max_bits_val) continue;

        int first_one_bit = bits - q - 1;
        if (first_one_bit < 0) continue;
        if (diff >= (1u << (bits - q))) continue;
        if ((diff & (1u << first_one_bit)) == 0) continue;

        put_bits(bw, bits, diff);
        return;
    }

    // Fallback — should never be hit for valid input
    fprintf(stderr, "ERROR: put_value fallback! val=%u, cb=0x%03x\n", value, codebook);
    int q = 31;
    int bits = exp_order + (q << 1) - switch_bits;
    if (bits > 0 && bits <= 32) {
        uint32_t max_val = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        put_bits(bw, bits, diff > max_val ? max_val : diff);
    } else {
        put_bits(bw, 1, 1);
    }
}
