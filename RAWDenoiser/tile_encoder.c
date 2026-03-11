#include "../include/prores_raw_enc.h"
#include "../include/denoise.h"

#define TODCCODEBOOK(x) ((x + 1) >> 1)
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static uint32_t encode_first_dc_code(int coeff) {
    if (coeff >= 1) {
        return (uint32_t)((coeff - 1) << 1);
    }
    return (uint32_t)(((-coeff) << 1) | 1);
}

static uint32_t encode_dc_delta_code(int dc_add, int sign_state) {
    int mag;
    int effective_sign;

    if (dc_add > 0) {
        mag = dc_add;
        effective_sign = 0;
    } else if (dc_add < 0) {
        mag = -dc_add;
        effective_sign = 1;
    } else {
        mag = 0;
        effective_sign = sign_state;
    }

    int parity = sign_state ^ effective_sign;
    if (parity) {
        return (uint32_t)((mag << 1) - 1);
    }
    return (uint32_t)(mag << 1);
}

// Encode one Bayer component for a tile.
// ac_buf: pre-allocated buffer for AC sequence (at least 16*63 int32_t),
//         or NULL to fall back to malloc.
int encode_component(BitWriter *bw, const uint16_t *bayer_data,
                     int width, int height, int tile_x, int tile_y,
                     int component, const int32_t *qmat,
                     float noise_sigma, int32_t *ac_buf) {

    const int comp_tile_w = MIN(TILE_WIDTH, width - tile_x);
    const int comp_tile_h = MIN(TILE_HEIGHT, height - tile_y);
    const int tile_width = comp_tile_w / 2;
    const int tile_height = comp_tile_h / 2;
    const int blocks_per_row = tile_width / 8;
    const int rows = tile_height / 8;
    const int nb_blocks = blocks_per_row * rows;

    if (nb_blocks <= 0) return 0;

    int32_t blocks[16][64];
    memset(blocks, 0, sizeof(blocks));

    // Encode each 8x8 block in this component
    for (int b = 0; b < nb_blocks && b < 16; b++) {
        int br = b / blocks_per_row;
        int bc = b % blocks_per_row;
        int block_x = tile_x + (bc * 16);
        int block_y = tile_y + (br * 16);

        int need_w = 16;
        int need_h = 16;
        if (block_x >= width || block_x + need_w > width) continue;
        if (block_y >= height || block_y + need_h > height) continue;

        const uint16_t *src = bayer_data + block_y * width + block_x;

        // Compute min/max input pixel values for this block (12-bit)
        // Used by overshoot control to prevent ringing below input range
        int row_off = (component >> 1) & 1;
        int col_off = component & 1;
        int32_t block_min_12 = 4095, block_max_12 = 0;
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int32_t v = src[(y * 2 + row_off) * width + (x * 2 + col_off)] >> 4;
                if (v < block_min_12) block_min_12 = v;
                if (v > block_max_12) block_max_12 = v;
            }
        }

        // Forward DCT
        forward_dct_8x8(blocks[b], src, width, component);

        // Denoise: soft-threshold AC coefficients (no-op when disabled)
        if (g_denoise_config.enabled && noise_sigma > 0) {
            denoise_dct_block(blocks[b], noise_sigma,
                              g_denoise_config.threshold_mul);
        }

        // Quantize
        quantize_block(blocks[b], qmat);

        // Overshoot control: prevent DCT ringing from creating
        // out-of-range values (colored dot artifacts at contrast edges)
        clamp_quantized_block(blocks[b], qmat, block_min_12, block_max_12);
    }

    // Encode DC as inverse of FFmpeg prores_raw decode_dc_coeffs().
    int prev_dc = 0;
    int sign = 0;
    uint32_t prev_dc_code = 0;

    if (nb_blocks > 0) {
        uint32_t code = encode_first_dc_code(blocks[0][0]);
        put_value(bw, code, 700);
        prev_dc = (code >> 1) ^ -(code & 1);
        prev_dc_code = code;
        sign = 0;
    }

    for (int b = 1; b < nb_blocks; b++) {
        int target_prev = blocks[b][0] - 1;
        int dc_add_intended = target_prev - prev_dc;
        uint32_t dc_code = encode_dc_delta_code(dc_add_intended, sign);
        int16_t dc_codebook;

        if ((b & 15) == 1) {
            dc_codebook = 100;
        } else {
            dc_codebook = prores_raw_dc_cb[MIN(TODCCODEBOOK(prev_dc_code), DC_CB_MAX)];
        }

        put_value(bw, dc_code, dc_codebook);

        // Mirror decoder state update exactly.
        sign ^= dc_code & 1;
        int dc_add = ((-sign) ^ TODCCODEBOOK(dc_code)) + sign;
        sign = dc_add < 0;
        prev_dc += dc_add;
        prev_dc_code = dc_code;
    }

    // Encode AC as inverse of FFmpeg prores_raw decode_ac_coeffs().
    int16_t ac_codebook = 49;
    int16_t ln_codebook = 66;
    int16_t rn_codebook = 0;
    int total_ac = nb_blocks * 63;

    /* Use pre-allocated buffer if available, else fallback to malloc */
    int32_t *ac_sequence;
    int ac_heap = 0;
    if (ac_buf) {
        ac_sequence = ac_buf;
    } else {
        ac_sequence = (int32_t *)malloc(total_ac * sizeof(int32_t));
        if (!ac_sequence) return 0;
        ac_heap = 1;
    }

    int seq_idx = 0;
    for (int coef = 1; coef < 64; coef++) {
        for (int b = 0; b < nb_blocks; b++) {
            ac_sequence[seq_idx++] = blocks[b][zigzag_scan[coef]];
        }
    }

    int n = nb_blocks;
    int nb_codes = nb_blocks * 64;

    while (n < nb_codes) {
        int seq_pos = n - nb_blocks;

        // If all remaining AC coefficients are zero, stop encoding.
        // The decoder fills remaining coefficients with zero from the
        // zero-padded bitstream.  Writing explicit ln=0 + rn codes for
        // trailing zeros produces an extra byte that Apple's hardware
        // ProRes decoder rejects (HWErrorCode 0x34).
        {
            int all_zero = 1;
            for (int i = seq_pos; i < total_ac; i++) {
                if (ac_sequence[i] != 0) { all_zero = 0; break; }
            }
            if (all_zero) break;
        }

        int ln = 0;
        while (seq_pos + ln < total_ac && ac_sequence[seq_pos + ln] != 0 && ln < 65535) {
            ln++;
        }

        put_value(bw, ln, ln_codebook);
        for (int i = 0; i < ln && n + i < nb_codes; i++) {
            int32_t val = ac_sequence[seq_pos + i];
            int ac_mag = abs(val) - 1;
            if (ac_mag < 0) ac_mag = 0;

            put_value(bw, ac_mag, ac_codebook);
            put_bits(bw, 1, val < 0 ? 1 : 0);
            ac_codebook = prores_raw_ac_cb[MIN(ac_mag, AC_CB_MAX)];
        }
        n += ln;

        if (n >= nb_codes) {
            break;
        }

        // After writing the last non-zero values, if all remaining AC
        // coefficients are zero, stop here — don't write the trailing
        // rn code.  The decoder reads zeros from the zero-padded
        // bitstream for the remaining positions.
        {
            int rem_pos = n - nb_blocks;
            int all_zero = 1;
            for (int j = rem_pos; j < total_ac; j++) {
                if (ac_sequence[j] != 0) { all_zero = 0; break; }
            }
            if (all_zero) break;
        }

        seq_pos = n - nb_blocks;
        int zero_run = 0;
        while (seq_pos + zero_run < total_ac && ac_sequence[seq_pos + zero_run] == 0) {
            zero_run++;
        }

        int rn = (zero_run > 0) ? (zero_run - 1) : 0;

        put_value(bw, rn, rn_codebook);
        rn_codebook = prores_raw_rn_cb[MIN(rn, RN_CB_MAX)];
        n += rn + 1;

        if (n >= nb_codes) {
            break;
        }

        // Explicit AC after the run.
        {
            int32_t val = ac_sequence[n - nb_blocks];
            int ac_mag = abs(val) - 1;
            if (ac_mag < 0) ac_mag = 0;

            put_value(bw, ac_mag, ac_codebook);
            put_bits(bw, 1, val < 0 ? 1 : 0);
            ac_codebook = prores_raw_ac_cb[MIN(ac_mag, AC_CB_MAX)];
            ln_codebook = prores_raw_ln_cb[MIN(ac_mag, LN_CB_MAX)];
            n++;
        }
    }

    if (ac_heap) free(ac_sequence);

    flush_bitwriter(bw);
    return bitwriter_tell(bw);
}

// Encode one tile (all 4 Bayer components).
// scratch: pre-allocated per-thread scratch buffers (NULL = fallback to malloc)
int encode_tile(const uint16_t *bayer_data, int width, int height,
                int tile_x, int tile_y, uint8_t *output,
                const int32_t *qmat, int scale, TileScratch *scratch) {

    uint8_t *out_ptr = output;
    uint8_t *tile_start = output;

    // Tile header: header_len in 8-byte units, then quantizer scale.
    *out_ptr++ = 0x40;
    *out_ptr++ = (uint8_t)scale;
    uint8_t *size_ptr = out_ptr;
    out_ptr += 6;

    // Estimate noise for this tile locally (thread-safe)
    float noise_sigma = 0;
    if (g_denoise_config.enabled) {
        int tw = TILE_WIDTH;
        int th = TILE_HEIGHT;
        if (tile_x + tw > width)  tw = width  - tile_x;
        if (tile_y + th > height) th = height - tile_y;
        noise_sigma = denoise_estimate_tile_noise(bayer_data, width,
                                                   tile_x, tile_y, tw, th);
    }

    // Use pre-allocated scratch or fallback to malloc
    uint8_t *comp_buffers[4];
    int heap_alloc = 0;
    if (scratch) {
        for (int i = 0; i < 4; i++)
            comp_buffers[i] = scratch->comp_buffers[i];
    } else {
        heap_alloc = 1;
        for (int i = 0; i < 4; i++) {
            comp_buffers[i] = (uint8_t*)malloc(COMP_BUF_SIZE);
            if (!comp_buffers[i]) {
                for (int j = 0; j < i; j++) free(comp_buffers[j]);
                return 0;
            }
        }
    }

    int comp_sizes[4] = {0};

    for (int c = 0; c < 4; c++) {
        BitWriter bw;
        init_bitwriter(&bw, comp_buffers[c], COMP_BUF_SIZE);

        comp_sizes[c] = encode_component(&bw, bayer_data, width, height,
                                         tile_x, tile_y, c, qmat,
                                         noise_sigma,
                                         scratch ? scratch->ac_sequence : NULL);
    }

    // Write component sizes (only 3, component 0 is implicit)
    write_be16(size_ptr + 0, comp_sizes[2]);  // Component order: 2,1,3
    write_be16(size_ptr + 2, comp_sizes[1]);
    write_be16(size_ptr + 4, comp_sizes[3]);

    // Copy component data in order: 2, 1, 3, 0
    memcpy(out_ptr, comp_buffers[2], comp_sizes[2]);
    out_ptr += comp_sizes[2];

    memcpy(out_ptr, comp_buffers[1], comp_sizes[1]);
    out_ptr += comp_sizes[1];

    memcpy(out_ptr, comp_buffers[3], comp_sizes[3]);
    out_ptr += comp_sizes[3];

    memcpy(out_ptr, comp_buffers[0], comp_sizes[0]);
    out_ptr += comp_sizes[0];

    int total_tile_size = (int)(out_ptr - tile_start);

    if (heap_alloc) {
        for (int i = 0; i < 4; i++)
            free(comp_buffers[i]);
    }

    return total_tile_size;
}
