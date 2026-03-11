#ifndef PRORES_RAW_DEC_H
#define PRORES_RAW_DEC_H

#include <stdint.h>

/* Decode a single ProRes RAW frame from compressed bitstream to raw Bayer RGGB16.
 * frame_data: compressed frame (starting with 4-byte size + 'prrf' marker)
 * frame_size: size of compressed data in bytes
 * bayer_out:  output buffer, must be width*height*sizeof(uint16_t) bytes
 * width, height: frame dimensions (full Bayer resolution)
 * Returns 0 on success, -1 on error. */
int prores_raw_decode_frame(const uint8_t *frame_data, int frame_size,
                            uint16_t *bayer_out, int width, int height);

#endif
