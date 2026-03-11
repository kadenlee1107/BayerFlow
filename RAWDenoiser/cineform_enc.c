/*
 * GoPro CineForm (CFHD) Encoder
 *
 * Encodes YUV 4:2:2 10-bit video into CineForm HD (CFHD) bitstream.
 * Self-contained pure C, no external dependencies.
 *
 * Pipeline: Input → Forward 2-6 DWT (3 levels) → Quantize → Compand → VLC → Tag bitstream
 *
 * References:
 *   - FFmpeg libavcodec/cfhdenc.c, cfhdencdsp.c (studied for bitstream format)
 *   - GoPro CineForm SDK (Apache 2.0 / MIT)
 *   - SMPTE ST 2073 (CineForm bitstream specification)
 */

#include "../include/cineform_enc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* CineForm tag IDs (SMPTE ST 2073)                                   */
/* ------------------------------------------------------------------ */

enum {
    TAG_SampleType          = 1,
    TAG_SampleIndexTable    = 2,
    TAG_BitstreamMarker     = 4,
    TAG_TransformType       = 10,
    TAG_NumFrames           = 11,
    TAG_ChannelCount        = 12,
    TAG_WaveletCount        = 13,
    TAG_SubbandCount        = 14,
    TAG_NumSpatial          = 15,
    TAG_GroupTrailer         = 18,
    TAG_ImageWidth          = 20,
    TAG_ImageHeight         = 21,
    TAG_LowpassSubband      = 25,
    TAG_NumLevels           = 26,
    TAG_LowpassWidth        = 27,
    TAG_LowpassHeight       = 28,
    TAG_PixelOffset         = 33,
    TAG_LowpassQuantization = 34,
    TAG_LowpassPrecision    = 35,
    TAG_WaveletType         = 37,
    TAG_WaveletNumber       = 38,
    TAG_WaveletLevel        = 39,
    TAG_NumBands            = 40,
    TAG_HighpassWidth       = 41,
    TAG_HighpassHeight      = 42,
    TAG_LowpassBorder       = 43,
    TAG_HighpassBorder      = 44,
    TAG_LowpassScale        = 45,
    TAG_LowpassDivisor      = 46,
    TAG_SubbandNumber       = 48,
    TAG_BandWidth           = 49,
    TAG_BandHeight          = 50,
    TAG_SubbandBand         = 51,
    TAG_BandEncoding        = 52,
    TAG_Quantization        = 53,
    TAG_BandScale           = 54,
    TAG_BandHeader          = 55,
    TAG_BandTrailer         = 56,
    TAG_ChannelNumber       = 62,
    TAG_SampleFlags         = 68,
    TAG_EncodedFormat       = 84,
    TAG_FirstWavelet        = 16,
    TAG_Precision           = 70,
    TAG_PrescaleTable       = 83,
    TAG_DisplayHeight       = 85,
    TAG_FrameNumber         = 69,
    TAG_BandCodingFlags     = 72,
};

/* Bitstream marker segment values */
enum {
    SEG_LowPassSection    = 0x1A4A,
    SEG_CoefficientData   = 0x0F0F,
    SEG_LowPassEnd        = 0x1B4B,
    SEG_HighPassBand      = 0x0D0D,
    SEG_BandData          = 0x0E0E,
    SEG_HighPassEnd       = 0x0C0C,
};

#define DWT_LEVELS  3
#define SUBBAND_COUNT 10

/* ------------------------------------------------------------------ */
/* VLC codebook (from FFmpeg cfhdenc.c)                               */
/* ------------------------------------------------------------------ */

static const struct { uint8_t len; uint32_t code; } cf_codebook[256] = {
    { 1, 0x00000000}, { 2, 0x00000002}, { 3, 0x00000007}, { 5, 0x00000019},
    { 6, 0x00000030}, { 6, 0x00000036}, { 7, 0x00000063}, { 7, 0x0000006B},
    { 7, 0x0000006F}, { 8, 0x000000D4}, { 8, 0x000000DC}, { 9, 0x00000189},
    { 9, 0x000001A0}, { 9, 0x000001AB}, {10, 0x00000310}, {10, 0x00000316},
    {10, 0x00000354}, {10, 0x00000375}, {10, 0x00000377}, {11, 0x00000623},
    {11, 0x00000684}, {11, 0x000006AB}, {11, 0x000006EC}, {12, 0x00000C44},
    {12, 0x00000C5C}, {12, 0x00000C5E}, {12, 0x00000D55}, {12, 0x00000DD1},
    {12, 0x00000DD3}, {12, 0x00000DDB}, {13, 0x0000188B}, {13, 0x000018BB},
    {13, 0x00001AA8}, {13, 0x00001BA0}, {13, 0x00001BA4}, {13, 0x00001BB5},
    {14, 0x00003115}, {14, 0x00003175}, {14, 0x0000317D}, {14, 0x00003553},
    {14, 0x00003768}, {15, 0x00006228}, {15, 0x000062E8}, {15, 0x000062F8},
    {15, 0x00006AA4}, {15, 0x00006E85}, {15, 0x00006E87}, {15, 0x00006ED3},
    {16, 0x0000C453}, {16, 0x0000C5D3}, {16, 0x0000C5F3}, {16, 0x0000DD08},
    {16, 0x0000DD0C}, {16, 0x0000DDA4}, {17, 0x000188A4}, {17, 0x00018BA5},
    {17, 0x00018BE5}, {17, 0x0001AA95}, {17, 0x0001AA97}, {17, 0x0001BA13},
    {17, 0x0001BB4A}, {17, 0x0001BB4B}, {18, 0x00031748}, {18, 0x000317C8},
    {18, 0x00035528}, {18, 0x0003552C}, {18, 0x00037424}, {18, 0x00037434},
    {18, 0x00037436}, {19, 0x00062294}, {19, 0x00062E92}, {19, 0x00062F92},
    {19, 0x0006AA52}, {19, 0x0006AA5A}, {19, 0x0006E84A}, {19, 0x0006E86A},
    {19, 0x0006E86E}, {20, 0x000C452A}, {20, 0x000C5D27}, {20, 0x000C5F26},
    {20, 0x000D54A6}, {20, 0x000D54B6}, {20, 0x000DD096}, {20, 0x000DD0D6},
    {20, 0x000DD0DE}, {21, 0x00188A56}, {21, 0x0018BA4D}, {21, 0x0018BE4E},
    {21, 0x0018BE4F}, {21, 0x001AA96E}, {21, 0x001BA12E}, {21, 0x001BA12F},
    {21, 0x001BA1AF}, {21, 0x001BA1BF}, {22, 0x00317498}, {22, 0x0035529C},
    {22, 0x0035529D}, {22, 0x003552DE}, {22, 0x003552DF}, {22, 0x0037435D},
    {22, 0x0037437D}, {23, 0x0062295D}, {23, 0x0062E933}, {23, 0x006AA53D},
    {23, 0x006AA53E}, {23, 0x006AA53F}, {23, 0x006E86B9}, {23, 0x006E86F8},
    {24, 0x00C452B8}, {24, 0x00C5D265}, {24, 0x00D54A78}, {24, 0x00D54A79},
    {24, 0x00DD0D70}, {24, 0x00DD0D71}, {24, 0x00DD0DF2}, {24, 0x00DD0DF3},
    {26, 0x03114BA2},
    {25, 0x0188A5B1}, {25, 0x0188A58B}, {25, 0x0188A595},
    {25, 0x0188A5D6}, {25, 0x0188A5D7}, {25, 0x0188A5A8}, {25, 0x0188A5AE},
    {25, 0x0188A5AF}, {25, 0x0188A5C4}, {25, 0x0188A5C5}, {25, 0x0188A587},
    {25, 0x0188A584}, {25, 0x0188A585}, {25, 0x0188A5C6}, {25, 0x0188A5C7},
    {25, 0x0188A5CC}, {25, 0x0188A5CD}, {25, 0x0188A581}, {25, 0x0188A582},
    {25, 0x0188A583}, {25, 0x0188A5CE}, {25, 0x0188A5CF}, {25, 0x0188A5C2},
    {25, 0x0188A5C3}, {25, 0x0188A5C1}, {25, 0x0188A5B4}, {25, 0x0188A5B5},
    {25, 0x0188A5E6}, {25, 0x0188A5E7}, {25, 0x0188A5E4}, {25, 0x0188A5E5},
    {25, 0x0188A5AB}, {25, 0x0188A5E0}, {25, 0x0188A5E1}, {25, 0x0188A5E2},
    {25, 0x0188A5E3}, {25, 0x0188A5B6}, {25, 0x0188A5B7}, {25, 0x0188A5FD},
    {25, 0x0188A57E}, {25, 0x0188A57F}, {25, 0x0188A5EC}, {25, 0x0188A5ED},
    {25, 0x0188A5FE}, {25, 0x0188A5FF}, {25, 0x0188A57D}, {25, 0x0188A59C},
    {25, 0x0188A59D}, {25, 0x0188A5E8}, {25, 0x0188A5E9}, {25, 0x0188A5EA},
    {25, 0x0188A5EB}, {25, 0x0188A5EF}, {25, 0x0188A57A}, {25, 0x0188A57B},
    {25, 0x0188A578}, {25, 0x0188A579}, {25, 0x0188A5BA}, {25, 0x0188A5BB},
    {25, 0x0188A5B8}, {25, 0x0188A5B9}, {25, 0x0188A588}, {25, 0x0188A589},
    {25, 0x018BA4C8}, {25, 0x018BA4C9}, {25, 0x0188A5FA}, {25, 0x0188A5FB},
    {25, 0x0188A5BC}, {25, 0x0188A5BD}, {25, 0x0188A598}, {25, 0x0188A599},
    {25, 0x0188A5F4}, {25, 0x0188A5F5}, {25, 0x0188A59B}, {25, 0x0188A5DE},
    {25, 0x0188A5DF}, {25, 0x0188A596}, {25, 0x0188A597}, {25, 0x0188A5F8},
    {25, 0x0188A5F9}, {25, 0x0188A5F1}, {25, 0x0188A58E}, {25, 0x0188A58F},
    {25, 0x0188A5DC}, {25, 0x0188A5DD}, {25, 0x0188A5F2}, {25, 0x0188A5F3},
    {25, 0x0188A58C}, {25, 0x0188A58D}, {25, 0x0188A5A4}, {25, 0x0188A5F0},
    {25, 0x0188A5A5}, {25, 0x0188A5A6}, {25, 0x0188A5A7}, {25, 0x0188A59A},
    {25, 0x0188A5A2}, {25, 0x0188A5A3}, {25, 0x0188A58A}, {25, 0x0188A5B0},
    {25, 0x0188A5A0}, {25, 0x0188A5A1}, {25, 0x0188A5DA}, {25, 0x0188A5DB},
    {25, 0x0188A59E}, {25, 0x0188A59F}, {25, 0x0188A5D8}, {25, 0x0188A5EE},
    {25, 0x0188A5D9}, {25, 0x0188A5F6}, {25, 0x0188A5F7}, {25, 0x0188A57C},
    {25, 0x0188A5C8}, {25, 0x0188A5C9}, {25, 0x0188A594}, {25, 0x0188A5FC},
    {25, 0x0188A5CA}, {25, 0x0188A5CB}, {25, 0x0188A5B2}, {25, 0x0188A5AA},
    {25, 0x0188A5B3}, {25, 0x0188A572}, {25, 0x0188A573}, {25, 0x0188A5C0},
    {25, 0x0188A5BE}, {25, 0x0188A5BF}, {25, 0x0188A592}, {25, 0x0188A580},
    {25, 0x0188A593}, {25, 0x0188A590}, {25, 0x0188A591}, {25, 0x0188A586},
    {25, 0x0188A5A9}, {25, 0x0188A5D2}, {25, 0x0188A5D3}, {25, 0x0188A5D4},
    {25, 0x0188A5D5}, {25, 0x0188A5AC}, {25, 0x0188A5AD}, {25, 0x0188A5D0},
};

#define CF_EOB_CODE  0x3114BA3
#define CF_EOB_LEN   26

/* Zero-run codebook: {code_length, code_bits, run_count} */
static const struct { uint8_t len; uint16_t code; uint16_t run; } cf_runbook_full[] = {
    { 1, 0x000,   1}, { 2, 0x000,   2}, { 3, 0x000,   3}, { 4, 0x000,   4},
    { 5, 0x000,   5}, { 6, 0x000,   6}, { 7, 0x000,   7}, { 8, 0x000,   8},
    { 9, 0x000,   9}, {10, 0x000,  10}, {11, 0x000,  11},
    { 7, 0x069,  12}, { 8, 0x0D1,  20}, { 9, 0x18A,  32},
    {10, 0x343,  60}, {11, 0x685, 100}, {13, 0x18BF, 180}, {13, 0x1BA5, 320},
};
#define CF_RUNBOOK_FULL_COUNT 18

/* Decompanding LUT: maps magnitude 0-255 → coefficient 0-1014 */
static uint16_t cf_decompand_lut[256];
/* Companding LUT: maps coefficient 0-1023 → magnitude 0-255 (inverse) */
static uint16_t cf_compand_lut[1024];
static int cf_luts_built = 0;

/* Quantization tables per subband [format_idx][plane][quality][subband] */
static const uint16_t quant_table[2][3][13][9] = {
    /* format_idx=0: YUV422P10 */
    {{
        { 16, 16,  8,  4,  4,  2,   6,   6,   9, },
        { 16, 16,  8,  4,  4,  2,   6,   6,   9, },
        { 16, 16,  8,  4,  4,  2,   7,   7,  10, },
        { 16, 16,  8,  4,  4,  2,   8,   8,  12, },
        { 16, 16,  8,  4,  4,  2,  16,  16,  26, },
        { 24, 24, 12,  6,  6,  3,  24,  24,  36, },
        { 24, 24, 12,  6,  6,  3,  24,  24,  36, },
        { 32, 32, 24,  8,  8,  6,  32,  32,  48, },
        { 32, 32, 24,  8,  8,  6,  32,  32,  48, },
        { 48, 48, 32, 12, 12,  8,  64,  64,  96, },
        { 48, 48, 32, 12, 12,  8,  64,  64,  96, },
        { 64, 64, 48, 16, 16, 12,  96,  96, 144, },
        { 64, 64, 48, 16, 16, 12, 128, 128, 192, },
    },
    {
        { 16, 16,  8,  4,  4,  2,   6,   6,   9, },
        { 16, 16,  8,  4,  4,  2,   6,   6,  12, },
        { 16, 16,  8,  4,  4,  2,   7,   7,  14, },
        { 16, 16,  8,  4,  4,  2,   8,   8,  16, },
        { 16, 16,  8,  4,  4,  2,  16,  16,  26, },
        { 24, 24, 12,  6,  6,  3,  24,  24,  36, },
        { 24, 24, 12,  6,  6,  3,  24,  24,  48, },
        { 32, 32, 24,  8,  8,  6,  32,  32,  48, },
        { 48, 48, 32, 12, 12,  8,  32,  32,  64, },
        { 48, 48, 32, 12, 12,  8,  64,  64,  96, },
        { 48, 48, 32, 12, 12,  8,  64,  64, 128, },
        { 64, 64, 48, 16, 16, 12,  96,  96, 160, },
        { 64, 64, 48, 16, 16, 12, 128, 128, 192, },
    },
    {
        { 16, 16,  8,  4,  4,  2,   6,   6,   9, },
        { 16, 16,  8,  4,  4,  2,   6,   6,  12, },
        { 16, 16,  8,  4,  4,  2,   7,   7,  14, },
        { 16, 16,  8,  4,  4,  2,   8,   8,  16, },
        { 16, 16,  8,  4,  4,  2,  16,  16,  26, },
        { 24, 24, 12,  6,  6,  3,  24,  24,  36, },
        { 24, 24, 12,  6,  6,  3,  24,  24,  48, },
        { 32, 32, 24,  8,  8,  6,  32,  32,  48, },
        { 48, 48, 32, 12, 12,  8,  32,  32,  64, },
        { 48, 48, 32, 12, 12,  8,  64,  64,  96, },
        { 48, 48, 32, 12, 12,  8,  64,  64, 128, },
        { 64, 64, 48, 16, 16, 12,  96,  96, 160, },
        { 64, 64, 48, 16, 16, 12, 128, 128, 192, },
    }},
    /* format_idx=1: GBRP12 / GBRAP12 (for future use) */
    {{
        { 16, 16,  8, 16, 16,  8,  24,  24,  36, },
        { 16, 16,  8, 16, 16,  8,  24,  24,  36, },
        { 16, 16,  8, 16, 16,  8,  32,  32,  48, },
        { 16, 16,  8, 16, 16,  8,  32,  32,  48, },
        { 16, 16,  8, 20, 20, 10,  80,  80, 128, },
        { 24, 24, 12, 24, 24, 12,  96,  96, 144, },
        { 24, 24, 12, 24, 24, 12,  96,  96, 144, },
        { 32, 32, 24, 32, 32, 24, 128, 128, 192, },
        { 32, 32, 24, 32, 32, 24, 128, 128, 192, },
        { 48, 48, 32, 48, 48, 32, 256, 256, 384, },
        { 48, 48, 32, 48, 48, 32, 256, 256, 384, },
        { 56, 56, 40, 56, 56, 40, 512, 512, 768, },
        { 64, 64, 48, 64, 64, 48, 512, 512, 768, },
    },
    {
        { 16, 16,  8, 16, 16,  8,  24,  24,  36, },
        { 16, 16,  8, 16, 16,  8,  48,  48,  72, },
        { 16, 16,  8, 16, 16,  8,  48,  48,  72, },
        { 16, 16,  8, 16, 16,  8,  64,  64,  96, },
        { 16, 16,  8, 20, 20, 10,  80,  80, 128, },
        { 24, 24, 12, 24, 24, 12,  96,  96, 144, },
        { 24, 24, 12, 24, 24, 12, 192, 192, 288, },
        { 32, 32, 24, 32, 32, 24, 128, 128, 192, },
        { 32, 32, 24, 32, 32, 24, 256, 256, 384, },
        { 48, 48, 32, 48, 48, 32, 256, 256, 384, },
        { 48, 48, 32, 48, 48, 32, 512, 512, 768, },
        { 56, 56, 40, 56, 56, 40, 512, 512, 768, },
        { 64, 64, 48, 64, 64, 48,1024,1024,1536, },
    },
    {
        { 16, 16,  8, 16, 16,  8,  24,  24,  36, },
        { 16, 16,  8, 16, 16,  8,  48,  48,  72, },
        { 16, 16,  8, 16, 16,  8,  48,  48,  72, },
        { 16, 16,  8, 16, 16,  8,  64,  64,  96, },
        { 16, 16, 10, 20, 20, 10,  80,  80, 128, },
        { 24, 24, 12, 24, 24, 12,  96,  96, 144, },
        { 24, 24, 12, 24, 24, 12, 192, 192, 288, },
        { 32, 32, 24, 32, 32, 24, 128, 128, 192, },
        { 32, 32, 24, 32, 32, 24, 256, 256, 384, },
        { 48, 48, 32, 48, 48, 32, 256, 256, 384, },
        { 48, 48, 32, 48, 48, 32, 512, 512, 768, },
        { 56, 56, 40, 56, 56, 40, 512, 512, 768, },
        { 64, 64, 48, 64, 64, 48,1024,1024,1536, },
    }},
};

/* ------------------------------------------------------------------ */
/* BitWriter (MSB-first, 64-bit accumulator)                          */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    int capacity;
    int byte_pos;
    uint64_t acc;
    int acc_bits;
} BitWriter;

static void bw_init(BitWriter *bw, uint8_t *buf, int cap) {
    bw->data = buf;
    bw->capacity = cap;
    bw->byte_pos = 0;
    bw->acc = 0;
    bw->acc_bits = 0;
}

static inline void bw_put(BitWriter *bw, int n, uint32_t val) {
    bw->acc = (bw->acc << n) | (uint64_t)(val & ((1U << n) - 1));
    bw->acc_bits += n;
    while (bw->acc_bits >= 8) {
        bw->acc_bits -= 8;
        if (bw->byte_pos < bw->capacity)
            bw->data[bw->byte_pos++] = (uint8_t)(bw->acc >> bw->acc_bits);
    }
}

static void bw_flush(BitWriter *bw) {
    if (bw->acc_bits > 0) {
        if (bw->byte_pos < bw->capacity)
            bw->data[bw->byte_pos++] = (uint8_t)(bw->acc << (8 - bw->acc_bits));
        bw->acc_bits = 0;
        bw->acc = 0;
    }
}

/* ------------------------------------------------------------------ */
/* Byte writer helpers                                                */
/* ------------------------------------------------------------------ */

static inline void put_be16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v);
}

static inline void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

static inline void put_tag(uint8_t *p, int16_t tag, uint16_t val) {
    put_be16(p, (uint16_t)tag);
    put_be16(p + 2, val);
}

/* ------------------------------------------------------------------ */
/* Companding / Decompanding LUT                                       */
/* ------------------------------------------------------------------ */

static void build_luts(void) {
    if (cf_luts_built) return;

    /* Decompanding: maps magnitude (0-255) → coefficient (0-1014) */
    for (int i = 0; i < 256; i++)
        cf_decompand_lut[i] = (uint16_t)(i + (768LL * i * i * i) / (256 * 256 * 256));

    /* Companding: inverse of decompanding. Maps coefficient → magnitude. */
    memset(cf_compand_lut, 0, sizeof(cf_compand_lut));
    int last = 0;
    for (int i = 0; i < 256; i++) {
        int idx = cf_decompand_lut[i];
        if (idx < 1024) cf_compand_lut[idx] = (uint16_t)i;
    }
    /* Gap-fill: any unmapped entry gets nearest-lower magnitude */
    for (int i = 0; i < 1024; i++) {
        if (cf_compand_lut[i])
            last = cf_compand_lut[i];
        else
            cf_compand_lut[i] = (uint16_t)last;
    }

    cf_luts_built = 1;
}

/* ------------------------------------------------------------------ */
/* Encoder context                                                     */
/* ------------------------------------------------------------------ */

/* Pre-built encoder codebook: index by two's-complement 9-bit value */
typedef struct { uint32_t bits; int size; } EncCodebook;

/* Pre-built runbook: index by run count (0-320) */
typedef struct { uint32_t bits; int size; int run; } EncRunbook;

struct CfEncoder {
    int width, height;
    int quality;
    int planes;

    /* Per-plane DWT buffers */
    int16_t *dwt_buf[4];   /* main buffer (subbands mapped into this) */
    int16_t *dwt_tmp[4];   /* temp buffer for DWT */

    /* Subband pointers (into dwt_buf) */
    int16_t *subband[4][SUBBAND_COUNT];

    /* Band dimensions */
    int band_width[DWT_LEVELS];
    int band_height[DWT_LEVELS];
    int band_a_width[DWT_LEVELS];   /* aligned width (with padding) */

    /* Per-plane quantization */
    unsigned plane_quant[4][SUBBAND_COUNT];

    /* Encoder lookup tables */
    EncCodebook cb[513];    /* 0-511 = coefficients, 512 = EOB */
    EncRunbook  rb[321];    /* run counts 0-320 */
};

/* ------------------------------------------------------------------ */
/* Forward 2-6 Biorthogonal DWT (from FFmpeg cfhdencdsp.c)            */
/* ------------------------------------------------------------------ */

static inline int16_t clip16(int v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* 1D forward filter: splits input into low + high subbands */
static void fwd_filter(const int16_t *input, int in_stride,
                       int16_t *low, int low_stride,
                       int16_t *high, int high_stride,
                       int len)
{
    /* First pair (boundary) */
    low[0] = clip16(input[0 * in_stride] + input[1 * in_stride]);
    high[0] = clip16((5 * input[0 * in_stride] - 11 * input[1 * in_stride] +
                      4 * input[2 * in_stride] +  4 * input[3 * in_stride] -
                      1 * input[4 * in_stride] -  1 * input[5 * in_stride] + 4) >> 3);

    /* Interior pairs */
    for (int i = 2; i < len - 2; i += 2) {
        low[(i >> 1) * low_stride] = clip16(input[i * in_stride] + input[(i + 1) * in_stride]);
        high[(i >> 1) * high_stride] = clip16(
            ((-input[(i - 2) * in_stride] - input[(i - 1) * in_stride] +
               input[(i + 2) * in_stride] + input[(i + 3) * in_stride] + 4) >> 3) +
             input[i * in_stride] - input[(i + 1) * in_stride]);
    }

    /* Last pair (boundary) */
    int last = len - 2;
    low[(last >> 1) * low_stride] = clip16(input[last * in_stride] + input[(last + 1) * in_stride]);
    high[(last >> 1) * high_stride] = clip16(
        (11 * input[last * in_stride] - 5 * input[(last + 1) * in_stride] -
          4 * input[(last - 1) * in_stride] - 4 * input[(last - 2) * in_stride] +
          1 * input[(last - 3) * in_stride] + 1 * input[(last - 4) * in_stride] + 4) >> 3);
}

static void fwd_horiz(const int16_t *input, int in_stride,
                      int16_t *low, int low_stride,
                      int16_t *high, int high_stride,
                      int width, int height)
{
    /* Copy each row to temp to avoid aliasing when output overlaps input */
    int16_t tmp[4096];  /* max width we support */
    for (int i = 0; i < height; i++) {
        memcpy(tmp, input, width * sizeof(int16_t));
        fwd_filter(tmp, 1, low, 1, high, 1, width);
        input += in_stride;
        low += low_stride;
        high += high_stride;
    }
}

static void fwd_vert(const int16_t *input, int in_stride,
                     int16_t *low, int low_stride,
                     int16_t *high, int high_stride,
                     int width, int height)
{
    /* Copy each column to temp to avoid aliasing when output overlaps input */
    int16_t tmp[4096];  /* max height we support */
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++)
            tmp[j] = input[i + j * in_stride];
        fwd_filter(tmp, 1, &low[i], low_stride, &high[i], high_stride, height);
    }
}

/* ------------------------------------------------------------------ */
/* VLC encoding                                                        */
/* ------------------------------------------------------------------ */

static void put_runcode(BitWriter *bw, int count, const EncRunbook *rb) {
    while (count > 0) {
        int idx = count < 320 ? count : 320;
        bw_put(bw, rb[idx].size, rb[idx].bits);
        count -= rb[idx].run;
    }
}

static void encode_band_vlc(BitWriter *bw, const int16_t *data, int width, int a_width,
                            int height, int stride, const EncCodebook *cb, const EncRunbook *rb)
{
    int count = 0;
    for (int m = 0; m < height; m++) {
        for (int j = 0; j < stride; j++) {
            int16_t val = j >= width ? 0 : data[j];

            /* Compand: map coefficient → magnitude, apply sign */
            int abs_val = val < 0 ? -val : val;
            int mag = abs_val < 1024 ? cf_compand_lut[abs_val] : 255;
            int index = val < 0 ? -mag : (val > 0 ? mag : 0);
            if (index < 0) index += 512;

            if (index == 0) {
                count++;
            } else {
                if (count > 0) {
                    put_runcode(bw, count, rb);
                    count = 0;
                }
                bw_put(bw, cb[index].size, cb[index].bits);
            }
        }
        data += a_width;
    }

    /* Flush trailing zeros */
    if (count > 0)
        put_runcode(bw, count, rb);

    /* EOB */
    bw_put(bw, cb[512].size, cb[512].bits);
    bw_flush(bw);
}

/* ------------------------------------------------------------------ */
/* Quantization                                                        */
/* ------------------------------------------------------------------ */

static void quantize_band(int16_t *data, int width, int a_width,
                          int height, unsigned quant)
{
    int16_t factor = (int16_t)((1U << 15) / quant);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int v = data[j];
            int sign = v < 0 ? -1 : 1;
            data[j] = clip16((v * factor + 16384 * sign) / 32768);
        }
        data += a_width;
    }
}

/* ------------------------------------------------------------------ */
/* Encoder API                                                         */
/* ------------------------------------------------------------------ */

int cf_enc_init(CfEncoder **enc_out, const CfEncConfig *cfg) {
    if (!enc_out || !cfg) return CF_ENC_ERR_PARAM;
    if (cfg->width & 15) return CF_ENC_ERR_PARAM;
    if (cfg->height < 32) return CF_ENC_ERR_PARAM;
    if (cfg->quality < 0 || cfg->quality > 12) return CF_ENC_ERR_PARAM;

    build_luts();

    CfEncoder *e = (CfEncoder *)calloc(1, sizeof(CfEncoder));
    if (!e) return CF_ENC_ERR_MEM;

    e->width = cfg->width;
    e->height = ((cfg->height + 7) / 8) * 8;  /* align to 8 */
    e->quality = cfg->quality;
    e->planes = 3;  /* YUV422 */

    /* Compute band dimensions for each level */
    for (int p = 0; p < e->planes; p++) {
        int pw = (p == 0) ? e->width : (e->width / 2);  /* chroma is half-width */
        int ph = e->height;
        int w8 = pw / 8 + 64;  /* padded width at finest level */
        int h8 = ph / 8;

        e->dwt_buf[p] = (int16_t *)calloc((size_t)h8 * 8 * w8 * 8, sizeof(int16_t));
        e->dwt_tmp[p] = (int16_t *)calloc((size_t)h8 * 8 * w8 * 8, sizeof(int16_t));
        if (!e->dwt_buf[p] || !e->dwt_tmp[p]) { cf_enc_close(e); *enc_out = NULL; return CF_ENC_ERR_MEM; }

        /* Map subbands into dwt_buf (matching FFmpeg cfhdenc.c layout) */
        int w2 = w8 * 4, h2 = h8 * 4;
        int w4 = w8 * 2, h4 = h8 * 2;

        e->subband[p][0] = e->dwt_buf[p];                      /* LL */
        e->subband[p][1] = e->dwt_buf[p] + 2 * w8 * h8;       /* level0 LH */
        e->subband[p][2] = e->dwt_buf[p] + 1 * w8 * h8;       /* level0 HL */
        e->subband[p][3] = e->dwt_buf[p] + 3 * w8 * h8;       /* level0 HH */
        e->subband[p][4] = e->dwt_buf[p] + 2 * w4 * h4;       /* level1 LH */
        e->subband[p][5] = e->dwt_buf[p] + 1 * w4 * h4;       /* level1 HL */
        e->subband[p][6] = e->dwt_buf[p] + 3 * w4 * h4;       /* level1 HH */
        e->subband[p][7] = e->dwt_buf[p] + 2 * w2 * h2;       /* level2 LH */
        e->subband[p][8] = e->dwt_buf[p] + 1 * w2 * h2;       /* level2 HL */
        e->subband[p][9] = e->dwt_buf[p] + 3 * w2 * h2;       /* level2 HH */

        /* Set per-subband quantization */
        int plane_idx = (p >= 3) ? 0 : p;
        for (int s = 0; s < 9; s++)
            e->plane_quant[p][1 + s] = quant_table[0][plane_idx][cfg->quality][s];
    }

    /* Compute band dimensions for each DWT level */
    {
        int pw = e->width;
        int ph = e->height;
        for (int l = 0; l < DWT_LEVELS; l++) {
            /* At level l, band dimensions = (width/8)<<l × (height>>(3-l)) */
            e->band_width[l] = (pw / 8) << l;
            e->band_height[l] = ph >> (DWT_LEVELS - l);
            e->band_a_width[l] = (pw / 8 + 64) << l;
        }
    }

    /* Build encoder codebook (two's complement 9-bit indexing) */
    for (int i = 0; i < 512; i++) {
        int value = (i & 256) ? (-256 + (i & 255)) : i;
        int mag = value < 0 ? -value : value;
        if (mag > 255) mag = 255;

        if (mag) {
            e->cb[i].bits = (cf_codebook[mag].code << 1) | (value > 0 ? 0 : 1);
            e->cb[i].size = cf_codebook[mag].len + 1;
        } else {
            e->cb[i].bits = cf_codebook[0].code;
            e->cb[i].size = cf_codebook[0].len;
        }
    }
    e->cb[512].bits = CF_EOB_CODE;
    e->cb[512].size = CF_EOB_LEN;

    /* Build encoder runbook */
    e->rb[0].run = 0;
    for (int i = 1, j = 0; i < 320 && j < CF_RUNBOOK_FULL_COUNT - 1; j++) {
        int run = cf_runbook_full[j].run;
        int end = cf_runbook_full[j + 1].run;
        while (i < end) {
            e->rb[i].run = run;
            e->rb[i].bits = cf_runbook_full[j].code;
            e->rb[i].size = cf_runbook_full[j].len;
            i++;
        }
    }
    e->rb[320].bits = cf_runbook_full[CF_RUNBOOK_FULL_COUNT - 1].code;
    e->rb[320].size = cf_runbook_full[CF_RUNBOOK_FULL_COUNT - 1].len;
    e->rb[320].run  = 320;

    *enc_out = e;
    return CF_ENC_OK;
}

int cf_enc_encode_frame(CfEncoder *enc,
                        const uint16_t *planes[3], const int strides[3],
                        uint8_t *output, int output_size)
{
    if (!enc || !planes || !strides || !output) return CF_ENC_ERR_PARAM;

    int pos = 0;
    #define CHECK_SPACE(n) do { if (pos + (n) > output_size) return CF_ENC_ERR_SIZE; } while(0)
    #define PUT_TAG(t, v) do { CHECK_SPACE(4); put_tag(output + pos, (t), (v)); pos += 4; } while(0)
    #define PUT_TAG_NEG(t, v) do { CHECK_SPACE(4); put_be16(output + pos, (uint16_t)(-(t))); put_be16(output + pos + 2, (v)); pos += 4; } while(0)

    /* ---- Forward DWT for each plane ---- */
    for (int p = 0; p < enc->planes; p++) {
        int pw = (p == 0) ? enc->width : (enc->width / 2);
        int ph = enc->height;
        int w8 = pw / 8 + 64;
        int h8 = ph / 8;
        int w4 = w8 * 2, h4 = h8 * 2;
        int w2 = w8 * 4, h2 = h8 * 4;

        /* FFmpeg convention: swap Cb/Cr for input reading */
        int act_p_in = (p == 1) ? 2 : (p == 2) ? 1 : p;

        /* Copy input into dwt_buf as int16_t (separate from dwt_tmp to avoid aliasing) */
        int16_t *buf = enc->dwt_buf[p];
        int src_stride = strides[act_p_in];
        const uint16_t *src = planes[act_p_in];
        for (int y = 0; y < ph; y++) {
            for (int x = 0; x < pw; x++)
                buf[x] = (int16_t)src[x];
            for (int x = pw; x < w2 * 2; x++)
                buf[x] = 0;
            buf += w2 * 2;
            src += src_stride;
        }

        /* Level 1 (finest): input(w2*2, h2*2) → low(w2, h2) + high bands */
        int16_t *input_l1 = enc->dwt_buf[p];  /* separate buffer from dwt_tmp */

        /* Horizontal split — output to dwt_tmp (l_h[6] and l_h[7] in FFmpeg) */
        int16_t *h_low = enc->dwt_tmp[p];           /* l_h[6] = dwt_tmp base */
        int16_t *h_high = enc->dwt_tmp[p] + 2 * w2 * h2;  /* l_h[7] = dwt_tmp + 2*w2*h2 */

        fwd_horiz(input_l1, w2 * 2, h_low, w2, h_high, w2, pw, ph);

        /* Vertical split of high → subband[7] (LH) + subband[9] (HH) */
        int16_t *v_input_h = h_high;
        fwd_vert(v_input_h, w2,
                 enc->subband[p][7], w2, enc->subband[p][9], w2,
                 pw / 2, ph);

        /* Vertical split of low → l_h[7] (temp) + subband[8] (HL) */
        int16_t *v_input_l = h_low;
        int16_t *l1_out = h_high;  /* reuse as temp */
        fwd_vert(v_input_l, w2,
                 l1_out, w2, enc->subband[p][8], w2,
                 pw / 2, ph);

        /* Prescale: divide by 4 before level 2 (PrescaleTable=0x2000 for YUV422P10) */
        {
            int16_t *prescale_buf = l1_out;
            int prescale_w = pw / 2;
            int prescale_h = ph / 2;
            for (int y = 0; y < prescale_h; y++) {
                for (int x = 0; x < prescale_w; x++)
                    prescale_buf[x] /= 4;
                prescale_buf += w2;
            }
        }

        /* Level 2: input(w4, h4) → subbands 4-6 */
        int16_t *input_l2 = l1_out;
        int16_t *h2_low = enc->dwt_tmp[p];              /* l_h[3] = dwt_tmp base */
        int16_t *h2_high = enc->dwt_tmp[p] + 2 * w4 * h4;  /* l_h[4] = dwt_tmp + 2*w4*h4 */

        fwd_horiz(input_l2, w2, h2_low, w4, h2_high, w4, pw / 2, ph / 2);

        fwd_vert(h2_high, w4,
                 enc->subband[p][4], w4, enc->subband[p][6], w4,
                 pw / 4, ph / 2);

        fwd_vert(h2_low, w4,
                 h2_high, w4, enc->subband[p][5], w4,
                 pw / 4, ph / 2);

        /* Level 3 (coarsest): input(w8*2, h8*2) → subbands 0-3 */
        int16_t *input_l3 = h2_high;
        int16_t *h3_low = enc->dwt_tmp[p];      /* l_h[0] */
        int16_t *h3_high = enc->dwt_tmp[p] + w8 * h8 * 2; /* l_h[1] */

        fwd_horiz(input_l3, w4, h3_low, w8, h3_high, w8, pw / 4, ph / 4);

        /* Vertical split: high → subband[1] (LH) + subband[3] (HH) */
        fwd_vert(h3_high, w8,
                 enc->subband[p][1], w8, enc->subband[p][3], w8,
                 pw / 8, ph / 4);

        /* Vertical split: low → subband[0] (LL) + subband[2] (HL) */
        fwd_vert(h3_low, w8,
                 enc->subband[p][0], w8, enc->subband[p][2], w8,
                 pw / 8, ph / 4);
    }

    /* ---- Quantize highpass subbands ---- */
    for (int p = 0; p < enc->planes; p++) {
        int pw = (p == 0) ? enc->width : (enc->width / 2);
        int w8 = pw / 8 + 64;
        (void)0; /* h8 unused */
        for (int l = 0; l < DWT_LEVELS; l++) {
            int bw = (pw / 8) << l;
            int bh = enc->height >> (DWT_LEVELS - l);
            int aw = w8 << l;
            for (int i = 0; i < 3; i++) {
                int sb = 1 + l * 3 + i;
                quantize_band(enc->subband[p][sb], bw, aw, bh, enc->plane_quant[p][sb]);
            }
        }
    }

    /* ---- Write CFHD bitstream ---- */

    /* Frame header */
    PUT_TAG(TAG_SampleType, 9);
    PUT_TAG(TAG_SampleIndexTable, enc->planes);
    int sample_index_pos = pos;
    for (int i = 0; i < enc->planes; i++) {
        CHECK_SPACE(4);
        put_be32(output + pos, 0);  /* placeholder for plane sizes */
        pos += 4;
    }

    PUT_TAG(TAG_TransformType, 0);
    PUT_TAG(TAG_NumFrames, 1);
    PUT_TAG(TAG_ChannelCount, enc->planes);
    PUT_TAG(TAG_EncodedFormat, 1);  /* YUV422 */
    PUT_TAG(TAG_WaveletCount, 3);
    PUT_TAG(TAG_SubbandCount, SUBBAND_COUNT);
    PUT_TAG(TAG_NumSpatial, 2);
    PUT_TAG(TAG_FirstWavelet, 3);
    PUT_TAG(TAG_ImageWidth, enc->width);
    PUT_TAG(TAG_ImageHeight, enc->height);
    PUT_TAG_NEG(TAG_DisplayHeight, enc->height);  /* negative tag */
    PUT_TAG(TAG_Precision, 10);
    PUT_TAG(TAG_PrescaleTable, 0x2000);
    PUT_TAG(TAG_SampleFlags, 1);

    /* ---- Per-plane encoding ---- */
    for (int p = 0; p < enc->planes; p++) {
        int pw = (p == 0) ? enc->width : (enc->width / 2);
        int w8 = pw / 8 + 64;
        (void)0; /* h8 unused */
        int plane_start = pos;

        if (p > 0) {
            PUT_TAG(TAG_SampleType, 3);
            PUT_TAG(TAG_ChannelNumber, p);
        }

        /* Lowpass section */
        PUT_TAG(TAG_BitstreamMarker, SEG_LowPassSection);

        int lp_width = pw / 8;
        int lp_height = enc->height / 8;

        PUT_TAG(TAG_LowpassSubband, 0);
        PUT_TAG(TAG_NumLevels, 3);
        PUT_TAG(TAG_LowpassWidth, lp_width);
        PUT_TAG(TAG_LowpassHeight, lp_height);
        PUT_TAG(TAG_PixelOffset, 0);
        PUT_TAG(TAG_LowpassQuantization, 1);
        PUT_TAG(TAG_LowpassPrecision, 16);

        PUT_TAG(TAG_BitstreamMarker, SEG_CoefficientData);

        /* Write lowpass coefficients as raw BE16 */
        {
            int16_t *lp_data = enc->subband[p][0];
            int lp_a_width = w8;
            CHECK_SPACE(lp_width * lp_height * 2);
            for (int y = 0; y < lp_height; y++) {
                for (int x = 0; x < lp_width; x++) {
                    put_be16(output + pos, (uint16_t)lp_data[x]);
                    pos += 2;
                }
                lp_data += lp_a_width;
            }
        }

        PUT_TAG(TAG_BitstreamMarker, SEG_LowPassEnd);

        /* Highpass levels */
        for (int l = 0; l < DWT_LEVELS; l++) {
            int bw = (pw / 8) << l;
            int bh = enc->height >> (DWT_LEVELS - l);
            int aw = w8 << l;
            int stride_aligned = ((bw + 7) / 8) * 8;

            PUT_TAG(TAG_BitstreamMarker, SEG_HighPassBand);
            PUT_TAG(TAG_WaveletType, l == 2 ? 5 : 3);
            PUT_TAG(TAG_WaveletNumber, 3 - l);
            PUT_TAG(TAG_WaveletLevel, 3 - l);
            PUT_TAG(TAG_NumBands, 4);
            PUT_TAG(TAG_HighpassWidth, bw);
            PUT_TAG(TAG_HighpassHeight, bh);
            PUT_TAG(TAG_LowpassBorder, 0);
            PUT_TAG(TAG_HighpassBorder, 0);
            PUT_TAG(TAG_LowpassScale, 1);
            PUT_TAG(TAG_LowpassDivisor, 1);

            /* 3 subbands per level: LH, HL, HH */
            for (int i = 0; i < 3; i++) {
                int sb = 1 + l * 3 + i;

                PUT_TAG(TAG_BitstreamMarker, SEG_BandData);
                PUT_TAG(TAG_SubbandNumber, i + 1);
                PUT_TAG(TAG_BandCodingFlags, 1);
                PUT_TAG(TAG_BandWidth, bw);
                PUT_TAG(TAG_BandHeight, bh);
                PUT_TAG(TAG_SubbandBand, sb);
                PUT_TAG(TAG_BandEncoding, 3);
                PUT_TAG(TAG_Quantization, enc->plane_quant[p][sb]);
                PUT_TAG(TAG_BandScale, 1);
                PUT_TAG(TAG_BandHeader, 0);

                /* VLC encode this band */
                int vlc_space = output_size - pos;
                if (vlc_space < 1024) return CF_ENC_ERR_SIZE;

                BitWriter bw_vlc;
                bw_init(&bw_vlc, output + pos, vlc_space);

                encode_band_vlc(&bw_vlc, enc->subband[p][sb], bw, aw, bh,
                                stride_aligned, enc->cb, enc->rb);

                pos += bw_vlc.byte_pos;

                /* Pad to 4-byte boundary */
                int pad = (4 - (pos & 3)) & 3;
                CHECK_SPACE(pad);
                while (pad--) output[pos++] = 0;

                PUT_TAG(TAG_BandTrailer, 0);
            }

            PUT_TAG(TAG_BitstreamMarker, SEG_HighPassEnd);
        }

        /* Record plane size */
        int plane_size = pos - plane_start;
        put_be32(output + sample_index_pos + p * 4, (uint32_t)plane_size);
    }

    PUT_TAG(TAG_GroupTrailer, 0);

    return pos;  /* packet size */
}

void cf_enc_close(CfEncoder *enc) {
    if (!enc) return;
    for (int p = 0; p < 4; p++) {
        free(enc->dwt_buf[p]);
        free(enc->dwt_tmp[p]);
    }
    free(enc);
}

/* ------------------------------------------------------------------ */
/* MOV Container Writer (streaming: ftyp → wide → mdat → moov)        */
/* ------------------------------------------------------------------ */

#define CF_MOV_MAX_FRAMES 100000

struct CfMovWriter {
    FILE *fp;
    CfEncoder *enc;
    int width, height;
    int frame_count;
    uint32_t *frame_sizes;
    uint64_t *frame_offsets;
    long mdat_pos;          /* file offset of mdat atom start */
    int timescale;
    int sample_duration;
    uint8_t *packet_buf;
    int packet_buf_size;
};

static void cf_compute_timing(double fps, int *timescale, int *sample_dur) {
    if      (fps > 59.93 && fps < 59.95) { *timescale = 60000; *sample_dur = 1001; }
    else if (fps > 49.99 && fps < 50.01) { *timescale = 50;    *sample_dur = 1;    }
    else if (fps > 47.95 && fps < 47.97) { *timescale = 48000; *sample_dur = 1001; }
    else if (fps > 29.96 && fps < 29.98) { *timescale = 30000; *sample_dur = 1001; }
    else if (fps > 24.99 && fps < 25.01) { *timescale = 25;    *sample_dur = 1;    }
    else if (fps > 23.97 && fps < 23.98) { *timescale = 24000; *sample_dur = 1001; }
    else if (fps > 23.99 && fps < 24.01) { *timescale = 24;    *sample_dur = 1;    }
    else { *timescale = (int)(fps + 0.5); *sample_dur = 1; }
}

int cf_mov_writer_open(CfMovWriter **out, const char *filename,
                       int width, int height, double fps, int quality)
{
    if (!out || !filename) return CF_ENC_ERR_PARAM;

    CfMovWriter *w = (CfMovWriter *)calloc(1, sizeof(CfMovWriter));
    if (!w) return CF_ENC_ERR_MEM;

    w->width = width;
    w->height = height;
    cf_compute_timing(fps, &w->timescale, &w->sample_duration);

    /* Init encoder */
    CfEncConfig cfg = { .width = width, .height = height, .quality = quality };
    int ret = cf_enc_init(&w->enc, &cfg);
    if (ret != CF_ENC_OK) { free(w); return ret; }

    /* Alloc tracking arrays */
    w->frame_sizes   = (uint32_t *)calloc(CF_MOV_MAX_FRAMES, sizeof(uint32_t));
    w->frame_offsets = (uint64_t *)calloc(CF_MOV_MAX_FRAMES, sizeof(uint64_t));
    w->packet_buf_size = width * height * 4 + 65536;
    w->packet_buf = (uint8_t *)malloc(w->packet_buf_size);
    if (!w->frame_sizes || !w->frame_offsets || !w->packet_buf) {
        cf_enc_close(w->enc);
        free(w->frame_sizes); free(w->frame_offsets); free(w->packet_buf);
        free(w); return CF_ENC_ERR_MEM;
    }

    /* Open file */
    w->fp = fopen(filename, "wb");
    if (!w->fp) {
        cf_enc_close(w->enc);
        free(w->frame_sizes); free(w->frame_offsets); free(w->packet_buf);
        free(w); return CF_ENC_ERR_PARAM;
    }

    /* Write ftyp atom */
    uint8_t ftyp[] = {
        0,0,0,0x14, 'f','t','y','p', 'q','t',' ',' ', 0,0,0,0, 'q','t',' ',' '
    };
    fwrite(ftyp, 1, sizeof(ftyp), w->fp);

    /* Write wide atom (padding before mdat) */
    uint8_t wide[8] = {0,0,0,8, 'w','i','d','e'};
    fwrite(wide, 1, 8, w->fp);

    /* Write mdat header (size=0 placeholder, patched at close) */
    w->mdat_pos = ftell(w->fp);
    uint8_t mdat_hdr[8] = {0,0,0,0, 'm','d','a','t'};
    fwrite(mdat_hdr, 1, 8, w->fp);

    *out = w;
    return CF_ENC_OK;
}

int cf_mov_writer_add_frame(CfMovWriter *w,
                            const uint16_t *planes[3], const int strides[3])
{
    if (!w || !planes || !strides) return CF_ENC_ERR_PARAM;
    if (w->frame_count >= CF_MOV_MAX_FRAMES) return CF_ENC_ERR_SIZE;

    int pkt_size = cf_enc_encode_frame(w->enc, planes, strides,
                                       w->packet_buf, w->packet_buf_size);
    if (pkt_size < 0) return pkt_size;

    w->frame_offsets[w->frame_count] = (uint64_t)ftell(w->fp);
    w->frame_sizes[w->frame_count]   = (uint32_t)pkt_size;
    fwrite(w->packet_buf, 1, pkt_size, w->fp);
    w->frame_count++;
    return CF_ENC_OK;
}

void cf_mov_writer_close(CfMovWriter *w) {
    if (!w) return;
    if (!w->fp) goto cleanup;

    int n = w->frame_count;
    int width = w->width, height = w->height;
    int ts = w->timescale, sd = w->sample_duration;
    int total_dur = n * sd;

    /* Patch mdat size */
    long mdat_end = ftell(w->fp);
    uint32_t mdat_size = (uint32_t)(mdat_end - w->mdat_pos);
    fseek(w->fp, w->mdat_pos, SEEK_SET);
    uint8_t sz[4];
    put_be32(sz, mdat_size);
    fwrite(sz, 1, 4, w->fp);
    fseek(w->fp, mdat_end, SEEK_SET);

    /* Build moov atom */
    int stsd_entry_size = 86;
    int stsd_size = 16 + stsd_entry_size;
    int stts_size = 24;
    int stsc_size = 28;
    int stsz_size = 20 + n * 4;
    int co64_size = 16 + n * 8;
    int stbl_size = 8 + stsd_size + stts_size + stsc_size + stsz_size + co64_size;
    int dinf_size = 36;
    int vmhd_size = 20;
    int minf_size = 8 + vmhd_size + dinf_size + stbl_size;
    int hdlr_size = 33;
    int mdhd_size = 32;
    int mdia_size = 8 + mdhd_size + hdlr_size + minf_size;
    int tkhd_size = 92;
    int trak_size = 8 + tkhd_size + mdia_size;
    int mvhd_size = 108;
    int moov_size = 8 + mvhd_size + trak_size;

    uint8_t *moov = (uint8_t *)calloc(moov_size, 1);
    if (!moov) goto cleanup;
    int mp = 0;

    #define MW32(v) do { put_be32(moov + mp, (v)); mp += 4; } while(0)
    #define MW16(v) do { put_be16(moov + mp, (v)); mp += 2; } while(0)
    #define MW8(v)  do { moov[mp++] = (uint8_t)(v); } while(0)
    #define MFCC(a,b,c,d) do { moov[mp++]=(a); moov[mp++]=(b); moov[mp++]=(c); moov[mp++]=(d); } while(0)

    /* moov */
    MW32(moov_size); MFCC('m','o','o','v');

    /* mvhd */
    MW32(mvhd_size); MFCC('m','v','h','d');
    MW32(0); MW32(0); MW32(0); /* version/flags, creation, modification */
    MW32(ts); MW32(total_dur);
    MW32(0x00010000); MW16(0x0100); /* rate=1.0, volume=1.0 */
    for (int i = 0; i < 10; i++) MW8(0);
    MW32(0x00010000); MW32(0); MW32(0);
    MW32(0); MW32(0x00010000); MW32(0);
    MW32(0); MW32(0); MW32(0x40000000);
    for (int i = 0; i < 24; i++) MW8(0);
    MW32(2); /* next_track_ID */

    /* trak */
    MW32(trak_size); MFCC('t','r','a','k');

    /* tkhd */
    MW32(tkhd_size); MFCC('t','k','h','d');
    MW32(3); MW32(0); MW32(0); /* flags, creation, modification */
    MW32(1); MW32(0); /* track_ID, reserved */
    MW32(total_dur);
    MW32(0); MW32(0); MW16(0); MW16(0); MW16(0); MW16(0);
    MW32(0x00010000); MW32(0); MW32(0);
    MW32(0); MW32(0x00010000); MW32(0);
    MW32(0); MW32(0); MW32(0x40000000);
    MW32(width << 16); MW32(height << 16);

    /* mdia */
    MW32(mdia_size); MFCC('m','d','i','a');

    /* mdhd */
    MW32(mdhd_size); MFCC('m','d','h','d');
    MW32(0); MW32(0); MW32(0);
    MW32(ts); MW32(total_dur);
    MW32(0);

    /* hdlr */
    MW32(hdlr_size); MFCC('h','d','l','r');
    MW32(0); MW32(0);
    MFCC('v','i','d','e');
    MW32(0); MW32(0); MW32(0);
    MW8(0); /* name */

    /* minf */
    MW32(minf_size); MFCC('m','i','n','f');

    /* vmhd */
    MW32(vmhd_size); MFCC('v','m','h','d');
    MW32(1); MW16(0); MW16(0); MW16(0); MW16(0);

    /* dinf */
    MW32(dinf_size); MFCC('d','i','n','f');
    MW32(28); MFCC('d','r','e','f');
    MW32(0); MW32(1);
    MW32(12); MFCC('u','r','l',' ');
    MW32(1);

    /* stbl */
    MW32(stbl_size); MFCC('s','t','b','l');

    /* stsd */
    MW32(stsd_size); MFCC('s','t','s','d');
    MW32(0); MW32(1); /* version, entry_count */
    {
        /* CFHD sample entry (86 bytes) */
        uint8_t entry[86];
        memset(entry, 0, sizeof(entry));
        put_be32(entry, 86);
        entry[4]='C'; entry[5]='F'; entry[6]='H'; entry[7]='D';
        entry[14]=0; entry[15]=1; /* data_ref_index */
        put_be16(entry+32, (uint16_t)width);
        put_be16(entry+34, (uint16_t)height);
        put_be32(entry+36, 0x00480000); /* 72 dpi */
        put_be32(entry+40, 0x00480000);
        entry[48]=0; entry[49]=1; /* frame_count */
        entry[82]=0; entry[83]=24; /* depth */
        entry[84]=0xFF; entry[85]=0xFF; /* color_table_id = -1 */
        memcpy(moov + mp, entry, 86); mp += 86;
    }

    /* stts */
    MW32(stts_size); MFCC('s','t','t','s');
    MW32(0); MW32(1);
    MW32(n); MW32(sd);

    /* stsc */
    MW32(stsc_size); MFCC('s','t','s','c');
    MW32(0); MW32(1);
    MW32(1); MW32(1); MW32(1);

    /* stsz */
    MW32(stsz_size); MFCC('s','t','s','z');
    MW32(0); MW32(0); MW32(n);
    for (int i = 0; i < n; i++)
        MW32(w->frame_sizes[i]);

    /* co64 */
    MW32(co64_size); MFCC('c','o','6','4');
    MW32(0); MW32(n);
    for (int i = 0; i < n; i++) {
        MW32((uint32_t)(w->frame_offsets[i] >> 32));
        MW32((uint32_t)(w->frame_offsets[i]));
    }

    fwrite(moov, 1, mp, w->fp);
    free(moov);

    #undef MW32
    #undef MW16
    #undef MW8
    #undef MFCC

cleanup:
    if (w->fp) fclose(w->fp);
    cf_enc_close(w->enc);
    free(w->frame_sizes);
    free(w->frame_offsets);
    free(w->packet_buf);
    free(w);
}
