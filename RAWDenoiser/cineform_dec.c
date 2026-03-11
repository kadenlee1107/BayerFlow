/*
 * GoPro CineForm (CFHD) Decoder
 *
 * Decodes CineForm HD compressed video from MOV containers.
 * Supports YUV 4:2:2 and Bayer RAW CFA modes.
 * Self-contained pure C, no external dependencies.
 *
 * Codec: 2-6 wavelet, 3-level decomposition.
 * Bitstream: Big-endian tag(16)/value(16) pairs + Huffman VLC coefficient data.
 *
 * References:
 *   - GoPro CineForm SDK (Apache 2.0 / MIT): github.com/gopro/cineform-sdk
 *   - FFmpeg libavcodec/cfhd.c (LGPL, studied as reference for bitstream format)
 *   - SMPTE ST 2073 (CineForm bitstream specification)
 */

#include "../include/cineform_dec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* CineForm tag IDs (SMPTE ST 2073)                                    */
/* ------------------------------------------------------------------ */

enum {
    TAG_SampleType          = 1,
    TAG_SampleIndexTable    = 2,
    TAG_BitstreamMarker     = 4,
    TAG_VersionMajor        = 5,
    TAG_VersionMinor        = 6,
    TAG_TransformType       = 10,
    TAG_NumFrames           = 11,
    TAG_ChannelCount        = 12,
    TAG_WaveletCount        = 13,
    TAG_SubbandCount        = 14,
    TAG_NumSpatial          = 15,
    TAG_GroupTrailer         = 18,
    TAG_FrameType           = 19,
    TAG_ImageWidth          = 20,
    TAG_ImageHeight         = 21,
    TAG_FrameIndex          = 23,
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
    TAG_FrameNumber         = 69,
    TAG_Precision           = 70,
    TAG_InputFormat         = 71,
    TAG_BandCodingFlags     = 72,
    TAG_PeakLevel           = 74,
    TAG_PeakOffsetLow       = 75,
    TAG_PeakOffsetHigh      = 76,
    TAG_Version             = 79,
    TAG_BandSecondPass      = 82,
    TAG_PrescaleTable       = 83,
    TAG_EncodedFormat       = 84,
    TAG_DisplayHeight       = 85,
    TAG_ChannelWidth        = 104,
    TAG_ChannelHeight       = 105,
};

/* Bitstream segment markers */
enum {
    SEG_LowPassSection  = 0x1A4A,  /* lowpass section start (params follow) */
    SEG_LowPassData     = 0x0F0F,  /* actual lowpass coefficient data */
    SEG_LowPassEnd      = 0x1B4B,  /* end of lowpass section */
    SEG_HighPassBand    = 0x0D0D,  /* highpass wavelet level start */
    SEG_CoefficientData = 0x0E0E,  /* coefficient band params follow */
    SEG_LevelEnd        = 0x0C0C,  /* end of wavelet level */
};

/* Encoded pixel formats */
enum {
    CF_FMT_YUV422   = 1,
    CF_FMT_BAYER    = 2,
    CF_FMT_RGB444   = 3,
    CF_FMT_RGBA4444 = 4,
};

/* ------------------------------------------------------------------ */
/* Bitstream reader                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    int            size;
    int            byte_pos;
    uint64_t       cache;    /* 64-bit cache supports codes up to 56 bits */
    int            bits_left;
} BitReader;

static void br_init(BitReader *br, const uint8_t *data, int size) {
    br->data = data;
    br->size = size;
    br->byte_pos = 0;
    br->cache = 0;
    br->bits_left = 0;
}

static inline void br_refill(BitReader *br) {
    while (br->bits_left <= 56 && br->byte_pos < br->size) {
        br->cache |= (uint64_t)br->data[br->byte_pos++] << (56 - br->bits_left);
        br->bits_left += 8;
    }
}

static inline void br_skip(BitReader *br, int n) {
    br->cache <<= n;
    br->bits_left -= n;
}

static inline uint32_t br_read(BitReader *br, int n) {
    br_refill(br);
    uint32_t val = (uint32_t)(br->cache >> (64 - n));
    br_skip(br, n);
    return val;
}

/* ------------------------------------------------------------------ */
/* VLC (Huffman) table for highpass coefficient decoding                */
/* ------------------------------------------------------------------ */
/*
 * Built from FFmpeg's CineForm encoder codebook (cfhdenc.c).
 * Magnitude 0 = single zero coefficient, 1-bit code '0'.
 * Magnitudes 1+ = unsigned code followed by sign bit (0=pos, 1=neg).
 * Zero runs ≥12 have dedicated multi-bit codes.
 * End-of-band = 26-bit code (handled as slow-path fallback).
 */

/* Full 256-entry encoder codebook from FFmpeg cfhdenc.c.
 * Entry format: {unsigned_code_length, unsigned_code_bits} per magnitude.
 * Note: entry 120 is the end-of-band prefix (26-bit, 0x03114BA2).
 * The actual end-of-band code is 0x03114BA3 (26 bits). */
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
    {26, 0x03114BA2}, /* magnitude 116: 26-bit unsigned (27-bit signed) */
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
#define CF_CODEBOOK_COUNT 256
/* End-of-band code: 26 bits, 0x3114BA3 */
#define CF_EOB_CODE  0x3114BA3
#define CF_EOB_LEN   26

/* Zero-run codebook: {code_length, code_bits, run_count} for runs ≥ 12 */
static const struct { uint8_t len; uint16_t code; uint16_t run; } cf_runbook[] = {
    { 7, 0x069,  12 }, { 8, 0x0D1,  20 }, { 9, 0x18A,  32 },
    {10, 0x343,  60 }, {11, 0x685, 100 },
    /* Runs 180 (13-bit) and 320 (13-bit) handled in slow path */
};
#define CF_RUNBOOK_COUNT 5

/* Long zero-run codes for slow-path fallback */
static const struct { uint8_t len; uint16_t code; uint16_t run; } cf_longrun[] = {
    {13, 0x18BF, 180 }, {13, 0x1BA5, 320 },
};
#define CF_LONGRUN_COUNT 2

/* Long coefficient codes for slow-path fallback (magnitudes 23-40, signed) */
/* Built dynamically in vlc_build_slow() */

#define VLC_BITS 12
#define VLC_SIZE (1 << VLC_BITS)

typedef struct {
    int16_t  level;
    uint16_t run;
    int8_t   len;   /* >0 = valid, 0 = need slow path */
} VLCEntry;

/* Slow-path entry for codes longer than VLC_BITS */
typedef struct {
    uint32_t code;
    uint8_t  len;
    int16_t  level;
    uint16_t run;
} VLCLong;

typedef struct {
    VLCEntry table[VLC_SIZE];
    VLCLong  slow[600];   /* long codes (magnitudes 23-255 signed + runs + EOB) */
    int      slow_count;
} VLCTable;

/* Cubic decompanding LUT for codebook 1 (used by FFmpeg encoder).
 * lut[i] = i + (768*i*i*i) / (256*256*256).
 * Maps VLC-decoded magnitude (0-255) → actual coefficient magnitude (0-1014). */
static int cf_decompand_lut[256];

static void vlc_fill(VLCTable *vt, uint32_t code, int len, int run, int level) {
    if (len < 1 || len > VLC_BITS) return;
    int shift = VLC_BITS - len;
    uint32_t base = code << shift;
    int count = 1 << shift;
    for (int j = 0; j < count; j++) {
        uint32_t idx = base | j;
        if (idx < (uint32_t)VLC_SIZE) {
            vt->table[idx].level = (int16_t)level;
            vt->table[idx].run   = (uint16_t)run;
            vt->table[idx].len   = (int8_t)len;
        }
    }
}

static void vlc_build(VLCTable *vt) {
    memset(vt, 0, sizeof(*vt));

    /* Initialize cubic decompanding LUT (codebook 1) */
    for (int i = 0; i < 256; i++)
        cf_decompand_lut[i] = i + (int)((768LL * i * i * i) / (256 * 256 * 256));

    /* Magnitude 0: code=0 (1 bit), one zero coefficient */
    vlc_fill(vt, 0, 1, 1, 0);

    /* Magnitudes 1+: unsigned code + sign bit.
     * All 255 magnitudes (1-255) are valid coefficients.
     * Magnitude 116 has a 26-bit unsigned code (0x3114BA2), producing
     * 27-bit signed codes — distinct from the 26-bit EOB (0x3114BA3).
     * They differ in the last unsigned bit: mag116=...0, EOB=...1. */
    for (int mag = 1; mag < CF_CODEBOOK_COUNT; mag++) {
        int ulen = cf_codebook[mag].len;
        uint32_t ucode = cf_codebook[mag].code;
        int slen = ulen + 1;

        if (slen <= VLC_BITS) {
            /* Fast path: fits in primary table */
            vlc_fill(vt, (ucode << 1) | 0, slen, 1,  mag);  /* positive */
            vlc_fill(vt, (ucode << 1) | 1, slen, 1, -mag);  /* negative */
        } else {
            /* Slow path */
            if (vt->slow_count < 598) {
                vt->slow[vt->slow_count++] = (VLCLong){(ucode<<1)|0, slen, mag, 1};
                vt->slow[vt->slow_count++] = (VLCLong){(ucode<<1)|1, slen, -mag, 1};
            }
        }
    }

    /* End-of-band code: 26 bits, 0x3114BA3.
     * run=0 signals end-of-band to caller. */
    if (vt->slow_count < 600) {
        vt->slow[vt->slow_count++] = (VLCLong){CF_EOB_CODE, CF_EOB_LEN, 0, 0};
    }

    /* Zero runs ≥ 12 (fast path) */
    for (int i = 0; i < CF_RUNBOOK_COUNT; i++) {
        if (cf_runbook[i].len <= VLC_BITS)
            vlc_fill(vt, cf_runbook[i].code, cf_runbook[i].len, cf_runbook[i].run, 0);
    }

    /* Long zero runs (slow path) */
    for (int i = 0; i < CF_LONGRUN_COUNT; i++) {
        if (vt->slow_count < 600)
            vt->slow[vt->slow_count++] = (VLCLong){
                cf_longrun[i].code, cf_longrun[i].len, 0, cf_longrun[i].run};
    }
}

/* Decode one (run, level) pair.
 * Returns 0=success, -1=end-of-band/error.
 * EOB is signaled by run=0 from the slow path (26-bit code 0x3114BA3). */
static int vlc_decode(const VLCTable *vt, BitReader *br, int *run_out, int *level_out) {
    br_refill(br);

    /* If no bits available, treat as end-of-band */
    if (br->bits_left <= 0)
        return -1;

    uint32_t peek = (uint32_t)(br->cache >> (64 - VLC_BITS));
    VLCEntry e = vt->table[peek & (VLC_SIZE - 1)];

    if (e.len > 0) {
        br_skip(br, e.len);
        *run_out   = e.run;
        *level_out = e.level;
        return 0;
    }

    /* Slow path: try long codes */
    int total_avail = br->bits_left + (br->size - br->byte_pos) * 8;
    for (int i = 0; i < vt->slow_count; i++) {
        int len = vt->slow[i].len;
        if (len > total_avail)
            continue;
        br_refill(br);
        if (len > br->bits_left)
            continue;
        uint32_t bits = (uint32_t)(br->cache >> (64 - len));
        if (bits == vt->slow[i].code) {
            br_skip(br, len);
            *run_out   = vt->slow[i].run;
            *level_out = vt->slow[i].level;
            if (vt->slow[i].run == 0)
                return -1;  /* end-of-band */
            return 0;
        }
    }

    /* No match — corrupt data or insufficient bits */
    br_skip(br, 1);
    return -1;
}

/* ------------------------------------------------------------------ */
/* 2-6 Inverse Wavelet Transform                                       */
/* ------------------------------------------------------------------ */
/*
 * CineForm 2-6 biorthogonal wavelet.
 * Reconstructs 2N samples from N lowpass + N highpass coefficients.
 *
 * Interior (1 <= i < N-1):
 *   even[2i]   = low[i] + high[i] + correction
 *   odd[2i+1]  = low[i] - high[i] - correction
 *   correction = (low[i-1] - low[i+1] + 4) >> 3
 */

/* Horizontal inverse DWT matching FFmpeg's cfhddsp.c filter().
 * Interior: even = (corr + low[i] + high[i]) >> IDWT_SHIFT
 *           odd  = (-corr + low[i] - high[i]) >> IDWT_SHIFT
 * where corr = (low[i-1] - low[i+1] + 4) >> 3 */
#define IDWT_SHIFT 1  /* 1 = matches FFmpeg filter(), 0 = no scaling */
static void idwt_horiz(int16_t *output, int out_stride,
                        const int16_t *low, int low_stride,
                        const int16_t *high, int high_stride,
                        int half_width, int height) {
    for (int y = 0; y < height; y++) {
        const int16_t *lp = low  + y * low_stride;
        const int16_t *hp = high + y * high_stride;
        int16_t *out = output + y * out_stride;
        int N = half_width;

        if (N < 2) {
            if (N == 1) {
                out[0] = (lp[0] + hp[0]) >> IDWT_SHIFT;
                out[1] = (lp[0] - hp[0]) >> IDWT_SHIFT;
            }
            continue;
        }

        /* Left boundary (i=0) */
        {
            int tmp;
            tmp = (11 * lp[0] - 4 * lp[1] + lp[2] + 4) >> 3;
            out[0] = (tmp + hp[0]) >> IDWT_SHIFT;
            tmp = (5 * lp[0] + 4 * lp[1] - lp[2] + 4) >> 3;
            out[1] = (tmp - hp[0]) >> IDWT_SHIFT;
        }

        /* Interior */
        for (int i = 1; i < N - 1; i++) {
            int tmp_e = (lp[i-1] - lp[i+1] + 4) >> 3;
            out[2*i]     = (tmp_e + lp[i] + hp[i]) >> IDWT_SHIFT;
            int tmp_o = (lp[i+1] - lp[i-1] + 4) >> 3;
            out[2*i + 1] = (tmp_o + lp[i] - hp[i]) >> IDWT_SHIFT;
        }

        /* Right boundary (i=N-1) */
        {
            int i = N - 1;
            int tmp;
            tmp = (5 * lp[i] + 4 * lp[i-1] - lp[i > 1 ? i-2 : 0] + 4) >> 3;
            out[2*i] = (tmp + hp[i]) >> IDWT_SHIFT;
            tmp = (11 * lp[i] - 4 * lp[i-1] + lp[i > 1 ? i-2 : 0] + 4) >> 3;
            out[2*i + 1] = (tmp - hp[i]) >> IDWT_SHIFT;
        }
    }
}

/* Vertical inverse DWT matching FFmpeg's cfhddsp.c filter().
 * Same formula as horizontal but applied column-wise. */
static void idwt_vert(int16_t *output, int out_stride,
                       const int16_t *low, int low_stride,
                       const int16_t *high, int high_stride,
                       int width, int half_height) {
    int N = half_height;
    if (N < 2) {
        if (N == 1) {
            for (int x = 0; x < width; x++) {
                output[x] = (low[x] + high[x]) >> IDWT_SHIFT;
                output[out_stride + x] = (low[x] - high[x]) >> IDWT_SHIFT;
            }
        }
        return;
    }

    for (int x = 0; x < width; x++) {
        /* Top boundary (i=0) */
        {
            int l0 = low[0 * low_stride + x];
            int l1 = low[1 * low_stride + x];
            int l2 = low[2 * low_stride + x];
            int h0 = high[0 * high_stride + x];
            int tmp;
            tmp = (11 * l0 - 4 * l1 + l2 + 4) >> 3;
            output[0 * out_stride + x] = (tmp + h0) >> IDWT_SHIFT;
            tmp = (5 * l0 + 4 * l1 - l2 + 4) >> 3;
            output[1 * out_stride + x] = (tmp - h0) >> IDWT_SHIFT;
        }

        /* Interior */
        for (int i = 1; i < N - 1; i++) {
            int lm = low[(i-1) * low_stride + x];
            int l0 = low[i * low_stride + x];
            int lp = low[(i+1) * low_stride + x];
            int h0 = high[i * high_stride + x];
            int tmp_e = (lm - lp + 4) >> 3;
            output[(2*i)     * out_stride + x] = (tmp_e + l0 + h0) >> IDWT_SHIFT;
            int tmp_o = (lp - lm + 4) >> 3;
            output[(2*i + 1) * out_stride + x] = (tmp_o + l0 - h0) >> IDWT_SHIFT;
        }

        /* Bottom boundary (i=N-1) */
        {
            int i = N - 1;
            int lm2 = low[(i > 1 ? i-2 : 0) * low_stride + x];
            int lm1 = low[(i-1) * low_stride + x];
            int l0  = low[i * low_stride + x];
            int h0  = high[i * high_stride + x];
            int tmp;
            tmp = (5 * l0 + 4 * lm1 - lm2 + 4) >> 3;
            output[(2*i) * out_stride + x] = (tmp + h0) >> IDWT_SHIFT;
            tmp = (11 * l0 - 4 * lm1 + lm2 + 4) >> 3;
            output[(2*i + 1) * out_stride + x] = (tmp - h0) >> IDWT_SHIFT;
        }
    }
}

/* ------------------------------------------------------------------ */
/* MOV container parser                                                */
/* ------------------------------------------------------------------ */

static uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

static uint16_t read_be16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

static uint64_t read_be64(const uint8_t *p) {
    return ((uint64_t)read_be32(p) << 32) | read_be32(p + 4);
}

/* ------------------------------------------------------------------ */
/* CfReader structure                                                  */
/* ------------------------------------------------------------------ */

#define CF_MAX_CHANNELS 4
#define CF_MAX_SUBBANDS 10
#define CF_DWT_LEVELS   3

struct CfReader {
    FILE    *fp;
    char     path[4096];
    CfInfo   info;

    /* MOV sample table */
    int      sample_count;
    uint32_t *sample_sizes;
    int64_t  *sample_offsets;
    float    fps;

    /* VLC table */
    VLCTable vlc;

    /* First-frame info populated flag */
    int      info_populated;
};

/* ------------------------------------------------------------------ */
/* MOV parsing                                                         */
/* ------------------------------------------------------------------ */

static int find_atom(FILE *fp, int64_t end_pos, uint32_t target,
                     int64_t *atom_start, int64_t *atom_size) {
    uint8_t hdr[8];
    while (ftello(fp) < end_pos) {
        if (fread(hdr, 1, 8, fp) != 8) return -1;
        uint64_t sz = read_be32(hdr);
        uint32_t tag = read_be32(hdr + 4);
        int64_t pos = ftello(fp) - 8;

        if (sz == 1) {
            uint8_t ext[8];
            if (fread(ext, 1, 8, fp) != 8) return -1;
            sz = read_be64(ext);
        }
        if (sz < 8) return -1;

        if (tag == target) {
            *atom_start = pos;
            *atom_size = (int64_t)sz;
            fseeko(fp, pos + 8, SEEK_SET);
            return 0;
        }
        fseeko(fp, pos + (int64_t)sz, SEEK_SET);
    }
    return -1;
}

static int parse_mov(CfReader *r) {
    FILE *fp = r->fp;
    fseeko(fp, 0, SEEK_END);
    int64_t file_size = ftello(fp);
    fseeko(fp, 0, SEEK_SET);

    int64_t moov_start, moov_size;
    if (find_atom(fp, file_size, 0x6D6F6F76, &moov_start, &moov_size) != 0) {
        fprintf(stderr, "cineform_dec: no moov atom found\n");
        return CF_ERR_FMT;
    }
    int64_t moov_end = moov_start + moov_size;

    int64_t trak_start, trak_size;
    while (find_atom(fp, moov_end, 0x7472616B, &trak_start, &trak_size) == 0) {
        int64_t trak_end = trak_start + trak_size;

        int64_t mdia_start, mdia_size;
        if (find_atom(fp, trak_end, 0x6D646961, &mdia_start, &mdia_size) != 0)
            continue;
        int64_t mdia_end = mdia_start + mdia_size;

        /* Check hdlr for video */
        int64_t save = ftello(fp);
        int64_t hdlr_start, hdlr_size;
        if (find_atom(fp, mdia_end, 0x68646C72, &hdlr_start, &hdlr_size) == 0) {
            uint8_t hdlr_data[16];
            fseeko(fp, hdlr_start + 8 + 4, SEEK_SET);
            if (fread(hdlr_data, 1, 8, fp) != 8) continue;
            uint32_t handler = read_be32(hdlr_data + 4);
            if (handler != 0x76696465) {
                fseeko(fp, trak_end, SEEK_SET);
                continue;
            }
        }
        fseeko(fp, save, SEEK_SET);

        int64_t minf_start, minf_size;
        if (find_atom(fp, mdia_end, 0x6D696E66, &minf_start, &minf_size) != 0)
            continue;
        int64_t minf_end = minf_start + minf_size;

        int64_t stbl_start, stbl_size;
        if (find_atom(fp, minf_end, 0x7374626C, &stbl_start, &stbl_size) != 0)
            continue;
        int64_t stbl_end = stbl_start + stbl_size;

        /* Check stsd for CFHD */
        int64_t stsd_start, stsd_size;
        fseeko(fp, stbl_start + 8, SEEK_SET);
        if (find_atom(fp, stbl_end, 0x73747364, &stsd_start, &stsd_size) == 0) {
            uint8_t stsd_data[86];
            fseeko(fp, stsd_start + 8 + 4, SEEK_SET);
            if (fread(stsd_data, 1, 86, fp) != 86) continue;
            uint32_t fourcc = read_be32(stsd_data + 4 + 4);
            if (fourcc != 0x43464844) {
                fseeko(fp, trak_end, SEEK_SET);
                continue;
            }
            r->info.width  = read_be16(stsd_data + 4 + 32);
            r->info.height = read_be16(stsd_data + 4 + 34);
        }

        /* Parse mdhd for timescale */
        fseeko(fp, mdia_start + 8, SEEK_SET);
        int64_t mdhd_start, mdhd_size;
        if (find_atom(fp, mdia_end, 0x6D646864, &mdhd_start, &mdhd_size) == 0) {
            uint8_t mdhd_buf[28];
            fseeko(fp, mdhd_start + 8, SEEK_SET);
            if (fread(mdhd_buf, 1, 28, fp) >= 24) {
                uint8_t version = mdhd_buf[0];
                uint32_t timescale;
                if (version == 0) {
                    timescale = read_be32(mdhd_buf + 4 + 8);
                } else {
                    timescale = read_be32(mdhd_buf + 4 + 16);
                }
                r->fps = (float)timescale;
            }
        }

        /* Parse stsz */
        fseeko(fp, stbl_start + 8, SEEK_SET);
        int64_t stsz_start, stsz_size;
        if (find_atom(fp, stbl_end, 0x7374737A, &stsz_start, &stsz_size) == 0) {
            uint8_t stsz_hdr[12];
            fseeko(fp, stsz_start + 8, SEEK_SET);
            if (fread(stsz_hdr, 1, 12, fp) != 12) continue;
            uint32_t default_size = read_be32(stsz_hdr + 4);
            uint32_t count = read_be32(stsz_hdr + 8);
            r->sample_count = (int)count;
            r->sample_sizes = (uint32_t *)malloc(count * sizeof(uint32_t));
            if (!r->sample_sizes) return CF_ERR_MEM;

            if (default_size > 0) {
                for (uint32_t i = 0; i < count; i++)
                    r->sample_sizes[i] = default_size;
            } else {
                uint8_t *sz_buf = (uint8_t *)malloc(count * 4);
                if (!sz_buf) return CF_ERR_MEM;
                if (fread(sz_buf, 4, count, fp) != count) { free(sz_buf); return CF_ERR_IO; }
                for (uint32_t i = 0; i < count; i++)
                    r->sample_sizes[i] = read_be32(sz_buf + i * 4);
                free(sz_buf);
            }
        }

        /* Parse stco / co64 */
        fseeko(fp, stbl_start + 8, SEEK_SET);
        int64_t co_start, co_size;
        int use_co64 = 0;
        if (find_atom(fp, stbl_end, 0x636F3634, &co_start, &co_size) == 0) {
            use_co64 = 1;
        } else {
            fseeko(fp, stbl_start + 8, SEEK_SET);
            if (find_atom(fp, stbl_end, 0x7374636F, &co_start, &co_size) != 0)
                continue;
        }

        {
            uint8_t co_hdr[8];
            fseeko(fp, co_start + 8, SEEK_SET);
            if (fread(co_hdr, 1, 8, fp) != 8) continue;
            uint32_t chunk_count = read_be32(co_hdr + 4);

            r->sample_offsets = (int64_t *)malloc(r->sample_count * sizeof(int64_t));
            if (!r->sample_offsets) return CF_ERR_MEM;

#ifdef CFHD_TRACE_TAGS
            fprintf(stderr, "MOV: chunk_count=%u sample_count=%d use_co64=%d\n",
                    chunk_count, r->sample_count, use_co64);
#endif
            if (chunk_count == (uint32_t)r->sample_count) {
                for (uint32_t i = 0; i < chunk_count; i++) {
                    if (use_co64) {
                        uint8_t buf8[8];
                        if (fread(buf8, 1, 8, fp) != 8) return CF_ERR_IO;
                        r->sample_offsets[i] = (int64_t)read_be64(buf8);
                    } else {
                        uint8_t buf4[4];
                        if (fread(buf4, 1, 4, fp) != 4) return CF_ERR_IO;
                        r->sample_offsets[i] = (int64_t)read_be32(buf4);
                    }
                }
            } else {
                int64_t *chunk_offsets = (int64_t *)malloc(chunk_count * sizeof(int64_t));
                if (!chunk_offsets) return CF_ERR_MEM;
                for (uint32_t i = 0; i < chunk_count; i++) {
                    if (use_co64) {
                        uint8_t buf8[8];
                        if (fread(buf8, 1, 8, fp) != 8) { free(chunk_offsets); return CF_ERR_IO; }
                        chunk_offsets[i] = (int64_t)read_be64(buf8);
                    } else {
                        uint8_t buf4[4];
                        if (fread(buf4, 1, 4, fp) != 4) { free(chunk_offsets); return CF_ERR_IO; }
                        chunk_offsets[i] = (int64_t)read_be32(buf4);
                    }
                }

                /* Parse stsc (sample-to-chunk) for correct samples-per-chunk.
                 * stsc entries: {first_chunk (1-based), samples_per_chunk, sample_desc_idx}.
                 * Each entry applies from first_chunk until the next entry. */
                uint32_t *spc_table = (uint32_t *)calloc(chunk_count, sizeof(uint32_t));
                if (!spc_table) { free(chunk_offsets); return CF_ERR_MEM; }

                /* Default: 1 sample per chunk */
                for (uint32_t i = 0; i < chunk_count; i++) spc_table[i] = 1;

                /* Try to read stsc atom */
                fseeko(fp, stbl_start + 8, SEEK_SET);
                int64_t stsc_start, stsc_size;
                if (find_atom(fp, stbl_end, 0x73747363, &stsc_start, &stsc_size) == 0) {
                    uint8_t stsc_hdr[8];
                    fseeko(fp, stsc_start + 8, SEEK_SET);
                    if (fread(stsc_hdr, 1, 8, fp) == 8) {
                        uint32_t stsc_entries = read_be32(stsc_hdr + 4);
                        uint32_t *stsc_data = (uint32_t *)malloc(stsc_entries * 12);
                        if (stsc_data && fread(stsc_data, 12, stsc_entries, fp) == stsc_entries) {
                            for (uint32_t e = 0; e < stsc_entries; e++) {
                                uint32_t first_chunk = read_be32((uint8_t *)stsc_data + e * 12) - 1; /* 0-based */
                                uint32_t spc_val = read_be32((uint8_t *)stsc_data + e * 12 + 4);
                                uint32_t next_first = (e + 1 < stsc_entries)
                                    ? read_be32((uint8_t *)stsc_data + (e + 1) * 12) - 1
                                    : chunk_count;
                                for (uint32_t ci = first_chunk; ci < next_first && ci < chunk_count; ci++)
                                    spc_table[ci] = spc_val;
                            }
                        }
                        free(stsc_data);
                    }
                }

                /* Expand chunk offsets to sample offsets using spc_table */
                int si = 0;
                for (uint32_t ci = 0; ci < chunk_count && si < r->sample_count; ci++) {
                    int64_t off = chunk_offsets[ci];
                    for (uint32_t j = 0; j < spc_table[ci] && si < r->sample_count; j++) {
                        r->sample_offsets[si] = off;
                        off += r->sample_sizes[si];
                        si++;
                    }
                }
                free(spc_table);
                free(chunk_offsets);
            }
        }

        /* Calculate FPS from stts */
        if (r->sample_count > 0 && r->fps > 0) {
            fseeko(fp, stbl_start + 8, SEEK_SET);
            int64_t stts_start, stts_size;
            if (find_atom(fp, stbl_end, 0x73747473, &stts_start, &stts_size) == 0) {
                uint8_t stts_hdr[8];
                fseeko(fp, stts_start + 8, SEEK_SET);
                if (fread(stts_hdr, 1, 8, fp) == 8) {
                    uint32_t entry_count = read_be32(stts_hdr + 4);
                    if (entry_count >= 1) {
                        uint8_t stts_entry[8];
                        if (fread(stts_entry, 1, 8, fp) == 8) {
                            uint32_t duration = read_be32(stts_entry + 4);
                            if (duration > 0)
                                r->info.fps = r->fps / (float)duration;
                        }
                    }
                }
            }
        }

        r->info.frame_count = r->sample_count;
        return CF_OK;
    }

    fprintf(stderr, "cineform_dec: no CFHD video track found\n");
    return CF_ERR_FMT;
}

/* ------------------------------------------------------------------ */
/* CineForm frame decoder                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int16_t *data;
    int      width;
    int      height;
} SubBand;

typedef struct {
    SubBand  subbands[CF_MAX_SUBBANDS]; /* 0=LL, 1-9=highpass */
    int      num_levels;
} ChannelDec;

static int decode_frame(CfReader *r, const uint8_t *data, int size,
                        uint16_t *output, int out_width, int out_height) {
    int pos = 0;
    int channel_count = 1;
    int current_channel = 0;
    int encoded_format = 0;
    int prescale_table = 0;
    int image_width = 0, image_height = 0;
    (void)image_width; (void)image_height;
    int lowpass_width = 0, lowpass_height = 0;
    int lowpass_quant = 1;
    int band_width = 0, band_height = 0;
    int quantization = 1;
    int precision = 10;
    int num_levels = 3;
    int subband_band = 0;  /* global subband index from TAG_SubbandBand */

    ChannelDec channels[CF_MAX_CHANNELS];
    memset(channels, 0, sizeof(channels));

    /* Parse tag/value pairs */
#ifdef CFHD_TRACE_TAGS
    fprintf(stderr, "decode_frame: size=%d bytes\n", size);
#endif
    while (pos + 4 <= size) {
        int16_t  raw_tag = (int16_t)read_be16(data + pos);
        uint16_t val     = read_be16(data + pos + 2);
        pos += 4;

        int abs_tag = raw_tag < 0 ? -raw_tag : raw_tag;

        switch (abs_tag) {
        case TAG_ChannelCount:        channel_count = val; break;
        case TAG_ImageWidth:          image_width = val; break;
        case TAG_ImageHeight:         image_height = val; break;
        case TAG_ChannelNumber:       current_channel = val; break;
        case TAG_LowpassWidth:        lowpass_width = val; break;
        case TAG_LowpassHeight:       lowpass_height = val; break;
        case TAG_LowpassQuantization: lowpass_quant = val; break;
        case TAG_BandWidth:           band_width = val; break;
        case TAG_BandHeight:          band_height = val; break;
        case TAG_Quantization:        quantization = val; break;
        case TAG_SubbandBand:         subband_band = val; break;
        case TAG_Precision:           precision = val; break;
        case TAG_EncodedFormat:       encoded_format = val; break;
        case TAG_PrescaleTable:       prescale_table = val; break;
        case TAG_NumLevels:           num_levels = val; break;

        case TAG_BitstreamMarker:
            if (val == SEG_LowPassData) {
                /* Actual lowpass coefficient data follows */
                if (current_channel >= CF_MAX_CHANNELS) break;
                if (lowpass_width <= 0 || lowpass_height <= 0) break;

                int lp_size = lowpass_width * lowpass_height;
                int16_t *lp_data = (int16_t *)malloc(lp_size * sizeof(int16_t));
                if (!lp_data) return CF_ERR_MEM;

                for (int i = 0; i < lp_size && pos + 2 <= size; i++) {
                    lp_data[i] = (int16_t)read_be16(data + pos);
                    pos += 2;
                }

                /* Dequantize lowpass */
                if (lowpass_quant > 1) {
                    for (int i = 0; i < lp_size; i++)
                        lp_data[i] = lp_data[i] * lowpass_quant;
                }

                channels[current_channel].subbands[0].data = lp_data;
                channels[current_channel].subbands[0].width = lowpass_width;
                channels[current_channel].subbands[0].height = lowpass_height;
                channels[current_channel].num_levels = num_levels;

                /* Align to 4-byte boundary */
                pos = (pos + 3) & ~3;
            }
            /* Other markers: just continue parsing tags */
            break;

        case TAG_BandHeader:
        case TAG_BandSecondPass: {
            /* Decode highpass band VLC data */
            if (current_channel >= CF_MAX_CHANNELS) break;
            if (subband_band < 1 || subband_band >= CF_MAX_SUBBANDS) break;
            if (band_width <= 0 || band_height <= 0) break;

            int bp_size = band_width * band_height;
            int16_t *bp_data = (int16_t *)calloc(bp_size, sizeof(int16_t));
            if (!bp_data) return CF_ERR_MEM;

            /* Find BandTrailer FIRST to bound the VLC data range.
             * This prevents the VLC decoder from reading past the band
             * boundary into subsequent bands' data. */
            int trailer_pos = -1;
            {
                int scan = (pos + 1) & ~1;  /* align to 16-bit boundary */
                int scan_iters = 0;
                while (scan + 4 <= size) {
                    int16_t st = (int16_t)read_be16(data + scan);
                    uint16_t sv = read_be16(data + scan + 2);
                    int sat = st < 0 ? -st : st;
                    if (sat == TAG_BandTrailer && sv == 0) {
                        trailer_pos = scan;
                        break;
                    }
                    scan += 2;
                    scan_iters++;
                }
#ifdef CFHD_TRACE_TAGS
                if (trailer_pos < 0) {
                    fprintf(stderr, "  PRE-SCAN ch%d band%d: start=%d size=%d iters=%d FAILED\n",
                            current_channel, subband_band, (pos + 1) & ~1, size, scan_iters);
                    /* Dump last 32 bytes of frame data */
                    int dstart = size - 32;
                    if (dstart < 0) dstart = 0;
                    fprintf(stderr, "  Last bytes [%d..%d]: ", dstart, size);
                    for (int di = dstart; di < size; di++)
                        fprintf(stderr, "%02x", data[di]);
                    fprintf(stderr, "\n");
                }
#endif
                (void)scan_iters;
            }

            /* Bound BitReader to VLC data region only */
            int vlc_size = (trailer_pos > pos) ? (trailer_pos - pos) : (size - pos);
#ifdef CFHD_TRACE_TAGS
            fprintf(stderr, "  BandRange ch%d band%d: vlc_start=%d trailer=%d vlc_size=%d\n",
                    current_channel, subband_band, pos, trailer_pos, vlc_size);
#endif
            BitReader br;
            br_init(&br, data + pos, vlc_size);

            int coeff_idx = 0;
            int safety = bp_size * 4;
            while (coeff_idx < bp_size && safety-- > 0) {
                int run = 0, level = 0;
                if (vlc_decode(&r->vlc, &br, &run, &level) != 0)
                    break;

                /* Dequantize with cubic decompanding (codebook 1).
                 * FFmpeg: dequant = lut[codebook][abs(level)] * sign(level) * quant */
                int abs_level = level < 0 ? -level : level;
                int sign = level < 0 ? -1 : 1;
                int decompanded = (abs_level < 256) ? cf_decompand_lut[abs_level] : abs_level;
                int32_t dq_level = (int32_t)(sign * decompanded) * quantization;
                for (int j = 0; j < run && coeff_idx < bp_size; j++)
                    bp_data[coeff_idx++] = (int16_t)dq_level;
            }

            /* Advance pos past VLC data to the BandTrailer */
            if (trailer_pos >= 0)
                pos = trailer_pos;

            channels[current_channel].subbands[subband_band].data = bp_data;
            channels[current_channel].subbands[subband_band].width = band_width;
            channels[current_channel].subbands[subband_band].height = band_height;
            break;
        }

        default:
            break;
        }
    }

    /* ---- Populate info on first decode ---- */
    if (!r->info_populated) {
        r->info.channels = channel_count;
        r->info.encoded_format = encoded_format;
        if (encoded_format == CF_FMT_BAYER) {
            r->info.pixel_format = CF_PIX_BAYER_RGGB16;
            r->info.is_bayer = 1;
        } else if (encoded_format == CF_FMT_YUV422) {
            r->info.pixel_format = CF_PIX_YUV422_10;
            r->info.is_bayer = 0;
        } else {
            r->info.pixel_format = CF_PIX_RGB48;
            r->info.is_bayer = 0;
        }
        r->info_populated = 1;
    }

    /* ---- Decode prescale table ---- */
    /* FFmpeg formula: prescale_table[i] = (val >> (14 - i*2)) & 3
     * where i=0 is finest level, i=N-1 is coarsest level.
     * Our lev goes 1(finest) to N(coarsest), so FFmpeg i = lev - 1. */
    int prescale[CF_DWT_LEVELS + 1];
    for (int i = 0; i <= CF_DWT_LEVELS; i++) prescale[i] = 0;
    if (prescale_table > 0) {
        for (int lev = 1; lev <= num_levels && lev <= CF_DWT_LEVELS; lev++) {
            prescale[lev] = (prescale_table >> (14 - (lev - 1) * 2)) & 3;
        }
    }

    /* ---- Inverse DWT: reconstruct each channel ---- */
    for (int ch = 0; ch < channel_count && ch < CF_MAX_CHANNELS; ch++) {
        ChannelDec *c = &channels[ch];
        int nl = c->num_levels;
        if (nl < 1) nl = CF_DWT_LEVELS;

        SubBand *ll = &c->subbands[0];
        if (!ll->data) {
            fprintf(stderr, "cineform_dec: ch%d missing lowpass data\n", ch);
            continue;
        }

        int cur_w = ll->width;
        int cur_h = ll->height;

        /* Allocate work buffers */
        int max_w = cur_w << nl;
        int max_h = cur_h << nl;
        int16_t *recon = (int16_t *)calloc((size_t)max_w * max_h, sizeof(int16_t));
        int16_t *v_low = (int16_t *)calloc((size_t)max_w * max_h, sizeof(int16_t));
        int16_t *v_high = (int16_t *)calloc((size_t)max_w * max_h, sizeof(int16_t));
        if (!recon || !v_low || !v_high) {
            free(recon); free(v_low); free(v_high);
            return CF_ERR_MEM;
        }

        /* Copy lowpass into recon */
        for (int y = 0; y < cur_h; y++)
            memcpy(recon + y * cur_w, ll->data + y * cur_w, cur_w * sizeof(int16_t));

        /* Reconstruct level by level (level 3 → 2 → 1) */
        for (int lev = nl; lev >= 1; lev--) {
            int sb_base = 1 + (nl - lev) * 3;
            SubBand *lh = &c->subbands[sb_base];      /* vertical highpass */
            SubBand *hl = &c->subbands[sb_base + 1];   /* horizontal highpass */
            SubBand *hh = &c->subbands[sb_base + 2];   /* diagonal highpass */

            int next_w = cur_w * 2;
            int next_h = cur_h * 2;

            if (!lh->data || !hl->data) {
                /* Missing highpass — just expand without detail */
                cur_w = next_w;
                cur_h = next_h;
                continue;
            }

            /* CineForm subband order: band1=HL, band2=LH, band3=HH.
             * Vertical iDWT pairs: vert(LL, LH) and vert(HL, HH).
             * So we use hl (=band2=LH) with LL, and lh (=band1=HL) with HH. */

            /* Step 1: Vertical iDWT of (LL, LH) → v_low (doubled height) */
            idwt_vert(v_low, cur_w,
                      recon, cur_w,
                      hl->data, hl->width,   /* hl = subbands[sb_base+1] = LH */
                      cur_w, cur_h);

            /* Step 2: Vertical iDWT of (HL, HH) → v_high (doubled height) */
            if (hh->data) {
                idwt_vert(v_high, cur_w,
                          lh->data, lh->width,   /* lh = subbands[sb_base] = HL */
                          hh->data, hh->width,
                          cur_w, cur_h);
            } else {
                /* No HH band — use HL as v_high directly (zero padding) */
                for (int y = 0; y < next_h && y < cur_h; y++)
                    memcpy(v_high + y * cur_w, lh->data + y * lh->width,
                           cur_w * sizeof(int16_t));
            }

            /* Step 3: Horizontal iDWT of (v_low, v_high) → recon (doubled width) */
            idwt_horiz(recon, next_w,
                       v_low, cur_w,
                       v_high, cur_w,
                       cur_w, next_h);

            /* Apply prescale shift for this level */
            if (prescale[lev] > 0) {
                int shift = prescale[lev];
                for (int i = 0; i < next_w * next_h; i++)
                    recon[i] <<= shift;
            }

            cur_w = next_w;
            cur_h = next_h;
        }

        /* Store reconstructed channel */
        c->subbands[0].width = cur_w;   /* reuse for output dimensions */
        c->subbands[0].height = cur_h;

        free(v_low);
        free(v_high);

        /* Convert int16 recon buffer to output.
         * For multi-channel output, store recon pointer temporarily. */
        /* We'll reuse subbands[0].data to hold the full recon buffer */
        free(ll->data);
        c->subbands[0].data = recon;
    }

    /* ---- Output ---- */
    if (encoded_format == CF_FMT_BAYER && channel_count >= 4) {
        /* Bayer: 4 channels interleaved into RGGB */
        int cw = channels[0].subbands[0].width;
        int ch_h = channels[0].subbands[0].height;
        for (int y = 0; y < ch_h && y * 2 + 1 < out_height; y++) {
            for (int x = 0; x < cw && x * 2 + 1 < out_width; x++) {
                int32_t rv  = channels[0].subbands[0].data ? channels[0].subbands[0].data[y*cw+x] : 0;
                int32_t g1v = channels[1].subbands[0].data ? channels[1].subbands[0].data[y*cw+x] : 0;
                int32_t g2v = channels[2].subbands[0].data ? channels[2].subbands[0].data[y*cw+x] : 0;
                int32_t bv  = channels[3].subbands[0].data ? channels[3].subbands[0].data[y*cw+x] : 0;
                int oy = y * 2, ox = x * 2;
                output[oy * out_width + ox]         = (uint16_t)(rv < 0 ? 0 : (rv > 65535 ? 65535 : rv));
                output[oy * out_width + ox + 1]     = (uint16_t)(g1v < 0 ? 0 : (g1v > 65535 ? 65535 : g1v));
                output[(oy+1) * out_width + ox]     = (uint16_t)(g2v < 0 ? 0 : (g2v > 65535 ? 65535 : g2v));
                output[(oy+1) * out_width + ox + 1] = (uint16_t)(bv < 0 ? 0 : (bv > 65535 ? 65535 : bv));
            }
        }
    } else if (encoded_format == CF_FMT_YUV422 && channel_count >= 3) {
        /* YUV 4:2:2: output 3 planes concatenated.
         * Layout: Y (width × height) + Cb (width/2 × height) + Cr (width/2 × height) */
        int upshift = (precision > 0 && precision < 16) ? (16 - precision) : 0;

        for (int ch = 0; ch < 3; ch++) {
            int16_t *ch_data = channels[ch].subbands[0].data;
            if (!ch_data) continue;
            int cw = channels[ch].subbands[0].width;
            int ch_h = channels[ch].subbands[0].height;

            /* Channel 0 (Y) is full width; channels 1,2 (Cb,Cr) are half width */
            int plane_w = (ch == 0) ? out_width : out_width / 2;
            int plane_h = out_height;
            int ow = cw < plane_w ? cw : plane_w;
            int oh = ch_h < plane_h ? ch_h : plane_h;

            /* Compute output offset: Y at 0, Cb at width*height, Cr at width*height + width/2*height */
            uint16_t *dst;
            if (ch == 0) dst = output;
            else if (ch == 1) dst = output + (size_t)out_width * out_height;
            else dst = output + (size_t)out_width * out_height + (size_t)(out_width / 2) * out_height;

            for (int y = 0; y < oh; y++) {
                for (int x = 0; x < ow; x++) {
                    int32_t v = (int32_t)ch_data[y * cw + x] << upshift;
                    dst[y * plane_w + x] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
                }
            }
        }
    } else {
        /* Fallback: single channel or unknown — output channel 0 */
        int16_t *y_data = channels[0].subbands[0].data;
        if (y_data) {
            int cw = channels[0].subbands[0].width;
            int ch_h = channels[0].subbands[0].height;
            int ow = cw < out_width ? cw : out_width;
            int oh = ch_h < out_height ? ch_h : out_height;
            int upshift = (precision > 0 && precision < 16) ? (16 - precision) : 0;

            for (int y = 0; y < oh; y++) {
                for (int x = 0; x < ow; x++) {
                    int32_t v = (int32_t)y_data[y * cw + x] << upshift;
                    output[y * out_width + x] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
                }
            }
        }
    }

    /* Cleanup subbands */
    for (int ch = 0; ch < CF_MAX_CHANNELS; ch++) {
        /* subbands[0].data now holds the recon buffer (or was freed) */
        free(channels[ch].subbands[0].data);
        for (int sb = 1; sb < CF_MAX_SUBBANDS; sb++)
            free(channels[ch].subbands[sb].data);
    }

    return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int cf_reader_open(CfReader **out, const char *path) {
    if (!out || !path) return CF_ERR_IO;

    const char *ext = strrchr(path, '.');
    if (!ext) return CF_ERR_FMT;
    if (strcasecmp(ext, ".mov") != 0 && strcasecmp(ext, ".avi") != 0 &&
        strcasecmp(ext, ".mp4") != 0)
        return CF_ERR_FMT;

    CfReader *r = (CfReader *)calloc(1, sizeof(CfReader));
    if (!r) return CF_ERR_MEM;
    strncpy(r->path, path, sizeof(r->path) - 1);

    r->fp = fopen(path, "rb");
    if (!r->fp) {
        free(r);
        return CF_ERR_IO;
    }

    /* Build VLC table from encoder codebook */
    vlc_build(&r->vlc);

    int ret = parse_mov(r);
    if (ret != CF_OK) {
        cf_reader_close(r);
        return ret;
    }

    /* Probe first frame header to detect encoded format (Bayer vs YUV) early,
     * so frame_reader can route to the correct pipeline before any full decode. */
    r->info.channels = 3; /* default: YUV 4:2:2 */
    if (r->sample_count > 0) {
        uint32_t probe_size = r->sample_sizes[0];
        if (probe_size > 256) probe_size = 256; /* only need header tags */
        uint8_t *probe = (uint8_t *)malloc(probe_size);
        if (probe) {
            fseeko(r->fp, r->sample_offsets[0], SEEK_SET);
            if (fread(probe, 1, probe_size, r->fp) == probe_size) {
                /* Scan tag/value pairs for EncodedFormat */
                for (int p = 0; p + 3 < (int)probe_size; p += 4) {
                    int16_t tag = (int16_t)((probe[p] << 8) | probe[p + 1]);
                    uint16_t val = (uint16_t)((probe[p + 2] << 8) | probe[p + 3]);
                    int abs_tag = tag < 0 ? -tag : tag;
                    if (abs_tag == TAG_EncodedFormat) {
                        if (val == CF_FMT_BAYER) {
                            r->info.pixel_format = CF_PIX_BAYER_RGGB16;
                            r->info.is_bayer = 1;
                            r->info.channels = 4;
                        } else if (val == CF_FMT_YUV422) {
                            r->info.pixel_format = CF_PIX_YUV422_10;
                            r->info.is_bayer = 0;
                            r->info.channels = 3;
                        } else {
                            r->info.pixel_format = CF_PIX_RGB48;
                            r->info.is_bayer = 0;
                            r->info.channels = 3;
                        }
                        r->info_populated = 1;
                        break;
                    }
                }
            }
            free(probe);
        }
    }

    *out = r;
    return CF_OK;
}

int cf_reader_get_info(const CfReader *r, CfInfo *info) {
    if (!r || !info) return CF_ERR_IO;
    *info = r->info;
    return CF_OK;
}

int cf_reader_read_frame(CfReader *r, int frame_idx, uint16_t *bayer_out) {
    if (!r || !bayer_out) return CF_ERR_IO;
    if (frame_idx < 0 || frame_idx >= r->sample_count) return CF_ERR_IO;

    uint32_t frame_size = r->sample_sizes[frame_idx];
    int64_t  frame_off  = r->sample_offsets[frame_idx];

#ifdef CFHD_TRACE_TAGS
    fprintf(stderr, "cf_reader_read_frame: frame=%d offset=%lld size=%u\n",
            frame_idx, (long long)frame_off, frame_size);
#endif

    uint8_t *frame_data = (uint8_t *)malloc(frame_size);
    if (!frame_data) return CF_ERR_MEM;

    fseeko(r->fp, frame_off, SEEK_SET);
    if (fread(frame_data, 1, frame_size, r->fp) != frame_size) {
        free(frame_data);
        return CF_ERR_IO;
    }

    int ret = decode_frame(r, frame_data, (int)frame_size,
                           bayer_out, r->info.width, r->info.height);
    free(frame_data);
    return ret;
}

int cf_reader_read_frame_rgb(CfReader *r, int frame_idx, uint16_t *rgb_out) {
    if (!r || !rgb_out) return CF_ERR_IO;

    int w = r->info.width, h = r->info.height;

    /* Decode to YUV planar buffer: Y (w*h) + Cb (w/2*h) + Cr (w/2*h) */
    size_t yuv_size = (size_t)w * h + (size_t)(w / 2) * h * 2;
    uint16_t *yuv = (uint16_t *)calloc(yuv_size, sizeof(uint16_t));
    if (!yuv) return CF_ERR_MEM;

    int ret = cf_reader_read_frame(r, frame_idx, yuv);
    if (ret != CF_OK) { free(yuv); return ret; }

    /* YUV 4:2:2 → RGB conversion (inverse BT.709) */
    uint16_t *y_plane  = yuv;
    uint16_t *cb_plane = yuv + (size_t)w * h;
    uint16_t *cr_plane = cb_plane + (size_t)(w / 2) * h;

    uint16_t *r_out = rgb_out;
    uint16_t *g_out = rgb_out + (size_t)w * h;
    uint16_t *b_out = g_out + (size_t)w * h;

    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            /* 16-bit YUV: Y is full range, Cb/Cr offset at 32768 */
            int32_t Y16  = (int32_t)y_plane[row * w + col];
            int32_t Cb16 = (int32_t)cb_plane[row * (w / 2) + col / 2] - 32768;
            int32_t Cr16 = (int32_t)cr_plane[row * (w / 2) + col / 2] - 32768;

            /* Inverse BT.709: */
            int32_t R = Y16 + (int32_t)(1.5748f * Cr16);
            int32_t G = Y16 - (int32_t)(0.1873f * Cb16) - (int32_t)(0.4681f * Cr16);
            int32_t B = Y16 + (int32_t)(1.8556f * Cb16);

            if (R < 0) R = 0; if (R > 65535) R = 65535;
            if (G < 0) G = 0; if (G > 65535) G = 65535;
            if (B < 0) B = 0; if (B > 65535) B = 65535;

            size_t idx = (size_t)row * w + col;
            r_out[idx] = (uint16_t)R;
            g_out[idx] = (uint16_t)G;
            b_out[idx] = (uint16_t)B;
        }
    }

    free(yuv);
    return CF_OK;
}

void cf_reader_close(CfReader *r) {
    if (!r) return;
    if (r->fp) fclose(r->fp);
    free(r->sample_sizes);
    free(r->sample_offsets);
    free(r);
}

int cf_reader_probe_frame_count(const char *path) {
    CfReader *r = NULL;
    int ret = cf_reader_open(&r, path);
    if (ret != CF_OK) return -1;
    int count = r->info.frame_count;
    cf_reader_close(r);
    return count;
}

int cf_reader_probe_dimensions(const char *path, int *width, int *height) {
    CfReader *r = NULL;
    int ret = cf_reader_open(&r, path);
    if (ret != CF_OK) return -1;
    if (width)  *width  = r->info.width;
    if (height) *height = r->info.height;
    cf_reader_close(r);
    return 0;
}
