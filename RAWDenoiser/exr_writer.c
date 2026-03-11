/*
 * Minimal EXR Writer — Half-Float RGB
 *
 * Writes uncompressed OpenEXR 2.0 scanline files with half-float R, G, B channels.
 * Pure C implementation, no external dependencies.
 *
 * OpenEXR file layout (uncompressed, scanline):
 *   1. Magic number (4 bytes): 0x762f3101
 *   2. Version (4 bytes): 2 | 0x00000000 (single-part, scanline)
 *   3. Header attributes (variable): name + type + size + value, null-terminated
 *   4. End of header: single null byte
 *   5. Scanline offset table: height × int64
 *   6. Scanline blocks: per line → { int32 y, int32 dataSize, half B[w], half G[w], half R[w] }
 *
 * Channel order in EXR is always alphabetical: B, G, R.
 */

#include "../include/exr_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- IEEE 754 half-float conversion ---- */

/* Convert float to IEEE 754 half-precision (16-bit).
 * Handles normals, denormals, infinity, and NaN. */
static uint16_t float_to_half(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t bits = v.u;

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x007FFFFF;

    if (exp > 15) {
        /* Overflow → infinity */
        return (uint16_t)(sign | 0x7C00);
    } else if (exp > -15) {
        /* Normal */
        uint32_t h_exp  = (uint32_t)(exp + 15) << 10;
        uint32_t h_mant = mant >> 13;
        /* Round to nearest even */
        if ((mant & 0x1000) && (mant & 0x2FFF))
            h_mant++;
        return (uint16_t)(sign | h_exp | h_mant);
    } else if (exp > -25) {
        /* Denormal */
        mant |= 0x00800000; /* implicit leading 1 */
        int shift = -exp - 15 + 10 + 23; /* total right shift */
        if (shift > 31) shift = 31;
        uint32_t h_mant = mant >> shift;
        return (uint16_t)(sign | h_mant);
    }
    /* Underflow → zero */
    return (uint16_t)sign;
}

/* Convert uint16 [0..65535] to half-float [0.0..1.0] */
static uint16_t u16_to_half(uint16_t val) {
    float f = (float)val / 65535.0f;
    return float_to_half(f);
}

/* ---- EXR attribute writing helpers ---- */

/* Write a null-terminated string to file */
static void write_str(FILE *fp, const char *s) {
    fwrite(s, 1, strlen(s) + 1, fp);
}

/* Write an EXR attribute: name (null-term) + type (null-term) + size (int32) + value */
static void write_attr_str(FILE *fp, const char *name, const char *type,
                            const void *data, int32_t size) {
    write_str(fp, name);
    write_str(fp, type);
    fwrite(&size, 4, 1, fp);
    fwrite(data, 1, size, fp);
}

/* EXR channel descriptor */
typedef struct {
    char name[256];
    int32_t pixel_type;  /* 0=UINT, 1=HALF, 2=FLOAT */
    uint8_t pLinear;
    uint8_t reserved[3];
    int32_t xSampling;
    int32_t ySampling;
} ExrChannel;

/* Write the "channels" attribute (chlist type) */
static void write_channels(FILE *fp, int num_channels, const ExrChannel *channels) {
    /* Compute total size: sum of (name_len+1 + 16 bytes per channel) + 1 null terminator */
    int32_t total_size = 1; /* trailing null byte */
    for (int i = 0; i < num_channels; i++)
        total_size += (int32_t)(strlen(channels[i].name) + 1 + 4 + 1 + 3 + 4 + 4);

    write_str(fp, "channels");
    write_str(fp, "chlist");
    fwrite(&total_size, 4, 1, fp);

    for (int i = 0; i < num_channels; i++) {
        write_str(fp, channels[i].name);
        fwrite(&channels[i].pixel_type, 4, 1, fp);
        fwrite(&channels[i].pLinear, 1, 1, fp);
        fwrite(channels[i].reserved, 1, 3, fp);
        fwrite(&channels[i].xSampling, 4, 1, fp);
        fwrite(&channels[i].ySampling, 4, 1, fp);
    }
    /* Null terminator for channel list */
    uint8_t null = 0;
    fwrite(&null, 1, 1, fp);
}

/* EXR box2i: xMin, yMin, xMax, yMax (4 × int32) */
static void write_box2i(FILE *fp, const char *name,
                         int32_t xMin, int32_t yMin, int32_t xMax, int32_t yMax) {
    write_str(fp, name);
    write_str(fp, "box2i");
    int32_t size = 16;
    fwrite(&size, 4, 1, fp);
    fwrite(&xMin, 4, 1, fp);
    fwrite(&yMin, 4, 1, fp);
    fwrite(&xMax, 4, 1, fp);
    fwrite(&yMax, 4, 1, fp);
}

static void write_int(FILE *fp, const char *name, int32_t val) {
    write_str(fp, name);
    write_str(fp, "int");
    int32_t size = 4;
    fwrite(&size, 4, 1, fp);
    fwrite(&val, 4, 1, fp);
}

/* Write a single-byte enum attribute (compression, lineOrder) */
static void write_enum_byte(FILE *fp, const char *name, const char *type, uint8_t val) {
    write_str(fp, name);
    write_str(fp, type);
    int32_t size = 1;
    fwrite(&size, 4, 1, fp);
    fwrite(&val, 1, 1, fp);
}

static void write_float_attr(FILE *fp, const char *name, float val) {
    write_str(fp, name);
    write_str(fp, "float");
    int32_t size = 4;
    fwrite(&size, 4, 1, fp);
    fwrite(&val, 4, 1, fp);
}

static void write_v2f(FILE *fp, const char *name, float x, float y) {
    write_str(fp, name);
    write_str(fp, "v2f");
    int32_t size = 8;
    fwrite(&size, 4, 1, fp);
    fwrite(&x, 4, 1, fp);
    fwrite(&y, 4, 1, fp);
}

/* ---- Main writer ---- */

int exr_write_frame(const char *path, const uint16_t *rgb_planar,
                    int width, int height) {
    if (!path || !rgb_planar || width <= 0 || height <= 0)
        return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "exr_writer: cannot open '%s' for writing\n", path);
        return -1;
    }

    size_t plane_pixels = (size_t)width * height;
    const uint16_t *r_plane = rgb_planar;
    const uint16_t *g_plane = rgb_planar + plane_pixels;
    const uint16_t *b_plane = rgb_planar + 2 * plane_pixels;

    /* ---- Magic + version ---- */
    int32_t magic = 20000630; /* 0x762f3101 */
    int32_t version = 2;      /* version 2, single-part scanline, no tiles */
    fwrite(&magic, 4, 1, fp);
    fwrite(&version, 4, 1, fp);

    /* ---- Header attributes ---- */

    /* Channels: B, G, R (alphabetical order required by EXR spec) */
    ExrChannel channels[3];
    memset(channels, 0, sizeof(channels));
    strcpy(channels[0].name, "B");
    channels[0].pixel_type = 1; /* HALF */
    channels[0].xSampling = 1;
    channels[0].ySampling = 1;
    strcpy(channels[1].name, "G");
    channels[1].pixel_type = 1;
    channels[1].xSampling = 1;
    channels[1].ySampling = 1;
    strcpy(channels[2].name, "R");
    channels[2].pixel_type = 1;
    channels[2].xSampling = 1;
    channels[2].ySampling = 1;
    write_channels(fp, 3, channels);

    /* Compression: NONE (0) */
    write_enum_byte(fp, "compression", "compression", 0);

    /* Data window and display window */
    write_box2i(fp, "dataWindow", 0, 0, width - 1, height - 1);
    write_box2i(fp, "displayWindow", 0, 0, width - 1, height - 1);

    /* Line order: INCREASING_Y (0) */
    write_enum_byte(fp, "lineOrder", "lineOrder", 0);

    /* Pixel aspect ratio */
    write_float_attr(fp, "pixelAspectRatio", 1.0f);

    /* Screen window */
    write_v2f(fp, "screenWindowCenter", 0.0f, 0.0f);
    write_float_attr(fp, "screenWindowWidth", 1.0f);

    /* End of header */
    uint8_t null = 0;
    fwrite(&null, 1, 1, fp);

    /* ---- Scanline offset table ---- */
    /* For uncompressed, each scanline is one block.
     * Offset table: height × int64, filled after we know positions. */
    long offset_table_pos = ftell(fp);
    int64_t *offsets = (int64_t *)calloc(height, sizeof(int64_t));
    if (!offsets) { fclose(fp); return -1; }

    /* Write placeholder offset table */
    fwrite(offsets, sizeof(int64_t), height, fp);

    /* ---- Scanline data ---- */
    /* Each scanline block: int32 y_coord + int32 pixel_data_size + B[w] + G[w] + R[w]
     * Each channel is width × half (2 bytes) */
    int32_t scanline_data_size = width * 3 * 2; /* 3 channels × width × sizeof(half) */
    uint16_t *half_line = (uint16_t *)malloc(width * 3 * sizeof(uint16_t));
    if (!half_line) { free(offsets); fclose(fp); return -1; }

    for (int y = 0; y < height; y++) {
        offsets[y] = (int64_t)ftell(fp);

        /* Write scanline header */
        int32_t y_coord = y;
        fwrite(&y_coord, 4, 1, fp);
        fwrite(&scanline_data_size, 4, 1, fp);

        /* Convert and interleave: B channel, then G channel, then R channel */
        int src_offset = y * width;
        uint16_t *dst = half_line;

        /* B channel */
        for (int x = 0; x < width; x++)
            dst[x] = u16_to_half(b_plane[src_offset + x]);
        dst += width;

        /* G channel */
        for (int x = 0; x < width; x++)
            dst[x] = u16_to_half(g_plane[src_offset + x]);
        dst += width;

        /* R channel */
        for (int x = 0; x < width; x++)
            dst[x] = u16_to_half(r_plane[src_offset + x]);

        fwrite(half_line, sizeof(uint16_t), width * 3, fp);
    }

    /* ---- Write actual offsets ---- */
    fseek(fp, offset_table_pos, SEEK_SET);
    fwrite(offsets, sizeof(int64_t), height, fp);

    free(offsets);
    free(half_line);
    fclose(fp);
    return 0;
}
