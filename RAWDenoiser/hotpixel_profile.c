#include "include/denoise_bridge.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Binary format (little-endian):
 * Header: magic "HPXL" (4 bytes)
 *         width (uint16), height (uint16), n_isos (uint16), pedestal (uint16)
 * Per ISO block:
 *         iso (uint32), n_hot (uint32)
 *         n_hot × (y uint16, x uint16)
 */

struct HotPixelProfile {
    int n_hot;
    uint16_t *hot_y;   /* y coordinates */
    uint16_t *hot_x;   /* x coordinates */
    int iso;
};

HotPixelProfile *hotpixel_profile_load(const char *path, int target_iso) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "hotpixel_profile: cannot open %s\n", path);
        return NULL;
    }

    /* Read header */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "HPXL", 4) != 0) {
        fprintf(stderr, "hotpixel_profile: invalid magic\n");
        fclose(f);
        return NULL;
    }

    uint16_t header[4];
    if (fread(header, 2, 4, f) != 4) {
        fprintf(stderr, "hotpixel_profile: truncated header\n");
        fclose(f);
        return NULL;
    }
    int n_isos = header[2];

    /* Find closest ISO */
    int best_iso = 0;
    int best_diff = 0x7FFFFFFF;
    long best_offset = 0;
    int best_n_hot = 0;

    for (int i = 0; i < n_isos; i++) {
        uint32_t iso_val, n_hot;
        if (fread(&iso_val, 4, 1, f) != 1) break;
        if (fread(&n_hot, 4, 1, f) != 1) break;

        int diff = abs((int)iso_val - target_iso);
        if (diff < best_diff) {
            best_diff = diff;
            best_iso = (int)iso_val;
            best_offset = ftell(f);
            best_n_hot = (int)n_hot;
        }

        /* Skip coordinate data */
        fseek(f, (long)n_hot * 4, SEEK_CUR);
    }

    if (best_iso == 0 || best_n_hot == 0) {
        fprintf(stderr, "hotpixel_profile: no ISO match for %d\n", target_iso);
        fclose(f);
        return NULL;
    }

    printf("hotpixel_profile: ISO %d → using profile ISO %d (%d hot pixels)\n",
           target_iso, best_iso, best_n_hot);

    /* Allocate and read hot pixel coordinates */
    HotPixelProfile *hp = calloc(1, sizeof(HotPixelProfile));
    if (!hp) { fclose(f); return NULL; }

    hp->n_hot = best_n_hot;
    hp->iso = best_iso;
    hp->hot_y = malloc(best_n_hot * sizeof(uint16_t));
    hp->hot_x = malloc(best_n_hot * sizeof(uint16_t));

    if (!hp->hot_y || !hp->hot_x) {
        hotpixel_profile_free(hp);
        fclose(f);
        return NULL;
    }

    fseek(f, best_offset, SEEK_SET);
    for (int i = 0; i < best_n_hot; i++) {
        uint16_t yx[2];
        if (fread(yx, 2, 2, f) != 2) {
            fprintf(stderr, "hotpixel_profile: truncated at pixel %d\n", i);
            hp->n_hot = i;
            break;
        }
        hp->hot_y[i] = yx[0];
        hp->hot_x[i] = yx[1];
    }

    fclose(f);
    return hp;
}

void hotpixel_profile_free(HotPixelProfile *hp) {
    if (!hp) return;
    free(hp->hot_y);
    free(hp->hot_x);
    free(hp);
}

/* Helper: median of up to 8 uint16 values */
static uint16_t median_u16(uint16_t *vals, int n) {
    /* Simple insertion sort for small n */
    for (int i = 1; i < n; i++) {
        uint16_t key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            j--;
        }
        vals[j + 1] = key;
    }
    if (n % 2 == 1) return vals[n / 2];
    return (uint16_t)(((uint32_t)vals[n/2 - 1] + vals[n/2]) / 2);
}

void hotpixel_profile_apply(const HotPixelProfile *hp,
                            uint16_t *frame, int width, int height) {
    if (!hp || hp->n_hot == 0) return;

    for (int i = 0; i < hp->n_hot; i++) {
        int y = hp->hot_y[i];
        int x = hp->hot_x[i];

        if (y >= height || x >= width) continue;

        /* Collect same-channel neighbors (step=2 for Bayer) */
        uint16_t neighbors[8];
        int n = 0;

        for (int dy = -2; dy <= 2; dy += 2) {
            for (int dx = -2; dx <= 2; dx += 2) {
                if (dy == 0 && dx == 0) continue;
                int ny = y + dy;
                int nx = x + dx;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    neighbors[n++] = frame[ny * width + nx];
                }
            }
        }

        if (n > 0) {
            frame[y * width + x] = median_u16(neighbors, n);
        }
    }
}
