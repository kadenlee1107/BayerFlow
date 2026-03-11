/*
 * RED R3D Reader — Stub Implementation
 *
 * This is a placeholder that returns R3D_ERR_SDK for all operations.
 * Replace with r3d_reader.mm (Obj-C++ wrapper around RED SDK) once
 * the RED SDK framework is integrated into the project.
 *
 * Download RED SDK (free): https://www.red.com/developer
 */

#include "../include/r3d_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct R3dReader {
    char     path[4096];
    R3dInfo  info;
};

static void r3d_sdk_not_available(void) {
    fprintf(stderr, "r3d_reader: RED SDK not integrated. "
            "Download from https://www.red.com/developer (free).\n");
}

int r3d_reader_open(R3dReader **out, const char *path) {
    if (!out || !path) return R3D_ERR_IO;

    /* Check extension */
    const char *ext = strrchr(path, '.');
    if (!ext) return R3D_ERR_FMT;
    if (strcasecmp(ext, ".r3d") != 0 && strcasecmp(ext, ".nraw") != 0)
        return R3D_ERR_FMT;

    r3d_sdk_not_available();
    return R3D_ERR_SDK;
}

int r3d_reader_get_info(const R3dReader *r, R3dInfo *info) {
    if (!r || !info) return R3D_ERR_IO;
    *info = r->info;
    return R3D_OK;
}

int r3d_reader_read_frame_rgb(R3dReader *r, int frame_idx,
                               uint16_t *rgb_out) {
    (void)r; (void)frame_idx; (void)rgb_out;
    r3d_sdk_not_available();
    return R3D_ERR_SDK;
}

void r3d_reader_close(R3dReader *r) {
    if (r) free(r);
}

int r3d_reader_probe_frame_count(const char *path) {
    (void)path;
    r3d_sdk_not_available();
    return -1;
}

int r3d_reader_probe_dimensions(const char *path, int *width, int *height) {
    (void)path; (void)width; (void)height;
    r3d_sdk_not_available();
    return -1;
}

int r3d_sdk_available(void) { return 0; }
