#import <Vision/Vision.h>
#import <CoreVideo/CoreVideo.h>
#include "../include/of_apple.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Create a float16 single-channel CVPixelBuffer from a 16-bit grayscale image.
 * IOSurface-backed so Vision/Metal can use it for optical flow.
 * Float16 preserves the full 16-bit dynamic range — dark pixels (2048-3000)
 * that collapse to only 3 levels in 8-bit get ~60 distinct float16 values,
 * giving Vision OF much better signal for motion estimation in shadows. */
static CVPixelBufferRef make_pixel_buffer(const uint16_t *src, int width, int height) {
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };

    CVPixelBufferRef pb = NULL;
    CVReturn ret = CVPixelBufferCreate(
        kCFAllocatorDefault,
        (size_t)width, (size_t)height,
        kCVPixelFormatType_OneComponent16Half,
        (__bridge CFDictionaryRef)attrs,
        &pb);

    if (ret != kCVReturnSuccess) return NULL;

    CVPixelBufferLockBaseAddress(pb, 0);
    __fp16 *dst = (__fp16 *)CVPixelBufferGetBaseAddress(pb);
    size_t bpr = CVPixelBufferGetBytesPerRow(pb);

    for (int y = 0; y < height; y++) {
        __fp16 *row = (__fp16 *)((uint8_t *)dst + (size_t)y * bpr);
        const uint16_t *src_row = src + (size_t)y * width;
        for (int x = 0; x < width; x++) {
            row[x] = (__fp16)((float)src_row[x] * (1.0f / 65535.0f));
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);

    return pb;
}

/* Downsample uint16 image by 2x using 2×2 box average. */
static uint16_t *downsample_2x(const uint16_t *src, int w, int h,
                                int *out_w, int *out_h) {
    int hw = w / 2, hh = h / 2;
    uint16_t *dst = (uint16_t *)malloc((size_t)hw * hh * sizeof(uint16_t));
    if (!dst) return NULL;
    for (int y = 0; y < hh; y++) {
        const uint16_t *r0 = src + (size_t)(y * 2) * w;
        const uint16_t *r1 = r0 + w;
        for (int x = 0; x < hw; x++) {
            int x2 = x * 2;
            dst[y * hw + x] = (uint16_t)(((uint32_t)r0[x2] + r0[x2+1]
                                         + r1[x2] + r1[x2+1] + 2) >> 2);
        }
    }
    *out_w = hw;
    *out_h = hh;
    return dst;
}

/* Bilinear upscale of flow field by 2×.
 * Flow magnitudes are doubled to convert half-res displacements
 * to full-res (green channel) displacements. */
static void upscale_flow_2x(const float *src_fx, const float *src_fy,
                             int sw, int sh,
                             float *dst_fx, float *dst_fy,
                             int fw, int fh) {
    for (int y = 0; y < fh; y++) {
        float sy = (y + 0.5f) * 0.5f - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        if (y0 < 0) { y0 = 0; fy = 0; }
        int y1 = y0 + 1;
        if (y1 >= sh) y1 = sh - 1;

        for (int x = 0; x < fw; x++) {
            float sx = (x + 0.5f) * 0.5f - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            if (x0 < 0) { x0 = 0; fx = 0; }
            int x1 = x0 + 1;
            if (x1 >= sw) x1 = sw - 1;

            float w00 = (1-fx)*(1-fy), w10 = fx*(1-fy);
            float w01 = (1-fx)*fy,     w11 = fx*fy;

            size_t i00 = (size_t)y0*sw + x0, i10 = (size_t)y0*sw + x1;
            size_t i01 = (size_t)y1*sw + x0, i11 = (size_t)y1*sw + x1;

            dst_fx[y*fw + x] = 2.0f * (w00*src_fx[i00] + w10*src_fx[i10]
                                      + w01*src_fx[i01] + w11*src_fx[i11]);
            dst_fy[y*fw + x] = 2.0f * (w00*src_fy[i00] + w10*src_fy[i10]
                                      + w01*src_fy[i01] + w11*src_fy[i11]);
        }
    }
}

/* Single-pair OF — kept for bootstrap / standalone callers. */
int compute_apple_flow(const uint16_t *frame1, const uint16_t *frame2,
                       int green_w, int green_h,
                       float *flow_x, float *flow_y)
{
    /* Delegate to batch path with 1 neighbor */
    const uint16_t *neighbors[1] = { frame2 };
    float *fx_out[1] = { flow_x };
    float *fy_out[1] = { flow_y };
    return compute_apple_flow_batch(frame1, neighbors, 1,
                                    green_w, green_h, fx_out, fy_out);
}

/* Batch OF: compute flow from center to multiple neighbors.
 * Center frame is downsampled and converted to CVPixelBuffer ONCE,
 * then reused for all neighbor pairs. Processes serially — Vision OF
 * runs on ANE which cannot parallelize concurrent requests from one process;
 * serializing avoids ANE contention and thread-spawn overhead. */
int compute_apple_flow_batch(const uint16_t *center,
                             const uint16_t *const *neighbors, int num_neighbors,
                             int green_w, int green_h,
                             float **fx_out, float **fy_out)
{
    if (num_neighbors <= 0) return 0;

    /* Downsample center to half-res — 4× fewer pixels for Vision,
     * faster OF. Flow is inherently smooth so 2× upscale back to green
     * resolution is lossless for denoising purposes. */
    int hw, hh;
    uint16_t *half_center = downsample_2x(center, green_w, green_h, &hw, &hh);
    if (!half_center) return -1;

    CVPixelBufferRef pb_center = make_pixel_buffer(half_center, hw, hh);
    free(half_center);
    if (!pb_center) return -1;

    size_t half_npix = (size_t)hw * hh;

    /* Single reusable scratch buffers — no need to pre-allocate per-neighbor */
    uint16_t *half_nbr = (uint16_t *)malloc(half_npix * sizeof(uint16_t));
    float    *tmp_fx   = (float *)malloc(half_npix * sizeof(float));
    float    *tmp_fy   = (float *)malloc(half_npix * sizeof(float));
    if (!half_nbr || !tmp_fx || !tmp_fy) {
        free(half_nbr); free(tmp_fx); free(tmp_fy);
        CFRelease(pb_center);
        return -1;
    }

    int err = 0;
    size_t full_npix = (size_t)green_w * green_h;

    for (int n = 0; n < num_neighbors; n++) {
        @autoreleasepool {
            /* Downsample neighbor to half-res */
            for (int y = 0; y < hh; y++) {
                const uint16_t *r0 = neighbors[n] + (size_t)(y * 2) * green_w;
                const uint16_t *r1 = r0 + green_w;
                for (int x = 0; x < hw; x++) {
                    int x2 = x * 2;
                    half_nbr[y * hw + x] = (uint16_t)(((uint32_t)r0[x2] + r0[x2+1]
                                                       + r1[x2] + r1[x2+1] + 2) >> 2);
                }
            }

            CVPixelBufferRef pb_nbr = make_pixel_buffer(half_nbr, hw, hh);
            if (!pb_nbr) { err = -1; continue; }

            NSError *error = nil;
            VNGenerateOpticalFlowRequest *request =
                [[VNGenerateOpticalFlowRequest alloc]
                 initWithTargetedCVPixelBuffer:pb_nbr options:@{}];
            /* Medium: faster than High, sufficient accuracy for denoising.
             * Our bilateral range kernel absorbs sub-pixel flow imprecision. */
            request.computationAccuracy = VNGenerateOpticalFlowRequestComputationAccuracyMedium;

            VNImageRequestHandler *handler =
                [[VNImageRequestHandler alloc]
                 initWithCVPixelBuffer:pb_center options:@{}];

            BOOL ok = [handler performRequests:@[request] error:&error];
            CFRelease(pb_nbr);

            if (!ok || error || !request.results || request.results.count == 0) {
                memset(fx_out[n], 0, full_npix * sizeof(float));
                memset(fy_out[n], 0, full_npix * sizeof(float));
                err = -1;
                continue;
            }

            VNPixelBufferObservation *obs = request.results[0];
            CVPixelBufferRef flowBuf = obs.pixelBuffer;
            CVPixelBufferLockBaseAddress(flowBuf, kCVPixelBufferLock_ReadOnly);

            int fw  = (int)CVPixelBufferGetWidth(flowBuf);
            int fh2 = (int)CVPixelBufferGetHeight(flowBuf);
            size_t bpr = CVPixelBufferGetBytesPerRow(flowBuf);
            const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(flowBuf);

            int copy_w = fw < hw ? fw : hw;
            int copy_h = fh2 < hh ? fh2 : hh;

            memset(tmp_fx, 0, half_npix * sizeof(float));
            memset(tmp_fy, 0, half_npix * sizeof(float));
            for (int y = 0; y < copy_h; y++) {
                const float *row = (const float *)(base + (size_t)y * bpr);
                for (int x = 0; x < copy_w; x++) {
                    tmp_fx[y * fw + x] = row[x * 2 + 0];
                    tmp_fy[y * fw + x] = row[x * 2 + 1];
                }
            }
            CVPixelBufferUnlockBaseAddress(flowBuf, kCVPixelBufferLock_ReadOnly);

            upscale_flow_2x(tmp_fx, tmp_fy, fw, fh2,
                            fx_out[n], fy_out[n], green_w, green_h);
        }
    }

    free(half_nbr); free(tmp_fx); free(tmp_fy);
    CFRelease(pb_center);
    return err;
}

