#ifndef DENOISE_BRIDGE_H
#define DENOISE_BRIDGE_H

#include <stdint.h>

/* Tuning parameters for temporal filter (shared with temporal_filter.h) */
#ifndef TEMPORAL_FILTER_TUNING_DEFINED
#define TEMPORAL_FILTER_TUNING_DEFINED
typedef struct {
    float chroma_boost;     /* R/B bilateral kernel multiplier (1.0 = same as luma) */
    float dist_sigma;       /* distance confidence sigma in green-pixel units */
    float flow_tightening;  /* per-pixel strictness = 1 + mag * flow_tightening */
} TemporalFilterTuning;
#endif

/* Configuration passed from Swift to the C denoising pipeline. */
typedef struct {
    int   window_size;       /* frames in temporal window (5, 7, or 9) */
    float strength;          /* bilateral filter strength (0.7 / 1.0 / 1.5) */
    float noise_sigma;       /* 0 = auto-estimate from first frame */
    float spatial_strength;  /* 0 = off */
    int   use_ml;            /* 1 = use CoreML FastDVDnet instead of bilateral filter */
    const char *dark_frame_path; /* path to dark frame .mov for subtraction, NULL = off */
    int   auto_dark_frame;       /* 1 = auto-estimate hot pixels from input, 0 = off */
    const char *hotpixel_profile_path; /* path to .bin hot pixel profile, NULL = off */
    int   detected_iso;          /* ISO from metadata, used to select profile entry */
    int   use_cnn_postfilter;    /* 1 = apply CNN post-filter after bilateral */
    int   protect_subjects;      /* 1 = boost denoise on detected persons */
    int   invert_mask;           /* 1 = invert segmentation (boost background instead) */
    float motion_avg;            /* average motion in pixels (from analyze_motion), 0 = unknown */
    int   start_frame;           /* first frame to process (0-based). 0 = beginning */
    int   end_frame;             /* last frame to process (exclusive). <=0 = all frames */
    int   temporal_filter_mode;  /* 0 = bilateral (default), 1 = Wiener DCT */
    int   output_format;         /* 0 = auto (match input), 1 = force MOV, 2 = force DNG, 3 = force BRAW, 4 = EXR sequence */
    int   collect_training_data; /* 1 = extract patch pairs for model training, 0 = off */
    float black_level;           /* sensor black level in 16-bit ADU (0 = use default 6032) */
    float shot_gain;             /* shot noise gain (0 = use default 180) */
    float read_noise;            /* read noise floor in 16-bit ADU (0 = use default 616) */
} DenoiseCConfig;

/* Progress callback called once per encoded frame.
 * current : frames completed so far (0-based)
 * total   : total frame count
 * ctx     : opaque pointer passed through from denoise_file()
 * Return 0 to continue, 1 to cancel. */
typedef int (*DenoiseCProgressCB)(int current, int total, void *ctx);

/* Return codes */
#define DENOISE_OK              0
#define DENOISE_ERR_INPUT_OPEN  (-1)
#define DENOISE_ERR_OUTPUT_OPEN (-2)
#define DENOISE_ERR_ALLOC       (-3)
#define DENOISE_ERR_ENCODE      (-4)
#define DENOISE_ERR_CANCELLED   (-5)

/* Run the full denoising pipeline on input_path, writing to output_path.
 * progress_cb is called after each frame; pass NULL to disable.
 * Returns DENOISE_OK (0) on success, negative on error/cancel. */
int denoise_file(
    const char         *input_path,
    const char         *output_path,
    const DenoiseCConfig *cfg,
    DenoiseCProgressCB  progress_cb,
    void               *progress_ctx);

/* Analyze clip motion by sampling frame pairs and computing optical flow.
 * Writes average and max per-frame mean flow magnitude (in pixels).
 * progress_cb reports analysis progress; pass NULL to disable.
 * Returns DENOISE_OK on success, negative on error. */
int analyze_motion(
    const char         *input_path,
    float              *avg_motion,
    float              *max_motion,
    DenoiseCProgressCB  progress_cb,
    void               *progress_ctx);

/* Generate a single denoised frame as a 1-frame ProRes RAW .mov file.
 * temp_output_path: where to write the temporary .mov (caller chooses path).
 * The resulting file can be decoded by AVFoundation to get the actual image.
 * Returns DENOISE_OK on success, negative on error. */
int denoise_preview_frame(
    const char          *input_path,
    int                  frame_index,
    const DenoiseCConfig *cfg,
    const char          *temp_output_path);

/* Probe video dimensions without decoding frames.
 * Returns 0 on success, -1 on error. */
int denoise_probe_dimensions(const char *path, int *width, int *height);

/* Probe frame count from input file/folder.
 * Returns frame count on success, -1 on error. */
int denoise_probe_frame_count(const char *path);

/* Detect input format. Returns 0=MOV, 1=CinemaDNG, 2=BRAW, -1=unknown. */
int denoise_probe_format(const char *path);

/* Probe camera model and ISO from MOV metadata.
 * camera_model: buffer to write camera model string (null-terminated)
 * model_len: size of camera_model buffer
 * iso: output ISO value (0 if not found)
 * Returns 0 on success, -1 on error. */
int denoise_probe_camera(const char *path, char *camera_model, int model_len,
                         int *iso);

/* Auto-estimate dark frame (hot pixel map) from video content.
 * Samples ~30 frames, computes per-pixel minimum, then detects hot pixels
 * as isolated bright outliers vs their Bayer-aware neighborhood.
 * Writes estimated dark frame to temp_output_path as raw uint16 binary.
 * out_hot_count: number of hot pixels detected (can be NULL).
 * Returns 0 on success, -1 on error. */
int denoise_estimate_dark_frame(const char *input_path,
                                const char *temp_output_path,
                                int *out_hot_count);

/* Load calibrated hot pixel profile from binary file.
 * Returns opaque handle, or NULL on error.
 * Call hotpixel_profile_free() when done. */
typedef struct HotPixelProfile HotPixelProfile;
HotPixelProfile *hotpixel_profile_load(const char *path, int iso);
void hotpixel_profile_free(HotPixelProfile *hp);

/* Apply hot pixel correction: replace hot pixels with Bayer-aware median.
 * Modifies frame in-place. */
void hotpixel_profile_apply(const HotPixelProfile *hp,
                            uint16_t *frame, int width, int height);

/* CPU fallback for temporal filter (called from Metal bridge when GPU unavailable).
 * This is a pure C function defined in temporal_filter.c. */
void temporal_filter_frame_cpu(
    uint16_t *output,
    const uint16_t **frames,
    const float **flows_x,
    const float **flows_y,
    int num_frames, int center_idx,
    int width, int height,
    float strength, float noise_sigma,
    const TemporalFilterTuning *tuning);

/* VST+Bilateral temporal filter (CPU fallback for GPU bridge).
 * Defined in temporal_techniques.c. */
void technique_vst_bilateral(
    uint16_t *output, const uint16_t **frames,
    const float **flows_x, const float **flows_y,
    int num_frames, int center_idx,
    int width, int height, float noise_sigma);

/* Parameter sweep: process 6 sequential frames with multiple configs,
 * writing raw uint16 output for each config to /tmp/sweep_configN/.
 * Returns number of configs tested, or negative on error. */
int denoise_parameter_sweep(
    const char *input_path,
    int start_frame,
    const DenoiseCConfig *base_cfg);

/* Technique sweep: compare 5 alternative temporal denoising techniques
 * against baseline NLM. Writes raw uint16 output per technique per frame
 * to /tmp/bayerflow_technique_sweep/.
 * Returns number of techniques tested, or negative on error. */
int denoise_technique_sweep(
    const char *input_path,
    int start_frame,
    const DenoiseCConfig *base_cfg);

/* ---- ProRes 4444 Writer (Swift bridge) ---- */
/* For RGB-output formats (R3D, etc.) that can't produce ProRes RAW. */
int  prores444_writer_open(const char *path, int width, int height, float fps);
int  prores444_writer_write_frame(const uint16_t *rgb_planar);
void prores444_writer_close(void);
int  prores444_writer_available(void);

/* ---- RED SDK availability ---- */
/* Returns 1 if RED SDK was linked at compile time, 0 if stub. */
int r3d_sdk_available(void);

/* ---- Noise Profiler ---- */
/* Noise model fitted from a flat patch in a raw Bayer frame. */
typedef struct {
    float sigma;        /* measured noise std-dev at patch brightness */
    float read_noise;   /* read noise floor */
    float shot_gain;    /* shot noise slope */
    float black_level;  /* estimated sensor black level */
    float mean_signal;  /* mean pixel value in patch (16-bit) */
    int   valid;        /* 1 if fit succeeded */
} CNoiseProfile;

/* Fit noise model from a flat patch (px,py,pw,ph in Bayer pixel coords). */
void noise_profile_from_patch(const uint16_t *bayer,
                               int bayer_w, int bayer_h,
                               int px, int py, int pw, int ph,
                               CNoiseProfile *out);

/* Read a single raw Bayer frame from a video file. Caller must free(). */
uint16_t *noise_profile_read_frame(const char *path, int frame_index,
                                   int *out_width, int *out_height);

#endif /* DENOISE_BRIDGE_H */
