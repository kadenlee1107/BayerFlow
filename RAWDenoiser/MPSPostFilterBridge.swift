import Foundation

// MARK: - C-callable bridge for MPS GPU 4-channel CNN post-filter
// Called from denoise_core.c as a faster replacement for CoreML path.

/// GPU-accelerated 4-channel CNN post-filter via MPSGraph.
/// Falls back to CoreML single-channel path if MPS is unavailable.
@_cdecl("postfilter_frame_mps")
func postfilter_frame_mps(
    _ bayer: UnsafeMutablePointer<UInt16>,
    _ width: Int32,
    _ height: Int32,
    _ blendFactor: Float
) {
    guard let filter = MPSPostFilter.shared else { return }
    filter.apply(bayer: bayer, width: Int(width), height: Int(height), blendFactor: blendFactor)
}

/// Returns 1 if MPS GPU post-filter is available, 0 otherwise.
@_cdecl("mps_postfilter_available")
func mps_postfilter_available() -> Int32 {
    return MPSPostFilter.shared != nil ? 1 : 0
}

/// Set camera noise model for calibrated noise sigma map in CNN input.
/// black_level, read_noise: raw 16-bit values (e.g. 6032, 616)
/// shot_gain: raw scale (e.g. 180.0)
@_cdecl("mps_postfilter_set_noise_model")
func mps_postfilter_set_noise_model(_ black_level: Float, _ read_noise: Float, _ shot_gain: Float) {
    guard let filter = MPSPostFilter.shared else { return }
    filter.noiseModel = MPSPostFilter.NoiseModelParams(
        black_level: black_level / 65535.0,
        read_noise: read_noise / 65535.0,
        shot_gain: shot_gain
    )
}

/// Zero-copy CNN: reads from temporal filter's shared MTLBuffer, writes result back.
/// Eliminates 2 frame-sized memcpys per frame (GPU→CPU→GPU roundtrip).
@_cdecl("postfilter_frame_mps_shared")
func postfilter_frame_mps_shared(
    _ sharedBufIdx: Int32,
    _ width: Int32,
    _ height: Int32,
    _ blendFactor: Float
) {
    guard let filter = MPSPostFilter.shared,
          let tf = MetalTemporalFilter.shared,
          let inputBuf = tf.sharedOutputBuffer(index: Int(sharedBufIdx)) else { return }
    filter.apply(inputBuffer: inputBuf, width: Int(width), height: Int(height), blendFactor: blendFactor)
}

/// Enable/disable person segmentation subject protection.
/// When enabled, detected persons get boosted denoising on skin/hair.
/// invert_mask: 1 = invert (boost background instead of subjects)
@_cdecl("mps_postfilter_set_protect_subjects")
func mps_postfilter_set_protect_subjects(_ enable: Int32, _ protection: Float, _ invert_mask: Int32) {
    guard let filter = MPSPostFilter.shared else { return }
    filter.protectSubjects = enable != 0
    filter.subjectProtection = max(0, min(1.0, protection))
    filter.invertMask = invert_mask != 0
}
