import Foundation

// MARK: - C-callable bridge for Metal temporal filter
// These @_cdecl functions allow C code (denoise_core.c) to call into Swift/Metal.

/// GPU-accelerated temporal filter. Falls back to CPU if Metal is unavailable.
/// Called from denoise_core.c in place of temporal_filter_frame().
/// guide: optional pointer to a denoised frame used for patch matching (NULL = use center).
@_cdecl("temporal_filter_frame_gpu")
func temporal_filter_frame_gpu(
    _ output: UnsafeMutablePointer<UInt16>,
    _ frames: UnsafeMutablePointer<UnsafePointer<UInt16>?>,
    _ flowsX: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ flowsY: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ numFrames: Int32,
    _ centerIdx: Int32,
    _ width: Int32,
    _ height: Int32,
    _ strength: Float,
    _ noiseSigma: Float,
    _ chromaBoost: Float,
    _ distSigma: Float,
    _ flowTightening: Float,
    _ guide: UnsafePointer<UInt16>?,
    _ centerWeight: Float
) {
    if let gpu = MetalTemporalFilter.shared {
        gpu.filterFrame(
            output: output,
            frames: UnsafePointer(frames),
            flowsX: UnsafePointer(flowsX),
            flowsY: UnsafePointer(flowsY),
            numFrames: Int(numFrames),
            centerIdx: Int(centerIdx),
            width: Int(width),
            height: Int(height),
            strength: strength,
            noiseSigma: noiseSigma,
            chromaBoost: chromaBoost,
            distSigma: distSigma,
            flowTightening: flowTightening,
            guide: guide,
            centerWeight: centerWeight
        )
    } else {
        // CPU fallback — call original C function
        var tuning = TemporalFilterTuning(chroma_boost: chromaBoost,
                                          dist_sigma: distSigma,
                                          flow_tightening: flowTightening)
        temporal_filter_frame_cpu(output, frames, flowsX, flowsY,
                                  numFrames, centerIdx, width, height,
                                  strength, noiseSigma, &tuning)
    }
}

/// Returns 1 if Metal GPU is available, 0 otherwise.
@_cdecl("metal_gpu_available")
func metal_gpu_available() -> Int32 {
    return MetalTemporalFilter.shared != nil ? 1 : 0
}

// MARK: - GPU-resident frame ring (zero-copy temporal filter)

/// Allocate GPU-backed frame ring buffers. Returns CPU-accessible pointers
/// via gpu_ring_frame_ptr / gpu_ring_denoised_ptr. Frames written through
/// these pointers live in Metal shared memory — no copy needed for GPU access.
@_cdecl("gpu_ring_init")
func gpu_ring_init(_ numSlots: Int32, _ width: Int32, _ height: Int32) {
    MetalTemporalFilter.shared?.ensureRing(
        numSlots: Int(numSlots), width: Int(width), height: Int(height))
}

/// Get CPU-writable pointer for raw frame ring slot (backed by MTLBuffer).
@_cdecl("gpu_ring_frame_ptr")
func gpu_ring_frame_ptr(_ slot: Int32) -> UnsafeMutablePointer<UInt16>? {
    return MetalTemporalFilter.shared?.frameRingPointer(slot: Int(slot))
}

/// Get CPU-writable pointer for denoised cache ring slot (backed by MTLBuffer).
@_cdecl("gpu_ring_denoised_ptr")
func gpu_ring_denoised_ptr(_ slot: Int32) -> UnsafeMutablePointer<UInt16>? {
    return MetalTemporalFilter.shared?.denoisedRingPointer(slot: Int(slot))
}

/// GPU VST+Bilateral temporal filter using ring buffers.
/// 4-pass pipeline: collect → preestimate → fuse → finalize.
@_cdecl("temporal_filter_vst_bilateral_gpu_ring")
func temporal_filter_vst_bilateral_gpu_ring(
    _ output: UnsafeMutablePointer<UInt16>,
    _ ringSlots: UnsafePointer<Int32>,
    _ useDenoised: UnsafePointer<Int32>,
    _ flowsX: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ flowsY: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ numFrames: Int32,
    _ centerIdx: Int32,
    _ width: Int32,
    _ height: Int32,
    _ noiseSigma: Float,
    _ blackLevel: Float,
    _ shotGain: Float,
    _ readNoise: Float
) {
    if let gpu = MetalTemporalFilter.shared {
        gpu.filterFrameRingVSTBilateral(
            output: output,
            ringSlots: ringSlots,
            useDenoised: useDenoised,
            flowsX: UnsafePointer(flowsX),
            flowsY: UnsafePointer(flowsY),
            numFrames: Int(numFrames),
            centerIdx: Int(centerIdx),
            width: Int(width),
            height: Int(height),
            noiseSigma: noiseSigma,
            blackLevel: blackLevel,
            shotGain: shotGain,
            readNoise: readNoise
        )
    } else {
        // CPU fallback — build frame pointer array from ring
        let n = Int(numFrames)
        var framePtrs = [UnsafePointer<UInt16>?](repeating: nil, count: n)
        for i in 0..<n {
            let slot = Int(ringSlots[i])
            if useDenoised[i] != 0 {
                framePtrs[i] = UnsafePointer(gpu_ring_denoised_ptr(Int32(slot)))
            } else {
                framePtrs[i] = UnsafePointer(gpu_ring_frame_ptr(Int32(slot)))
            }
        }
        framePtrs.withUnsafeMutableBufferPointer { buf in
            technique_vst_bilateral(output, buf.baseAddress, flowsX, flowsY,
                                     numFrames, centerIdx, width, height, noiseSigma)
        }
    }
}

/// Zero-copy variant: writes to shared MTLBuffer instead of CPU pointer.
/// Returns the CPU-accessible pointer to the shared buffer.
@_cdecl("temporal_filter_vst_bilateral_gpu_ring_shared")
func temporal_filter_vst_bilateral_gpu_ring_shared(
    _ sharedBufIdx: Int32,
    _ ringSlots: UnsafePointer<Int32>,
    _ useDenoised: UnsafePointer<Int32>,
    _ flowsX: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ flowsY: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ numFrames: Int32,
    _ centerIdx: Int32,
    _ width: Int32,
    _ height: Int32,
    _ noiseSigma: Float,
    _ blackLevel: Float,
    _ shotGain: Float,
    _ readNoise: Float
) -> UnsafeMutablePointer<UInt16>? {
    guard let gpu = MetalTemporalFilter.shared else { return nil }
    // Use a dummy output pointer — won't be written to when sharedOutputIndex ≥ 0
    var dummy: UInt16 = 0
    gpu.filterFrameRingVSTBilateral(
        output: &dummy,
        ringSlots: ringSlots,
        useDenoised: useDenoised,
        flowsX: UnsafePointer(flowsX),
        flowsY: UnsafePointer(flowsY),
        numFrames: Int(numFrames),
        centerIdx: Int(centerIdx),
        width: Int(width),
        height: Int(height),
        noiseSigma: noiseSigma,
        blackLevel: blackLevel,
        shotGain: shotGain,
        readNoise: readNoise,
        sharedOutputIndex: Int(sharedBufIdx)
    )
    return gpu.sharedOutputPointer(index: Int(sharedBufIdx))
}

/// GPU temporal filter using ring buffers — frames already in GPU memory.
/// Only flow fields need CPU→GPU copy.
@_cdecl("temporal_filter_frame_gpu_ring")
func temporal_filter_frame_gpu_ring(
    _ output: UnsafeMutablePointer<UInt16>,
    _ ringSlots: UnsafePointer<Int32>,
    _ useDenoised: UnsafePointer<Int32>,
    _ flowsX: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ flowsY: UnsafeMutablePointer<UnsafePointer<Float>?>,
    _ numFrames: Int32,
    _ centerIdx: Int32,
    _ width: Int32,
    _ height: Int32,
    _ strength: Float,
    _ noiseSigma: Float,
    _ chromaBoost: Float,
    _ distSigma: Float,
    _ flowTightening: Float,
    _ guide: UnsafePointer<UInt16>?,
    _ centerWeight: Float
) {
    if let gpu = MetalTemporalFilter.shared {
        gpu.filterFrameRing(
            output: output,
            ringSlots: ringSlots,
            useDenoised: useDenoised,
            flowsX: UnsafePointer(flowsX),
            flowsY: UnsafePointer(flowsY),
            numFrames: Int(numFrames),
            centerIdx: Int(centerIdx),
            width: Int(width),
            height: Int(height),
            strength: strength,
            noiseSigma: noiseSigma,
            chromaBoost: chromaBoost,
            distSigma: distSigma,
            flowTightening: flowTightening,
            guide: guide,
            centerWeight: centerWeight
        )
    } else {
        // CPU fallback — build frame pointer array from ring for legacy path
        // (Not expected in practice — Metal is always available on supported Macs)
        var tuning = TemporalFilterTuning(chroma_boost: chromaBoost,
                                          dist_sigma: distSigma,
                                          flow_tightening: flowTightening)
        // Build frames array from ring pointers
        let n = Int(numFrames)
        var framePtrs = [UnsafePointer<UInt16>?](repeating: nil, count: n)
        for i in 0..<n {
            let slot = Int(ringSlots[i])
            if useDenoised[i] != 0 {
                framePtrs[i] = UnsafePointer(gpu_ring_denoised_ptr(Int32(slot)))
            } else {
                framePtrs[i] = UnsafePointer(gpu_ring_frame_ptr(Int32(slot)))
            }
        }
        framePtrs.withUnsafeMutableBufferPointer { buf in
            temporal_filter_frame_cpu(output, buf.baseAddress, flowsX, flowsY,
                                      numFrames, centerIdx, width, height,
                                      strength, noiseSigma, &tuning)
        }
    }
}
