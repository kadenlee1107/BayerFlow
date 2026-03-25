import Foundation
import Metal
import Accelerate

nonisolated struct MetalFilterParams: Sendable {
    var width: UInt32
    var height: UInt32
    var lut_scale_luma: Float
    var lut_scale_chroma: Float
    var dist_lut_scale: Float
    var flow_tightening: Float
    var h_luma: Float           // bilateral bandwidth = noise_sigma * strength
    var h_chroma: Float         // bilateral bandwidth for R/B channels
}

nonisolated struct VSTBilateralParams: Sendable {
    var width: UInt32
    var height: UInt32
    var noise_sigma: Float
    var h: Float             // bilateral bandwidth in VST domain (1.0)
    var z_reject: Float      // hard rejection threshold (3.0)
    var flow_sigma2: Float   // 2*sigma_flow^2 for distance attenuation (8.0)
    var sigma_g2: Float      // 2*sigma_g^2 for structural term (0.5)
    var black_level: Float   // sensor black level (default 6032)
    var shot_gain: Float     // shot noise gain (default 180)
    var read_noise: Float    // read noise floor (default 616)
}

nonisolated final class MetalTemporalFilter: @unchecked Sendable {
    static let shared: MetalTemporalFilter? = MetalTemporalFilter()

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let filterPipeline: MTLComputePipelineState
    private let normPipeline: MTLComputePipelineState

    // VST+Bilateral pipeline states (optional — NLM works without them)
    private let vstCollectPipeline: MTLComputePipelineState?
    private let vstPreestimatePipeline: MTLComputePipelineState?
    private let vstFusePipeline: MTLComputePipelineState?
    private let vstFinalizePipeline: MTLComputePipelineState?

    // FP16 VST+Bilateral pipeline states (2x ALU throughput on Apple GPU)
    private let vstCollectFP16Pipeline: MTLComputePipelineState?
    private let vstPreestimateFP16Pipeline: MTLComputePipelineState?
    private let vstFuseFP16Pipeline: MTLComputePipelineState?
    private let vstFinalizeFP16Pipeline: MTLComputePipelineState?
    private(set) var useFP16: Bool = true

    // Persistent buffers — resized per-clip, reused per-frame
    private var centerBuf: MTLBuffer?
    private var guideBuf: MTLBuffer?      // guide frame for patch matching (can differ from center)
    private var valSumBuf: MTLBuffer?
    private var wSumBuf: MTLBuffer?
    private var outputBuf: MTLBuffer?
    private var weightLutLumaBuf: MTLBuffer?
    private var threshLutBuf: MTLBuffer?
    private var distLutBuf: MTLBuffer?
    private var weightLutChromaBuf: MTLBuffer?

    // VST+Bilateral extra buffers
    private var zPreestimateBuf: MTLBuffer?
    private var maxFlowBuf: MTLBuffer?

    // Async GPU state — used by commitOnly / waitGPU pair
    private var asyncSemaphore: DispatchSemaphore? = nil
    private var asyncSucceeded = true
    private var asyncOutput: UnsafeMutablePointer<UInt16>? = nil
    private var asyncFramePixels: Int = 0

    // Pre-allocated flow buffer pools — sized per-clip, reused every frame
    private static let maxNeighbors = 14  // MAX_WINDOW(15) - 1 center
    private var flowXPool: [MTLBuffer] = []
    private var flowYPool: [MTLBuffer] = []

    // GPU-resident frame ring — frames live in MTLBuffers from allocation.
    // CPU writes directly via .contents(); GPU reads via buffer binding.
    // Eliminates ~620MB of per-frame memcpy (13 of 14 neighbors are repeats).
    private var frameRing: [MTLBuffer] = []     // W+1 slots for raw Bayer frames
    private var denoisedRing: [MTLBuffer] = []  // W+1 slots for cached denoised frames
    private var ringSlotCount: Int = 0

    // Shared output buffers for zero-copy TF→CNN pipeline (2 for ping-pong)
    private(set) var sharedOutputBufs: [MTLBuffer] = []

    private var currentWidth: Int = 0
    private var currentHeight: Int = 0

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.queue = queue

        guard let library = device.makeDefaultLibrary(),
              let filterFunc = library.makeFunction(name: "temporal_filter_kernel"),
              let normFunc = library.makeFunction(name: "temporal_filter_normalize") else {
            return nil
        }

        do {
            self.filterPipeline = try device.makeComputePipelineState(function: filterFunc)
            self.normPipeline = try device.makeComputePipelineState(function: normFunc)
        } catch {
            return nil
        }

        // VST+Bilateral pipelines (optional — NLM works without them)
        if let f1 = library.makeFunction(name: "vst_bilateral_collect"),
           let f2 = library.makeFunction(name: "vst_bilateral_preestimate"),
           let f3 = library.makeFunction(name: "vst_bilateral_fuse"),
           let f4 = library.makeFunction(name: "vst_bilateral_finalize") {
            self.vstCollectPipeline = try? device.makeComputePipelineState(function: f1)
            self.vstPreestimatePipeline = try? device.makeComputePipelineState(function: f2)
            self.vstFusePipeline = try? device.makeComputePipelineState(function: f3)
            self.vstFinalizePipeline = try? device.makeComputePipelineState(function: f4)
        } else {
            self.vstCollectPipeline = nil
            self.vstPreestimatePipeline = nil
            self.vstFusePipeline = nil
            self.vstFinalizePipeline = nil
        }

        // FP16 VST+Bilateral pipelines (2x ALU throughput)
        if let f1 = library.makeFunction(name: "vst_bilateral_collect_fp16"),
           let f2 = library.makeFunction(name: "vst_bilateral_preestimate_fp16"),
           let f3 = library.makeFunction(name: "vst_bilateral_fuse_fp16"),
           let f4 = library.makeFunction(name: "vst_bilateral_finalize_fp16") {
            self.vstCollectFP16Pipeline = try? device.makeComputePipelineState(function: f1)
            self.vstPreestimateFP16Pipeline = try? device.makeComputePipelineState(function: f2)
            self.vstFuseFP16Pipeline = try? device.makeComputePipelineState(function: f3)
            self.vstFinalizeFP16Pipeline = try? device.makeComputePipelineState(function: f4)
        } else {
            self.vstCollectFP16Pipeline = nil
            self.vstPreestimateFP16Pipeline = nil
            self.vstFuseFP16Pipeline = nil
            self.vstFinalizeFP16Pipeline = nil
            self.useFP16 = false
        }
    }

    /// Wait for a previously committed async GPU temporal filter to complete.
    /// Must be called after commitOnly=true filterFrameRingVSTBilateral before
    /// accessing the output buffer. Returns false on GPU error or timeout.
    func waitGPU() -> Bool {
        guard let sem = asyncSemaphore else { return true }
        if sem.wait(timeout: .now() + 120.0) == .timedOut {
            fputs("GPU TIMEOUT [async filterFrameRingVSTBilateral]\n", stderr)
            asyncSemaphore = nil
            return false
        }
        asyncSemaphore = nil
        // Perform deferred memcpy now that GPU is done
        if let out = asyncOutput, let outputBuf, asyncFramePixels > 0 {
            let src = outputBuf.contents().bindMemory(to: UInt16.self, capacity: asyncFramePixels)
            memcpy(out, src, asyncFramePixels * MemoryLayout<UInt16>.size)
            asyncOutput = nil
        }
        return asyncSucceeded
    }

    /// Commit command buffer with error logging and timeout.
    /// Returns true on success, false on GPU error or timeout.
    private func commitAndWait(_ cmdBuf: MTLCommandBuffer, label: String, timeoutSeconds: Double = 120.0) -> Bool {
        let semaphore = DispatchSemaphore(value: 0)
        var succeeded = true

        cmdBuf.addCompletedHandler { cb in
            if cb.status == .error {
                succeeded = false
                fputs("GPU ERROR [\(label)]: \(cb.error?.localizedDescription ?? "unknown")\n", stderr)
            }
            semaphore.signal()
        }

        cmdBuf.commit()

        if semaphore.wait(timeout: .now() + timeoutSeconds) == .timedOut {
            fputs("GPU TIMEOUT [\(label)]: did not complete in \(Int(timeoutSeconds))s\n", stderr)
            return false
        }
        return succeeded
    }

    private func ensureBuffers(width: Int, height: Int) {
        guard width != currentWidth || height != currentHeight else { return }
        currentWidth = width
        currentHeight = height

        let frameBytes = width * height * MemoryLayout<UInt16>.size
        let floatBytes = width * height * MemoryLayout<Float>.size

        centerBuf   = device.makeBuffer(length: frameBytes, options: .storageModeShared)
        guideBuf    = device.makeBuffer(length: frameBytes, options: .storageModeShared)
        valSumBuf   = device.makeBuffer(length: floatBytes, options: .storageModeShared)
        wSumBuf     = device.makeBuffer(length: floatBytes, options: .storageModeShared)
        outputBuf   = device.makeBuffer(length: frameBytes, options: .storageModeShared)

        if centerBuf == nil || guideBuf == nil || valSumBuf == nil || wSumBuf == nil || outputBuf == nil {
            fputs("GPU OOM: failed to allocate core buffers (\(frameBytes)B frame + \(floatBytes)B float)\n", stderr)
            return
        }

        // LUT buffers (constant size)
        weightLutLumaBuf   = device.makeBuffer(length: 256 * MemoryLayout<Float>.size, options: .storageModeShared)
        threshLutBuf       = device.makeBuffer(length: 256 * MemoryLayout<Float>.size, options: .storageModeShared)
        distLutBuf         = device.makeBuffer(length: 128 * MemoryLayout<Float>.size, options: .storageModeShared)
        weightLutChromaBuf = device.makeBuffer(length: 256 * MemoryLayout<Float>.size, options: .storageModeShared)

        // VST+Bilateral extra buffers
        zPreestimateBuf = device.makeBuffer(length: floatBytes, options: .storageModeShared)
        maxFlowBuf      = device.makeBuffer(length: floatBytes, options: .storageModeShared)

        // Shared output buffers for zero-copy TF→CNN (2 for ping-pong)
        sharedOutputBufs.removeAll()
        for _ in 0..<2 {
            if let buf = device.makeBuffer(length: frameBytes, options: .storageModeShared) {
                sharedOutputBufs.append(buf)
            }
        }

        // Pre-allocate flow buffer pools (reused every frame)
        let greenW = width / 2
        let greenH = height / 2
        let flowBytes = greenW * greenH * MemoryLayout<Float>.size

        flowXPool.removeAll()
        flowYPool.removeAll()
        for _ in 0..<Self.maxNeighbors {
            guard let fxb = device.makeBuffer(length: flowBytes, options: .storageModeShared),
                  let fyb = device.makeBuffer(length: flowBytes, options: .storageModeShared)
            else { break }
            flowXPool.append(fxb)
            flowYPool.append(fyb)
        }
    }

    /// Allocate GPU-resident frame rings. Called once per clip from C.
    /// Returns CPU-accessible pointers via frameRingPointer/denoisedRingPointer.
    func ensureRing(numSlots: Int, width: Int, height: Int) {
        ensureBuffers(width: width, height: height)
        let frameBytes = width * height * MemoryLayout<UInt16>.size
        // Reallocate if slot count changed OR frame size changed (different resolution clip)
        let sizeMatch = frameRing.first.map { $0.length == frameBytes } ?? false
        guard numSlots != ringSlotCount || !sizeMatch else { return }

        // Release old ring buffers before allocating new ones
        frameRing.removeAll()
        denoisedRing.removeAll()
        ringSlotCount = numSlots

        frameRing = (0..<numSlots).compactMap { _ in
            device.makeBuffer(length: frameBytes, options: .storageModeShared)
        }
        denoisedRing = (0..<numSlots).compactMap { _ in
            device.makeBuffer(length: frameBytes, options: .storageModeShared)
        }

        if frameRing.count < numSlots || denoisedRing.count < numSlots {
            fputs("GPU OOM: ring buffer alloc failed (need \(numSlots) × \(frameBytes) bytes)\n", stderr)
        }
    }

    func frameRingPointer(slot: Int) -> UnsafeMutablePointer<UInt16>? {
        guard slot < frameRing.count else { return nil }
        return frameRing[slot].contents().bindMemory(to: UInt16.self,
                                                      capacity: currentWidth * currentHeight)
    }

    func denoisedRingPointer(slot: Int) -> UnsafeMutablePointer<UInt16>? {
        guard slot < denoisedRing.count else { return nil }
        return denoisedRing[slot].contents().bindMemory(to: UInt16.self,
                                                         capacity: currentWidth * currentHeight)
    }

    /// GPU temporal filter using ring buffers — zero frame copies.
    /// Frames are already in GPU-visible MTLBuffers (frameRing / denoisedRing).
    /// Only flow fields (from Vision OF on CPU) need memcpy.
    func filterFrameRing(
        output: UnsafeMutablePointer<UInt16>,
        ringSlots: UnsafePointer<Int32>,
        useDenoised: UnsafePointer<Int32>,
        flowsX: UnsafePointer<UnsafePointer<Float>?>,
        flowsY: UnsafePointer<UnsafePointer<Float>?>,
        numFrames: Int, centerIdx: Int,
        width: Int, height: Int,
        strength: Float, noiseSigma: Float,
        chromaBoost: Float = 1.0,
        distSigma: Float = 1.5,
        flowTightening: Float = 3.0,
        guide: UnsafePointer<UInt16>? = nil,
        centerWeight: Float = 0.3
    ) {
        ensureBuffers(width: width, height: height)

        guard let valSumBuf, let wSumBuf, let outputBuf, let guideBuf,
              let weightLutLumaBuf, let threshLutBuf, let distLutBuf,
              let weightLutChromaBuf else { return }

        let framePixels = width * height
        let greenW = width / 2
        let greenPixels = greenW * (height / 2)

        // Resolve center ring buffer (already on GPU — no copy!)
        let centerSlot = Int(ringSlots[centerIdx])
        let centerRingBuf = useDenoised[centerIdx] != 0
            ? denoisedRing[centerSlot] : frameRing[centerSlot]
        let centerPtr = centerRingBuf.contents().bindMemory(to: UInt16.self,
                                                             capacity: framePixels)

        // --- LUTs (same as filterFrame) ---
        let hLuma = noiseSigma * strength
        let hChroma = hLuma * chromaBoost

        func buildWeightLut(_ buf: MTLBuffer, h: Float) {
            let h2 = h * h
            let maxDiff = 3.0 * h
            let step = maxDiff / 256.0
            let lut = buf.contents().bindMemory(to: Float.self, capacity: 256)
            for i in 0..<256 {
                let diff = Float(i) * step
                lut[i] = expf(-(diff * diff) / (2.0 * h2))
            }
        }
        buildWeightLut(weightLutLumaBuf, h: hLuma)
        buildWeightLut(weightLutChromaBuf, h: hChroma)
        let lutScaleLuma   = Float(256.0 / (3.0 * hLuma))
        let lutScaleChroma = Float(256.0 / (3.0 * hChroma))

        let blackLevel: Float = 6032.0
        let readNoiseFrac: Float = 0.55
        let readNoiseVar = readNoiseFrac * noiseSigma * noiseSigma
        let shotGain: Float = 180.0
        let baseReject = 3.0 * noiseSigma
        let tLut = threshLutBuf.contents().bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 {
            let v = (Float(i) + 0.5) * (65535.0 / 256.0)
            let signal = max(0.0, v - blackLevel)
            let calibrated = 3.0 * sqrtf(readNoiseVar + shotGain * signal)
            let legacy = baseReject * sqrtf(1.0 + v / 32768.0)
            tLut[i] = min(calibrated, legacy)
        }

        let dLut = distLutBuf.contents().bindMemory(to: Float.self, capacity: 128)
        let distMax = 3.0 * distSigma
        for i in 0..<128 {
            let d = Float(i) * distMax / 128.0
            dLut[i] = expf(-(d * d) / (2.0 * distSigma * distSigma))
        }
        let distLutScale = 128.0 / distMax

        // Guide frame: guide pointer or center (from ring)
        let guideDst = guideBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
        let frameByteSize = framePixels * MemoryLayout<UInt16>.size
        if let guide {
            memcpy(guideDst, guide, frameByteSize)
        } else {
            memcpy(guideDst, centerPtr, frameByteSize)
        }

        // Initialize val_sum/w_sum from center (read directly from ring buffer — no copy)
        let vSum = valSumBuf.contents().bindMemory(to: Float.self, capacity: framePixels)
        let wSum = wSumBuf.contents().bindMemory(to: Float.self, capacity: framePixels)
        vDSP_vfltu16(centerPtr, 1, vSum, 1, vDSP_Length(framePixels))
        var cw = centerWeight
        vDSP_vsmul(vSum, 1, &cw, vSum, 1, vDSP_Length(framePixels))
        vDSP_vfill(&cw, wSum, 1, vDSP_Length(framePixels))

        // Resolve neighbor ring buffers + upload ONLY flow fields
        let flowByteSize = greenPixels * MemoryLayout<Float>.size
        var neighborBufs: [MTLBuffer] = []
        var neighborCount = 0

        for f in 0..<numFrames {
            if f == centerIdx { continue }
            guard let fx = flowsX[f], let fy = flowsY[f],
                  neighborCount < flowXPool.count else { continue }

            // Frame: bind ring buffer directly (ZERO COPY)
            let slot = Int(ringSlots[f])
            let buf = useDenoised[f] != 0 ? denoisedRing[slot] : frameRing[slot]
            neighborBufs.append(buf)

            // Flow: still needs memcpy (Vision OF outputs to CPU)
            memcpy(flowXPool[neighborCount].contents(), fx, flowByteSize)
            memcpy(flowYPool[neighborCount].contents(), fy, flowByteSize)
            neighborCount += 1
        }

        // --- GPU dispatch ---
        guard let cmdBuf = queue.makeCommandBuffer() else {
            fputs("GPU: makeCommandBuffer failed (filterFrameRing)\n", stderr)
            return
        }

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        var params = MetalFilterParams(
            width: UInt32(width),
            height: UInt32(height),
            lut_scale_luma: lutScaleLuma,
            lut_scale_chroma: lutScaleChroma,
            dist_lut_scale: distLutScale,
            flow_tightening: flowTightening,
            h_luma: hLuma,
            h_chroma: hChroma
        )

        if let encoder = cmdBuf.makeComputeCommandEncoder() {
            for i in 0..<neighborCount {
                encoder.setComputePipelineState(filterPipeline)
                encoder.setBuffer(centerRingBuf,      offset: 0, index: 0)
                encoder.setBuffer(neighborBufs[i],    offset: 0, index: 1)
                encoder.setBuffer(flowXPool[i],       offset: 0, index: 2)
                encoder.setBuffer(flowYPool[i],       offset: 0, index: 3)
                encoder.setBuffer(valSumBuf,          offset: 0, index: 4)
                encoder.setBuffer(wSumBuf,            offset: 0, index: 5)
                encoder.setBytes(&params, length: MemoryLayout<MetalFilterParams>.size, index: 6)
                encoder.setBuffer(weightLutLumaBuf,   offset: 0, index: 7)
                encoder.setBuffer(threshLutBuf,       offset: 0, index: 8)
                encoder.setBuffer(distLutBuf,         offset: 0, index: 9)
                encoder.setBuffer(weightLutChromaBuf, offset: 0, index: 10)
                encoder.setBuffer(guideBuf,           offset: 0, index: 11)
                encoder.dispatchThreads(
                    MTLSize(width: width, height: height, depth: 1),
                    threadsPerThreadgroup: threadgroupSize
                )
            }
            encoder.endEncoding()
        }

        if let normEncoder = cmdBuf.makeComputeCommandEncoder() {
            var dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
            var cwForNorm = cw
            normEncoder.setComputePipelineState(normPipeline)
            normEncoder.setBuffer(valSumBuf, offset: 0, index: 0)
            normEncoder.setBuffer(wSumBuf,   offset: 0, index: 1)
            normEncoder.setBuffer(outputBuf, offset: 0, index: 2)
            normEncoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            normEncoder.setBuffer(centerRingBuf, offset: 0, index: 4)
            normEncoder.setBytes(&cwForNorm, length: MemoryLayout<Float>.size, index: 5)
            var hLumaForNorm = hLuma
            normEncoder.setBytes(&hLumaForNorm, length: MemoryLayout<Float>.size, index: 6)
            normEncoder.dispatchThreads(
                MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )
            normEncoder.endEncoding()
        }

        if !commitAndWait(cmdBuf, label: "filterFrameRing") {
            // GPU failed — output center frame as fallback
            let src = centerRingBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
            memcpy(output, src, framePixels * MemoryLayout<UInt16>.size)
            return
        }

        let outSrc = outputBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
        memcpy(output, outSrc, framePixels * MemoryLayout<UInt16>.size)
    }

    /// Run the full temporal filter on GPU (legacy path — copies all frames).
    /// guide: optional denoised frame for patch matching (nil = use center frame).
    func filterFrame(
        output: UnsafeMutablePointer<UInt16>,
        frames: UnsafePointer<UnsafePointer<UInt16>?>,
        flowsX: UnsafePointer<UnsafePointer<Float>?>,
        flowsY: UnsafePointer<UnsafePointer<Float>?>,
        numFrames: Int, centerIdx: Int,
        width: Int, height: Int,
        strength: Float, noiseSigma: Float,
        chromaBoost: Float = 1.0,
        distSigma: Float = 1.5,
        flowTightening: Float = 3.0,
        guide: UnsafePointer<UInt16>? = nil,
        centerWeight: Float = 0.3
    ) {
        ensureBuffers(width: width, height: height)

        guard let centerBuf, let guideBuf, let valSumBuf, let wSumBuf, let outputBuf,
              let weightLutLumaBuf, let threshLutBuf, let distLutBuf,
              let weightLutChromaBuf else { return }

        let framePixels = width * height
        let greenW = width / 2
        let greenPixels = greenW * (height / 2)

        // Chroma boost: passed as parameter (1.0 = same kernel as luma)

        // Bilateral weight LUTs: luma (Gr/Gb) and chroma (R/B)
        let hLuma = noiseSigma * strength
        let hChroma = hLuma * chromaBoost

        func buildWeightLut(_ buf: MTLBuffer, h: Float) {
            let h2 = h * h
            let maxDiff = 3.0 * h
            let step = maxDiff / 256.0
            let lut = buf.contents().bindMemory(to: Float.self, capacity: 256)
            for i in 0..<256 {
                let diff = Float(i) * step
                lut[i] = expf(-(diff * diff) / (2.0 * h2))
            }
        }
        buildWeightLut(weightLutLumaBuf, h: hLuma)
        buildWeightLut(weightLutChromaBuf, h: hChroma)
        let lutScaleLuma   = Float(256.0 / (3.0 * hLuma))
        let lutScaleChroma = Float(256.0 / (3.0 * hChroma))

        // Signal-dependent rejection threshold LUT (3× sigma).
        // Calibrated from S5II dark frames + static scene temporal variance.
        // Uses proper black level (6032) and affine noise model:
        //   σ(v) = sqrt(read_noise² + shot_gain * max(0, v - black_level))
        // The old model sqrt(1 + v/32768) had no black level and over-estimated
        // dark pixel thresholds by ~1.5×, causing bright background to leak into
        // dark subject edges (halo artifact).
        let blackLevel: Float = 6032.0
        // Read noise from dark frame temporal variance at matching ISO.
        // Scale with auto-estimated sigma so it adapts to different ISOs.
        let readNoiseFrac: Float = 0.55  // read noise fraction; balances halo rejection vs dark denoising
        let readNoiseVar = readNoiseFrac * noiseSigma * noiseSigma
        // Shot noise gain calibrated from scene data (spatially flat regions):
        // var rises ~180 per count above black level in the signal range 6032-15000.
        // Above ~17000 the S5II's dual-gain sensor reduces noise, so we cap
        // with the old model to avoid runaway thresholds at bright pixels.
        let shotGain: Float = 180.0
        let baseReject = 3.0 * noiseSigma
        let tLut = threshLutBuf.contents().bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 {
            let v = (Float(i) + 0.5) * (65535.0 / 256.0)
            // Calibrated model: correct for dark pixels (proper black level)
            let signal = max(0.0, v - blackLevel)
            let calibrated = 3.0 * sqrtf(readNoiseVar + shotGain * signal)
            // Old model: caps bright pixels where dual-gain reduces noise
            let legacy = baseReject * sqrtf(1.0 + v / 32768.0)
            // Use the tighter of the two at each signal level
            tLut[i] = min(calibrated, legacy)
        }

        // Distance LUT (distSigma passed as parameter)
        let dLut = distLutBuf.contents().bindMemory(to: Float.self, capacity: 128)
        let distMax = 3.0 * distSigma
        for i in 0..<128 {
            let d = Float(i) * distMax / 128.0
            dLut[i] = expf(-(d * d) / (2.0 * distSigma * distSigma))
        }
        let distLutScale = 128.0 / distMax

        // Initialize val_sum and w_sum with center frame (weight = 1)
        // Using vDSP for vectorized uint16→float conversion instead of scalar loop
        guard let centerFrame = frames[centerIdx] else { return }

        let centerDst = centerBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
        memcpy(centerDst, centerFrame, framePixels * MemoryLayout<UInt16>.size)

        // Guide frame: use provided guide (e.g., pass-1 denoised) or fall back to center
        let guideDst = guideBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
        if let guide {
            memcpy(guideDst, guide, framePixels * MemoryLayout<UInt16>.size)
        } else {
            memcpy(guideDst, centerFrame, framePixels * MemoryLayout<UInt16>.size)
        }

        let vSum = valSumBuf.contents().bindMemory(to: Float.self, capacity: framePixels)
        let wSum = wSumBuf.contents().bindMemory(to: Float.self, capacity: framePixels)

        // Center frame weight: adaptive based on per-frame motion.
        // Low motion (0.3) → cleaner recursive neighbors dominate.
        // High motion (1.0) → trust current frame when neighbors unreliable.
        // Vectorized: convert center frame uint16 → float, then scale by center weight
        vDSP_vfltu16(centerFrame, 1, vSum, 1, vDSP_Length(framePixels))
        var cw = centerWeight
        vDSP_vsmul(vSum, 1, &cw, vSum, 1, vDSP_Length(framePixels))
        // Fill w_sum with center weight
        vDSP_vfill(&cw, wSum, 1, vDSP_Length(framePixels))

        // Legacy path: copy all frame + flow data into temporary Metal buffers.
        // (The pipelined ring path avoids this via filterFrameRing.)
        let frameByteSize = framePixels * MemoryLayout<UInt16>.size
        let flowByteSize = greenPixels * MemoryLayout<Float>.size

        var neighborBufs: [MTLBuffer] = []
        var neighborCount = 0
        for f in 0..<numFrames {
            if f == centerIdx { continue }
            guard let neighborFrame = frames[f],
                  let fx = flowsX[f],
                  let fy = flowsY[f],
                  neighborCount < flowXPool.count else { continue }

            guard let nb = device.makeBuffer(bytes: neighborFrame, length: frameByteSize,
                                              options: .storageModeShared) else { continue }
            neighborBufs.append(nb)
            memcpy(flowXPool[neighborCount].contents(), fx, flowByteSize)
            memcpy(flowYPool[neighborCount].contents(), fy, flowByteSize)
            neighborCount += 1
        }

        // Single command buffer for ALL neighbor dispatches + normalization
        guard let cmdBuf = queue.makeCommandBuffer() else {
            fputs("GPU: makeCommandBuffer failed (filterFrame)\n", stderr)
            return
        }

        // Dispatch all neighbor filter kernels in one command encoder
        // (Metal guarantees sequential execution within a single encoder)
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        var params = MetalFilterParams(
            width: UInt32(width),
            height: UInt32(height),
            lut_scale_luma: lutScaleLuma,
            lut_scale_chroma: lutScaleChroma,
            dist_lut_scale: distLutScale,
            flow_tightening: flowTightening,
            h_luma: hLuma,
            h_chroma: hChroma
        )

        if let encoder = cmdBuf.makeComputeCommandEncoder() {
            for i in 0..<neighborCount {
                encoder.setComputePipelineState(filterPipeline)
                encoder.setBuffer(centerBuf,         offset: 0, index: 0)
                encoder.setBuffer(neighborBufs[i],   offset: 0, index: 1)
                encoder.setBuffer(flowXPool[i],      offset: 0, index: 2)
                encoder.setBuffer(flowYPool[i],      offset: 0, index: 3)
                encoder.setBuffer(valSumBuf,            offset: 0, index: 4)
                encoder.setBuffer(wSumBuf,              offset: 0, index: 5)
                encoder.setBytes(&params, length: MemoryLayout<MetalFilterParams>.size, index: 6)
                encoder.setBuffer(weightLutLumaBuf,     offset: 0, index: 7)
                encoder.setBuffer(threshLutBuf,         offset: 0, index: 8)
                encoder.setBuffer(distLutBuf,           offset: 0, index: 9)
                encoder.setBuffer(weightLutChromaBuf,   offset: 0, index: 10)
                encoder.setBuffer(guideBuf,             offset: 0, index: 11)
                encoder.dispatchThreads(
                    MTLSize(width: width, height: height, depth: 1),
                    threadsPerThreadgroup: threadgroupSize
                )
            }
            encoder.endEncoding()
        }

        // Normalization pass in same command buffer
        if let normEncoder = cmdBuf.makeComputeCommandEncoder() {
            var dims = SIMD2<UInt32>(UInt32(width), UInt32(height))
            var cwForNorm = cw  // pass center weight for dark-pixel ghost suppression
            normEncoder.setComputePipelineState(normPipeline)
            normEncoder.setBuffer(valSumBuf, offset: 0, index: 0)
            normEncoder.setBuffer(wSumBuf,   offset: 0, index: 1)
            normEncoder.setBuffer(outputBuf, offset: 0, index: 2)
            normEncoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            normEncoder.setBuffer(centerBuf, offset: 0, index: 4)
            normEncoder.setBytes(&cwForNorm, length: MemoryLayout<Float>.size, index: 5)
            var hLumaForNorm = hLuma  // noise bandwidth for anti-glow clamp
            normEncoder.setBytes(&hLumaForNorm, length: MemoryLayout<Float>.size, index: 6)
            normEncoder.dispatchThreads(
                MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )
            normEncoder.endEncoding()
        }

        // Single wait for ALL dispatches (filter + normalize) with error handling
        if !commitAndWait(cmdBuf, label: "filterFrame") {
            // GPU failed — output center frame as fallback
            memcpy(output, centerFrame, framePixels * MemoryLayout<UInt16>.size)
            return
        }

        // Copy output back
        let outSrc = outputBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
        memcpy(output, outSrc, framePixels * MemoryLayout<UInt16>.size)
    }

    // MARK: - VST+Bilateral GPU (4-pass pipeline with research improvements)

    /// GPU temporal filter using VST+Bilateral with:
    ///   1. Structural term (gradient comparison)
    ///   2. Self-guided reference (robust pre-estimate)
    ///   3. Multi-hypothesis sampling (M=4)
    /// Returns CPU pointer to shared output buffer (for zero-copy pipeline).
    func sharedOutputPointer(index: Int) -> UnsafeMutablePointer<UInt16>? {
        guard index < sharedOutputBufs.count else { return nil }
        return sharedOutputBufs[index].contents().bindMemory(
            to: UInt16.self, capacity: currentWidth * currentHeight)
    }

    /// Returns the MTLBuffer for a shared output slot (for CNN to read directly).
    func sharedOutputBuffer(index: Int) -> MTLBuffer? {
        guard index < sharedOutputBufs.count else { return nil }
        return sharedOutputBufs[index]
    }

    func filterFrameRingVSTBilateral(
        output: UnsafeMutablePointer<UInt16>,
        ringSlots: UnsafePointer<Int32>,
        useDenoised: UnsafePointer<Int32>,
        flowsX: UnsafePointer<UnsafePointer<Float>?>,
        flowsY: UnsafePointer<UnsafePointer<Float>?>,
        numFrames: Int, centerIdx: Int,
        width: Int, height: Int,
        noiseSigma: Float,
        blackLevel: Float = 6032.0,
        shotGain: Float = 180.0,
        readNoise: Float = 616.0,
        sharedOutputIndex: Int = -1,  // ≥0: write to sharedOutputBufs[idx], skip memcpy
        commitOnly: Bool = false,      // true: commit GPU work and return; call waitGPU() later
        maxNeighbors: Int = 14        // motion-adaptive GPU window (default: all neighbors)
    ) {
        ensureBuffers(width: width, height: height)

        guard let vstCollectPipeline, let vstPreestimatePipeline,
              let vstFusePipeline, let vstFinalizePipeline,
              let valSumBuf, let wSumBuf, let outputBuf,
              let zPreestimateBuf, let maxFlowBuf else { return }

        let framePixels = width * height
        let greenW = width / 2
        let greenPixels = greenW * (height / 2)
        let floatBytes = framePixels * MemoryLayout<Float>.size

        // Zero accumulators for Phase 1
        memset(valSumBuf.contents(), 0, floatBytes)   // z_sum
        memset(wSumBuf.contents(), 0, floatBytes)      // z_count
        memset(maxFlowBuf.contents(), 0, floatBytes)

        // Resolve center ring buffer
        let centerSlot = Int(ringSlots[centerIdx])
        let centerRingBuf = useDenoised[centerIdx] != 0
            ? denoisedRing[centerSlot] : frameRing[centerSlot]

        // Resolve neighbor ring buffers + upload flow fields.
        // Iterate by increasing distance from center so maxNeighbors always
        // captures the closest (most useful) neighbors first.
        let flowByteSize = greenPixels * MemoryLayout<Float>.size
        var neighborBufs: [MTLBuffer] = []
        var neighborCount = 0
        let limit = min(maxNeighbors, flowXPool.count)

        for dist in 1...(numFrames / 2 + 1) {
            if neighborCount >= limit { break }
            for sign in [-1, 1] {
                if neighborCount >= limit { break }
                let f = centerIdx + dist * sign
                guard f >= 0 && f < numFrames else { continue }
                guard let fx = flowsX[f], let fy = flowsY[f] else { continue }

                let slot = Int(ringSlots[f])
                let buf = useDenoised[f] != 0 ? denoisedRing[slot] : frameRing[slot]
                neighborBufs.append(buf)

                memcpy(flowXPool[neighborCount].contents(), fx, flowByteSize)
                memcpy(flowYPool[neighborCount].contents(), fy, flowByteSize)
                neighborCount += 1
            }
        }

        var params = VSTBilateralParams(
            width: UInt32(width),
            height: UInt32(height),
            noise_sigma: noiseSigma,
            h: 1.0,
            z_reject: 3.0,
            flow_sigma2: 8.0,       // 2 * 2.0^2
            sigma_g2: 0.5,          // 2 * 0.5^2
            black_level: blackLevel,
            shot_gain: shotGain,
            read_noise: readNoise
        )

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(width: width, height: height, depth: 1)

        guard let cmdBuf = queue.makeCommandBuffer() else {
            fputs("GPU: makeCommandBuffer failed (filterFrameRingVSTBilateral)\n", stderr)
            return
        }

        // Select fp16 or fp32 pipeline
        let collectPipe = (useFP16 && vstCollectFP16Pipeline != nil) ? vstCollectFP16Pipeline! : vstCollectPipeline
        let preestPipe  = (useFP16 && vstPreestimateFP16Pipeline != nil) ? vstPreestimateFP16Pipeline! : vstPreestimatePipeline
        let fusePipe    = (useFP16 && vstFuseFP16Pipeline != nil) ? vstFuseFP16Pipeline! : vstFusePipeline
        let finPipe     = (useFP16 && vstFinalizeFP16Pipeline != nil) ? vstFinalizeFP16Pipeline! : vstFinalizePipeline

        // Phase 1a: Collect z-values (per-neighbor dispatches)
        if let enc = cmdBuf.makeComputeCommandEncoder() {
            for i in 0..<neighborCount {
                enc.setComputePipelineState(collectPipe)
                enc.setBuffer(centerRingBuf,    offset: 0, index: 0)
                enc.setBuffer(neighborBufs[i],  offset: 0, index: 1)
                enc.setBuffer(flowXPool[i],     offset: 0, index: 2)
                enc.setBuffer(flowYPool[i],     offset: 0, index: 3)
                enc.setBuffer(valSumBuf,        offset: 0, index: 4)  // z_sum
                enc.setBuffer(wSumBuf,          offset: 0, index: 5)  // z_count
                enc.setBuffer(maxFlowBuf,       offset: 0, index: 6)
                enc.setBytes(&params, length: MemoryLayout<VSTBilateralParams>.size, index: 7)
                enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            }
            enc.endEncoding()
        }

        // Phase 1b: Preestimate (also zeros z_sum/z_count for Phase 2)
        if let enc = cmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(preestPipe)
            enc.setBuffer(centerRingBuf,    offset: 0, index: 0)
            enc.setBuffer(valSumBuf,        offset: 0, index: 1)  // z_sum in, zeroed on out
            enc.setBuffer(wSumBuf,          offset: 0, index: 2)  // z_count in, zeroed on out
            enc.setBuffer(zPreestimateBuf,  offset: 0, index: 3)
            enc.setBytes(&params, length: MemoryLayout<VSTBilateralParams>.size, index: 4)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        // Phase 2: Fuse with structural + self-guided + multi-hypothesis
        if let enc = cmdBuf.makeComputeCommandEncoder() {
            for i in 0..<neighborCount {
                enc.setComputePipelineState(fusePipe)
                enc.setBuffer(centerRingBuf,    offset: 0, index: 0)
                enc.setBuffer(neighborBufs[i],  offset: 0, index: 1)
                enc.setBuffer(flowXPool[i],     offset: 0, index: 2)
                enc.setBuffer(flowYPool[i],     offset: 0, index: 3)
                enc.setBuffer(valSumBuf,        offset: 0, index: 4)  // val_sum
                enc.setBuffer(wSumBuf,          offset: 0, index: 5)  // w_sum
                enc.setBuffer(zPreestimateBuf,  offset: 0, index: 6)
                enc.setBytes(&params, length: MemoryLayout<VSTBilateralParams>.size, index: 7)
                enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            }
            enc.endEncoding()
        }

        // Phase 3: Center weight floor + inverse Anscombe
        // Write to shared output buffer if index specified, otherwise outputBuf
        let finalOutBuf: MTLBuffer
        if sharedOutputIndex >= 0 && sharedOutputIndex < sharedOutputBufs.count {
            finalOutBuf = sharedOutputBufs[sharedOutputIndex]
        } else {
            finalOutBuf = outputBuf
        }

        if let enc = cmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(finPipe)
            enc.setBuffer(centerRingBuf,    offset: 0, index: 0)
            enc.setBuffer(valSumBuf,        offset: 0, index: 1)
            enc.setBuffer(wSumBuf,          offset: 0, index: 2)
            enc.setBuffer(maxFlowBuf,       offset: 0, index: 3)
            enc.setBuffer(finalOutBuf,      offset: 0, index: 4)
            enc.setBytes(&params, length: MemoryLayout<VSTBilateralParams>.size, index: 5)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            enc.endEncoding()
        }

        if commitOnly && sharedOutputIndex < 0 {
            // Async path: commit GPU work, return immediately.
            // Caller must call waitGPU() before reading output.
            let sem = DispatchSemaphore(value: 0)
            asyncSemaphore = sem
            asyncSucceeded = true
            asyncOutput = output
            asyncFramePixels = framePixels
            cmdBuf.addCompletedHandler { [weak self] cb in
                if cb.status == .error {
                    self?.asyncSucceeded = false
                    fputs("GPU ERROR [async filterFrameRingVSTBilateral]: \(cb.error?.localizedDescription ?? "unknown")\n", stderr)
                }
                sem.signal()
            }
            cmdBuf.commit()
            return
        }

        if !commitAndWait(cmdBuf, label: "filterFrameRingVSTBilateral") {
            // GPU failed — output center frame as fallback
            let src = centerRingBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
            memcpy(output, src, framePixels * MemoryLayout<UInt16>.size)
            return
        }

        // Skip memcpy when using shared output — data is already in CPU-visible shared buffer
        if sharedOutputIndex < 0 {
            let outSrc = outputBuf.contents().bindMemory(to: UInt16.self, capacity: framePixels)
            memcpy(output, outSrc, framePixels * MemoryLayout<UInt16>.size)
        }
    }
}
