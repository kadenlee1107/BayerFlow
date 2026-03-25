import Foundation
import Metal

/// GPU-accelerated hierarchical block matching optical flow.
/// Replaces Apple Vision ANE optical flow for ~100x speedup.
///
/// 5-level Gaussian pyramid + coarse-to-fine 8x8 block matching.
/// Output: dense per-pixel flow at green-channel resolution.
nonisolated final class MetalBlockMatchOF: @unchecked Sendable {
    static let shared: MetalBlockMatchOF? = MetalBlockMatchOF()

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let downsamplePipeline: MTLComputePipelineState
    private let blockMatchPipeline: MTLComputePipelineState
    private let interpolatePipeline: MTLComputePipelineState
    private let warpFlowPipeline: MTLComputePipelineState

    private static let pyramidLevels = 3  // 3 levels sufficient: 1440→720→360
    private static let blockSize: UInt32 = 8
    /// Search range at coarsest level (exhaustive), finer levels use 3
    private static let coarseSearchRange: UInt32 = 12
    private static let fineSearchRange: UInt32 = 3

    // Reusable pyramid buffers
    private var pyramidBufs: [[MTLBuffer]] = [[], []]  // [center, neighbor] × levels
    private var pyramidDims: [(Int, Int)] = []          // (width, height) per level
    private var mvBufs: [MTLBuffer] = []                // MV buffer per level
    private var currentWidth: Int = 0
    private var currentHeight: Int = 0

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let dsFunc = library.makeFunction(name: "pyramid_downsample"),
              let bmFunc = library.makeFunction(name: "block_match"),
              let intFunc = library.makeFunction(name: "interpolate_flow"),
              let wfFunc = library.makeFunction(name: "warp_flow") else {
            return nil
        }
        self.device = device
        self.queue = queue
        do {
            self.downsamplePipeline = try device.makeComputePipelineState(function: dsFunc)
            self.blockMatchPipeline = try device.makeComputePipelineState(function: bmFunc)
            self.interpolatePipeline = try device.makeComputePipelineState(function: intFunc)
            self.warpFlowPipeline = try device.makeComputePipelineState(function: wfFunc)
        } catch {
            return nil
        }
    }

    /// Ensure pyramid buffers are allocated for the given resolution.
    private func ensureBuffers(width: Int, height: Int) {
        if width == currentWidth && height == currentHeight { return }
        currentWidth = width
        currentHeight = height

        pyramidDims = []
        pyramidBufs = [[], []]
        mvBufs = []

        var w = width, h = height
        for _ in 0..<Self.pyramidLevels {
            pyramidDims.append((w, h))
            let bytes = w * h * MemoryLayout<UInt16>.size
            pyramidBufs[0].append(device.makeBuffer(length: bytes, options: .storageModeShared)!)
            pyramidBufs[1].append(device.makeBuffer(length: bytes, options: .storageModeShared)!)

            let bw = (w + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            let bh = (h + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            let mvBytes = bw * bh * MemoryLayout<SIMD2<Int32>>.size
            mvBufs.append(device.makeBuffer(length: mvBytes, options: .storageModeShared)!)

            w /= 2
            h /= 2
        }
    }

    // Pre-allocated flow output buffers (avoids per-call allocation)
    private var flowXBuf: MTLBuffer?
    private var flowYBuf: MTLBuffer?
    private var centerPyramidBuilt = false

    /// Build center pyramid once, then call computeFlowForNeighbor for each neighbor.
    func buildCenterPyramid(center: UnsafePointer<UInt16>, width: Int, height: Int) {
        ensureBuffers(width: width, height: height)
        let pixelBytes = width * height * MemoryLayout<UInt16>.size
        memcpy(pyramidBufs[0][0].contents(), center, pixelBytes)

        // Ensure flow output buffers exist
        let flowBytes = width * height * MemoryLayout<Float>.size
        if flowXBuf == nil || flowXBuf!.length < flowBytes {
            flowXBuf = device.makeBuffer(length: flowBytes, options: .storageModeShared)
            flowYBuf = device.makeBuffer(length: flowBytes, options: .storageModeShared)
        }

        guard let cmdBuf = queue.makeCommandBuffer() else { return }
        let tg = MTLSize(width: 16, height: 16, depth: 1)

        // Build center pyramid only
        for lvl in 0..<(Self.pyramidLevels - 1) {
            let (sw, sh) = pyramidDims[lvl]
            guard let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
            enc.setComputePipelineState(downsamplePipeline)
            enc.setBuffer(pyramidBufs[0][lvl], offset: 0, index: 0)
            enc.setBuffer(pyramidBufs[0][lvl + 1], offset: 0, index: 1)
            var srcDims = SIMD2<UInt32>(UInt32(sw), UInt32(sh))
            enc.setBytes(&srcDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.dispatchThreads(MTLSize(width: sw/2, height: sh/2, depth: 1),
                                threadsPerThreadgroup: tg)
            enc.endEncoding()
        }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        centerPyramidBuilt = true
    }

    /// Compute flow for one neighbor (center pyramid must already be built).
    func computeFlowForNeighbor(
        neighbor: UnsafePointer<UInt16>,
        width: Int, height: Int,
        flowX: UnsafeMutablePointer<Float>,
        flowY: UnsafeMutablePointer<Float>
    ) {
        guard centerPyramidBuilt, let flowXBuf, let flowYBuf else { return }

        let pixelBytes = width * height * MemoryLayout<UInt16>.size
        memcpy(pyramidBufs[1][0].contents(), neighbor, pixelBytes)

        guard let cmdBuf = queue.makeCommandBuffer() else { return }
        let tg = MTLSize(width: 16, height: 16, depth: 1)

        // Build neighbor pyramid
        for lvl in 0..<(Self.pyramidLevels - 1) {
            let (sw, sh) = pyramidDims[lvl]
            guard let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
            enc.setComputePipelineState(downsamplePipeline)
            enc.setBuffer(pyramidBufs[1][lvl], offset: 0, index: 0)
            enc.setBuffer(pyramidBufs[1][lvl + 1], offset: 0, index: 1)
            var srcDims = SIMD2<UInt32>(UInt32(sw), UInt32(sh))
            enc.setBytes(&srcDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.dispatchThreads(MTLSize(width: sw/2, height: sh/2, depth: 1),
                                threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        // Coarse-to-fine block matching
        for lvl in stride(from: Self.pyramidLevels - 1, through: 0, by: -1) {
            let (lw, lh) = pyramidDims[lvl]
            let bw = (lw + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            let bh = (lh + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            let isCoarsest = (lvl == Self.pyramidLevels - 1)

            var params = BlockMatchParams(
                width: UInt32(lw), height: UInt32(lh),
                block_size: Self.blockSize,
                search_range: isCoarsest ? Self.coarseSearchRange : Self.fineSearchRange,
                blocks_w: UInt32(bw), blocks_h: UInt32(bh),
                prev_blocks_w: 0, prev_blocks_h: 0
            )
            if !isCoarsest {
                params.prev_blocks_w = UInt32((pyramidDims[lvl+1].0 + Int(Self.blockSize) - 1) / Int(Self.blockSize))
                params.prev_blocks_h = UInt32((pyramidDims[lvl+1].1 + Int(Self.blockSize) - 1) / Int(Self.blockSize))
            }

            guard let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
            enc.setComputePipelineState(blockMatchPipeline)
            enc.setBuffer(pyramidBufs[0][lvl], offset: 0, index: 0)
            enc.setBuffer(pyramidBufs[1][lvl], offset: 0, index: 1)
            enc.setBuffer(isCoarsest ? nil : mvBufs[lvl + 1], offset: 0, index: 2)
            enc.setBuffer(mvBufs[lvl], offset: 0, index: 3)
            enc.setBytes(&params, length: MemoryLayout<BlockMatchParams>.size, index: 4)
            enc.dispatchThreads(MTLSize(width: bw, height: bh, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.endEncoding()
        }

        // Interpolate to dense flow
        if let enc = cmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(interpolatePipeline)
            enc.setBuffer(mvBufs[0], offset: 0, index: 0)
            enc.setBuffer(flowXBuf, offset: 0, index: 1)
            enc.setBuffer(flowYBuf, offset: 0, index: 2)
            let bw = (width + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            var dims = SIMD4<UInt32>(UInt32(width), UInt32(height), UInt32(bw), Self.blockSize)
            enc.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
            enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let flowBytes = width * height * MemoryLayout<Float>.size
        memcpy(flowX, flowXBuf.contents(), flowBytes)
        memcpy(flowY, flowYBuf.contents(), flowBytes)
    }

    /// Compute dense optical flow from center to neighbor.
    /// Input: green-channel uint16 frames at `width × height`.
    /// Output: per-pixel flow_x, flow_y (float arrays, caller-allocated).
    func computeFlow(
        center: UnsafePointer<UInt16>,
        neighbor: UnsafePointer<UInt16>,
        width: Int, height: Int,
        flowX: UnsafeMutablePointer<Float>,
        flowY: UnsafeMutablePointer<Float>
    ) {
        ensureBuffers(width: width, height: height)

        // Copy input frames to level-0 pyramid buffers
        let pixelBytes = width * height * MemoryLayout<UInt16>.size
        memcpy(pyramidBufs[0][0].contents(), center, pixelBytes)
        memcpy(pyramidBufs[1][0].contents(), neighbor, pixelBytes)

        guard let cmdBuf = queue.makeCommandBuffer() else { return }

        let tg = MTLSize(width: 16, height: 16, depth: 1)

        // Build pyramids for both frames
        for frame in 0..<2 {
            for lvl in 0..<(Self.pyramidLevels - 1) {
                let (sw, sh) = pyramidDims[lvl]
                guard let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
                enc.setComputePipelineState(downsamplePipeline)
                enc.setBuffer(pyramidBufs[frame][lvl], offset: 0, index: 0)
                enc.setBuffer(pyramidBufs[frame][lvl + 1], offset: 0, index: 1)
                var srcDims = SIMD2<UInt32>(UInt32(sw), UInt32(sh))
                enc.setBytes(&srcDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
                let dw = sw / 2, dh = sh / 2
                enc.dispatchThreads(MTLSize(width: dw, height: dh, depth: 1),
                                    threadsPerThreadgroup: tg)
                enc.endEncoding()
            }
        }

        // Coarse-to-fine block matching
        for lvl in stride(from: Self.pyramidLevels - 1, through: 0, by: -1) {
            let (lw, lh) = pyramidDims[lvl]
            let bw = (lw + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            let bh = (lh + Int(Self.blockSize) - 1) / Int(Self.blockSize)

            let isCoarsest = (lvl == Self.pyramidLevels - 1)
            let searchRange = isCoarsest ? Self.coarseSearchRange : Self.fineSearchRange

            var params = BlockMatchParams(
                width: UInt32(lw),
                height: UInt32(lh),
                block_size: Self.blockSize,
                search_range: searchRange,
                blocks_w: UInt32(bw),
                blocks_h: UInt32(bh),
                prev_blocks_w: 0,
                prev_blocks_h: 0
            )

            if !isCoarsest {
                let (_, _) = pyramidDims[lvl + 1]
                let pbw = (pyramidDims[lvl + 1].0 + Int(Self.blockSize) - 1) / Int(Self.blockSize)
                let pbh = (pyramidDims[lvl + 1].1 + Int(Self.blockSize) - 1) / Int(Self.blockSize)
                params.prev_blocks_w = UInt32(pbw)
                params.prev_blocks_h = UInt32(pbh)
            }

            guard let enc = cmdBuf.makeComputeCommandEncoder() else { continue }
            enc.setComputePipelineState(blockMatchPipeline)
            enc.setBuffer(pyramidBufs[0][lvl], offset: 0, index: 0)  // center
            enc.setBuffer(pyramidBufs[1][lvl], offset: 0, index: 1)  // neighbor
            if isCoarsest {
                enc.setBuffer(nil, offset: 0, index: 2)  // no previous MVs
            } else {
                enc.setBuffer(mvBufs[lvl + 1], offset: 0, index: 2)  // coarser MVs
            }
            enc.setBuffer(mvBufs[lvl], offset: 0, index: 3)  // output MVs
            enc.setBytes(&params, length: MemoryLayout<BlockMatchParams>.size, index: 4)
            enc.dispatchThreads(MTLSize(width: bw, height: bh, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.endEncoding()
        }

        // Interpolate block MVs to dense per-pixel flow
        let flowBytes = width * height * MemoryLayout<Float>.size
        guard let flowXBuf = device.makeBuffer(length: flowBytes, options: .storageModeShared),
              let flowYBuf = device.makeBuffer(length: flowBytes, options: .storageModeShared) else { return }

        if let enc = cmdBuf.makeComputeCommandEncoder() {
            enc.setComputePipelineState(interpolatePipeline)
            enc.setBuffer(mvBufs[0], offset: 0, index: 0)  // finest level MVs
            enc.setBuffer(flowXBuf, offset: 0, index: 1)
            enc.setBuffer(flowYBuf, offset: 0, index: 2)
            let bw = (width + Int(Self.blockSize) - 1) / Int(Self.blockSize)
            var dims = SIMD4<UInt32>(UInt32(width), UInt32(height), UInt32(bw), Self.blockSize)
            enc.setBytes(&dims, length: MemoryLayout<SIMD4<UInt32>>.size, index: 3)
            enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Copy results
        memcpy(flowX, flowXBuf.contents(), flowBytes)
        memcpy(flowY, flowYBuf.contents(), flowBytes)
    }
}

// C-compatible struct matching Metal's BlockMatchParams
private struct BlockMatchParams {
    var width: UInt32
    var height: UInt32
    var block_size: UInt32
    var search_range: UInt32
    var blocks_w: UInt32
    var blocks_h: UInt32
    var prev_blocks_w: UInt32
    var prev_blocks_h: UInt32
}
