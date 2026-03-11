import Foundation
import Metal
import MetalPerformanceShadersGraph
import Accelerate

/// GPU-accelerated CNN post-filter using MPSGraph.
/// Supports two architectures:
///   - UNet-Lite: 5ch input (4 Bayer + noise map), 4ch output, multi-scale with skip connections
///   - DnCNN (legacy): 1ch input batched as [4,1,H,W], single-scale
/// Pipeline: Bayer extract (full sub-channel res) → CNN → noise blend/interleave → output.
nonisolated final class MPSPostFilter: @unchecked Sendable {
    static let shared: MPSPostFilter? = MPSPostFilter()

    private let device: MTLDevice
    private let queue: MTLCommandQueue

    // Metal compute pipelines for Bayer pre/post processing
    private let extractPSO: MTLComputePipelineState
    private let blendPSO: MTLComputePipelineState
    private let maskedBlendPSO: MTLComputePipelineState?

    // MPSGraph for CNN inference
    private let graph: MPSGraph
    private let inputPlaceholder: MPSGraphTensor
    private let noiseOutput: MPSGraphTensor

    // Architecture type
    private enum GraphType { case restormer, unet, dncnn }
    private let graphType: GraphType
    private let inputChannels: Int  // 5 for UNet, 1 for DnCNN

    // Persistent buffers (resized per-clip)
    private var bayerInputBuf: MTLBuffer?
    private var cnnInputBuf: MTLBuffer?
    private var cnnOutputBuf: MTLBuffer?
    private var bayerOutputBuf: MTLBuffer?
    private var currentWidth: Int = 0
    private var currentHeight: Int = 0
    private var currentCnnW: Int = 0
    private var currentCnnH: Int = 0
    private var debugFrameCount: Int = -1  // -1 to dump on first frame after rebuild

    // Temporal EMA: smooth CNN noise predictions across frames to reduce "heat shimmer"
    private var prevNoiseBuf: UnsafeMutablePointer<Float>?
    private var prevNoiseCount: Int = 0
    private var prevNoiseValid: Bool = false

    // Camera noise model for calibrated noise sigma map
    struct NoiseModelParams {
        var black_level: Float  // normalized [0,1] (raw / 65535)
        var read_noise: Float   // normalized [0,1] (raw / 65535)
        var shot_gain: Float    // raw scale (e.g. 180.0)
    }
    var noiseModel = NoiseModelParams(
        black_level: 6032.0 / 65535.0,  // S1M2 default
        read_noise: 616.0 / 65535.0,
        shot_gain: 180.0
    )

    // Person segmentation mask support
    var protectSubjects: Bool = false
    var subjectProtection: Float = 0.9
    var invertMask: Bool = false
    private var maskGPUBuf: MTLBuffer?


    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            debugLog("MPSPostFilter: no Metal device")
            return nil
        }
        self.device = device
        self.queue = queue

        // Load Metal compute kernels
        guard let library = device.makeDefaultLibrary(),
              let extractFunc = library.makeFunction(name: "bayer_extract_downsample"),
              let blendFunc = library.makeFunction(name: "noise_subtract_blend_interleave") else {
            debugLog("MPSPostFilter: Metal kernels not found")
            return nil
        }

        do {
            self.extractPSO = try device.makeComputePipelineState(function: extractFunc)
            self.blendPSO = try device.makeComputePipelineState(function: blendFunc)
        } catch {
            debugLog("MPSPostFilter: pipeline creation failed: \(error)")
            return nil
        }

        // Masked blend (optional — for subject protection)
        if let maskedFunc = library.makeFunction(name: "noise_subtract_blend_masked") {
            self.maskedBlendPSO = try? device.makeComputePipelineState(function: maskedFunc)
        } else {
            self.maskedBlendPSO = nil
        }

        // Try Restormer first, then UNet-Lite, then DnCNN
        let graph = MPSGraph()
        self.graph = graph

        if let weightsURL = Bundle.main.url(forResource: "postfilter_restormer_weights", withExtension: "bin"),
           let weights = loadRestormerWeights(from: weightsURL) {
            // Build Restormer-Lite graph
            self.graphType = .restormer
            self.inputChannels = weights.inChannels  // 5

            let inputTensor = graph.placeholder(
                shape: [1, NSNumber(value: weights.inChannels), -1, -1],
                dataType: .float32,
                name: "input"
            )
            self.inputPlaceholder = inputTensor
            self.noiseOutput = MPSPostFilter.buildRestormerGraph(
                graph: graph, input: inputTensor, weights: weights
            )

            debugLog("MPSPostFilter: initialized Restormer-Lite v\(weights.version) "
                  + "(\(weights.inChannels)→\(weights.outChannels)ch, "
                  + "\(weights.numBlocks) transformer blocks, \(weights.tensors.count) tensors) — float16 inference")

        } else if let weightsURL = Bundle.main.url(forResource: "postfilter_unet_weights", withExtension: "bin"),
           let weights = loadUNetWeights(from: weightsURL) {
            // Build UNet-Lite graph
            self.graphType = .unet
            self.inputChannels = weights.inChannels  // 5

            let inputTensor = graph.placeholder(
                shape: [1, NSNumber(value: weights.inChannels), -1, -1],
                dataType: .float32,
                name: "input"
            )
            self.inputPlaceholder = inputTensor
            self.noiseOutput = MPSPostFilter.buildUNetGraph(
                graph: graph, input: inputTensor, weights: weights
            )

            let upsampleType = weights.version == 2 ? "pixel shuffle" : "bilinear"
            debugLog("MPSPostFilter: initialized UNet-Lite v\(weights.version) "
                  + "(\(weights.inChannels)→\(weights.outChannels)ch, "
                  + "\(weights.layers.count) conv layers, \(upsampleType)) — float16 inference")

        } else if let weightsURL = Bundle.main.url(forResource: "postfilter_1ch_weights", withExtension: "bin"),
                  let weights = loadDnCNNWeights(from: weightsURL),
                  weights.inChannels == 1 {
            // Legacy DnCNN graph
            self.graphType = .dncnn
            self.inputChannels = 1

            let inputTensor = graph.placeholder(
                shape: [-1, 1, -1, -1],
                dataType: .float32,
                name: "input"
            )
            self.inputPlaceholder = inputTensor
            self.noiseOutput = MPSPostFilter.buildDnCNNGraph(
                graph: graph, input: inputTensor, weights: weights
            )

            debugLog("MPSPostFilter: initialized DnCNN "
                  + "(\(weights.hiddenChannels) hidden, \(weights.numLayers) layers) "
                  + "— batched 4ch float16 inference")

        } else {
            debugLog("MPSPostFilter: no weights found in bundle")
            return nil
        }
    }

    // MARK: - UNet-Lite Graph Builder

    /// Build UNet graph with skip connections.
    /// Supports 2-level (v1/v2, 10 layers) and 3-level (v3, 14 layers).
    /// Architecture matches training/unet_lite.py exactly.
    private static func buildUNetGraph(
        graph: MPSGraph,
        input: MPSGraphTensor,
        weights: UNetWeights
    ) -> MPSGraphTensor {

        // Cast input to float16
        var h = graph.cast(input, to: .float16, name: "input_f16")

        // Helper: Conv3x3 + optional ReLU
        func conv(_ x: MPSGraphTensor, layer: UNetLayer, name: String, relu: Bool) -> MPSGraphTensor {
            let convDesc = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1, strideInY: 1,
                dilationRateInX: 1, dilationRateInY: 1,
                groups: 1,
                paddingLeft: 1, paddingRight: 1,
                paddingTop: 1, paddingBottom: 1,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            )!

            let wShape = [
                NSNumber(value: layer.outChannels),
                NSNumber(value: layer.inChannels),
                3 as NSNumber, 3 as NSNumber
            ]
            let wF32 = graph.constant(layer.weight, shape: wShape, dataType: .float32)
            let wF16 = graph.cast(wF32, to: .float16, name: "\(name)_w")

            var out = graph.convolution2D(x, weights: wF16, descriptor: convDesc, name: "\(name)_conv")

            let bShape = [1 as NSNumber, NSNumber(value: layer.outChannels), 1 as NSNumber, 1 as NSNumber]
            let bF32 = graph.constant(layer.bias, shape: bShape, dataType: .float32)
            let bF16 = graph.cast(bF32, to: .float16, name: "\(name)_b")
            out = graph.addition(out, bF16, name: "\(name)_bias")

            if relu {
                out = graph.reLU(with: out, name: "\(name)_relu")
            }
            return out
        }

        // Helper: 2x2 average pooling (stride 2)
        func pool2x(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let poolDesc = MPSGraphPooling2DOpDescriptor(
                kernelWidth: 2, kernelHeight: 2,
                strideInX: 2, strideInY: 2,
                paddingStyle: .TF_VALID,
                dataLayout: .NCHW
            )!
            return graph.avgPooling2D(withSourceTensor: x, descriptor: poolDesc, name: name)
        }

        let usePixelShuffle = weights.version >= 2

        // Helper: pixel shuffle (depth-to-space) — learned upsample 2x.
        func pixelShuffle(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            return graph.depth(toSpace2DTensor:
                x,
                widthAxis: 3,
                heightAxis: 2,
                depthAxis: 1,
                blockSize: 2,
                usePixelShuffleOrder: true,
                name: "\(name)_ps"
            )
        }

        // Helper: bilinear upsample to match a target tensor's spatial dims (v1 fallback)
        func upsampleToMatch(_ x: MPSGraphTensor, target: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let targetShape = graph.shapeOf(target, name: "\(name)_tshape")
            let hw = graph.sliceTensor(targetShape, dimension: 0, start: 2, length: 2, name: "\(name)_hw")
            return graph.resize(x, sizeTensor: hw, mode: .bilinear,
                                centerResult: true, alignCorners: false,
                                layout: .NCHW, name: "\(name)_up")
        }

        let layers = weights.layers

        if weights.numLevels == 3 {
            // --- 3-level UNet (v3, 14 layers) ---

            // Encoder L0 (full res)
            h = conv(h, layer: layers[0], name: "enc0a", relu: true)
            h = conv(h, layer: layers[1], name: "enc0b", relu: true)
            let skip0 = h
            h = pool2x(h, name: "pool0")

            // Encoder L1 (half res)
            h = conv(h, layer: layers[2], name: "enc1a", relu: true)
            h = conv(h, layer: layers[3], name: "enc1b", relu: true)
            let skip1 = h
            h = pool2x(h, name: "pool1")

            // Encoder L2 (quarter res)
            h = conv(h, layer: layers[4], name: "enc2a", relu: true)
            h = conv(h, layer: layers[5], name: "enc2b", relu: true)
            let skip2 = h
            h = pool2x(h, name: "pool2")

            // Bottleneck (eighth res)
            h = conv(h, layer: layers[6], name: "bot_a", relu: true)
            h = conv(h, layer: layers[7], name: "bot_b", relu: true)

            // Decoder L2
            h = pixelShuffle(h, name: "up2")  // [N, 128, H/4, W/4]
            h = graph.concatTensors([h, skip2], dimension: 1, name: "cat2")  // 128+128=256ch
            h = conv(h, layer: layers[8], name: "dec2a", relu: true)
            h = conv(h, layer: layers[9], name: "dec2b", relu: true)

            // Decoder L1
            h = pixelShuffle(h, name: "up1")  // [N, 64, H/2, W/2]
            h = graph.concatTensors([h, skip1], dimension: 1, name: "cat1")  // 64+64=128ch
            h = conv(h, layer: layers[10], name: "dec1a", relu: true)
            h = conv(h, layer: layers[11], name: "dec1b", relu: true)

            // Decoder L0
            h = pixelShuffle(h, name: "up0")  // [N, 32, H, W]
            h = graph.concatTensors([h, skip0], dimension: 1, name: "cat0")  // 32+32=64ch
            h = conv(h, layer: layers[12], name: "dec0a", relu: true)
            h = conv(h, layer: layers[13], name: "dec0b", relu: false)

        } else {
            // --- 2-level UNet (v1/v2, 10 layers) ---

            // Encoder L0 (full res)
            h = conv(h, layer: layers[0], name: "enc0a", relu: true)
            h = conv(h, layer: layers[1], name: "enc0b", relu: true)
            let skip0 = h
            h = pool2x(h, name: "pool0")

            // Encoder L1 (half res)
            h = conv(h, layer: layers[2], name: "enc1a", relu: true)
            h = conv(h, layer: layers[3], name: "enc1b", relu: true)
            let skip1 = h
            h = pool2x(h, name: "pool1")

            // Bottleneck (quarter res)
            h = conv(h, layer: layers[4], name: "bot_a", relu: true)
            h = conv(h, layer: layers[5], name: "bot_b", relu: true)

            // Decoder L1
            if usePixelShuffle {
                h = pixelShuffle(h, name: "up1")
            } else {
                h = upsampleToMatch(h, target: skip1, name: "up1")
            }
            h = graph.concatTensors([h, skip1], dimension: 1, name: "cat1")
            h = conv(h, layer: layers[6], name: "dec1a", relu: true)
            h = conv(h, layer: layers[7], name: "dec1b", relu: true)

            // Decoder L0
            if usePixelShuffle {
                h = pixelShuffle(h, name: "up0")
            } else {
                h = upsampleToMatch(h, target: skip0, name: "up0")
            }
            h = graph.concatTensors([h, skip0], dimension: 1, name: "cat0")
            h = conv(h, layer: layers[8], name: "dec0a", relu: true)
            h = conv(h, layer: layers[9], name: "dec0b", relu: false)
        }

        // Cast back to float32
        h = graph.cast(h, to: .float32, name: "output_f32")
        return h
    }

    // MARK: - Restormer-Lite Graph Builder

    /// Build Restormer-Lite graph with transformer attention blocks.
    /// Architecture matches training/restormer_lite.py exactly.
    private static func buildRestormerGraph(
        graph: MPSGraph,
        input: MPSGraphTensor,
        weights: RestormerWeights
    ) -> MPSGraphTensor {

        var idx = 0  // current tensor index

        /// Read next tensor from weights array, verify type.
        func nextTensor(_ expectedType: RestormerTensorType, _ label: String) -> RestormerTensor {
            let t = weights.tensors[idx]
            assert(t.type == expectedType, "Restormer tensor \(idx) (\(label)): expected type \(expectedType), got \(t.type)")
            idx += 1
            return t
        }

        /// Make an MPSGraph constant from tensor data with given shape (float16).
        func constant(_ t: RestormerTensor, shape: [NSNumber], name: String) -> MPSGraphTensor {
            let f32 = graph.constant(t.data, shape: shape, dataType: .float32)
            return graph.cast(f32, to: .float16, name: "\(name)_f16")
        }

        // Cast input to float16
        let inputF16 = graph.cast(input, to: .float16, name: "input_f16")

        // --- Helper: Conv (1×1 or 3×3) ---
        func conv(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let wt = nextTensor(.convWeight, "\(name)_w")
            let bt = nextTensor(.convBias, "\(name)_b")

            let kH = wt.shape[2], kW = wt.shape[3]
            let pad = kH / 2  // 0 for 1×1, 1 for 3×3

            let convDesc = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1, strideInY: 1,
                dilationRateInX: 1, dilationRateInY: 1,
                groups: 1,
                paddingLeft: pad, paddingRight: pad,
                paddingTop: pad, paddingBottom: pad,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            )!

            let wShape = wt.shape.map { NSNumber(value: $0) }
            let w = constant(wt, shape: wShape, name: "\(name)_w")

            var out = graph.convolution2D(x, weights: w, descriptor: convDesc, name: "\(name)_conv")

            let outCh = bt.shape[0]
            let bShape = [1 as NSNumber, NSNumber(value: outCh), 1 as NSNumber, 1 as NSNumber]
            let b = constant(bt, shape: bShape, name: "\(name)_b")
            out = graph.addition(out, b, name: "\(name)_bias")

            return out
        }

        // --- Helper: Depthwise Conv 3×3 ---
        // Implemented as grouped conv2d with groups=C (same as depthwise)
        func dwConv(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let wt = nextTensor(.dwConvWeight, "\(name)_w")
            let bt = nextTensor(.dwConvBias, "\(name)_b")

            let channels = wt.shape[1]  // [1, C, 3, 3] in MPS layout

            // Use standard conv2d with groups=channels for depthwise
            // Weights must be [C, 1, 3, 3] for groups=C conv (OIHW where I=1 per group)
            // Our binary stores [1, C, 3, 3], so transpose to [C, 1, 3, 3]
            let wShape: [NSNumber] = [NSNumber(value: channels), 1 as NSNumber,
                                      NSNumber(value: wt.shape[2]), NSNumber(value: wt.shape[3])]
            // Stored as [1, C, kH, kW] → need [C, 1, kH, kW] for grouped conv
            let wOrig = graph.constant(wt.data, shape: [1 as NSNumber, NSNumber(value: channels),
                                                         NSNumber(value: wt.shape[2]), NSNumber(value: wt.shape[3])],
                                       dataType: .float32)
            let wTransposed = graph.transposeTensor(wOrig, dimension: 0, withDimension: 1, name: "\(name)_wt")
            let w = graph.cast(wTransposed, to: .float16, name: "\(name)_w_f16")

            let convDesc = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1, strideInY: 1,
                dilationRateInX: 1, dilationRateInY: 1,
                groups: channels,
                paddingLeft: 1, paddingRight: 1,
                paddingTop: 1, paddingBottom: 1,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            )!

            var out = graph.convolution2D(x, weights: w, descriptor: convDesc, name: "\(name)_dw")

            let bShape = [1 as NSNumber, NSNumber(value: channels), 1 as NSNumber, 1 as NSNumber]
            let b = constant(bt, shape: bShape, name: "\(name)_b")
            out = graph.addition(out, b, name: "\(name)_bias")

            return out
        }

        // --- Helper: LayerNorm (spatial, per-channel) ---
        func layerNorm(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let gammaT = nextTensor(.lnGamma, "\(name)_gamma")
            let betaT = nextTensor(.lnBeta, "\(name)_beta")

            let channels = gammaT.shape[1]
            let gShape = [1 as NSNumber, NSNumber(value: channels), 1 as NSNumber, 1 as NSNumber]
            let gamma = constant(gammaT, shape: gShape, name: "\(name)_gamma")
            let beta = constant(betaT, shape: gShape, name: "\(name)_beta")

            // Mean and variance over spatial dims [2, 3]
            let mean = graph.mean(of: x, axes: [2, 3], name: "\(name)_mean")
            let variance = graph.variance(of: x, mean: mean, axes: [2, 3], name: "\(name)_var")

            let eps = graph.constant(1e-5, dataType: .float16)
            let varEps = graph.addition(variance, eps, name: "\(name)_var_eps")
            let invStd = graph.reverseSquareRoot(with: varEps, name: "\(name)_invstd")
            let centered = graph.subtraction(x, mean, name: "\(name)_centered")
            let normed = graph.multiplication(centered, invStd, name: "\(name)_normed")
            let scaled = graph.multiplication(normed, gamma, name: "\(name)_scaled")
            return graph.addition(scaled, beta, name: "\(name)_out")
        }

        // --- Helper: MDTA (Multi-DConv Transposed Attention) ---
        func mdta(_ x: MPSGraphTensor, channels: Int, heads: Int, name: String) -> MPSGraphTensor {
            let headDim = channels / heads

            // QKV projection: 1×1 conv → [N, 3C, H, W]
            let qkv = conv(x, name: "\(name)_qkv")

            // Split into Q, K, V
            let q_raw = graph.sliceTensor(qkv, dimension: 1, start: 0, length: channels, name: "\(name)_q")
            let k_raw = graph.sliceTensor(qkv, dimension: 1, start: channels, length: channels, name: "\(name)_k")
            let v_raw = graph.sliceTensor(qkv, dimension: 1, start: channels * 2, length: channels, name: "\(name)_v")

            // DWConv3x3 on each for local context
            let q = dwConv(q_raw, name: "\(name)_dwq")
            let k = dwConv(k_raw, name: "\(name)_dwk")
            let v = dwConv(v_raw, name: "\(name)_dwv")

            // Temperature (learnable per-head scalar)
            let tempT = nextTensor(.temperature, "\(name)_temp")
            let tempShape = [1 as NSNumber, NSNumber(value: heads), 1 as NSNumber, 1 as NSNumber]
            let temp = constant(tempT, shape: tempShape, name: "\(name)_temp")

            // Reshape to [N*heads, headDim, H*W] — merge N and heads for batched matmul
            // From [1, C, H, W] → [heads, headDim, H*W]
            let qr = graph.reshape(q, shape: [NSNumber(value: heads), NSNumber(value: headDim), -1], name: "\(name)_qr")
            let kr = graph.reshape(k, shape: [NSNumber(value: heads), NSNumber(value: headDim), -1], name: "\(name)_kr")
            let vr = graph.reshape(v, shape: [NSNumber(value: heads), NSNumber(value: headDim), -1], name: "\(name)_vr")

            // L2 normalize Q and K along last dim for stable attention
            let qNormSq = graph.reductionSum(with: graph.multiplication(qr, qr, name: nil), axes: [2], name: "\(name)_qns")
            let qNorm = graph.reverseSquareRoot(with: graph.addition(qNormSq, graph.constant(1e-12, dataType: .float16), name: nil), name: "\(name)_qnorm")
            let qn = graph.multiplication(qr, qNorm, name: "\(name)_qnormalized")

            let kNormSq = graph.reductionSum(with: graph.multiplication(kr, kr, name: nil), axes: [2], name: "\(name)_kns")
            let kNorm = graph.reverseSquareRoot(with: graph.addition(kNormSq, graph.constant(1e-12, dataType: .float16), name: nil), name: "\(name)_knorm")
            let kn = graph.multiplication(kr, kNorm, name: "\(name)_knormalized")

            // Channel attention: Q @ K^T → [heads, headDim, headDim]
            let kt = graph.transposeTensor(kn, dimension: 1, withDimension: 2, name: "\(name)_kt")
            let attnRaw = graph.matrixMultiplication(primary: qn, secondary: kt, name: "\(name)_qkt")

            // Scale by temperature
            let attnScaled = graph.multiplication(attnRaw, temp, name: "\(name)_attn_scaled")

            // Softmax along last dim
            let attn = graph.softMax(with: attnScaled, axis: 2, name: "\(name)_softmax")

            // Apply attention: attn @ V → [heads, headDim, H*W]
            let attended = graph.matrixMultiplication(primary: attn, secondary: vr, name: "\(name)_attnv")

            // Reshape back to [1, C, H, W]
            let outReshaped = graph.reshape(attended, shapeTensor: graph.shapeOf(x, name: "\(name)_xshape"), name: "\(name)_reshape")

            // Output projection: 1×1 conv
            return conv(outReshaped, name: "\(name)_out")
        }

        // --- Helper: GDFN (Gated Feed-Forward with SimpleGate) ---
        func gdfn(_ x: MPSGraphTensor, channels: Int, name: String) -> MPSGraphTensor {
            // Expand: 1×1 conv C→2C
            var h = conv(x, name: "\(name)_expand")

            // DWConv 3×3
            h = dwConv(h, name: "\(name)_dw")

            // SimpleGate: split channels in half, multiply
            let x1 = graph.sliceTensor(h, dimension: 1, start: 0, length: channels, name: "\(name)_g1")
            let x2 = graph.sliceTensor(h, dimension: 1, start: channels, length: channels, name: "\(name)_g2")
            h = graph.multiplication(x1, x2, name: "\(name)_gate")

            // Project: 1×1 conv C→C
            return conv(h, name: "\(name)_proj")
        }

        // --- Helper: RLTB (Restormer-Lite Transformer Block) ---
        func rltb(_ x: MPSGraphTensor, channels: Int, heads: Int, name: String) -> MPSGraphTensor {
            // Sub-block A: LayerNorm → MDTA → residual
            let normed1 = layerNorm(x, name: "\(name)_ln1")
            let attnOut = mdta(normed1, channels: channels, heads: heads, name: "\(name)_mdta")
            let afterAttn = graph.addition(x, attnOut, name: "\(name)_res1")

            // Sub-block B: LayerNorm → GDFN → residual
            let normed2 = layerNorm(afterAttn, name: "\(name)_ln2")
            let ffnOut = gdfn(normed2, channels: channels, name: "\(name)_gdfn")
            return graph.addition(afterAttn, ffnOut, name: "\(name)_res2")
        }

        // --- Helper: Average Pool 2×2 ---
        func pool2x(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            let desc = MPSGraphPooling2DOpDescriptor(
                kernelWidth: 2, kernelHeight: 2,
                strideInX: 2, strideInY: 2,
                paddingStyle: .TF_VALID,
                dataLayout: .NCHW
            )!
            return graph.avgPooling2D(withSourceTensor: x, descriptor: desc, name: name)
        }

        // --- Helper: Pixel Shuffle 2x ---
        func pixelShuffle(_ x: MPSGraphTensor, name: String) -> MPSGraphTensor {
            return graph.depth(toSpace2DTensor: x,
                widthAxis: 3, heightAxis: 2, depthAxis: 1,
                blockSize: 2, usePixelShuffleOrder: true,
                name: "\(name)_ps")
        }

        // ================================================================
        // Build the full Restormer-Lite graph
        // ================================================================

        var h = inputF16

        // Encoder L0 (full res): Conv1x1(5→48) + RLTB(48, heads=2)
        h = conv(h, name: "enc0_proj")
        h = rltb(h, channels: 48, heads: 2, name: "enc0_block")
        let skip0 = h
        h = pool2x(h, name: "pool0")

        // Encoder L1 (half res): Conv3x3(48→96) + RLTB(96, heads=4)
        h = conv(h, name: "enc1_proj")
        h = rltb(h, channels: 96, heads: 4, name: "enc1_block")
        let skip1 = h
        h = pool2x(h, name: "pool1")

        // Encoder L2 (quarter res): Conv3x3(96→192) + RLTB(192, heads=6)
        h = conv(h, name: "enc2_proj")
        h = rltb(h, channels: 192, heads: 6, name: "enc2_block")
        let skip2 = h
        h = pool2x(h, name: "pool2")

        // Bottleneck (eighth res): RLTB(192, heads=6) + Conv3x3(192→768)
        h = rltb(h, channels: 192, heads: 6, name: "bot_block")
        h = conv(h, name: "bot_up")
        h = pixelShuffle(h, name: "bot")  // → 192ch at quarter res

        // Decoder L2: cat(skip2)→384, Conv1x1(384→192), RLTB, Conv3x3(192→384) → PS→96ch
        h = graph.concatTensors([h, skip2], dimension: 1, name: "cat2")
        h = conv(h, name: "dec2_fuse")
        h = rltb(h, channels: 192, heads: 6, name: "dec2_block")
        h = conv(h, name: "dec2_up")
        h = pixelShuffle(h, name: "dec2")  // → 96ch at half res

        // Decoder L1: cat(skip1)→192, Conv1x1(192→96), RLTB, Conv3x3(96→192) → PS→48ch
        h = graph.concatTensors([h, skip1], dimension: 1, name: "cat1")
        h = conv(h, name: "dec1_fuse")
        h = rltb(h, channels: 96, heads: 4, name: "dec1_block")
        h = conv(h, name: "dec1_up")
        h = pixelShuffle(h, name: "dec1")  // → 48ch at full res

        // Decoder L0: cat(skip0)→96, Conv1x1(96→48), RLTB, Conv3x3(48→4)
        h = graph.concatTensors([h, skip0], dimension: 1, name: "cat0")
        h = conv(h, name: "dec0_fuse")
        h = rltb(h, channels: 48, heads: 2, name: "dec0_block")
        h = conv(h, name: "dec0_out")

        assert(idx == weights.tensors.count,
               "Restormer graph used \(idx) tensors but weights has \(weights.tensors.count)")

        // Cast back to float32
        h = graph.cast(h, to: .float32, name: "output_f32")
        return h
    }

    // MARK: - DnCNN Graph Builder (legacy)

    private static func buildDnCNNGraph(
        graph: MPSGraph,
        input: MPSGraphTensor,
        weights: DnCNNWeights
    ) -> MPSGraphTensor {
        var h = graph.cast(input, to: .float16, name: "input_f16")

        for (i, layer) in weights.layers.enumerated() {
            let convDesc = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1, strideInY: 1,
                dilationRateInX: 1, dilationRateInY: 1,
                groups: 1,
                paddingLeft: 1, paddingRight: 1,
                paddingTop: 1, paddingBottom: 1,
                paddingStyle: .explicit,
                dataLayout: .NCHW,
                weightsLayout: .OIHW
            )!

            let wShape = [
                NSNumber(value: layer.outChannels),
                NSNumber(value: layer.inChannels),
                NSNumber(value: layer.kernelH),
                NSNumber(value: layer.kernelW)
            ]
            let wF32 = graph.constant(layer.weight, shape: wShape, dataType: .float32)
            let wTensor = graph.cast(wF32, to: .float16, name: "w\(i)_f16")
            h = graph.convolution2D(h, weights: wTensor, descriptor: convDesc, name: "conv\(i)")

            let bShape = [1 as NSNumber, NSNumber(value: layer.outChannels), 1 as NSNumber, 1 as NSNumber]
            let bF32 = graph.constant(layer.bias, shape: bShape, dataType: .float32)
            let bTensor = graph.cast(bF32, to: .float16, name: "b\(i)_f16")
            h = graph.addition(h, bTensor, name: "bias\(i)")

            if i < weights.layers.count - 1 {
                h = graph.reLU(with: h, name: "relu\(i)")
            }
        }

        h = graph.cast(h, to: .float32, name: "output_f32")
        return h
    }

    // MARK: - Tiling

    /// Max CNN tile size in sub-channel pixels. Tiles larger than this are split.
    /// 512×512 × 5ch × 4bytes = 5 MB input per tile — safe on any GPU.
    private static let maxTileSize = 128
    /// Overlap halo on each side for depthwise conv context (3×3 kernel → 1px, but use 8 for safety across encoder levels).
    private static let tileHalo = 8

    // Tile buffers (allocated once, reused)
    private var tileInputBuf: MTLBuffer?
    private var tileOutputBuf: MTLBuffer?
    private var tileMaxW: Int = 0
    private var tileMaxH: Int = 0
    private var tiledLogPrinted: Bool = false

    private func ensureTileBuffers(tileW: Int, tileH: Int) {
        guard tileW > tileMaxW || tileH > tileMaxH else { return }
        tileMaxW = max(tileW, tileMaxW)
        tileMaxH = max(tileH, tileMaxH)
        let inCh = (graphType == .dncnn) ? 4 : 5
        tileInputBuf = device.makeBuffer(length: inCh * tileMaxW * tileMaxH * MemoryLayout<Float>.size, options: .storageModeShared)
        tileOutputBuf = device.makeBuffer(length: 4 * tileMaxW * tileMaxH * MemoryLayout<Float>.size, options: .storageModeShared)
    }

    // MARK: - Full-frame Inference (small frames)

    private func runFullInference(cnnIn: MTLBuffer, cnnOut: MTLBuffer, cnnW: Int, cnnH: Int, channelFloats: Int) {
        autoreleasepool {
            let inputShape: [NSNumber]
            let outputChannels: Int

            switch graphType {
            case .restormer, .unet:
                inputShape = [1, NSNumber(value: inputChannels), NSNumber(value: cnnH), NSNumber(value: cnnW)]
                outputChannels = 4
            case .dncnn:
                inputShape = [4, 1, NSNumber(value: cnnH), NSNumber(value: cnnW)]
                outputChannels = 4
            }

            let inputTensorData = MPSGraphTensorData(cnnIn, shape: inputShape, dataType: .float32)
            let feeds: [MPSGraphTensor: MPSGraphTensorData] = [inputPlaceholder: inputTensorData]
            let results = graph.run(with: queue, feeds: feeds, targetTensors: [noiseOutput], targetOperations: nil)

            guard let noiseTensorData = results[noiseOutput] else {
                debugLog("MPSPostFilter: CNN inference failed")
                return
            }

            let cnnOutPtr = cnnOut.contents().bindMemory(to: Float.self, capacity: outputChannels * channelFloats)
            noiseTensorData.mpsndarray().readBytes(cnnOutPtr, strideBytes: nil)
        }
    }

    // MARK: - Tiled Inference (large frames)

    /// Process CNN in overlapping tiles to avoid GPU OOM on large frames.
    /// Each tile is maxTileSize×maxTileSize with tileHalo overlap on each side.
    /// The Restormer uses transposed attention (C×C, not spatial), so tiling has zero quality loss.
    private func runTiledInference(cnnIn: MTLBuffer, cnnOut: MTLBuffer, cnnW: Int, cnnH: Int, channelFloats: Int) {
        let tileSize = Self.maxTileSize
        let halo = Self.tileHalo
        let inCh = inputChannels  // 5 for restormer/unet
        let outCh = 4

        // Compute tile grid
        let tilesX = (cnnW + tileSize - 1) / tileSize
        let tilesY = (cnnH + tileSize - 1) / tileSize

        // Ensure tile buffers are large enough (tile + 2*halo on each side)
        let maxPaddedW = min(tileSize + 2 * halo, cnnW)
        let maxPaddedH = min(tileSize + 2 * halo, cnnH)
        ensureTileBuffers(tileW: maxPaddedW, tileH: maxPaddedH)

        guard let tileBufIn = tileInputBuf, let tileBufOut = tileOutputBuf else {
            debugLog("MPSPostFilter: tile buffer allocation failed")
            return
        }

        let srcPtr = cnnIn.contents().bindMemory(to: Float.self, capacity: inCh * channelFloats)
        let dstPtr = cnnOut.contents().bindMemory(to: Float.self, capacity: outCh * channelFloats)
        let tileInPtr = tileBufIn.contents().bindMemory(to: Float.self, capacity: inCh * maxPaddedW * maxPaddedH)

        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                // Core tile region (no halo)
                let coreX0 = tx * tileSize
                let coreY0 = ty * tileSize
                let coreX1 = min(coreX0 + tileSize, cnnW)
                let coreY1 = min(coreY0 + tileSize, cnnH)

                // Padded region (with halo, clamped to image bounds)
                let padX0 = max(coreX0 - halo, 0)
                let padY0 = max(coreY0 - halo, 0)
                let padX1 = min(coreX1 + halo, cnnW)
                let padY1 = min(coreY1 + halo, cnnH)
                let padW = padX1 - padX0
                let padH = padY1 - padY0

                // Copy tile input data (all channels) from planar cnnIn to tile buffer
                for ch in 0..<inCh {
                    let srcBase = ch * channelFloats  // start of this channel in planar layout
                    let dstBase = ch * padW * padH
                    for row in 0..<padH {
                        let srcOff = srcBase + (padY0 + row) * cnnW + padX0
                        let dstOff = dstBase + row * padW
                        memcpy(tileInPtr + dstOff, srcPtr + srcOff, padW * MemoryLayout<Float>.size)
                    }
                }

                // Run inference on this tile
                autoreleasepool {
                    let tileShape: [NSNumber] = [1, NSNumber(value: inCh), NSNumber(value: padH), NSNumber(value: padW)]
                    let tileTensorData = MPSGraphTensorData(tileBufIn, shape: tileShape, dataType: .float32)
                    let feeds: [MPSGraphTensor: MPSGraphTensorData] = [inputPlaceholder: tileTensorData]
                    let results = graph.run(with: queue, feeds: feeds, targetTensors: [noiseOutput], targetOperations: nil)

                    guard let noiseTensorData = results[noiseOutput] else {
                        debugLog("MPSPostFilter: tiled inference failed at tile (\(tx),\(ty))")
                        return
                    }

                    // Read tile output into tile output buffer
                    let tileOutPtr = tileBufOut.contents().bindMemory(to: Float.self, capacity: outCh * padW * padH)
                    noiseTensorData.mpsndarray().readBytes(tileOutPtr, strideBytes: nil)

                    // Copy core region (strip halo) from tile output back to cnnOut
                    let haloLeft = coreX0 - padX0
                    let haloTop = coreY0 - padY0
                    let coreW = coreX1 - coreX0
                    let coreH = coreY1 - coreY0

                    for ch in 0..<outCh {
                        let srcBase = ch * padW * padH
                        let dstBase = ch * channelFloats
                        for row in 0..<coreH {
                            let srcOff = srcBase + (haloTop + row) * padW + haloLeft
                            let dstOff = dstBase + (coreY0 + row) * cnnW + coreX0
                            memcpy(dstPtr + dstOff, tileOutPtr + srcOff, coreW * MemoryLayout<Float>.size)
                        }
                    }
                }
            }
        }

        if !tiledLogPrinted {
            tiledLogPrinted = true
            debugLog("MPSPostFilter: tiled inference \(tilesX)×\(tilesY) tiles (\(tileSize)+\(halo)px halo) on \(cnnW)×\(cnnH)")
        }
    }

    // MARK: - Buffer Management

    private func ensureBuffers(width: Int, height: Int) {
        guard width != currentWidth || height != currentHeight else { return }
        currentWidth = width
        currentHeight = height

        let cnnW = width / 2
        let cnnH = height / 2
        currentCnnW = cnnW
        currentCnnH = cnnH

        let bayerBytes = width * height * MemoryLayout<UInt16>.size
        // Input: 5 channels for UNet/Restormer (4 Bayer + noise map), 4 for DnCNN
        let cnnInputCh = (graphType == .dncnn) ? 4 : 5
        let cnnInputBytes = cnnInputCh * cnnW * cnnH * MemoryLayout<Float>.size
        // Output: always 4 channels (noise predictions)
        let cnnOutputBytes = 4 * cnnW * cnnH * MemoryLayout<Float>.size

        bayerInputBuf  = device.makeBuffer(length: bayerBytes, options: .storageModeShared)
        cnnInputBuf    = device.makeBuffer(length: cnnInputBytes, options: .storageModeShared)
        cnnOutputBuf   = device.makeBuffer(length: cnnOutputBytes, options: .storageModeShared)
        bayerOutputBuf = device.makeBuffer(length: bayerBytes, options: .storageModeShared)

        // Temporal EMA buffer for CNN noise predictions
        let noiseFloats = 4 * cnnW * cnnH
        prevNoiseBuf?.deallocate()
        prevNoiseBuf = UnsafeMutablePointer<Float>.allocate(capacity: noiseFloats)
        prevNoiseCount = noiseFloats
        prevNoiseValid = false  // invalidate on resolution change

        debugLog("MPSPostFilter: buffers for \(width)×\(height), CNN \(cnnW)×\(cnnH) "
              + "(\(cnnInputCh)ch input)")
    }

    // MARK: - Apply

    /// Apply CNN post-filter. Modifies `bayer` in-place.
    func apply(bayer: UnsafeMutablePointer<UInt16>, width: Int, height: Int, blendFactor: Float = 0.9) {
        ensureBuffers(width: width, height: height)

        guard let bayerIn = bayerInputBuf,
              let cnnIn = cnnInputBuf,
              let cnnOut = cnnOutputBuf,
              let bayerOut = bayerOutputBuf else { return }

        let cnnW = currentCnnW
        let cnnH = currentCnnH
        let bayerBytes = width * height * MemoryLayout<UInt16>.size
        let channelFloats = cnnW * cnnH

        // 1. Upload Bayer to GPU
        memcpy(bayerIn.contents(), bayer, bayerBytes)

        // 2. Bayer extract (Metal compute) → planar channels in cnnIn
        guard let cmdBuf = queue.makeCommandBuffer() else { return }

        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

        if let enc = cmdBuf.makeComputeCommandEncoder() {
            var fullDims = SIMD2<UInt32>(UInt32(width), UInt32(height))
            var cnnDims = SIMD2<UInt32>(UInt32(cnnW), UInt32(cnnH))

            enc.setComputePipelineState(extractPSO)
            enc.setBuffer(bayerIn, offset: 0, index: 0)
            enc.setBuffer(cnnIn, offset: 0, index: 1)
            enc.setBytes(&fullDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.setBytes(&cnnDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            var nm = noiseModel
            enc.setBytes(&nm, length: MemoryLayout<NoiseModelParams>.size, index: 4)
            enc.dispatchThreads(
                MTLSize(width: cnnW, height: cnnH, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // 3. CNN inference (tiled for large frames to avoid GPU OOM)
        let needsTiling = (graphType == .restormer || graphType == .unet) &&
                          (cnnW > Self.maxTileSize || cnnH > Self.maxTileSize)

        if needsTiling {
            runTiledInference(cnnIn: cnnIn, cnnOut: cnnOut, cnnW: cnnW, cnnH: cnnH, channelFloats: channelFloats)
        } else {
            runFullInference(cnnIn: cnnIn, cnnOut: cnnOut, cnnW: cnnW, cnnH: cnnH, channelFloats: channelFloats)
        }

        // 3b. Temporal EMA: smooth CNN noise predictions to reduce frame-to-frame wobble.
        //     smoothed[i] = alpha * current[i] + (1-alpha) * previous[i]
        //     alpha=0.85 → light smoothing, kills heat shimmer without hurting denoising.
        if let prevBuf = prevNoiseBuf {
            let cnnOutPtr = cnnOut.contents().bindMemory(to: Float.self, capacity: prevNoiseCount)
            if prevNoiseValid {
                var alpha: Float = 0.85
                // vDSP_vintb: C[i] = A[i] + alpha*(B[i] - A[i]) = A*(1-alpha) + B*alpha
                // A=previous, B=current, C=current (in-place)
                vDSP_vintb(prevBuf, 1, cnnOutPtr, 1, &alpha, cnnOutPtr, 1, vDSP_Length(prevNoiseCount))
            }
            // Store current (blended) predictions for next frame
            memcpy(prevBuf, cnnOutPtr, prevNoiseCount * MemoryLayout<Float>.size)
            prevNoiseValid = true
        }

        // 4. Person segmentation mask (if enabled)
        var useMaskedBlend = false
        if protectSubjects, let maskedPSO = maskedBlendPSO {
            // Run segmentation on current frame
            let bayerPtr = bayerIn.contents().bindMemory(to: UInt16.self, capacity: width * height)
            if let maskPtr = PersonSegmentor.shared.segment(bayer: bayerPtr, width: width, height: height) {
                let maskBytes = cnnW * cnnH * MemoryLayout<Float>.size
                if maskGPUBuf == nil || maskGPUBuf!.length < maskBytes {
                    maskGPUBuf = device.makeBuffer(length: maskBytes, options: .storageModeShared)
                }
                if let gpuMask = maskGPUBuf {
                    let maskCount = cnnW * cnnH
                    if invertMask {
                        // Invert: 1.0 on background, 0.0 on people
                        let dst = gpuMask.contents().bindMemory(to: Float.self, capacity: maskCount)
                        for i in 0..<maskCount {
                            dst[i] = 1.0 - maskPtr[i]
                        }
                    } else {
                        memcpy(gpuMask.contents(), maskPtr, maskBytes)
                    }
                    useMaskedBlend = true
                }
            }
            _ = maskedPSO  // suppress unused warning
        }

        // 5. Noise subtract + blend + interleave (Metal compute)
        guard let cmdBuf2 = queue.makeCommandBuffer() else { return }

        if let enc = cmdBuf2.makeComputeCommandEncoder() {
            var fullDims = SIMD2<UInt32>(UInt32(width), UInt32(height))
            var cnnDims = SIMD2<UInt32>(UInt32(cnnW), UInt32(cnnH))
            var blend = blendFactor

            if useMaskedBlend, let maskedPSO = maskedBlendPSO, let gpuMask = maskGPUBuf {
                var protection = subjectProtection
                enc.setComputePipelineState(maskedPSO)
                enc.setBuffer(cnnOut, offset: 0, index: 0)
                enc.setBuffer(bayerIn, offset: 0, index: 1)
                enc.setBuffer(bayerOut, offset: 0, index: 2)
                enc.setBytes(&fullDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
                enc.setBytes(&cnnDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
                enc.setBytes(&blend, length: MemoryLayout<Float>.size, index: 5)
                enc.setBuffer(gpuMask, offset: 0, index: 6)
                enc.setBytes(&protection, length: MemoryLayout<Float>.size, index: 7)
            } else {
                enc.setComputePipelineState(blendPSO)
                enc.setBuffer(cnnOut, offset: 0, index: 0)
                enc.setBuffer(bayerIn, offset: 0, index: 1)
                enc.setBuffer(bayerOut, offset: 0, index: 2)
                enc.setBytes(&fullDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
                enc.setBytes(&cnnDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
                enc.setBytes(&blend, length: MemoryLayout<Float>.size, index: 5)
            }

            enc.dispatchThreads(
                MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )
            enc.endEncoding()
        }

        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        // 5. Copy denoised result back to CPU
        memcpy(bayer, bayerOut.contents(), bayerBytes)
    }

    /// Zero-copy variant: reads from an MTLBuffer (temporal filter output),
    /// writes result back to the same buffer. Eliminates 2 frame memcpys.
    func apply(inputBuffer: MTLBuffer, width: Int, height: Int, blendFactor: Float = 0.9) {
        ensureBuffers(width: width, height: height)

        guard let cnnIn = cnnInputBuf,
              let cnnOut = cnnOutputBuf,
              let bayerOut = bayerOutputBuf else { return }

        let cnnW = currentCnnW
        let cnnH = currentCnnH
        let bayerBytes = width * height * MemoryLayout<UInt16>.size
        let channelFloats = cnnW * cnnH

        // 1. Skip upload — inputBuffer IS the GPU data (shared MTLBuffer)

        // 2. Bayer extract → planar channels in cnnIn
        guard let cmdBuf = queue.makeCommandBuffer() else { return }
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

        if let enc = cmdBuf.makeComputeCommandEncoder() {
            var fullDims = SIMD2<UInt32>(UInt32(width), UInt32(height))
            var cnnDims = SIMD2<UInt32>(UInt32(cnnW), UInt32(cnnH))

            enc.setComputePipelineState(extractPSO)
            enc.setBuffer(inputBuffer, offset: 0, index: 0)  // read from shared TF output
            enc.setBuffer(cnnIn, offset: 0, index: 1)
            enc.setBytes(&fullDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
            enc.setBytes(&cnnDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
            var nm = noiseModel
            enc.setBytes(&nm, length: MemoryLayout<NoiseModelParams>.size, index: 4)
            enc.dispatchThreads(
                MTLSize(width: cnnW, height: cnnH, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )
            enc.endEncoding()
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // 3. CNN inference
        let needsTiling = (graphType == .restormer || graphType == .unet) &&
                          (cnnW > Self.maxTileSize || cnnH > Self.maxTileSize)

        if needsTiling {
            runTiledInference(cnnIn: cnnIn, cnnOut: cnnOut, cnnW: cnnW, cnnH: cnnH, channelFloats: channelFloats)
        } else {
            runFullInference(cnnIn: cnnIn, cnnOut: cnnOut, cnnW: cnnW, cnnH: cnnH, channelFloats: channelFloats)
        }

        // 3b. Temporal EMA
        if let prevBuf = prevNoiseBuf {
            let cnnOutPtr = cnnOut.contents().bindMemory(to: Float.self, capacity: prevNoiseCount)
            if prevNoiseValid {
                var alpha: Float = 0.85
                vDSP_vintb(prevBuf, 1, cnnOutPtr, 1, &alpha, cnnOutPtr, 1, vDSP_Length(prevNoiseCount))
            }
            memcpy(prevBuf, cnnOutPtr, prevNoiseCount * MemoryLayout<Float>.size)
            prevNoiseValid = true
        }

        // 4. Person segmentation (if enabled)
        var useMaskedBlend = false
        if protectSubjects, let maskedPSO = maskedBlendPSO {
            let bayerPtr = inputBuffer.contents().bindMemory(to: UInt16.self, capacity: width * height)
            if let maskPtr = PersonSegmentor.shared.segment(bayer: bayerPtr, width: width, height: height) {
                let maskBytes = cnnW * cnnH * MemoryLayout<Float>.size
                if maskGPUBuf == nil || maskGPUBuf!.length < maskBytes {
                    maskGPUBuf = device.makeBuffer(length: maskBytes, options: .storageModeShared)
                }
                if let gpuMask = maskGPUBuf {
                    let maskCount = cnnW * cnnH
                    if invertMask {
                        let dst = gpuMask.contents().bindMemory(to: Float.self, capacity: maskCount)
                        for i in 0..<maskCount { dst[i] = 1.0 - maskPtr[i] }
                    } else {
                        memcpy(gpuMask.contents(), maskPtr, maskBytes)
                    }
                    useMaskedBlend = true
                }
            }
            _ = maskedPSO
        }

        // 5. Noise subtract + blend (read from inputBuffer, write to bayerOut)
        guard let cmdBuf2 = queue.makeCommandBuffer() else { return }

        if let enc = cmdBuf2.makeComputeCommandEncoder() {
            var fullDims = SIMD2<UInt32>(UInt32(width), UInt32(height))
            var cnnDims = SIMD2<UInt32>(UInt32(cnnW), UInt32(cnnH))
            var blend = blendFactor

            if useMaskedBlend, let maskedPSO = maskedBlendPSO, let gpuMask = maskGPUBuf {
                var protection = subjectProtection
                enc.setComputePipelineState(maskedPSO)
                enc.setBuffer(cnnOut, offset: 0, index: 0)
                enc.setBuffer(inputBuffer, offset: 0, index: 1)  // read from shared TF output
                enc.setBuffer(bayerOut, offset: 0, index: 2)
                enc.setBytes(&fullDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
                enc.setBytes(&cnnDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
                enc.setBytes(&blend, length: MemoryLayout<Float>.size, index: 5)
                enc.setBuffer(gpuMask, offset: 0, index: 6)
                enc.setBytes(&protection, length: MemoryLayout<Float>.size, index: 7)
            } else {
                enc.setComputePipelineState(blendPSO)
                enc.setBuffer(cnnOut, offset: 0, index: 0)
                enc.setBuffer(inputBuffer, offset: 0, index: 1)  // read from shared TF output
                enc.setBuffer(bayerOut, offset: 0, index: 2)
                enc.setBytes(&fullDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 3)
                enc.setBytes(&cnnDims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 4)
                enc.setBytes(&blend, length: MemoryLayout<Float>.size, index: 5)
            }

            enc.dispatchThreads(
                MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: threadgroupSize
            )
            enc.endEncoding()
        }

        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()

        // 6. Copy result back to the shared buffer (shared→shared, fast)
        memcpy(inputBuffer.contents(), bayerOut.contents(), bayerBytes)
    }
}
