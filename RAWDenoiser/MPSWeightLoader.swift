import Foundation

// MARK: - Restormer-Lite weight loader

/// Tensor types in the RSTL binary format.
enum RestormerTensorType: Int {
    case convWeight = 0      // OIHW layout
    case convBias = 1        // 1D
    case dwConvWeight = 2    // [1, C, kH, kW] (MPS layout)
    case dwConvBias = 3      // 1D
    case lnGamma = 4         // [1, C, 1, 1]
    case lnBeta = 5          // [1, C, 1, 1]
    case temperature = 6     // [heads, 1, 1]
}

/// Single tensor from the RSTL binary.
struct RestormerTensor {
    let type: RestormerTensorType
    let shape: [Int]
    let data: Data           // float32
    var count: Int { shape.reduce(1, *) }
}

/// Parsed Restormer-Lite weight file.
struct RestormerWeights {
    let version: Int
    let inChannels: Int
    let outChannels: Int
    let numLevels: Int
    let numBlocks: Int
    let tensors: [RestormerTensor]
}

/// Parse the RSTL binary .bin file exported by export_restormer_weights.py.
nonisolated func loadRestormerWeights(from url: URL) -> RestormerWeights? {
    guard let data = try? Data(contentsOf: url) else {
        debugLog("MPSWeightLoader: cannot read \(url.path)")
        return nil
    }

    var offset = 0

    func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let val = data.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: offset, as: UInt32.self)
        }
        offset += 4
        return val
    }

    func readFloatData(count: Int) -> Data? {
        let byteCount = count * MemoryLayout<Float>.size
        guard offset + byteCount <= data.count else { return nil }
        let slice = data[offset..<(offset + byteCount)]
        offset += byteCount
        return Data(slice)
    }

    // Header (32 bytes)
    guard offset + 4 <= data.count else { return nil }
    let magic = data[offset..<(offset + 4)]
    offset += 4
    guard String(data: magic, encoding: .ascii) == "RSTL" else {
        debugLog("MPSWeightLoader: bad magic (expected RSTL)")
        return nil
    }

    guard let version = readUInt32(), version == 1,
          let inCh = readUInt32(),
          let outCh = readUInt32(),
          let numLevels = readUInt32(),
          let numBlocks = readUInt32(),
          let numTensors = readUInt32(),
          let _ = readUInt32() /* reserved */ else {
        debugLog("MPSWeightLoader: bad RSTL header")
        return nil
    }

    var tensors: [RestormerTensor] = []

    for i in 0..<Int(numTensors) {
        guard let rawType = readUInt32(),
              let ttype = RestormerTensorType(rawValue: Int(rawType)) else {
            debugLog("MPSWeightLoader: RSTL tensor \(i) bad type")
            return nil
        }

        guard let ndims = readUInt32() else { return nil }
        var shape: [Int] = []
        for _ in 0..<Int(ndims) {
            guard let dim = readUInt32() else { return nil }
            shape.append(Int(dim))
        }

        guard let dataCount = readUInt32() else { return nil }
        guard let tensorData = readFloatData(count: Int(dataCount)) else {
            debugLog("MPSWeightLoader: RSTL tensor \(i) truncated data")
            return nil
        }

        tensors.append(RestormerTensor(
            type: ttype,
            shape: shape,
            data: tensorData
        ))
    }

    debugLog("MPSWeightLoader: loaded Restormer-Lite v\(version) "
          + "(\(inCh)→\(outCh)ch, \(numLevels) levels, \(numBlocks) blocks, \(tensors.count) tensors)")

    return RestormerWeights(
        version: Int(version),
        inChannels: Int(inCh),
        outChannels: Int(outCh),
        numLevels: Int(numLevels),
        numBlocks: Int(numBlocks),
        tensors: tensors
    )
}


// MARK: - DnCNN weight loader

/// Parsed layer from the binary weight file.
struct DnCNNLayer {
    let weight: Data    // float32, OIHW layout
    let bias: Data      // float32
    let outChannels: Int
    let inChannels: Int
    let kernelH: Int
    let kernelW: Int
}

/// Parsed DnCNN weight file header + layers.
struct DnCNNWeights {
    let inChannels: Int
    let hiddenChannels: Int
    let numLayers: Int
    let layers: [DnCNNLayer]
}

/// Parse the binary .bin file exported by export_mps_weights.py.
///
/// Format:
///   Header: "DCNN" + version(u32) + in_channels(u32) + hidden_channels(u32) + num_layers(u32)
///   Per layer: weight_count(u32) + weights(float32[]) + bias_count(u32) + biases(float32[])
nonisolated func loadDnCNNWeights(from url: URL) -> DnCNNWeights? {
    guard let data = try? Data(contentsOf: url) else {
        debugLog("MPSWeightLoader: cannot read \(url.path)")
        return nil
    }

    var offset = 0

    func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let val = data.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: offset, as: UInt32.self)
        }
        offset += 4
        return val
    }

    func readFloatData(count: Int) -> Data? {
        let byteCount = count * MemoryLayout<Float>.size
        guard offset + byteCount <= data.count else { return nil }
        let slice = data[offset..<(offset + byteCount)]
        offset += byteCount
        return Data(slice)
    }

    // Header
    guard offset + 4 <= data.count else { return nil }
    let magic = data[offset..<(offset + 4)]
    offset += 4
    guard String(data: magic, encoding: .ascii) == "DCNN" else {
        debugLog("MPSWeightLoader: bad magic")
        return nil
    }

    guard let version = readUInt32(), version == 1,
          let inCh = readUInt32(),
          let hiddenCh = readUInt32(),
          let numLayers = readUInt32() else {
        debugLog("MPSWeightLoader: bad header")
        return nil
    }

    var layers: [DnCNNLayer] = []

    for i in 0..<Int(numLayers) {
        guard let wCount = readUInt32() else { return nil }
        guard let wData = readFloatData(count: Int(wCount)) else { return nil }
        guard let bCount = readUInt32() else { return nil }
        guard let bData = readFloatData(count: Int(bCount)) else { return nil }

        // Infer dimensions: weight is OIHW with 3×3 kernel
        let outCh = Int(bCount)
        let inCh_layer: Int
        if i == 0 {
            inCh_layer = Int(inCh)
        } else if i == Int(numLayers) - 1 {
            inCh_layer = Int(hiddenCh)
        } else {
            inCh_layer = Int(hiddenCh)
        }

        layers.append(DnCNNLayer(
            weight: wData,
            bias: bData,
            outChannels: outCh,
            inChannels: inCh_layer,
            kernelH: 3,
            kernelW: 3
        ))
    }

    debugLog("MPSWeightLoader: loaded \(layers.count) layers "
          + "(\(inCh)→\(hiddenCh)ch, \(numLayers) layers)")

    return DnCNNWeights(
        inChannels: Int(inCh),
        hiddenChannels: Int(hiddenCh),
        numLayers: Int(numLayers),
        layers: layers
    )
}


// MARK: - UNet-Lite weight loader

/// Parsed UNet conv layer.
struct UNetLayer {
    let weight: Data    // float32, OIHW layout
    let bias: Data      // float32
    let outChannels: Int
    let inChannels: Int
}

/// Parsed UNet-Lite weight file.
struct UNetWeights {
    let version: Int      // 1 = bilinear, 2 = pixel shuffle
    let inChannels: Int   // 5 (4 Bayer + 1 noise map)
    let outChannels: Int  // 4 (noise residuals per Bayer channel)
    let numLevels: Int    // 2 (encoder/decoder depth)
    let layers: [UNetLayer]  // 10 conv layers in fixed order
}

/// Parse the UNet binary .bin file exported by export_unet_weights.py.
///
/// Format:
///   Header: "UNET" + version(u32) + in_channels(u32) + out_channels(u32) + num_levels(u32)
///   Per layer (10 total): weight_count(u32) + weights(float32[]) + bias_count(u32) + biases(float32[])
///
/// Layer order (architecture hardcoded, same as Python):
///   v1 (bilinear): bot_b=128→64,  dec1_b=64→32
///   v2 (pixel shuffle): bot_b=128→256, dec1_b=64→128  (4x channels for PixelShuffle(2))
nonisolated func loadUNetWeights(from url: URL) -> UNetWeights? {
    guard let data = try? Data(contentsOf: url) else {
        debugLog("MPSWeightLoader: cannot read \(url.path)")
        return nil
    }

    var offset = 0

    func readUInt32() -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let val = data.withUnsafeBytes { ptr -> UInt32 in
            ptr.load(fromByteOffset: offset, as: UInt32.self)
        }
        offset += 4
        return val
    }

    func readFloatData(count: Int) -> Data? {
        let byteCount = count * MemoryLayout<Float>.size
        guard offset + byteCount <= data.count else { return nil }
        let slice = data[offset..<(offset + byteCount)]
        offset += byteCount
        return Data(slice)
    }

    // Header
    guard offset + 4 <= data.count else { return nil }
    let magic = data[offset..<(offset + 4)]
    offset += 4
    guard String(data: magic, encoding: .ascii) == "UNET" else {
        debugLog("MPSWeightLoader: bad magic (expected UNET)")
        return nil
    }

    guard let version = readUInt32(), (version == 1 || version == 2 || version == 3),
          let inCh = readUInt32(),
          let outCh = readUInt32(),
          let numLevels = readUInt32() else {
        debugLog("MPSWeightLoader: bad UNET header")
        return nil
    }

    // Layer dimensions depend on version:
    // v1 (bilinear upsample): bot_b=128→64, dec1_b=64→32
    // v2 (pixel shuffle): bot_b=128→256 (64*4), dec1_b=64→128 (32*4)
    // v3 (3-level pixel shuffle): 14 layers, 32→64→128→256→512
    let layerDims: [(inCh: Int, outCh: Int)]
    if version == 3 {
        layerDims = [
            (Int(inCh), 32),   // enc0_a
            (32, 32),          // enc0_b
            (32, 64),          // enc1_a
            (64, 64),          // enc1_b
            (64, 128),         // enc2_a
            (128, 128),        // enc2_b
            (128, 256),        // bot_a
            (256, 512),        // bot_b → PixelShuffle(2) → 128ch
            (256, 128),        // dec2_a (128 shuffle + 128 skip = 256)
            (128, 256),        // dec2_b → PixelShuffle(2) → 64ch
            (128, 64),         // dec1_a (64 shuffle + 64 skip = 128)
            (64, 128),         // dec1_b → PixelShuffle(2) → 32ch
            (64, 32),          // dec0_a (32 shuffle + 32 skip = 64)
            (32, Int(outCh)),  // dec0_b
        ]
        debugLog("MPSWeightLoader: UNET v3 (3-level pixel shuffle)")
    } else if version == 2 {
        layerDims = [
            (Int(inCh), 32),   // enc0_a
            (32, 32),          // enc0_b
            (32, 64),          // enc1_a
            (64, 64),          // enc1_b
            (64, 128),         // bot_a
            (128, 256),        // bot_b → PixelShuffle(2) → 64ch
            (128, 64),         // dec1_a (64 shuffle + 64 skip = 128)
            (64, 128),         // dec1_b → PixelShuffle(2) → 32ch
            (64, 32),          // dec0_a (32 shuffle + 32 skip = 64)
            (32, Int(outCh)),  // dec0_b
        ]
        debugLog("MPSWeightLoader: UNET v2 (pixel shuffle)")
    } else {
        layerDims = [
            (Int(inCh), 32),   // enc0_a
            (32, 32),          // enc0_b
            (32, 64),          // enc1_a
            (64, 64),          // enc1_b
            (64, 128),         // bot_a
            (128, 64),         // bot_b → bilinear 2x
            (128, 64),         // dec1_a (64 upsample + 64 skip)
            (64, 32),          // dec1_b → bilinear 2x
            (64, 32),          // dec0_a (32 upsample + 32 skip)
            (32, Int(outCh)),  // dec0_b
        ]
        debugLog("MPSWeightLoader: UNET v1 (bilinear)")
    }

    var layers: [UNetLayer] = []

    for i in 0..<layerDims.count {
        guard let wCount = readUInt32() else { return nil }
        guard let wData = readFloatData(count: Int(wCount)) else { return nil }
        guard let bCount = readUInt32() else { return nil }
        guard let bData = readFloatData(count: Int(bCount)) else { return nil }

        let expectedOut = layerDims[i].outCh
        let expectedIn = layerDims[i].inCh

        guard Int(bCount) == expectedOut else {
            debugLog("MPSWeightLoader: UNET layer \(i) bias count \(bCount) != expected \(expectedOut)")
            return nil
        }

        let expectedW = expectedOut * expectedIn * 3 * 3
        guard Int(wCount) == expectedW else {
            debugLog("MPSWeightLoader: UNET layer \(i) weight count \(wCount) != expected \(expectedW)")
            return nil
        }

        layers.append(UNetLayer(
            weight: wData,
            bias: bData,
            outChannels: expectedOut,
            inChannels: expectedIn
        ))
    }

    debugLog("MPSWeightLoader: loaded UNet-Lite (\(inCh)→\(outCh)ch, "
          + "\(numLevels) levels, \(layers.count) conv layers)")

    return UNetWeights(
        version: Int(version),
        inChannels: Int(inCh),
        outChannels: Int(outCh),
        numLevels: Int(numLevels),
        layers: layers
    )
}
