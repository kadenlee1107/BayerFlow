import Foundation
import CoreML

/// CoreML-based FastDVDnet temporal denoiser.
/// Processes each Bayer sub-channel independently with tiled inference.
/// Uses MLShapedArray for safe, fast data transfer (no raw dataPointer on IOSurface memory).
nonisolated final class MLDenoiser: @unchecked Sendable {
    static let shared: MLDenoiser? = MLDenoiser()

    private let model: MLModel
    private let tileSize: Int = 1024
    private let overlap: Int = 128

    private init?() {
        guard let modelURL = Bundle.main.url(forResource: "FastDVDnet", withExtension: "mlmodelc") else {
            debugLog("MLDenoiser: FastDVDnet.mlmodelc not found in bundle")
            return nil
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all  // ANE/GPU/CPU — MLShapedArray handles memory safely

        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            debugLog("MLDenoiser: loaded FastDVDnet model (ANE/GPU/CPU)")
        } catch {
            debugLog("MLDenoiser: failed to load model: \(error)")
            return nil
        }
    }

    /// Denoise one Bayer sub-channel using 5-frame temporal window with tiled inference.
    func denoiseSubchannel(
        frames: [UnsafePointer<UInt16>],
        width: Int, height: Int,
        sigmaNorm: Float,
        output: UnsafeMutablePointer<UInt16>
    ) {
        guard frames.count == 5 else { return }

        let step = tileSize - overlap

        // Pad dimensions to tile grid
        let padH = (step - (height % step)) % step
        let padW = (step - (width % step)) % step
        let ph = height + padH
        let pw = width + padW

        // Convert all 5 frames to float32 [0, 1] with reflect padding
        var paddedFrames = [[Float]]()
        for f in 0..<5 {
            var padded = [Float](repeating: 0, count: ph * pw)
            let src = frames[f]
            for y in 0..<ph {
                for x in 0..<pw {
                    let sy = y < height ? y : max(0, 2 * height - y - 2)
                    let sx = x < width ? x : max(0, 2 * width - x - 2)
                    let clampY = min(height - 1, max(0, sy))
                    let clampX = min(width - 1, max(0, sx))
                    padded[y * pw + x] = Float(src[clampY * width + clampX]) / 65535.0
                }
            }
            paddedFrames.append(padded)
        }

        // Accumulation buffers
        var result = [Float](repeating: 0, count: ph * pw)
        var weight = [Float](repeating: 0, count: ph * pw)

        // Blending window (raised cosine)
        var winH = [Float](repeating: 1.0, count: tileSize)
        var winW = [Float](repeating: 1.0, count: tileSize)
        if overlap > 0 {
            for i in 0..<overlap {
                let t = Float(i) / Float(overlap)
                let w = 0.5 - 0.5 * cos(Float.pi * t)
                winH[i] = w
                winH[tileSize - 1 - i] = w
                winW[i] = w
                winW[tileSize - 1 - i] = w
            }
        }

        // Build tile positions (ensure edge coverage)
        func tilePositions(_ length: Int) -> [Int] {
            if length <= tileSize { return [0] }
            var positions = [Int]()
            var pos = 0
            while pos < length - tileSize {
                positions.append(pos)
                pos += step
            }
            positions.append(length - tileSize)
            return Array(Set(positions)).sorted()
        }

        let ys = tilePositions(ph)
        let xs = tilePositions(pw)

        let ts = tileSize

        // Build noise map as MLShapedArray (constant, created once)
        let noiseArray = MLShapedArray<Float>(repeating: sigmaNorm, shape: [1, 1, ts, ts])
        let noiseMultiArray = MLMultiArray(noiseArray)

        // Process each tile
        for y in ys {
            for x in xs {
                // Build input frames as flat Float array in [1, 5, ts, ts] row-major order
                var tileData = [Float](repeating: 0, count: 5 * ts * ts)
                for f in 0..<5 {
                    let fOffset = f * ts * ts
                    for ty in 0..<ts {
                        let srcRowOffset = (y + ty) * pw + x
                        let dstRowOffset = fOffset + ty * ts
                        for tx in 0..<ts {
                            tileData[dstRowOffset + tx] = paddedFrames[f][srcRowOffset + tx]
                        }
                    }
                }

                // Create MLShapedArray from our flat buffer, then convert to MLMultiArray
                let framesArray = MLShapedArray<Float>(scalars: tileData, shape: [1, 5, ts, ts])
                let framesMultiArray = MLMultiArray(framesArray)

                // Run inference
                let inputDict: [String: MLFeatureValue] = [
                    "frames": MLFeatureValue(multiArray: framesMultiArray),
                    "noise_map": MLFeatureValue(multiArray: noiseMultiArray),
                ]
                guard let provider = try? MLDictionaryFeatureProvider(dictionary: inputDict),
                      let prediction = try? model.prediction(from: provider),
                      let outputMultiArray = prediction.featureValue(for: "denoised")?.multiArrayValue else { continue }

                // Copy output into MLShapedArray (safe copy from IOSurface → Swift memory)
                let outputShaped = MLShapedArray<Float>(outputMultiArray)

                // Read output via withUnsafeShapedBufferPointer (fast pointer access on Swift-owned memory)
                outputShaped.withUnsafeShapedBufferPointer { buffer, _, strides in
                    let s2 = strides[2]
                    let s3 = strides[3]
                    for ty in 0..<ts {
                        let wH = winH[ty]
                        let outRowOffset = ty * s2
                        let resultRowOffset = (y + ty) * pw + x
                        for tx in 0..<ts {
                            let val = buffer[outRowOffset + tx * s3]
                            let w = wH * winW[tx]
                            result[resultRowOffset + tx] += val * w
                            weight[resultRowOffset + tx] += w
                        }
                    }
                }
            }
        }

        // Normalize and convert back to uint16
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * pw + x
                var val = weight[idx] > 0 ? result[idx] / weight[idx] : 0
                val = max(0, min(1, val))
                output[y * width + x] = UInt16(val * 65535.0 + 0.5)
            }
        }
    }
}
