import Foundation
import CoreML
import Accelerate

/// CoreML-based ResidualDnCNN post-filter.
/// Removes residual noise left by the bilateral temporal filter.
/// Runs CNN at half resolution for speed, applies noise residual at full res.
nonisolated final class PostFilterDenoiser: @unchecked Sendable {
    static let shared: PostFilterDenoiser? = PostFilterDenoiser()

    private let model: MLModel

    private init?() {
        guard let modelURL = Bundle.main.url(forResource: "PostFilterDnCNN",
                                              withExtension: "mlmodelc") else {
            debugLog("PostFilterDenoiser: PostFilterDnCNN.mlmodelc not found in bundle")
            return nil
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            debugLog("PostFilterDenoiser: loaded PostFilterDnCNN model")
        } catch {
            debugLog("PostFilterDenoiser: failed to load model: \(error)")
            return nil
        }
    }

    /// Denoise one Bayer sub-channel using half-resolution CNN.
    func denoiseSubchannel(
        input: UnsafePointer<UInt16>,
        width: Int, height: Int,
        output: UnsafeMutablePointer<UInt16>
    ) {
        let t0 = CFAbsoluteTimeGetCurrent()

        let fullPixels = width * height
        let halfW = width / 2
        let halfH = height / 2
        let halfPixels = halfW * halfH
        let alignTo = 64
        let ph = ((halfH + alignTo - 1) / alignTo) * alignTo
        let pw = ((halfW + alignTo - 1) / alignTo) * alignTo

        // 1. Convert full-res uint16 → float [0,1] using vDSP (vectorized)
        let fullFloat = UnsafeMutablePointer<Float>.allocate(capacity: fullPixels)
        vDSP_vfltu16(input, 1, fullFloat, 1, vDSP_Length(fullPixels))
        var invScale: Float = 1.0 / 65535.0
        vDSP_vsmul(fullFloat, 1, &invScale, fullFloat, 1, vDSP_Length(fullPixels))

        // 2. Downsample: 2×2 box filter (raw pointer loops)
        let inputData = UnsafeMutablePointer<Float>.allocate(capacity: ph * pw)
        memset(inputData, 0, ph * pw * MemoryLayout<Float>.size)

        for y in 0..<halfH {
            let y0 = y * 2
            let y1 = min(y0 + 1, height - 1)
            let row0 = fullFloat + y0 * width
            let row1 = fullFloat + y1 * width
            let dst = inputData + y * pw
            for x in 0..<halfW {
                let x0 = x * 2
                let x1 = x0 + 1  // halfW = width/2, so x0+1 < width
                dst[x] = (row0[x0] + row0[x1] + row1[x0] + row1[x1]) * 0.25
            }
        }
        // Clamp-pad rows
        if ph > halfH {
            let lastRow = inputData + (halfH - 1) * pw
            for y in halfH..<ph {
                memcpy(inputData + y * pw, lastRow, pw * MemoryLayout<Float>.size)
            }
        }
        // Clamp-pad cols
        for y in 0..<ph {
            let row = inputData + y * pw
            let lastVal = row[halfW - 1]
            for x in halfW..<pw {
                row[x] = lastVal
            }
        }

        let t1 = CFAbsoluteTimeGetCurrent()

        // 3. Run CNN at half-res
        let inputArray = MLShapedArray<Float>(
            scalars: UnsafeBufferPointer(start: inputData, count: ph * pw),
            shape: [1, 1, ph, pw])
        let inputMultiArray = MLMultiArray(inputArray)

        let inputDict: [String: MLFeatureValue] = [
            "input": MLFeatureValue(multiArray: inputMultiArray),
        ]
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: inputDict),
              let prediction = try? model.prediction(from: provider),
              let outputMultiArray = prediction.featureValue(for: "output")?.multiArrayValue
        else {
            memcpy(output, input, fullPixels * MemoryLayout<UInt16>.size)
            fullFloat.deallocate()
            inputData.deallocate()
            return
        }

        let t2 = CFAbsoluteTimeGetCurrent()

        // 4. Compute half-res residual + nearest-neighbor upsample + subtract
        let outputShaped = MLShapedArray<Float>(outputMultiArray)
        let residual = UnsafeMutablePointer<Float>.allocate(capacity: halfPixels)

        outputShaped.withUnsafeShapedBufferPointer { buffer, _, strides in
            let s2 = strides[2]
            let s3 = strides[3]

            // Compute residual = half_input - denoised
            var sumRes: Double = 0
            var sumResSq: Double = 0
            var sumInp: Double = 0
            var maxAbsRes: Float = 0
            for y in 0..<halfH {
                let srcRow = y * s2
                let inpRow = y * pw
                let resRow = y * halfW
                for x in 0..<halfW {
                    var d = buffer[srcRow + x * s3]
                    if d < 0 { d = 0 } else if d > 1 { d = 1 }
                    let r = inputData[inpRow + x] - d
                    residual[resRow + x] = r
                    sumRes += Double(r)
                    sumResSq += Double(r) * Double(r)
                    sumInp += Double(inputData[inpRow + x])
                    let ar = r < 0 ? -r : r
                    if ar > maxAbsRes { maxAbsRes = ar }
                }
            }
            let n = Double(halfW * halfH)
            let meanRes = sumRes / n
            let stdRes = sqrt(sumResSq / n - meanRes * meanRes)
            let meanInp = sumInp / n
            debugLog(String(format: "  DIAG: noise_pred std=%.6f mean=%.6f max=%.6f input_mean=%.4f (%dx%d)",
                         stdRes, meanRes, maxAbsRes, meanInp, halfW, halfH))
        }

        let t3 = CFAbsoluteTimeGetCurrent()

        // 5. Upsample residual (nearest-neighbor) + subtract from full-res + convert to uint16
        // Work row by row with raw pointers for maximum speed
        let resultFloat = UnsafeMutablePointer<Float>.allocate(capacity: fullPixels)

        for y in 0..<height {
            let sy = y >> 1
            let resRow = residual + sy * halfW
            let srcRow = fullFloat + y * width
            let dstRow = resultFloat + y * width
            for x in 0..<width {
                dstRow[x] = srcRow[x] - resRow[x >> 1]
            }
        }

        // Clamp [0, 1] and convert to uint16 using vDSP
        var lo: Float = 0, hi: Float = 1
        vDSP_vclip(resultFloat, 1, &lo, &hi, resultFloat, 1, vDSP_Length(fullPixels))
        var scale65535: Float = 65535.0
        vDSP_vsmul(resultFloat, 1, &scale65535, resultFloat, 1, vDSP_Length(fullPixels))
        vDSP_vfixu16(resultFloat, 1, output, 1, vDSP_Length(fullPixels))

        let t4 = CFAbsoluteTimeGetCurrent()

        resultFloat.deallocate()
        residual.deallocate()
        inputData.deallocate()
        fullFloat.deallocate()

        debugLog(String(format: "  PostFilter detail: prep=%.3fs predict=%.3fs residual=%.3fs output=%.3fs",
                     t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    }
}
