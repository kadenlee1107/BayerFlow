import Foundation
import Vision
import CoreVideo
import Accelerate

// MARK: - Person Segmentation via Apple Vision

/// Generates a person segmentation mask from a Bayer frame using Apple's Vision framework.
/// Returns a float mask at half-resolution (cnnW × cnnH) where 1.0 = person, 0.0 = background.
/// The mask is feathered at edges for smooth blending.
final class PersonSegmentor {
    static let shared = PersonSegmentor()

    /// Cached mask buffer (reused across frames)
    private var maskBuffer: UnsafeMutablePointer<Float>?
    private var maskCount: Int = 0
    private var maskWidth: Int = 0
    private var maskHeight: Int = 0

    /// Generate person segmentation mask from Bayer uint16 frame.
    /// Returns pointer to float mask at (cnnW × cnnH) resolution, or nil if no person detected.
    /// Mask values: 0.0 = background (full denoise), 1.0 = person (reduced denoise).
    func segment(bayer: UnsafePointer<UInt16>, width: Int, height: Int) -> UnsafeMutablePointer<Float>? {
        let cnnW = width / 2
        let cnnH = height / 2

        // Ensure mask buffer
        let needed = cnnW * cnnH
        if needed != maskCount {
            maskBuffer?.deallocate()
            maskBuffer = UnsafeMutablePointer<Float>.allocate(capacity: needed)
            maskCount = needed
            maskWidth = cnnW
            maskHeight = cnnH
        }

        // Create grayscale CVPixelBuffer from green channel (average Gr+Gb)
        // Use 8-bit grayscale — sufficient for person segmentation
        guard let pixelBuffer = makeGrayscaleBuffer(bayer: bayer, width: width, height: height,
                                                     outW: cnnW, outH: cnnH) else {
            return nil
        }

        // Run person segmentation
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced  // good tradeoff: ~5ms on ANE
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let result = request.results?.first else {
            // No results = no person detected, fill with zeros
            maskBuffer?.initialize(repeating: 0.0, count: maskCount)
            return maskBuffer
        }

        let segBuffer = result.pixelBuffer

        // Convert segmentation mask to float and resize to cnnW × cnnH
        CVPixelBufferLockBaseAddress(segBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(segBuffer, .readOnly) }

        let segW = CVPixelBufferGetWidth(segBuffer)
        let segH = CVPixelBufferGetHeight(segBuffer)
        let segBPR = CVPixelBufferGetBytesPerRow(segBuffer)
        guard let segBase = CVPixelBufferGetBaseAddress(segBuffer) else {
            maskBuffer?.initialize(repeating: 0.0, count: maskCount)
            return maskBuffer
        }

        let segPtr = segBase.assumingMemoryBound(to: UInt8.self)

        // Bilinear resize from segmentation output to cnnW × cnnH
        guard let mask = maskBuffer else { return nil }

        for y in 0..<cnnH {
            let srcY = Float(y) * Float(segH) / Float(cnnH)
            let sy0 = min(Int(srcY), segH - 1)
            let sy1 = min(sy0 + 1, segH - 1)
            let fy = srcY - Float(sy0)

            for x in 0..<cnnW {
                let srcX = Float(x) * Float(segW) / Float(cnnW)
                let sx0 = min(Int(srcX), segW - 1)
                let sx1 = min(sx0 + 1, segW - 1)
                let fx = srcX - Float(sx0)

                // Bilinear interpolation
                let v00 = Float(segPtr[sy0 * segBPR + sx0]) / 255.0
                let v10 = Float(segPtr[sy0 * segBPR + sx1]) / 255.0
                let v01 = Float(segPtr[sy1 * segBPR + sx0]) / 255.0
                let v11 = Float(segPtr[sy1 * segBPR + sx1]) / 255.0

                let top = v00 * (1 - fx) + v10 * fx
                let bot = v01 * (1 - fx) + v11 * fx
                mask[y * cnnW + x] = top * (1 - fy) + bot * fy
            }
        }

        // Feather edges with a 5×5 box blur for smooth transitions
        boxBlur(mask: mask, width: cnnW, height: cnnH, radius: 2)

        return mask
    }

    /// Create 8-bit grayscale CVPixelBuffer by averaging 2×2 Bayer blocks.
    private func makeGrayscaleBuffer(bayer: UnsafePointer<UInt16>, width: Int, height: Int,
                                      outW: Int, outH: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let ret = CVPixelBufferCreate(kCFAllocatorDefault, outW, outH,
                                       kCVPixelFormatType_OneComponent8,
                                       attrs as CFDictionary, &pb)
        guard ret == kCVReturnSuccess, let pixelBuffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let dst = base.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)

        for y in 0..<outH {
            let by = y * 2
            for x in 0..<outW {
                let bx = x * 2
                // Average 2×2 Bayer block → grayscale
                let idx = by * width + bx
                let sum = Int(bayer[idx]) + Int(bayer[idx + 1]) +
                          Int(bayer[idx + width]) + Int(bayer[idx + width + 1])
                // Convert from 16-bit to 8-bit
                dst[y * bpr + x] = UInt8(min(255, sum >> 10))  // /4 then >>8
            }
        }

        return pixelBuffer
    }

    /// In-place box blur for feathering mask edges.
    private func boxBlur(mask: UnsafeMutablePointer<Float>, width: Int, height: Int, radius: Int) {
        let count = width * height
        let temp = UnsafeMutablePointer<Float>.allocate(capacity: count)
        defer { temp.deallocate() }

        let kernelSize = 2 * radius + 1
        let invK = 1.0 / Float(kernelSize)

        // Horizontal pass
        for y in 0..<height {
            var sum: Float = 0
            let row = y * width
            // Initialize window
            for k in 0...radius { sum += mask[row + k] }
            for k in 1...radius { sum += mask[row] }  // clamp left edge

            for x in 0..<width {
                temp[row + x] = sum * invK
                let addX = min(x + radius + 1, width - 1)
                let subX = max(x - radius, 0)
                sum += mask[row + addX] - mask[row + subX]
            }
        }

        // Vertical pass
        for x in 0..<width {
            var sum: Float = 0
            for k in 0...radius { sum += temp[k * width + x] }
            for k in 1...radius { sum += temp[x] }

            for y in 0..<height {
                mask[y * width + x] = sum * invK
                let addY = min(y + radius + 1, height - 1)
                let subY = max(y - radius, 0)
                sum += temp[addY * width + x] - temp[subY * width + x]
            }
        }
    }
}
