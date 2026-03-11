import Foundation
import Accelerate

// MARK: - C-callable bridge for CoreML ResidualDnCNN post-filter
// Called from denoise_core.c after bilateral temporal filter + spatial denoise.

/// CNN post-filter. Extracts 4 Bayer sub-channels, runs ResidualDnCNN on each,
/// repacks to Bayer output with 70/30 blend.
@_cdecl("postfilter_frame_cnn")
func postfilter_frame_cnn(
    _ output: UnsafeMutablePointer<UInt16>,
    _ input: UnsafePointer<UInt16>,
    _ width: Int32,
    _ height: Int32
) {
    guard let denoiser = PostFilterDenoiser.shared else {
        debugLog("PostFilter: model not available, passing through")
        memcpy(output, input, Int(width) * Int(height) * MemoryLayout<UInt16>.size)
        return
    }

    let pfStart = CFAbsoluteTimeGetCurrent()
    let w = Int(width)
    let h = Int(height)
    let subW = w / 2
    let subH = h / 2
    let subPixels = subW * subH

    let chanNames = ["R", "Gr", "Gb", "B"]

    for comp in 0..<4 {
        let dy = (comp >> 1) & 1
        let dx = comp & 1

        // Extract sub-channel (raw pointer loop — strided gather)
        let subInput = UnsafeMutablePointer<UInt16>.allocate(capacity: subPixels)
        for y in 0..<subH {
            let srcRow = input + (y * 2 + dy) * w + dx
            let dstRow = subInput + y * subW
            for x in 0..<subW {
                dstRow[x] = srcRow[x * 2]
            }
        }

        // Run CNN
        let subOutput = UnsafeMutablePointer<UInt16>.allocate(capacity: subPixels)
        denoiser.denoiseSubchannel(
            input: UnsafePointer(subInput),
            width: subW, height: subH,
            output: subOutput
        )

        // Blend CNN output with bilateral input using vDSP (70% CNN, 30% bilateral)
        let cnnFloat = UnsafeMutablePointer<Float>.allocate(capacity: subPixels)
        let origFloat = UnsafeMutablePointer<Float>.allocate(capacity: subPixels)
        let blendResult = UnsafeMutablePointer<Float>.allocate(capacity: subPixels)

        vDSP_vfltu16(subOutput, 1, cnnFloat, 1, vDSP_Length(subPixels))
        vDSP_vfltu16(subInput, 1, origFloat, 1, vDSP_Length(subPixels))

        // blendResult = 0.7 * cnn + 0.3 * orig
        var blendCNN: Float = 0.7
        var blendOrig: Float = 0.3
        vDSP_vsmul(cnnFloat, 1, &blendCNN, cnnFloat, 1, vDSP_Length(subPixels))
        vDSP_vsmul(origFloat, 1, &blendOrig, origFloat, 1, vDSP_Length(subPixels))
        vDSP_vadd(cnnFloat, 1, origFloat, 1, blendResult, 1, vDSP_Length(subPixels))

        // Clamp to [0, 65535] and add 0.5 for rounding
        var lo: Float = 0, hi: Float = 65535
        vDSP_vclip(blendResult, 1, &lo, &hi, blendResult, 1, vDSP_Length(subPixels))
        var half: Float = 0.5
        vDSP_vsadd(blendResult, 1, &half, blendResult, 1, vDSP_Length(subPixels))

        // Convert back to uint16
        let blendU16 = UnsafeMutablePointer<UInt16>.allocate(capacity: subPixels)
        vDSP_vfixu16(blendResult, 1, blendU16, 1, vDSP_Length(subPixels))

        // Scatter back to Bayer positions (strided write)
        for y in 0..<subH {
            let srcRow = blendU16 + y * subW
            let dstRow = output + (y * 2 + dy) * w + dx
            for x in 0..<subW {
                dstRow[x * 2] = srcRow[x]
            }
        }

        subInput.deallocate()
        subOutput.deallocate()
        cnnFloat.deallocate()
        origFloat.deallocate()
        blendResult.deallocate()
        blendU16.deallocate()

        debugLog("PostFilter: \(chanNames[comp]) done — "
              + "\(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - pfStart))s elapsed")
    }

    debugLog("PostFilter: complete — "
          + "\(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - pfStart))s total")
}

/// Returns 1 if the CoreML PostFilterDnCNN model is available, 0 otherwise.
@_cdecl("cnn_postfilter_available")
func cnn_postfilter_available() -> Int32 {
    return PostFilterDenoiser.shared != nil ? 1 : 0
}
