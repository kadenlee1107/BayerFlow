import Foundation

// MARK: - C-callable bridge for CoreML FastDVDnet denoiser
// Called from denoise_core.c as an alternative to the bilateral temporal filter.

/// ML-based temporal denoiser. Extracts 4 Bayer sub-channels from each frame,
/// runs FastDVDnet on each sub-channel independently, then repacks to Bayer output.
@_cdecl("denoise_frame_ml")
func denoise_frame_ml(
    _ output: UnsafeMutablePointer<UInt16>,
    _ frames: UnsafeMutablePointer<UnsafePointer<UInt16>?>,
    _ numFrames: Int32,
    _ centerIdx: Int32,
    _ width: Int32,
    _ height: Int32,
    _ noiseSigma: Float
) {
    guard let denoiser = MLDenoiser.shared else {
        // Fallback: copy center frame unchanged
        debugLog("ML: model not available, copying center frame as-is")
        if let center = frames[Int(centerIdx)] {
            memcpy(output, center, Int(width) * Int(height) * MemoryLayout<UInt16>.size)
        }
        return
    }
    debugLog("ML: starting inference — \(width)x\(height), \(numFrames) frames, sigma=\(noiseSigma), sigmaNorm=\(noiseSigma / 65535.0)")
    let mlStart = CFAbsoluteTimeGetCurrent()

    let w = Int(width)
    let h = Int(height)
    let subW = w / 2
    let subH = h / 2
    let subPixels = subW * subH
    let sigmaNorm = noiseSigma / 65535.0

    // Build the 5-frame window centered on centerIdx
    // If fewer than 5 frames available, mirror edges
    var windowIndices = [Int]()
    let center = Int(centerIdx)
    let nf = Int(numFrames)
    for i in (center - 2)...(center + 2) {
        let clamped = max(0, min(nf - 1, i))
        windowIndices.append(clamped)
    }

    // Process each Bayer sub-channel: 0=R(0,0), 1=Gr(0,1), 2=Gb(1,0), 3=B(1,1)
    for comp in 0..<4 {
        let dy = (comp >> 1) & 1
        let dx = comp & 1

        // Extract sub-channel from each of the 5 window frames
        var subFrames = [UnsafeMutablePointer<UInt16>]()
        for fi in 0..<5 {
            let frameIdx = windowIndices[fi]
            guard let frame = frames[frameIdx] else {
                // Shouldn't happen, but safety
                let buf = UnsafeMutablePointer<UInt16>.allocate(capacity: subPixels)
                buf.initialize(repeating: 0, count: subPixels)
                subFrames.append(buf)
                continue
            }

            let buf = UnsafeMutablePointer<UInt16>.allocate(capacity: subPixels)
            for y in 0..<subH {
                for x in 0..<subW {
                    buf[y * subW + x] = frame[(y * 2 + dy) * w + (x * 2 + dx)]
                }
            }
            subFrames.append(buf)
        }

        // Run ML denoising on this sub-channel
        let subOutput = UnsafeMutablePointer<UInt16>.allocate(capacity: subPixels)
        let subFramePtrs = subFrames.map { UnsafePointer($0) }
        denoiser.denoiseSubchannel(
            frames: subFramePtrs,
            width: subW, height: subH,
            sigmaNorm: sigmaNorm,
            output: subOutput
        )

        // Repack into Bayer output
        for y in 0..<subH {
            for x in 0..<subW {
                output[(y * 2 + dy) * w + (x * 2 + dx)] = subOutput[y * subW + x]
            }
        }

        // Cleanup
        subOutput.deallocate()
        for buf in subFrames {
            buf.deallocate()
        }

        let chanNames = ["R", "Gr", "Gb", "B"]
        debugLog("ML: channel \(chanNames[comp]) done — \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - mlStart))s elapsed")
    }
    debugLog("ML: inference complete — \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - mlStart))s total")
}

/// Returns 1 if the CoreML FastDVDnet model is available, 0 otherwise.
@_cdecl("ml_denoiser_available")
func ml_denoiser_available() -> Int32 {
    return MLDenoiser.shared != nil ? 1 : 0
}
