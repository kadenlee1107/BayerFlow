/*
 * ProRes 4444 Writer
 *
 * Writes denoised RGB frames to ProRes 4444 XQ via AVAssetWriter.
 * Used for formats that output debayered RGB (e.g., RED R3D).
 *
 * Called from C (denoise_core.c) via @_cdecl bridge functions.
 */

import AVFoundation
import CoreVideo

class ProRes444Writer {
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var width: Int
    private var height: Int
    private var frameIndex: Int = 0
    private var fps: Double = 24.0

    init?(outputPath: String, width: Int, height: Int, fps: Double = 24.0) {
        self.width = width
        self.height = height
        self.fps = fps

        let url = URL(fileURLWithPath: outputPath)

        // Remove existing file
        try? FileManager.default.removeItem(at: url)

        do {
            assetWriter = try AVAssetWriter(outputURL: url, fileType: .mov)
        } catch {
            fputs("ProRes444Writer: failed to create AVAssetWriter: \(error)\n", stderr)
            return nil
        }

        // Video output settings for ProRes 4444 XQ
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.proRes4444,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_P3_D65,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_Linear,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020
            ] as [String: Any]
        ]

        videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: videoSettings
        )
        videoInput?.expectsMediaDataInRealTime = false

        // Pixel buffer adaptor for 16-bit RGBA
        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBALE,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]

        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput!,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes
        )

        assetWriter?.add(videoInput!)

        guard assetWriter?.startWriting() == true else {
            fputs("ProRes444Writer: startWriting failed: \(assetWriter?.error?.localizedDescription ?? "unknown")\n", stderr)
            return nil
        }
        assetWriter?.startSession(atSourceTime: .zero)

        fputs("ProRes444Writer: initialized \(width)×\(height) ProRes 4444 → \(outputPath)\n", stderr)
    }

    /// Write a frame from planar 16-bit RGB data.
    /// rgb_planar: [R plane: w*h] [G plane: w*h] [B plane: w*h], each uint16
    func writeFrame(rgbPlanar: UnsafePointer<UInt16>) -> Bool {
        guard let adaptor = pixelBufferAdaptor,
              let pool = adaptor.pixelBufferPool else {
            fputs("ProRes444Writer: no pixel buffer pool\n", stderr)
            return false
        }

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
        guard status == kCVReturnSuccess, let pb = pixelBuffer else {
            fputs("ProRes444Writer: CVPixelBufferPoolCreatePixelBuffer failed: \(status)\n", stderr)
            return false
        }

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }

        guard let baseAddr = CVPixelBufferGetBaseAddress(pb) else { return false }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)

        let planeSize = width * height
        let rPlane = rgbPlanar
        let gPlane = rgbPlanar.advanced(by: planeSize)
        let bPlane = rgbPlanar.advanced(by: 2 * planeSize)

        // Convert planar RGB16 → interleaved RGBA16 (64RGBALE = R16 G16 B16 A16, little-endian)
        let dst = baseAddr.assumingMemoryBound(to: UInt16.self)
        for y in 0 ..< height {
            let rowOffset = y * bytesPerRow / 2  // bytesPerRow is in bytes, dst is UInt16
            let srcOffset = y * width
            for x in 0 ..< width {
                let dstIdx = rowOffset + x * 4
                let srcIdx = srcOffset + x
                dst[dstIdx + 0] = rPlane[srcIdx]    // R
                dst[dstIdx + 1] = gPlane[srcIdx]    // G
                dst[dstIdx + 2] = bPlane[srcIdx]    // B
                dst[dstIdx + 3] = 0xFFFF             // A (opaque)
            }
        }

        // Presentation time
        let pts = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps))

        // Wait for input to be ready
        while !(videoInput?.isReadyForMoreMediaData ?? false) {
            Thread.sleep(forTimeInterval: 0.001)
        }

        let success = adaptor.append(pb, withPresentationTime: pts)
        if !success {
            fputs("ProRes444Writer: append failed at frame \(frameIndex): \(assetWriter?.error?.localizedDescription ?? "unknown")\n", stderr)
            return false
        }

        frameIndex += 1
        return true
    }

    /// Finalize and close the output file.
    func finalize() {
        videoInput?.markAsFinished()

        let semaphore = DispatchSemaphore(value: 0)
        assetWriter?.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()

        if assetWriter?.status == .completed {
            fputs("ProRes444Writer: wrote \(frameIndex) frames successfully\n", stderr)
        } else {
            fputs("ProRes444Writer: finalize failed: \(assetWriter?.error?.localizedDescription ?? "unknown")\n", stderr)
        }
    }

    // Singleton for C bridge access
    static var shared: ProRes444Writer?
}

// MARK: - C Bridge Functions

@_cdecl("prores444_writer_open")
func prores444_writer_open(
    _ path: UnsafePointer<CChar>,
    _ width: Int32,
    _ height: Int32,
    _ fps: Float
) -> Int32 {
    let outputPath = String(cString: path)
    if let writer = ProRes444Writer(outputPath: outputPath,
                                      width: Int(width),
                                      height: Int(height),
                                      fps: Double(fps)) {
        ProRes444Writer.shared = writer
        return 0
    }
    return -1
}

@_cdecl("prores444_writer_write_frame")
func prores444_writer_write_frame(_ rgb_planar: UnsafePointer<UInt16>) -> Int32 {
    guard let writer = ProRes444Writer.shared else { return -1 }
    return writer.writeFrame(rgbPlanar: rgb_planar) ? 0 : -1
}

@_cdecl("prores444_writer_close")
func prores444_writer_close() {
    ProRes444Writer.shared?.finalize()
    ProRes444Writer.shared = nil
}

@_cdecl("prores444_writer_available")
func prores444_writer_available() -> Int32 {
    return 1  // Always available on macOS with AVFoundation
}
