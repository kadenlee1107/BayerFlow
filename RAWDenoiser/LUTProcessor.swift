import AppKit
import simd

// MARK: - LUT application to NSImage

enum LUTProcessor {

    /// Apply a CubeLUT to an NSImage with blending.
    /// Returns a new NSImage with the LUT applied (preview only — not baked into output).
    static func apply(lut: CubeLUT, to image: NSImage, blend: Float) -> NSImage? {
        guard blend > 0.001 else { return image }
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }

        let w = cgImage.width
        let h = cgImage.height
        let bytesPerRow = w * 4

        // Draw into RGBA8 bitmap
        guard let context = CGContext(
            data: nil,
            width: w, height: h,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let pixels = context.data else { return nil }

        let ptr = pixels.bindMemory(to: UInt8.self, capacity: w * h * 4)
        let domainRange = lut.domainMax - lut.domainMin
        let invDomain = SIMD3<Float>(1, 1, 1) / max(domainRange, SIMD3<Float>(repeating: 1e-6))
        let clampBlend = min(max(blend, 0), 1)

        switch lut.type {
        case .threeD:
            apply3D(ptr: ptr, w: w, h: h, lut: lut, invDomain: invDomain, blend: clampBlend)
        case .oneD:
            apply1D(ptr: ptr, w: w, h: h, lut: lut, invDomain: invDomain, blend: clampBlend)
        }

        guard let outCG = context.makeImage() else { return nil }
        return NSImage(cgImage: outCG, size: image.size)
    }

    // MARK: - 3D LUT with trilinear interpolation

    private static func apply3D(
        ptr: UnsafeMutablePointer<UInt8>, w: Int, h: Int,
        lut: CubeLUT, invDomain: SIMD3<Float>, blend: Float
    ) {
        let s = lut.size
        let sF = Float(s - 1)
        let data = lut.data

        for i in 0..<(w * h) {
            let off = i * 4
            let origR = Float(ptr[off])     / 255.0
            let origG = Float(ptr[off + 1]) / 255.0
            let origB = Float(ptr[off + 2]) / 255.0

            // Normalize to domain
            let tr = min(max((origR - lut.domainMin.x) * invDomain.x, 0), 1) * sF
            let tg = min(max((origG - lut.domainMin.y) * invDomain.y, 0), 1) * sF
            let tb = min(max((origB - lut.domainMin.z) * invDomain.z, 0), 1) * sF

            let r0 = min(Int(tr), s - 2); let r1 = r0 + 1; let fr = tr - Float(r0)
            let g0 = min(Int(tg), s - 2); let g1 = g0 + 1; let fg = tg - Float(g0)
            let b0 = min(Int(tb), s - 2); let b1 = b0 + 1; let fb = tb - Float(b0)

            // 8-corner trilinear lookup (R-fastest, G-middle, B-slowest)
            let c000 = data[r0 + g0 * s + b0 * s * s]
            let c100 = data[r1 + g0 * s + b0 * s * s]
            let c010 = data[r0 + g1 * s + b0 * s * s]
            let c110 = data[r1 + g1 * s + b0 * s * s]
            let c001 = data[r0 + g0 * s + b1 * s * s]
            let c101 = data[r1 + g0 * s + b1 * s * s]
            let c011 = data[r0 + g1 * s + b1 * s * s]
            let c111 = data[r1 + g1 * s + b1 * s * s]

            // Lerp along R
            let c00 = mix(c000, c100, t: fr)
            let c10 = mix(c010, c110, t: fr)
            let c01 = mix(c001, c101, t: fr)
            let c11 = mix(c011, c111, t: fr)

            // Lerp along G
            let c0 = mix(c00, c10, t: fg)
            let c1 = mix(c01, c11, t: fg)

            // Lerp along B
            let lutOut = mix(c0, c1, t: fb)

            // Blend with original
            let finalR = origR + (lutOut.x - origR) * blend
            let finalG = origG + (lutOut.y - origG) * blend
            let finalB = origB + (lutOut.z - origB) * blend

            ptr[off]     = UInt8(min(max(finalR * 255, 0), 255))
            ptr[off + 1] = UInt8(min(max(finalG * 255, 0), 255))
            ptr[off + 2] = UInt8(min(max(finalB * 255, 0), 255))
        }
    }

    // MARK: - 1D LUT with linear interpolation

    private static func apply1D(
        ptr: UnsafeMutablePointer<UInt8>, w: Int, h: Int,
        lut: CubeLUT, invDomain: SIMD3<Float>, blend: Float
    ) {
        let s = lut.size
        let sF = Float(s - 1)
        let data = lut.data

        for i in 0..<(w * h) {
            let off = i * 4
            let origR = Float(ptr[off])     / 255.0
            let origG = Float(ptr[off + 1]) / 255.0
            let origB = Float(ptr[off + 2]) / 255.0

            let tr = min(max((origR - lut.domainMin.x) * invDomain.x, 0), 1) * sF
            let tg = min(max((origG - lut.domainMin.y) * invDomain.y, 0), 1) * sF
            let tb = min(max((origB - lut.domainMin.z) * invDomain.z, 0), 1) * sF

            let ir0 = min(Int(tr), s - 2); let lutR = data[ir0].x + (data[ir0 + 1].x - data[ir0].x) * (tr - Float(ir0))
            let ig0 = min(Int(tg), s - 2); let lutG = data[ig0].y + (data[ig0 + 1].y - data[ig0].y) * (tg - Float(ig0))
            let ib0 = min(Int(tb), s - 2); let lutB = data[ib0].z + (data[ib0 + 1].z - data[ib0].z) * (tb - Float(ib0))

            ptr[off]     = UInt8(min(max((origR + (lutR - origR) * blend) * 255, 0), 255))
            ptr[off + 1] = UInt8(min(max((origG + (lutG - origG) * blend) * 255, 0), 255))
            ptr[off + 2] = UInt8(min(max((origB + (lutB - origB) * blend) * 255, 0), 255))
        }
    }
}
