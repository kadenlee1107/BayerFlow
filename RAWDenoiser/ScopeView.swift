import SwiftUI
import AppKit

// MARK: - Scope mode

enum ScopeMode: String, CaseIterable, Identifiable {
    case histogram = "Histogram"
    case waveform  = "Waveform"
    var id: String { rawValue }
}

// MARK: - Scope data

struct ScopeData {
    var rHist: [Float]       // 256 bins, normalized 0..1
    var gHist: [Float]
    var bHist: [Float]
    var lumaHist: [Float]
    var waveformColumns: [[UInt8]]  // per-output-column: luma values
    let imageWidth: Int
    let imageHeight: Int

    /// Compute scope data from an NSImage on a background thread.
    static func compute(from image: NSImage, targetWidth: Int = 300) -> ScopeData? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }

        let w = cgImage.width
        let h = cgImage.height
        guard w > 0 && h > 0 else { return nil }

        let bytesPerRow = w * 4
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

        // Histogram: 256 bins per channel
        var rBins = [Int](repeating: 0, count: 256)
        var gBins = [Int](repeating: 0, count: 256)
        var bBins = [Int](repeating: 0, count: 256)
        var lumaBins = [Int](repeating: 0, count: 256)

        // Waveform: collect luma per output column
        let stride = max(1, w / targetWidth)
        let outCols = max(1, w / stride)
        var waveform = [[UInt8]](repeating: [], count: outCols)

        for y in 0..<h {
            for x in 0..<w {
                let off = (y * w + x) * 4
                let r = Int(ptr[off])
                let g = Int(ptr[off + 1])
                let b = Int(ptr[off + 2])
                let luma = Int(Float(r) * 0.2126 + Float(g) * 0.7152 + Float(b) * 0.0722)

                rBins[r] += 1
                gBins[g] += 1
                bBins[b] += 1
                lumaBins[min(luma, 255)] += 1

                let col = x / stride
                if col < outCols {
                    waveform[col].append(UInt8(min(luma, 255)))
                }
            }
        }

        // Normalize histograms (skip bin 0 and 255 to avoid clipping spikes dominating)
        let rMax = Float(rBins[1..<255].max() ?? 1)
        let gMax = Float(gBins[1..<255].max() ?? 1)
        let bMax = Float(bBins[1..<255].max() ?? 1)
        let lMax = Float(lumaBins[1..<255].max() ?? 1)
        let globalMax = max(rMax, max(gMax, max(bMax, lMax)))

        let rNorm = rBins.map { min(Float($0) / globalMax, 1.0) }
        let gNorm = gBins.map { min(Float($0) / globalMax, 1.0) }
        let bNorm = bBins.map { min(Float($0) / globalMax, 1.0) }
        let lNorm = lumaBins.map { min(Float($0) / globalMax, 1.0) }

        return ScopeData(
            rHist: rNorm, gHist: gNorm, bHist: bNorm, lumaHist: lNorm,
            waveformColumns: waveform,
            imageWidth: w, imageHeight: h
        )
    }
}

// MARK: - Scope view

struct ScopeView: View {
    let data: ScopeData
    let mode: ScopeMode

    var body: some View {
        Canvas { context, size in
            switch mode {
            case .histogram:
                drawHistogram(context: context, size: size)
            case .waveform:
                drawWaveform(context: context, size: size)
            }
        }
        .frame(height: 120)
        .background(Color.black.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityLabel("\(mode.rawValue) scope")
        .accessibilityElement(children: .ignore)
    }

    // MARK: - Histogram renderer

    private func drawHistogram(context: GraphicsContext, size: CGSize) {
        let w = size.width
        let h = size.height
        let binWidth = w / 256.0

        // Draw each channel as a filled area (R, G, B) with transparency
        let channels: [(bins: [Float], color: Color)] = [
            (data.rHist, Color.red),
            (data.gHist, Color.green),
            (data.bHist, Color.blue),
        ]

        for (bins, color) in channels {
            var path = Path()
            path.move(to: CGPoint(x: 0, y: h))
            for i in 0..<256 {
                let x = CGFloat(i) * binWidth
                let y = h - CGFloat(bins[i]) * h * 0.9  // 90% max height
                path.addLine(to: CGPoint(x: x, y: y))
            }
            path.addLine(to: CGPoint(x: w, y: h))
            path.closeSubpath()
            context.fill(path, with: .color(color.opacity(0.3)))
        }

        // Luma outline on top
        var lumaPath = Path()
        for i in 0..<256 {
            let x = CGFloat(i) * binWidth
            let y = h - CGFloat(data.lumaHist[i]) * h * 0.9
            if i == 0 {
                lumaPath.move(to: CGPoint(x: x, y: y))
            } else {
                lumaPath.addLine(to: CGPoint(x: x, y: y))
            }
        }
        context.stroke(lumaPath, with: .color(.white.opacity(0.7)), lineWidth: 1)
    }

    // MARK: - Waveform renderer

    private func drawWaveform(context: GraphicsContext, size: CGSize) {
        let w = size.width
        let h = size.height
        let colCount = data.waveformColumns.count
        guard colCount > 0 else { return }

        let colWidth = w / CGFloat(colCount)

        // For each output column, draw scattered dots at luma positions
        for col in 0..<colCount {
            let values = data.waveformColumns[col]
            let x = CGFloat(col) * colWidth + colWidth * 0.5

            // Count occurrences at each luma level for density
            var density = [Int](repeating: 0, count: 256)
            for v in values { density[Int(v)] += 1 }
            let maxDensity = max(Float(density.max() ?? 1), 1)

            for luma in 0..<256 where density[luma] > 0 {
                let y = h - (CGFloat(luma) / 255.0) * h * 0.95 - h * 0.025
                let alpha = min(Float(density[luma]) / maxDensity * 3.0, 1.0) * 0.6
                let rect = CGRect(x: x - colWidth * 0.4, y: y - 0.5,
                                  width: colWidth * 0.8, height: 1)
                context.fill(Path(rect), with: .color(.green.opacity(Double(alpha))))
            }
        }

        // Draw 10% and 90% reference lines
        for level in [0.1, 0.9] {
            let y = h - CGFloat(level) * h * 0.95 - h * 0.025
            var line = Path()
            line.move(to: CGPoint(x: 0, y: y))
            line.addLine(to: CGPoint(x: w, y: y))
            context.stroke(line, with: .color(.white.opacity(0.15)), lineWidth: 0.5)
        }
    }
}
