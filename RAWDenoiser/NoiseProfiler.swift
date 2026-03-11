import Foundation
import SwiftUI
import Combine

// MARK: - C bridge

// noise_profile_from_patch and noise_profile_read_frame are declared in noise_profile.h
// which is included via the bridging header.

// MARK: - Swift result type

struct NoiseProfileResult {
    let sigma: Float          // measured noise std-dev at patch brightness
    let readNoise: Float      // read noise floor
    let shotGain: Float       // shot noise slope
    let blackLevel: Float     // estimated black level
    let meanSignal: Float     // mean pixel value in patch (16-bit)
    let isValid: Bool
    var patchWidth: Int = 0   // Bayer pixels (for quality indicator)
    var patchHeight: Int = 0

    /// Formatted summary for display
    var summaryLine: String {
        guard isValid else { return "Too small — select a larger flat area" }
        return String(format: "σ = %.0f  ·  read = %.0f  ·  shot = %.3f  ·  BL = %.0f",
                      sigma, readNoise, shotGain, blackLevel)
    }

    /// Quality hint based on patch size (number of 16×16 blocks)
    var qualityHint: String {
        let blocks = (patchWidth / 16) * (patchHeight / 16)
        switch blocks {
        case 0..<4:    return "Too small"
        case 4..<30:   return "Fair — try a larger area for better accuracy"
        case 30..<100: return "Good"
        default:       return "Excellent"
        }
    }

    var qualityColor: Color {
        let blocks = (patchWidth / 16) * (patchHeight / 16)
        switch blocks {
        case 0..<4:    return .red
        case 4..<30:   return .yellow
        default:       return .green
        }
    }

    var sigmaLabel: String {
        guard isValid else { return "—" }
        return String(format: "%.0f", sigma)
    }
}

// MARK: - Profiler engine

@MainActor
class NoiseProfiler: ObservableObject {
    @Published var result: NoiseProfileResult? = nil
    @Published var isAnalyzing: Bool = false

    /// Run noise analysis on a patch of the raw Bayer frame.
    /// - Parameters:
    ///   - inputURL: The video file path
    ///   - frameIndex: Which frame to decode
    ///   - patchRect: Rect in Bayer pixel coordinates (full-res 16-bit frame)
    func analyze(inputURL: URL, frameIndex: Int, patchRect: CGRect) {
        isAnalyzing = true
        result = nil
        let path = inputURL.path
        let px = Int(patchRect.minX)
        let py = Int(patchRect.minY)
        let pw = Int(patchRect.width)
        let ph = Int(patchRect.height)

        Task.detached(priority: .userInitiated) { [weak self] in
            var w: Int32 = 0
            var h: Int32 = 0

            guard let bayer = noise_profile_read_frame(path, Int32(frameIndex), &w, &h) else {
                await MainActor.run { [weak self] in
                    self?.isAnalyzing = false
                }
                return
            }
            defer { bayer.deallocate() }

            var profile = CNoiseProfile()
            noise_profile_from_patch(bayer, w, h,
                                     Int32(px), Int32(py),
                                     Int32(pw), Int32(ph),
                                     &profile)

            var result = NoiseProfileResult(
                sigma:       profile.sigma,
                readNoise:   profile.read_noise,
                shotGain:    profile.shot_gain,
                blackLevel:  profile.black_level,
                meanSignal:  profile.mean_signal,
                isValid:     profile.valid != 0
            )
            result.patchWidth  = pw
            result.patchHeight = ph

            await MainActor.run { [weak self] in
                self?.result = result
                self?.isAnalyzing = false
            }
        }
    }

    func reset() {
        result = nil
        isAnalyzing = false
    }
}

// MARK: - Selection overlay view

/// Transparent overlay that lets the user drag a selection rectangle
/// for noise profiling. Reports the rect in the image's own pixel coordinates.
struct NoiseProfileSelectionOverlay: View {
    let imageSize: CGSize          // actual pixel dimensions of the Bayer frame
    @Binding var selectionRect: CGRect?   // in image pixel coords
    var onAnalyze: (CGRect) -> Void

    @State private var dragStart: CGPoint = .zero
    @State private var dragCurrent: CGPoint = .zero
    @State private var isDragging: Bool = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Dim overlay with cut-out for the selected region
                if let rect = viewRect(in: geo) {
                    Color.black.opacity(0.35)
                        .mask(
                            Rectangle()
                                .overlay(
                                    Rectangle()
                                        .frame(width: rect.width, height: rect.height)
                                        .position(x: rect.midX, y: rect.midY)
                                        .blendMode(.destinationOut)
                                )
                        )
                        .allowsHitTesting(false)

                    // Selection border
                    Rectangle()
                        .strokeBorder(
                            Color.yellow,
                            style: StrokeStyle(lineWidth: 1.5, dash: [6, 3])
                        )
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                        .allowsHitTesting(false)

                    // Analyze button above the rect
                    VStack(spacing: 4) {
                        Button {
                            if let imgRect = selectionRect {
                                onAnalyze(imgRect)
                            }
                        } label: {
                            Label("Analyze Patch", systemImage: "waveform.and.magnifyingglass")
                                .font(.caption.bold())
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.yellow)
                        .colorScheme(.dark)

                        Text("\(Int(selectionRect?.width ?? 0)) × \(Int(selectionRect?.height ?? 0)) px")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.7))
                    }
                    .position(x: rect.midX, y: max(rect.minY - 36, 28))
                } else {
                    // No selection yet — show instruction
                    VStack(spacing: 6) {
                        Image(systemName: "viewfinder")
                            .font(.title2)
                        Text("Drag to select a flat, textureless region")
                            .font(.caption)
                    }
                    .foregroundStyle(.white.opacity(0.8))
                    .padding(10)
                    .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                    .allowsHitTesting(false)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 4)
                    .onChanged { value in
                        if !isDragging {
                            dragStart = value.startLocation
                            isDragging = true
                        }
                        dragCurrent = value.location
                        let vRect = normalizedRect(from: dragStart, to: dragCurrent, in: geo)
                        selectionRect = viewToImageRect(vRect, geo: geo)
                    }
                    .onEnded { value in
                        isDragging = false
                        let vRect = normalizedRect(from: dragStart, to: value.location, in: geo)
                        selectionRect = viewToImageRect(vRect, geo: geo)
                    }
            )
        }
    }

    // Convert drag points to a normalized CGRect within the geometry
    private func normalizedRect(from a: CGPoint, to b: CGPoint, in geo: GeometryProxy) -> CGRect {
        let minX = max(0, min(a.x, b.x))
        let minY = max(0, min(a.y, b.y))
        let maxX = min(geo.size.width,  max(a.x, b.x))
        let maxY = min(geo.size.height, max(a.y, b.y))
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    // Return the selection rect in view coordinates (for drawing)
    private func viewRect(in geo: GeometryProxy) -> CGRect? {
        guard let imgRect = selectionRect else { return nil }
        // Scale from image coords to view coords
        let scaleX = geo.size.width  / max(imageSize.width,  1)
        let scaleY = geo.size.height / max(imageSize.height, 1)
        // Use uniform scale (aspect-fit)
        let scale = min(scaleX, scaleY)
        let offsetX = (geo.size.width  - imageSize.width  * scale) / 2
        let offsetY = (geo.size.height - imageSize.height * scale) / 2
        return CGRect(
            x:      imgRect.minX * scale + offsetX,
            y:      imgRect.minY * scale + offsetY,
            width:  imgRect.width  * scale,
            height: imgRect.height * scale
        )
    }

    // Convert view-coordinate rect to image-pixel rect
    private func viewToImageRect(_ viewRect: CGRect, geo: GeometryProxy) -> CGRect {
        let scaleX = geo.size.width  / max(imageSize.width,  1)
        let scaleY = geo.size.height / max(imageSize.height, 1)
        let scale  = min(scaleX, scaleY)
        let offsetX = (geo.size.width  - imageSize.width  * scale) / 2
        let offsetY = (geo.size.height - imageSize.height * scale) / 2
        guard scale > 0 else { return .zero }
        return CGRect(
            x:      (viewRect.minX - offsetX) / scale,
            y:      (viewRect.minY - offsetY) / scale,
            width:  viewRect.width  / scale,
            height: viewRect.height / scale
        )
    }
}
