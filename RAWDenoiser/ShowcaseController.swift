import SwiftUI
import Combine

/// Drives the app through a scripted demo sequence for screen recording.
/// Launch with `--showcase /path/to/file.MOV` or use Process → Showcase Demo menu.
@MainActor
final class ShowcaseController: ObservableObject {

    @Published var isActive = false
    @Published var pendingFilePath: String? = nil

    // — Signals for FormatHubView —
    @Published var carouselTarget: Int? = nil
    @Published var triggerSelect = false

    // — Cursor drag animation (phase-only, view handles interpolation) —
    @Published var cursorPhase: CursorPhase = .hidden
    var filePath: String = ""  // not @Published — no need to trigger re-renders

    enum CursorPhase: Equatable {
        case hidden
        case entering   // cursor + file icon sweeping toward drop zone
        case hovering   // paused over drop zone (highlight active)
        case dropped    // released — fade out
    }

    // — Signals for SessionView —
    @Published var resetToHub = false
    @Published var simulateDragOver = false
    @Published var triggerFileLoad: URL? = nil
    @Published var targetPreset: DenoisePreset? = nil
    @Published var targetStrength: Float? = nil
    @Published var targetWindowSize: Double? = nil
    @Published var targetSpatialStrength: Float? = nil
    @Published var targetProtectSubjects: Bool? = nil
    @Published var triggerPreview = false
    @Published var triggerDenoise = false

    // — Feedback from views —
    @Published var hubVisible = true
    @Published var analysisComplete = false
    @Published var previewComplete = false
    @Published var denoiseComplete = false

    private var task: Task<Void, Never>?

    init() {
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "--showcase"), idx + 1 < args.count {
            pendingFilePath = args[idx + 1]
        }
    }

    func start(filePath: String) {
        self.filePath = filePath

        // Reset all signals
        cursorPhase = .hidden
        carouselTarget = nil
        triggerSelect = false
        resetToHub = true
        simulateDragOver = false
        triggerFileLoad = nil
        targetPreset = nil
        targetStrength = nil
        targetWindowSize = nil
        targetSpatialStrength = nil
        targetProtectSubjects = nil
        triggerPreview = false
        triggerDenoise = false
        analysisComplete = false
        previewComplete = false
        denoiseComplete = false
        hubVisible = true

        isActive = true
        task = Task { await runSequence(filePath: filePath) }
    }

    func stop() {
        task?.cancel()
        task = nil
        isActive = false
        cursorPhase = .hidden
    }

    // MARK: - Sequence

    private func runSequence(filePath: String) async {
        // Wait for hub to fully render
        await wait(1500)

        // ——— Phase 1: Carousel Tour ———
        let count = rawFormats.count
        for i in 1..<count {
            guard alive else { return }
            carouselTarget = i
            await wait(500)
        }

        // Return to ProRes RAW (index 0)
        guard alive else { return }
        carouselTarget = 0
        await wait(1200)

        // Select the ProRes RAW orb (triggers zoom animation)
        guard alive else { return }
        triggerSelect = true
        await wait(100)
        triggerSelect = false

        // Wait for session view to appear
        await waitFor { !self.hubVisible }
        await wait(1200)

        // ——— Phase 2: Cursor Drag & File Drop ———
        // View handles smooth cubic bezier animation via Animatable — zero lag
        guard alive else { return }
        cursorPhase = .entering
        // Cursor sweeps in over 2.0s; highlight drop zone at ~75% (1.5s)
        await wait(1500)
        guard alive else { return }
        simulateDragOver = true
        await wait(500)

        // Hover over drop zone with gentle bob
        guard alive else { return }
        cursorPhase = .hovering
        await wait(800)

        // Drop the file
        guard alive else { return }
        cursorPhase = .dropped
        triggerFileLoad = URL(fileURLWithPath: filePath)
        await wait(500)

        // Clean up cursor
        cursorPhase = .hidden
        simulateDragOver = false
        triggerFileLoad = nil

        // Wait for motion analysis to complete
        await waitFor { self.analysisComplete }
        await wait(1500)

        // ——— Phase 3: Slider Showcase ———
        guard alive else { return }

        // Cycle through presets: Light → Standard → Strong
        targetPreset = .light
        await wait(1200)
        guard alive else { return }

        targetPreset = .standard
        await wait(1200)
        guard alive else { return }

        targetPreset = .strong
        await wait(800)

        // Smooth-slide temporal strength: 1.5 → 0.7 → 1.2
        guard alive else { return }
        await animateFloat(from: 1.5, to: 0.7, steps: 12, stepMs: 80) {
            self.targetStrength = $0
        }
        await wait(400)
        guard alive else { return }
        await animateFloat(from: 0.7, to: 1.2, steps: 10, stepMs: 80) {
            self.targetStrength = $0
        }
        await wait(600)

        // Smooth-slide window size: 15 → 7 → 11
        guard alive else { return }
        await animateSteppedDouble(from: 15, to: 7, step: -2, stepMs: 300) {
            self.targetWindowSize = $0
        }
        await wait(400)
        guard alive else { return }
        await animateSteppedDouble(from: 7, to: 11, step: 2, stepMs: 300) {
            self.targetWindowSize = $0
        }
        await wait(600)

        // Toggle protect subjects ON, pause, OFF
        guard alive else { return }
        targetProtectSubjects = true
        await wait(2000)
        guard alive else { return }
        targetProtectSubjects = false
        await wait(800)

        // Smooth-slide spatial strength: 0 → 0.5 → 0
        guard alive else { return }
        await animateFloat(from: 0.0, to: 0.5, steps: 8, stepMs: 80) {
            self.targetSpatialStrength = $0
        }
        await wait(500)
        guard alive else { return }
        await animateFloat(from: 0.5, to: 0.0, steps: 8, stepMs: 80) {
            self.targetSpatialStrength = $0
        }
        await wait(1000)

        // ——— Phase 4: Preview ———
        guard alive else { return }
        triggerPreview = true
        await wait(100)
        triggerPreview = false

        await waitFor { self.previewComplete }
        await wait(3000)

        // ——— Phase 5: Denoise ———
        guard alive else { return }
        triggerDenoise = true
        await wait(100)
        triggerDenoise = false

        await waitFor { self.denoiseComplete }
        await wait(3000)

        isActive = false
    }

    // MARK: - Helpers

    private var alive: Bool { !Task.isCancelled && isActive }

    private func wait(_ ms: Int) async {
        try? await Task.sleep(for: .milliseconds(ms))
    }

    private func waitFor(_ condition: @escaping () -> Bool, timeoutMs: Int = 120_000) async {
        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)
        while !condition() && alive && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(200))
        }
    }

    /// Smoothly interpolate a Float value using smoothstep easing.
    private func animateFloat(from: Float, to: Float, steps: Int, stepMs: Int,
                              setter: @escaping (Float) -> Void) async {
        for i in 1...steps {
            guard alive else { return }
            let t = Float(i) / Float(steps)
            let eased = t * t * (3 - 2 * t)
            setter(from + (to - from) * eased)
            await wait(stepMs)
        }
    }

    /// Step through discrete Double values (e.g., window size in increments of 2).
    private func animateSteppedDouble(from: Double, to: Double, step: Double, stepMs: Int,
                                      setter: @escaping (Double) -> Void) async {
        var current = from
        let goingUp = step > 0
        while alive {
            current += step
            if goingUp ? current > to : current < to { break }
            setter(current)
            await wait(stepMs)
        }
        setter(to)
    }
}

// MARK: - Bezier Position Modifier (Animatable — runs on render thread)

/// Animates position along a cubic bezier curve using Core Animation.
/// Zero SwiftUI re-renders during the animation — butter smooth.
private struct BezierPositionModifier: ViewModifier, Animatable {
    var progress: CGFloat
    let size: CGSize

    var animatableData: CGFloat {
        get { progress }
        set { progress = newValue }
    }

    func body(content: Content) -> some View {
        content.position(bezierPosition(t: progress))
    }

    private func bezierPosition(t: CGFloat) -> CGPoint {
        // Cubic bezier for a wide, natural arc from top-right into the drop zone
        // Start: outside window, top-right (like dragging from Finder)
        let p0 = CGPoint(x: size.width + 60, y: -20)
        // Control 1: sweeps inward with a wide arc
        let p1 = CGPoint(x: size.width * 0.85, y: size.height * 0.15)
        // Control 2: approaches drop zone from above-right
        let p2 = CGPoint(x: size.width * 0.72, y: size.height * 0.42)
        // End: center of right panel drop zone
        let p3 = CGPoint(x: size.width * 0.66, y: size.height * 0.5)

        let s = 1 - t
        return CGPoint(
            x: s*s*s * p0.x + 3*s*s*t * p1.x + 3*s*t*t * p2.x + t*t*t * p3.x,
            y: s*s*s * p0.y + 3*s*s*t * p1.y + 3*s*t*t * p2.y + t*t*t * p3.y
        )
    }
}

// MARK: - Cursor Drag Overlay

/// Animated cursor + file icon that sweeps into the drop zone during showcase.
/// Uses Animatable for GPU-accelerated bezier curve interpolation.
struct ShowcaseCursorView: View {
    let fileName: String
    let phase: ShowcaseController.CursorPhase

    @State private var progress: CGFloat = 0
    @State private var hoverBob: CGFloat = 0
    @State private var dropped = false

    var body: some View {
        GeometryReader { geo in
            VStack(alignment: .leading, spacing: 0) {
                // macOS cursor arrow
                Image(systemName: "cursorarrow")
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.6), radius: 1, x: 1, y: 1)

                // File drag ghost
                HStack(spacing: 8) {
                    Image(systemName: "film.fill")
                        .foregroundStyle(.orange)
                    Text(fileName)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .font(.callout)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.white.opacity(0.2), lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.25), radius: 8, y: 4)
                .offset(x: 16, y: -6)
            }
            .modifier(BezierPositionModifier(progress: progress, size: geo.size))
            .offset(y: hoverBob)
            .opacity(dropped ? 0 : 1)
            .scaleEffect(dropped ? 0.85 : 1.0)
        }
        .allowsHitTesting(false)
        .onAppear {
            if phase == .entering {
                startEntering()
            }
        }
        .onChange(of: phase) { _, newPhase in
            switch newPhase {
            case .entering:
                startEntering()
            case .hovering:
                // Gentle floating bob while hovering over drop zone
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    hoverBob = -4
                }
            case .dropped:
                // Stop bob, scale down + fade out
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    hoverBob = 0
                    dropped = true
                }
            case .hidden:
                withAnimation(nil) {
                    progress = 0
                    hoverBob = 0
                    dropped = false
                }
            }
        }
    }

    private func startEntering() {
        progress = 0
        hoverBob = 0
        dropped = false
        // Smooth spring-based sweep — decelerates naturally into the drop zone
        withAnimation(.timingCurve(0.25, 0.1, 0.25, 1.0, duration: 2.0)) {
            progress = 1
        }
    }
}
