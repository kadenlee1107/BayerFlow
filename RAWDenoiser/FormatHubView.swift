import SwiftUI

// MARK: - Format Hub View (3D Carousel)

struct FormatHubView: View {
    let onSelectFormat: (FormatItem) -> Void
    @EnvironmentObject private var showcase: ShowcaseController

    @State private var rotationAngle: Double = 0
    @State private var dragOffset: Double = 0
    @State private var zoomOpacity: Double = 1.0
    @State private var scrollAccumulator: Double = 0
    @State private var isScrolling = false

    private let formats = rawFormats
    private let step = 2.0 * .pi / Double(rawFormats.count)
    private let carouselRadius: Double = 180

    /// Index of the item currently at the front of the carousel.
    private var focusedIndex: Int {
        let total = rotationAngle + dragOffset
        // Negate because positive rotation moves items left → next index
        let raw = -total / step
        let idx = ((Int(raw.rounded()) % formats.count) + formats.count) % formats.count
        return idx
    }

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                Spacer()

                carouselView

                // Focused item label
                focusedLabel
                    .padding(.top, 16)
                    .frame(height: 44)

                Spacer()
            }

        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            InfiniteGridBackground()
        }
        .onKeyPress(.leftArrow) { moveCarousel(direction: -1); return .handled }
        .onKeyPress(.rightArrow) { moveCarousel(direction: 1); return .handled }
        .onKeyPress(characters: CharacterSet(charactersIn: "aA")) { _ in moveCarousel(direction: -1); return .handled }
        .onKeyPress(characters: CharacterSet(charactersIn: "dD")) { _ in moveCarousel(direction: 1); return .handled }
        .onKeyPress(.return) { selectFocusedOrb(); return .handled }
        .focusable()
        .onScrollWheel { delta in
            guard !isScrolling else { return }
            scrollAccumulator += delta
            let threshold: Double = 15
            if scrollAccumulator > threshold {
                scrollAccumulator = 0
                isScrolling = true
                withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                    rotationAngle += step
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    isScrolling = false
                }
            } else if scrollAccumulator < -threshold {
                scrollAccumulator = 0
                isScrolling = true
                withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                    rotationAngle -= step
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    isScrolling = false
                }
            }
        }
        // Showcase mode: auto-rotate and auto-select
        .onChange(of: showcase.carouselTarget) { _, target in
            if let target, showcase.isActive {
                rotateToIndex(target)
            }
        }
        .onChange(of: showcase.triggerSelect) { _, trigger in
            if trigger && showcase.isActive {
                selectFocusedOrb()
            }
        }
    }

    // MARK: - Carousel

    private var carouselView: some View {
        TimelineView(.animation(minimumInterval: 0.05)) { context in
            let time = context.date.timeIntervalSinceReferenceDate
            ZStack {
                ForEach(sortedIndices(time: time), id: \.self) { index in
                    carouselItem(index: index, time: time)
                }
            }
            .frame(width: 500, height: 200)
        }
        .gesture(
            DragGesture()
                .onChanged { value in
                    // Convert horizontal drag to rotation (pixels → radians)
                    dragOffset = value.translation.width / 120.0
                }
                .onEnded { value in
                    rotationAngle += dragOffset
                    dragOffset = 0
                    snapToNearest()
                }
        )
    }

    /// Sort indices by z-depth, hiding items behind the camera.
    private func sortedIndices(time: TimeInterval) -> [Int] {
        (0..<formats.count)
            .filter { zDepth(index: $0) > -0.3 } // hide rear items (keeps ~5 visible)
            .sorted { a, b in
                zDepth(index: a) < zDepth(index: b)
            }
    }

    private func zDepth(index: Int) -> Double {
        let itemAngle = step * Double(index)
        let angle = itemAngle + rotationAngle + dragOffset
        return cos(angle)
    }

    private func carouselItem(index: Int, time: TimeInterval) -> some View {
        let itemAngle = step * Double(index)
        let angle = itemAngle + rotationAngle + dragOffset
        let x = sin(angle) * carouselRadius
        let z = cos(angle) // -1 (back) to +1 (front)
        let normalizedZ = (z + 1.0) / 2.0 // 0=back, 1=front
        let scale = 0.55 + 0.45 * normalizedZ
        let itemOpacity = 0.35 + 0.65 * normalizedZ
        let isFront = normalizedZ > 0.85
        // Gentle floating bob
        let phase = (Double(index) / Double(formats.count)) * 2.0 * .pi
        let floatY = sin(time * 2.0 / 3.0 * .pi + phase) * 3.0
        // Slight Y shift for depth perspective (rear items slightly higher)
        let depthY = (1.0 - normalizedZ) * -12.0

        let groundShadowOpacity = 0.18 * normalizedZ // stronger shadow for front items

        return VStack(spacing: 0) {
            FormatBubbleContent(
                format: formats[index],
                isHovered: isFront
            )
            .frame(width: 110)

            // Ground shadow — ellipse cast on the "floor"
            Ellipse()
                .fill(Color.black.opacity(groundShadowOpacity))
                .frame(width: 60 * scale, height: 10 * scale)
                .blur(radius: 8)
                .offset(y: 8)
        }
        .scaleEffect(scale)
        .opacity(itemOpacity)
        .offset(x: x, y: floatY + depthY)
        .zIndex(Double(normalizedZ * 100))
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: rotationAngle)
        .onTapGesture {
            if isFront {
                let format = formats[index]
                if format.isActive {
                    onSelectFormat(format)
                }
            } else {
                rotateToIndex(index)
            }
        }
    }

    // MARK: - Focused Label

    @ViewBuilder
    private var focusedLabel: some View {
        let idx = focusedIndex
        let format = formats[idx]
        VStack(spacing: 4) {
            Text(format.name)
                .font(.title3.bold())
                .foregroundStyle(.primary)
            if !format.isActive {
                Text("Coming Soon")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: idx)
    }

    // MARK: - Snap & Rotate

    private func snapToNearest() {
        let snapped = (rotationAngle / step).rounded() * step
        withAnimation(.spring(response: 0.4, dampingFraction: 0.75)) {
            rotationAngle = snapped
        }
    }

    private func rotateToIndex(_ index: Int) {
        let targetAngle = -step * Double(index)
        let diff = targetAngle - rotationAngle
        let normalizedDiff = atan2(sin(diff), cos(diff))
        withAnimation(.spring(response: 0.5, dampingFraction: 0.75)) {
            rotationAngle += normalizedDiff
        }
    }

    private func moveCarousel(direction: Int) {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            rotationAngle -= step * Double(direction)
        }
    }

    private func selectFocusedOrb() {
        let format = formats[focusedIndex]
        if format.isActive {
            onSelectFormat(format)
        }
    }
}

// MARK: - 3D Orb Bubble Content

private struct FormatBubbleContent: View {
    let format: FormatItem
    let isHovered: Bool

    private let orbSize: CGFloat = 80

    var body: some View {
        VStack(spacing: 0) {
            orbView
        }
    }

    private var orbView: some View {
        ZStack {
            // Layer 1: Base color with vertical gradient (darker at bottom for 3D depth)
            Circle()
                .fill(
                    LinearGradient(
                        colors: [
                            format.color.opacity(format.isActive ? 1.0 : 0.5),
                            format.color.opacity(format.isActive ? 0.6 : 0.3)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

            // Layer 2: Specular highlight — radial gradient from top-left
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color.white.opacity(0.55),
                            Color.white.opacity(0.15),
                            Color.clear
                        ],
                        center: UnitPoint(x: 0.35, y: 0.25),
                        startRadius: 0,
                        endRadius: orbSize * 0.55
                    )
                )

            // Layer 3: Logo image or fallback initials
            logoOrInitials

            // Layer 4: Glossy rim — top highlight arc
            Circle()
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            Color.white.opacity(0.4),
                            Color.white.opacity(0.1),
                            Color.clear,
                            Color.clear
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: 1.5
                )

            // Layer 5: Inner shadow at bottom edge
            Circle()
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            Color.clear,
                            Color.clear,
                            Color.black.opacity(0.2),
                            Color.black.opacity(0.3)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    lineWidth: 2
                )
        }
        .frame(width: orbSize, height: orbSize)
        // Shadow 1: Colored glow
        .shadow(
            color: format.color.opacity(isHovered ? 0.6 : 0.25),
            radius: isHovered ? 16 : 6,
            y: isHovered ? 2 : 3
        )
        // Shadow 2: Dark ground shadow
        .shadow(
            color: Color.black.opacity(isHovered ? 0.35 : 0.2),
            radius: isHovered ? 20 : 12,
            y: isHovered ? 12 : 8
        )
    }

    @ViewBuilder
    private var logoOrInitials: some View {
        let img = NSImage(named: format.imageName)
        if img != nil {
            Image(format.imageName)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 36, height: 36)
                .shadow(color: .black.opacity(0.3), radius: 2, y: 1)
        } else {
            Text(format.initials)
                .font(.system(
                    size: format.initials.count > 2 ? 18 : 22,
                    weight: .bold,
                    design: .rounded
                ))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.3), radius: 2, y: 1)
        }
    }
}

// MARK: - Background

private struct InfiniteGridBackground: View {
    var body: some View {
        Canvas { context, size in
            let w = size.width
            let h = size.height

            // 1. Fog gradient: gray top → white floor
            context.fill(
                Path(CGRect(origin: .zero, size: size)),
                with: .linearGradient(
                    Gradient(colors: [
                        Color(white: 0.55),
                        Color(white: 0.70),
                        Color(white: 0.88),
                        Color.white
                    ]),
                    startPoint: .zero,
                    endPoint: CGPoint(x: 0, y: h * 0.55)
                )
            )

            // 2. Perspective grid on the floor
            // Horizon sits where fog meets the floor
            let horizonY = h * 0.52
            let lineColor = Color(white: 0.78)
            let farBottom = h * 3.0

            // Vertical lines: gentle convergence, extend to corners
            let vCount = 14
            let bottomSpread = w * 4.0
            let horizonSpread = w * 0.75
            for i in 0...vCount {
                let t = CGFloat(i) / CGFloat(vCount)
                let bottomX = (w - bottomSpread) / 2 + bottomSpread * t
                let horizonX = (w - horizonSpread) / 2 + horizonSpread * t

                // Fade out as lines approach the horizon (into the fog)
                var path = Path()
                path.move(to: CGPoint(x: horizonX, y: horizonY))
                path.addLine(to: CGPoint(x: bottomX, y: farBottom))

                // Draw with gradient opacity: faint near horizon, stronger near camera
                let segments = 20
                for s in 0..<segments {
                    let t0 = CGFloat(s) / CGFloat(segments)
                    let t1 = CGFloat(s + 1) / CGFloat(segments)
                    let y0 = horizonY + (h - horizonY) * t0
                    let y1 = horizonY + (h - horizonY) * t1
                    if y0 > h { break }
                    let x0 = horizonX + (bottomX - horizonX) * ((y0 - horizonY) / (farBottom - horizonY))
                    let x1 = horizonX + (bottomX - horizonX) * ((y1 - horizonY) / (farBottom - horizonY))
                    // Fade: 0 at horizon → 0.4 at bottom
                    let alpha = t0 * 0.4

                    var seg = Path()
                    seg.move(to: CGPoint(x: x0, y: y0))
                    seg.addLine(to: CGPoint(x: x1, y: min(y1, h)))
                    context.stroke(seg, with: .color(lineColor.opacity(alpha)), lineWidth: 0.5)
                }
            }

            // Horizontal lines: perspective spacing, fade near horizon
            let hCount = 12
            for i in 1...hCount {
                let t = CGFloat(i) / CGFloat(hCount)
                let y = horizonY + (h - horizonY) * pow(t, 1.8)
                if y > h { continue }

                // Fade near horizon into the fog
                let alpha = t * 0.4

                var path = Path()
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: w, y: y))
                context.stroke(path, with: .color(lineColor.opacity(alpha)), lineWidth: 0.5)
            }
        }
    }
}

// MARK: - Scroll Wheel Support (two-finger trackpad swipe)

private struct ScrollWheelModifier: ViewModifier {
    let action: (CGFloat) -> Void

    func body(content: Content) -> some View {
        content.background(
            ScrollWheelMonitorView(action: action)
        )
    }
}

/// Uses NSEvent.addLocalMonitorForEvents to reliably capture scroll wheel events,
/// even when SwiftUI views on top absorb hit-testing.
private struct ScrollWheelMonitorView: NSViewRepresentable {
    let action: (CGFloat) -> Void

    func makeNSView(context: Context) -> ScrollMonitorNSView {
        let view = ScrollMonitorNSView()
        view.onScroll = action
        return view
    }

    func updateNSView(_ nsView: ScrollMonitorNSView, context: Context) {
        nsView.onScroll = action
    }
}

private class ScrollMonitorNSView: NSView {
    var onScroll: ((CGFloat) -> Void)?
    private var monitor: Any?

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil && monitor == nil {
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                guard let self, let action = self.onScroll,
                      self.window != nil else { return event }
                // Only handle if mouse is within this view's bounds
                let loc = self.convert(event.locationInWindow, from: nil)
                guard self.bounds.contains(loc) else { return event }
                let dx = event.scrollingDeltaX
                let dy = event.scrollingDeltaY
                // Use whichever axis has larger magnitude
                let delta = abs(dx) >= abs(dy) ? dx : -dy
                action(delta)
                return event
            }
        } else if window == nil {
            removeMonitor()
        }
    }

    private func removeMonitor() {
        if let m = monitor {
            NSEvent.removeMonitor(m)
            monitor = nil
        }
    }

    deinit { removeMonitor() }
}

extension View {
    func onScrollWheel(action: @escaping (CGFloat) -> Void) -> some View {
        modifier(ScrollWheelModifier(action: action))
    }
}
