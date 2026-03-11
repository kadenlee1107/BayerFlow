import SwiftUI
import AppKit

/// Draggable wipe comparison view: overlays "before" and "after" images
/// with a vertical divider the user can drag left/right.
struct CompareView: View {
    let before: NSImage
    let after: NSImage

    @State private var dividerFraction: CGFloat = 0.5

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let dividerX = w * dividerFraction

            ZStack {
                // "After" image (full width, behind)
                Image(nsImage: after)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: w, height: h)

                // "Before" image clipped to left of divider
                Image(nsImage: before)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: w, height: h)
                    .clipShape(
                        HorizontalClip(rightEdge: dividerX)
                    )

                // Divider line
                Rectangle()
                    .fill(.white)
                    .frame(width: 2, height: h)
                    .position(x: dividerX, y: h / 2)
                    .shadow(color: .black.opacity(0.5), radius: 3, x: 0, y: 0)

                // Drag handle (pill on divider)
                Capsule()
                    .fill(.white)
                    .frame(width: 6, height: 36)
                    .shadow(color: .black.opacity(0.4), radius: 4, x: 0, y: 0)
                    .position(x: dividerX, y: h / 2)

                // Labels
                HStack {
                    Text("Original")
                        .font(.caption2.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.black.opacity(0.5), in: Capsule())
                        .padding(8)
                    Spacer()
                    Text("Denoised")
                        .font(.caption2.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.black.opacity(0.5), in: Capsule())
                        .padding(8)
                }
                .frame(width: w, alignment: .top)
                .position(x: w / 2, y: 16)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let fraction = value.location.x / w
                        dividerFraction = min(max(fraction, 0.05), 0.95)
                    }
            )
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Before and after comparison")
        .accessibilityValue("\(Int(dividerFraction * 100))% original visible")
        .accessibilityHint("Drag to adjust comparison split between original and denoised")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: dividerFraction = min(dividerFraction + 0.1, 0.95)
            case .decrement: dividerFraction = max(dividerFraction - 0.1, 0.05)
            @unknown default: break
            }
        }
    }
}

/// Clips content to the left side (from 0 to rightEdge).
private struct HorizontalClip: Shape {
    var rightEdge: CGFloat

    func path(in rect: CGRect) -> Path {
        Path(CGRect(x: 0, y: 0, width: rightEdge, height: rect.height))
    }
}
