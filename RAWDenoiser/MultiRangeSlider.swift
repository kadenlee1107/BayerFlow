import SwiftUI

// MARK: - Data Model

struct FrameRange: Identifiable, Equatable {
    let id: UUID
    var start: Int
    var end: Int   // exclusive

    init(start: Int, end: Int) {
        self.id = UUID()
        self.start = start
        self.end = end
    }

    var count: Int { max(end - start, 0) }
}

// MARK: - Multi-Range Timeline Slider

struct MultiRangeSlider: View {
    @Binding var ranges: [FrameRange]
    let totalFrames: Int
    var onScrub: ((Int) -> Void)? = nil
    @State private var selectedId: UUID? = nil
    @State private var dragEdge: DragEdge? = nil

    private enum DragEdge {
        case left(UUID, initialStart: Int)
        case right(UUID, initialEnd: Int)
        case body(UUID, startOffset: Int)
    }

    private let barHeight: CGFloat = 32
    private let handleWidth: CGFloat = 8
    private let minRangeFrames: Int = 2

    var body: some View {
        VStack(spacing: 6) {
            // Timeline bar
            GeometryReader { geo in
                let w = geo.size.width
                ZStack(alignment: .leading) {
                    // Track background
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(nsColor: .separatorColor).opacity(0.3))
                        .frame(height: barHeight)

                    // Range segments
                    ForEach(ranges) { range in
                        rangeSegment(range: range, barWidth: w)
                    }

                    // Frame ticks (every 10% or so)
                    if totalFrames > 0 {
                        frameTicks(barWidth: w)
                    }
                }
                .frame(height: barHeight)
                .contentShape(Rectangle())
                .onTapGesture { loc in
                    let frame = Int(loc.x / w * CGFloat(totalFrames))
                    tapOnTimeline(at: frame)
                }
                .gesture(
                    DragGesture(minimumDistance: 2)
                        .onChanged { value in
                            if dragEdge == nil {
                                let startX = value.startLocation.x
                                let hitTol: CGFloat = 12
                                // Prioritize selected range for handle hits
                                let ordered: [FrameRange] = {
                                    if let selId = selectedId,
                                       let sel = ranges.first(where: { $0.id == selId }) {
                                        return [sel] + ranges.filter { $0.id != selId }
                                    }
                                    return ranges
                                }()
                                // Check handles first
                                for range in ordered {
                                    let leftX = CGFloat(range.start) / CGFloat(totalFrames) * w
                                    let rightX = CGFloat(range.end) / CGFloat(totalFrames) * w
                                    if abs(startX - leftX) < hitTol {
                                        dragEdge = .left(range.id, initialStart: range.start)
                                        selectedId = range.id
                                        break
                                    }
                                    if abs(startX - rightX) < hitTol {
                                        dragEdge = .right(range.id, initialEnd: range.end)
                                        selectedId = range.id
                                        break
                                    }
                                }
                                // Then check body
                                if dragEdge == nil {
                                    let startFrame = Int(startX / w * CGFloat(totalFrames))
                                    if let hit = ordered.first(where: { startFrame >= $0.start && startFrame < $0.end }) {
                                        dragEdge = .body(hit.id, startOffset: hit.start)
                                        selectedId = hit.id
                                    }
                                }
                            }

                            switch dragEdge {
                            case .left(let id, _):
                                let frame = Int(value.location.x / w * CGFloat(totalFrames))
                                updateLeft(id: id, to: frame)
                                if let idx = ranges.firstIndex(where: { $0.id == id }) {
                                    onScrub?(ranges[idx].start)
                                }
                            case .right(let id, _):
                                let frame = Int(value.location.x / w * CGFloat(totalFrames))
                                updateRight(id: id, to: frame)
                                if let idx = ranges.firstIndex(where: { $0.id == id }) {
                                    onScrub?(ranges[idx].end - 1)
                                }
                            case .body(let id, let startOffset):
                                let deltaFrames = Int(value.translation.width / w * CGFloat(totalFrames))
                                if let range = ranges.first(where: { $0.id == id }) {
                                    let rangeLen = range.count
                                    var newStart = startOffset + deltaFrames
                                    newStart = max(0, min(newStart, totalFrames - rangeLen))
                                    if let idx = ranges.firstIndex(where: { $0.id == id }) {
                                        ranges[idx].start = newStart
                                        ranges[idx].end = newStart + rangeLen
                                        onScrub?(newStart)
                                    }
                                }
                            case nil:
                                break
                            }
                        }
                        .onEnded { _ in
                            if case .body(let id, _) = dragEdge {
                                resolveBodyOverlap(id: id)
                            }
                            dragEdge = nil
                        }
                )
            }
            .frame(height: barHeight)

            // Info row
            HStack(spacing: 8) {
                let total = ranges.reduce(0) { $0 + $1.count }
                if ranges.isEmpty {
                    Text("All \(totalFrames) frames")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    Text("\(total) of \(totalFrames) frames (\(ranges.count) section\(ranges.count == 1 ? "" : "s"))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                // Delete selected
                if let selId = selectedId, ranges.contains(where: { $0.id == selId }) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            ranges.removeAll { $0.id == selId }
                            selectedId = nil
                        }
                    } label: {
                        Image(systemName: "trash")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                    .help("Remove selected range")
                }

                // Add range
                Button {
                    addRange()
                } label: {
                    Image(systemName: "plus")
                        .font(.caption)
                        .foregroundStyle(Color.accentColor)
                }
                .buttonStyle(.plain)
                .help("Add a range section")

                // Clear all
                if !ranges.isEmpty {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            ranges.removeAll()
                            selectedId = nil
                        }
                    } label: {
                        Text("Clear")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear all ranges (process full clip)")
                }
            }

            // Range detail for selected
            if let selId = selectedId, let idx = ranges.firstIndex(where: { $0.id == selId }) {
                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Text("In")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        TextField("", value: Binding(
                            get: { ranges[idx].start },
                            set: { newVal in
                                let clamped = max(0, min(newVal, ranges[idx].end - minRangeFrames))
                                ranges[idx].start = clamped
                            }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospacedDigit())
                        .frame(width: 60)
                    }
                    HStack(spacing: 4) {
                        Text("Out")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        TextField("", value: Binding(
                            get: { ranges[idx].end },
                            set: { newVal in
                                let clamped = min(totalFrames, max(newVal, ranges[idx].start + minRangeFrames))
                                ranges[idx].end = clamped
                            }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospacedDigit())
                        .frame(width: 60)
                    }
                    Text("\(ranges[idx].count) frames")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Range segment view

    @ViewBuilder
    private func rangeSegment(range: FrameRange, barWidth: CGFloat) -> some View {
        let isSel = selectedId == range.id
        let x = CGFloat(range.start) / CGFloat(totalFrames) * barWidth
        let w = CGFloat(range.count) / CGFloat(totalFrames) * barWidth

        ZStack {
            // Range body
            RoundedRectangle(cornerRadius: 3)
                .fill(isSel ? Color.accentColor.opacity(0.5) : Color.accentColor.opacity(0.3))
                .frame(width: max(w, 4), height: barHeight - 4)
                .overlay(
                    RoundedRectangle(cornerRadius: 3)
                        .strokeBorder(isSel ? Color.accentColor : Color.accentColor.opacity(0.6),
                                      lineWidth: isSel ? 2 : 1)
                )

            // Frame label inside
            if w > 60 {
                Text("\(range.start)–\(range.end)")
                    .font(.system(size: 9).monospacedDigit())
                    .foregroundStyle(isSel ? .primary : .secondary)
            }

            // Left handle (visual only)
            handle(edge: .leading)
                .offset(x: -w / 2 + handleWidth / 2)

            // Right handle (visual only)
            handle(edge: .trailing)
                .offset(x: w / 2 - handleWidth / 2)
        }
        .frame(width: max(w, 4), height: barHeight - 4)
        .allowsHitTesting(false)
        .position(x: x + w / 2, y: barHeight / 2)
    }

    @ViewBuilder
    private func handle(edge: HorizontalEdge) -> some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(Color.accentColor)
            .frame(width: handleWidth, height: barHeight - 8)
    }

    @ViewBuilder
    private func frameTicks(barWidth: CGFloat) -> some View {
        let interval = tickInterval(total: totalFrames, barWidth: barWidth)
        ForEach(Array(stride(from: 0, through: totalFrames, by: interval)), id: \.self) { frame in
            let x = CGFloat(frame) / CGFloat(totalFrames) * barWidth
            VStack(spacing: 0) {
                Rectangle()
                    .fill(Color(nsColor: .separatorColor).opacity(0.5))
                    .frame(width: 1, height: 6)
                Spacer()
            }
            .position(x: x, y: barHeight / 2)
        }
        .allowsHitTesting(false)
    }

    // MARK: - Helpers

    private func frameFromX(_ x: CGFloat, barWidth: CGFloat) -> Int {
        let fraction = x / barWidth
        return max(0, min(Int(fraction * CGFloat(totalFrames)), totalFrames))
    }

    private func tickInterval(total: Int, barWidth: CGFloat) -> Int {
        let maxTicks = Int(barWidth / 40)
        guard maxTicks > 0 else { return total }
        let raw = total / maxTicks
        // Snap to nice numbers
        let nice = [1, 2, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000]
        return nice.last(where: { $0 <= raw }) ?? nice.first!
    }

    private func tapOnTimeline(at frame: Int) {
        if let hit = ranges.first(where: { frame >= $0.start && frame < $0.end }) {
            let wasSelected = selectedId == hit.id
            selectedId = wasSelected ? nil : hit.id
            if !wasSelected {
                onScrub?(hit.start)
            }
        } else {
            selectedId = nil
        }
    }

    private func addRange() {
        let defaultLen = max(totalFrames / 10, minRangeFrames)
        let insertStart: Int

        if ranges.isEmpty {
            insertStart = 0
        } else {
            let sorted = ranges.sorted { $0.start < $1.start }
            // Find first gap large enough
            var candidate: Int? = nil
            // Check before first range
            if sorted.first!.start >= minRangeFrames {
                candidate = 0
            }
            if candidate == nil {
                for i in 0..<sorted.count {
                    let gapStart = sorted[i].end
                    let gapEnd = (i + 1 < sorted.count) ? sorted[i + 1].start : totalFrames
                    if gapEnd - gapStart >= minRangeFrames {
                        candidate = gapStart
                        break
                    }
                }
            }
            guard let found = candidate else { return }
            insertStart = found
        }

        let insertEnd = min(insertStart + defaultLen, totalFrames)
        guard insertEnd - insertStart >= minRangeFrames else { return }

        let newRange = FrameRange(start: insertStart, end: insertEnd)
        withAnimation(.easeInOut(duration: 0.2)) {
            ranges.append(newRange)
            selectedId = newRange.id
        }
        onScrub?(insertStart)
    }

    private func updateLeft(id: UUID, to frame: Int) {
        guard let idx = ranges.firstIndex(where: { $0.id == id }) else { return }
        var clamped = max(0, min(frame, ranges[idx].end - minRangeFrames))

        // Prevent overlapping other ranges
        for (i, other) in ranges.enumerated() where i != idx {
            if clamped < other.end && ranges[idx].end > other.start {
                clamped = max(clamped, other.end)
            }
        }
        clamped = min(clamped, ranges[idx].end - minRangeFrames)
        ranges[idx].start = clamped
    }

    private func updateRight(id: UUID, to frame: Int) {
        guard let idx = ranges.firstIndex(where: { $0.id == id }) else { return }
        var clamped = min(totalFrames, max(frame, ranges[idx].start + minRangeFrames))

        // Prevent overlapping other ranges
        for (i, other) in ranges.enumerated() where i != idx {
            if clamped > other.start && ranges[idx].start < other.end {
                clamped = min(clamped, other.start)
            }
        }
        clamped = max(clamped, ranges[idx].start + minRangeFrames)
        ranges[idx].end = clamped
    }

    /// On body drag release, snap the range to the nearest non-overlapping position.
    private func resolveBodyOverlap(id: UUID) {
        guard let idx = ranges.firstIndex(where: { $0.id == id }) else { return }
        let rangeLen = ranges[idx].count
        let others = ranges.filter { $0.id != id }.sorted { $0.start < $1.start }
        var start = ranges[idx].start

        for _ in 0...others.count {
            guard let overlap = others.first(where: {
                start < $0.end && (start + rangeLen) > $0.start
            }) else { break }

            let snapLeft = overlap.start - rangeLen
            let snapRight = overlap.end

            if abs(snapLeft - start) <= abs(snapRight - start) && snapLeft >= 0 {
                start = snapLeft
            } else if snapRight + rangeLen <= totalFrames {
                start = snapRight
            } else if snapLeft >= 0 {
                start = snapLeft
            } else {
                break
            }
        }

        start = max(0, min(start, totalFrames - rangeLen))
        withAnimation(.easeInOut(duration: 0.15)) {
            ranges[idx].start = start
            ranges[idx].end = start + rangeLen
        }
    }

}
