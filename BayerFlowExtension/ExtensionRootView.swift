import SwiftUI
import UniformTypeIdentifiers
import AppKit

// MARK: - Clip model

struct FCPClip: Identifiable {
    let id = UUID()
    let name: String
    let url: URL
    let format: String    // e.g. "ProRes RAW", "BRAW", "CinemaDNG"
}

// MARK: - Extension state

@MainActor
final class ExtensionState: ObservableObject {
    @Published var clips: [FCPClip] = []
    @Published var strength: Double = 1.5
    @Published var windowSize: Double = 15
    @Published var spatialStrength: Double = 0.0
    @Published var useCNN: Bool = true
    @Published var isProcessing: Bool = false
    @Published var progress: Double = 0
    @Published var currentFrame: Int = 0
    @Published var totalFrames: Int = 0
    @Published var statusMessage: String = "Drop RAW clips here or browse"
    @Published var outputFormat: Int = 0  // 0=auto, 1=MOV, 2=DNG

    /// Detect format from file extension
    func detectFormat(_ url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "mov":  return "ProRes RAW"
        case "braw": return "BRAW"
        case "dng":  return "CinemaDNG"
        case "ari":  return "ARRIRAW"
        case "crm":  return "Canon CRM"
        case "r3d":  return "RED R3D"
        case "mxf":  return "MXF"
        default:     return "Unknown"
        }
    }

    func addClip(_ url: URL) {
        let format = detectFormat(url)
        let clip = FCPClip(name: url.lastPathComponent, url: url, format: format)
        clips.append(clip)
        statusMessage = "\(clips.count) clip\(clips.count == 1 ? "" : "s") ready"
    }

    func removeClip(_ clip: FCPClip) {
        clips.removeAll { $0.id == clip.id }
        statusMessage = clips.isEmpty ? "Drop RAW clips here or browse" : "\(clips.count) clip\(clips.count == 1 ? "" : "s") ready"
    }

    /// Launch BayerFlow main app with the clips to process
    func launchBayerFlow() {
        guard !clips.isEmpty else { return }

        // Build URL scheme or use NSWorkspace to launch main app with arguments
        let mainAppURL = Bundle.main.bundleURL
            .deletingLastPathComponent()  // PlugIns/
            .deletingLastPathComponent()  // Contents/
            .deletingLastPathComponent()  // BayerFlow.app/

        // Pass clip paths as arguments via distributed notification
        let clipPaths = clips.map { $0.url.path }
        let config: [String: Any] = [
            "clips": clipPaths,
            "strength": strength,
            "windowSize": Int(windowSize),
            "spatialStrength": spatialStrength,
            "useCNN": useCNN,
            "outputFormat": outputFormat
        ]

        // Post notification that main app listens for
        DistributedNotificationCenter.default().postNotificationName(
            Notification.Name("com.bayerflow.app.fcpDenoise"),
            object: nil,
            userInfo: config,
            deliverImmediately: true
        )

        // Also launch the main app if not running
        let ws = NSWorkspace.shared
        if ws.runningApplications.first(where: { $0.bundleIdentifier == "com.bayerflow.app" }) == nil {
            let openConfig = NSWorkspace.OpenConfiguration()
            openConfig.arguments = ["--fcp-denoise"] + clipPaths
            ws.openApplication(at: mainAppURL, configuration: openConfig) { _, error in
                if let error = error {
                    Task { @MainActor in
                        self.statusMessage = "Failed to launch BayerFlow: \(error.localizedDescription)"
                    }
                }
            }
        }

        isProcessing = true
        statusMessage = "Launched BayerFlow — processing..."

        // Listen for progress updates from main app
        DistributedNotificationCenter.default().addObserver(
            forName: Notification.Name("com.bayerflow.app.fcpProgress"),
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self else { return }
            if let info = notification.userInfo {
                if let p = info["progress"] as? Double { self.progress = p }
                if let f = info["frame"] as? Int { self.currentFrame = f }
                if let t = info["total"] as? Int { self.totalFrames = t }
                if let done = info["done"] as? Bool, done {
                    self.isProcessing = false
                    self.statusMessage = "Done! Output ready in source folder."
                    if let outputPath = info["output"] as? String {
                        self.statusMessage = "Done! → \(URL(fileURLWithPath: outputPath).lastPathComponent)"
                    }
                }
            }
        }
    }
}

// MARK: - Root view

struct ExtensionRootView: View {
    @StateObject private var state = ExtensionState()
    @State private var isDragOver = false

    private let rawExtensions: Set<String> = ["mov", "braw", "dng", "ari", "crm", "r3d", "mxf"]

    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerBar

            Divider().background(Color.white.opacity(0.1))

            // Main content
            ScrollView {
                VStack(spacing: 16) {
                    dropZone
                    if !state.clips.isEmpty {
                        clipList
                        settingsPanel
                        processButton
                    }
                }
                .padding(16)
            }

            // Status bar
            statusBar
        }
        .frame(minWidth: 480, minHeight: 640)
        .background(Color(nsColor: NSColor.windowBackgroundColor))
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            Image(systemName: "wand.and.rays")
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(.orange)
            Text("BayerFlow")
                .font(.system(size: 16, weight: .bold))
            Spacer()
            Text("FCP Extension")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - Drop zone

    private var dropZone: some View {
        VStack(spacing: 12) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 36))
                .foregroundColor(isDragOver ? .orange : .secondary)

            Text(state.clips.isEmpty ? "Drop RAW clips from FCP" : "Drop more clips")
                .font(.headline)
                .foregroundColor(isDragOver ? .orange : .primary)

            Text("ProRes RAW, BRAW, CinemaDNG, ARRIRAW, CRM, R3D")
                .font(.caption)
                .foregroundColor(.secondary)

            Button("Browse Files...") {
                browseFiles()
            }
            .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    isDragOver ? Color.orange : Color.secondary.opacity(0.3),
                    style: StrokeStyle(lineWidth: 2, dash: [8, 4])
                )
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isDragOver ? Color.orange.opacity(0.05) : Color.clear)
                )
        )
        .onDrop(of: [.fileURL], isTargeted: $isDragOver) { providers in
            handleDrop(providers)
        }
    }

    // MARK: - Clip list

    private var clipList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("CLIPS")
                .font(.caption)
                .foregroundColor(.secondary)
                .fontWeight(.semibold)

            ForEach(state.clips) { clip in
                HStack {
                    Image(systemName: "film")
                        .foregroundColor(.orange)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(clip.name)
                            .font(.system(size: 13, weight: .medium))
                            .lineLimit(1)
                        Text(clip.format)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button {
                        state.removeClip(clip)
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
            }
        }
    }

    // MARK: - Settings

    private var settingsPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("DENOISE SETTINGS")
                .font(.caption)
                .foregroundColor(.secondary)
                .fontWeight(.semibold)

            // Strength
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Strength")
                        .font(.system(size: 13))
                    Spacer()
                    Text(String(format: "%.1f", state.strength))
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Slider(value: $state.strength, in: 0.5...3.0, step: 0.1)
                    .tint(.orange)
            }

            // Temporal window
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Temporal Window")
                        .font(.system(size: 13))
                    Spacer()
                    Text("\(Int(state.windowSize)) frames")
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Slider(value: $state.windowSize, in: 3...31, step: 2)
                    .tint(.orange)
            }

            // Spatial strength
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Spatial")
                        .font(.system(size: 13))
                    Spacer()
                    Text(String(format: "%.1f", state.spatialStrength))
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Slider(value: $state.spatialStrength, in: 0.0...2.0, step: 0.1)
                    .tint(.orange)
            }

            // CNN toggle
            Toggle("CNN Post-Filter", isOn: $state.useCNN)
                .font(.system(size: 13))
                .tint(.orange)

            // Output format
            Picker("Output", selection: $state.outputFormat) {
                Text("Auto (match input)").tag(0)
                Text("ProRes RAW .mov").tag(1)
                Text("CinemaDNG .dng").tag(2)
            }
            .font(.system(size: 13))
            .pickerStyle(.menu)
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.06)))
    }

    // MARK: - Process button

    private var processButton: some View {
        VStack(spacing: 8) {
            Button {
                state.launchBayerFlow()
            } label: {
                HStack {
                    if state.isProcessing {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 16, height: 16)
                        if state.totalFrames > 0 {
                            Text("Frame \(state.currentFrame)/\(state.totalFrames)")
                        } else {
                            Text("Processing...")
                        }
                    } else {
                        Image(systemName: "wand.and.stars")
                        Text("Denoise in BayerFlow")
                    }
                }
                .font(.system(size: 14, weight: .semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .disabled(state.isProcessing || state.clips.isEmpty)

            if state.isProcessing && state.totalFrames > 0 {
                ProgressView(value: state.progress, total: 1.0)
                    .tint(.orange)
            }
        }
    }

    // MARK: - Status bar

    private var statusBar: some View {
        HStack {
            Circle()
                .fill(state.isProcessing ? Color.orange : Color.green)
                .frame(width: 6, height: 6)
            Text(state.statusMessage)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.secondary.opacity(0.05))
    }

    // MARK: - Actions

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        var handled = false
        for provider in providers {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                guard let data = item as? Data,
                      let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                let ext = url.pathExtension.lowercased()
                guard rawExtensions.contains(ext) else { return }
                Task { @MainActor in
                    state.addClip(url)
                }
            }
            handled = true
        }
        return handled
    }

    private func browseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [
            .movie,
            UTType(filenameExtension: "braw") ?? .data,
            UTType(filenameExtension: "dng") ?? .data,
            UTType(filenameExtension: "ari") ?? .data,
            UTType(filenameExtension: "crm") ?? .data,
            UTType(filenameExtension: "r3d") ?? .data,
            UTType(filenameExtension: "mxf") ?? .data,
        ]
        panel.begin { response in
            guard response == .OK else { return }
            for url in panel.urls {
                state.addClip(url)
            }
        }
    }
}
