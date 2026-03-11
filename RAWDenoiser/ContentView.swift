import SwiftUI
import UniformTypeIdentifiers

// MARK: - SessionView

struct SessionView: View {
    let session: DenoisingSession
    var onFileLoaded: ((String, Color) -> Void)?

    @EnvironmentObject private var license: LicenseManager
    @EnvironmentObject private var showcase: ShowcaseController
    @StateObject private var engine = DenoiseEngine()
    @StateObject private var fcpDetector = FCPLibraryDetector()

    @State private var inputURL:  URL? = nil
    @State private var outputURL: URL? = nil
    @State private var isDragOver = false
    @State private var showingFilePicker   = false
    @State private var showingOutputPicker = false

    // Slider values
    @State private var strength: Float = 1.5
    @State private var windowSize: Double = 15
    @State private var spatialStrength: Float = 0.0
    @State private var selectedPreset: DenoisePreset = .strong

    // Dark frame

    // Preview
    @State private var previewFrameIndex: Double = 0
    @State private var frameCount: Int = 0

    // Trim ranges (empty = full clip)
    @State private var selectedRanges: [FrameRange] = []
    @State private var scrubFrame: Double = 0

    // Video info
    @State private var videoWidth: Int = 0
    @State private var videoHeight: Int = 0

    // Output format for CinemaDNG (0=DNG, 1=ProRes RAW MOV)
    @State private var dngOutputFormat: Int = 0
    // Output format for BRAW (0=BRAW, 1=ProRes RAW MOV)
    @State private var brawOutputFormat: Int = 0

    // Watch folder
    @State private var showingFolderPicker = false

    // Settings
    @AppStorage("autoRevealInFinder") private var autoRevealInFinder = false
    @AppStorage("defaultOutputDirectory") private var defaultOutputDirectory = ""
    @AppStorage("rememberSettings") private var rememberSettings = true
    @AppStorage("savedStrength") private var savedStrength: Double = 1.0
    @AppStorage("savedWindowSize") private var savedWindowSize: Double = 9
    @AppStorage("savedCNNPostfilter") private var savedCNNPostfilter: Bool = false

    // Done animation
    @State private var showCheckmark = false

    // Processing timing + stats
    @State private var processingStartTime: Date? = nil
    @State private var processingElapsed: TimeInterval = 0
    @State private var elapsedTimer: Timer? = nil
    @State private var lastProcessingStats: (frames: Int, fps: Double, elapsed: TimeInterval)? = nil

    // Input validation alert
    @State private var inputError: String? = nil

    // Sidebar disclosure state
    @State private var denoiseSettingsExpanded = true
    @State private var outputExpanded = true

    // Hub
    @State private var showHub = true
    @State private var showSDKAlert = false


    // LUT preview
    @State private var lutEnabled = false
    @State private var lutURL: URL? = nil
    @State private var lutBlend: Float = 1.0
    @State private var loadedLUT: CubeLUT? = nil
    @State private var lutAppliedPreview: NSImage? = nil
    @State private var showingLUTPicker = false
    @State private var lutExpanded = false

    // Scope
    @State private var scopeEnabled = false
    @State private var scopeMode: ScopeMode = .histogram

    // Noise profiler
    @StateObject private var noiseProfiler = NoiseProfiler()
    @State private var noiseProfileMode: Bool = false
    @State private var noiseProfileSelection: CGRect? = nil
    @State private var rawProfileImage: NSImage? = nil   // raw frame loaded on-demand for patch selection
    @State private var scopeData: ScopeData? = nil

    // Auto-save
    @State private var autoSaveTimer: Timer? = nil
    @State private var lastSaveHash: Int = 0

    // Menu bar action bridge
    private let appActions = AppActions()

    private let movType = UTType(filenameExtension: "mov") ?? .movie
    private let dngType = UTType(filenameExtension: "dng") ?? .data
    private let brawType = UTType(filenameExtension: "braw") ?? .data
    private let crmType = UTType(filenameExtension: "crm") ?? .data
    private let ariType = UTType(filenameExtension: "ari") ?? .data
    private let mxfType = UTType(filenameExtension: "mxf") ?? .data
    private let r3dType = UTType(filenameExtension: "r3d") ?? .data
    private let nrawType = UTType(filenameExtension: "nraw") ?? .data

    @ViewBuilder
    private var mainContent: some View {
        if showHub {
            FormatHubView { format in
                if format.needsSDK && r3d_sdk_available() == 0 {
                    showSDKAlert = true
                } else {
                    resetSession()
                    showHub = false
                }
            }
            .frame(width: 960, height: 840)
            .alert("RED SDK Required", isPresented: $showSDKAlert) {
                Link("Download SDK", destination: URL(string: "https://www.red.com/developer")!)
                Button("OK", role: .cancel) {}
            } message: {
                Text("RED R3D support requires the RED SDK, which is free to download from red.com/developer. After installing, rebuild BayerFlow to enable R3D decoding.")
            }
        } else {
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 300)
                Divider()
                rightPanel
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .frame(width: 960, height: 840)
            .overlay {
                if showcase.isActive && showcase.cursorPhase != .hidden {
                    ShowcaseCursorView(
                        fileName: URL(fileURLWithPath: showcase.filePath).lastPathComponent,
                        phase: showcase.cursorPhase
                    )
                }
            }
        }
    }

    private var bodyCore: some View {
        mainContent
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                if !showHub {
                    Button {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            showHub = true
                        }
                    } label: {
                        Label("Hub", systemImage: "chevron.left")
                    }
                    .help("Back to format hub")
                }
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                if !showHub {
                    Button {
                        showingFilePicker = true
                    } label: {
                        Label("Open", systemImage: "folder")
                    }
                    .help("Open a ProRes RAW file (⌘O)")

                    if engine.isGPUAvailable {
                        Text("GPU")
                            .font(.caption2.bold())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.15), in: Capsule())
                            .foregroundStyle(.green)
                            .accessibilityLabel("GPU acceleration active")
                    }
                }
            }

            ToolbarItemGroup(placement: .primaryAction) {
                if !showHub {
                    if case .processing = engine.state {
                        Button {
                            engine.cancel()
                        } label: {
                            Label("Cancel", systemImage: "xmark.circle")
                        }
                        .help("Cancel processing (⌘.)")
                    } else {
                        Button {
                            triggerDenoise()
                        } label: {
                            Label("Denoise", systemImage: "wand.and.sparkles")
                        }
                        .help("Start denoising (⌘D)")
                        .disabled(inputURL == nil || !license.canDenoise || engine.isQueueRunning)
                    }
                }

                licenseChip
            }
        }
        .navigationTitle("BayerFlow")
        .focusedValue(\.appActions, appActions)
        .onAppear { wireAppActions() }
        .onChange(of: inputURL) { _, _ in wireAppActions() }
        .onChange(of: engine.state) { oldState, newState in
            wireAppActions()
            if case .processing = newState, processingStartTime == nil {
                processingStartTime = Date()
                processingElapsed = 0
                elapsedTimer?.invalidate()
                elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                    if let start = processingStartTime {
                        processingElapsed = Date().timeIntervalSince(start)
                    }
                }
            } else if case .processing = newState {
                // keep existing start time + timer
            } else {
                // Capture stats when transitioning from processing to done
                if case .processing(_, let frame, let total, _, let fps) = oldState,
                   case .done = newState,
                   let start = processingStartTime {
                    let elapsed = Date().timeIntervalSince(start)
                    let finalFrames = total > 0 ? total : frame
                    lastProcessingStats = (frames: finalFrames, fps: fps, elapsed: elapsed)
                }
                elapsedTimer?.invalidate()
                elapsedTimer = nil
                processingStartTime = nil
            }
        }
        .sheet(isPresented: $license.showActivation) {
            LicenseView()
                .environmentObject(license)
        }
        .alert("Trial Expired", isPresented: .constant(license.isTrialExpired && !license.showActivation)) {
            Button("Buy License") {
                if let url = URL(string: "https://bayerflow.com/buy") {
                    NSWorkspace.shared.open(url)
                }
            }
            Button("Activate License") { license.showActivation = true }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Your 14-day trial has ended. Purchase a license at bayerflow.com or activate an existing key.")
        }
        .alert("Unsupported File", isPresented: Binding(
            get: { inputError != nil },
            set: { if !$0 { inputError = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(inputError ?? "")
        }
        .task { fcpDetector.scan() }
        .onAppear {
            if rememberSettings {
                strength = Float(savedStrength)
                windowSize = savedWindowSize
                engine.useCNNPostfilter = savedCNNPostfilter
                selectedPreset = DenoisePreset.matching(strength: strength, windowSize: windowSize)
            }
        }
        .onChange(of: strength) { _, _ in if rememberSettings { savedStrength = Double(strength) } }
        .onChange(of: windowSize) { _, _ in if rememberSettings { savedWindowSize = windowSize } }
        .onChange(of: engine.useCNNPostfilter) { _, val in if rememberSettings { savedCNNPostfilter = val } }
        .fileImporter(isPresented: $showingFilePicker,
                      allowedContentTypes: [movType, .movie, dngType, brawType, crmType, ariType, mxfType, r3dType, nrawType]) { result in
            if case .success(let url) = result { setInput(url) }
        }
        .fileExporter(isPresented: $showingOutputPicker,
                      document: EmptyDocument(),
                      contentType: movType,
                      defaultFilename: suggestedOutputName()) { result in
            if case .success(let url) = result { outputURL = url }
        }
    }

    var body: some View {
        bodyCore
        .fileImporter(isPresented: $showingLUTPicker,
                      allowedContentTypes: [UTType(filenameExtension: "cube") ?? .data]) { result in
            if case .success(let url) = result {
                do {
                    loadedLUT = try CubeLUTLoader.load(from: url)
                    lutURL = url
                    lutEnabled = true
                    reapplyLUT()
                } catch {
                    inputError = "Failed to load LUT: \(error.localizedDescription)"
                }
            }
        }
        .onChange(of: engine.previewImage) { _, _ in
            reapplyLUT()
            computeScopeData()
        }
        .onChange(of: lutEnabled) { _, _ in reapplyLUT(); computeScopeData() }
        .onChange(of: lutBlend) { _, _ in reapplyLUT(); computeScopeData() }
        .onAppear {
            // Restore from pending snapshot if recovering
            if let snapshot = session.pendingRestore {
                restoreState(from: snapshot)
            }
            // Start auto-save timer
            autoSaveTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
                let hash = computeStateHash()
                guard hash != lastSaveHash else { return }
                lastSaveHash = hash
                let snapshot = snapshotState()
                SessionPersistenceManager.shared.save(snapshot)
            }
        }
        .onDisappear {
            autoSaveTimer?.invalidate()
            autoSaveTimer = nil
            SessionPersistenceManager.shared.delete(id: session.id.uuidString)
        }
        // — Showcase mode handlers —
        .onChange(of: showcase.resetToHub) { _, reset in
            if reset && showcase.isActive {
                showHub = true
                showcase.hubVisible = true
                showcase.resetToHub = false
            }
        }
        .onChange(of: showcase.simulateDragOver) { _, over in
            if showcase.isActive { isDragOver = over }
        }
        .onChange(of: showcase.triggerFileLoad) { _, url in
            if let url, showcase.isActive { setInput(url) }
        }
        .onChange(of: showcase.targetPreset) { _, preset in
            if let preset, showcase.isActive {
                withAnimation(.easeInOut(duration: 0.3)) {
                    selectedPreset = preset
                    strength = preset.strength
                    windowSize = preset.windowSize
                }
            }
        }
        .onChange(of: showcase.targetStrength) { _, val in
            if let val, showcase.isActive { strength = val }
        }
        .onChange(of: showcase.targetWindowSize) { _, val in
            if let val, showcase.isActive { windowSize = val }
        }
        .onChange(of: showcase.targetProtectSubjects) { _, val in
            if let val, showcase.isActive { engine.protectSubjects = val }
        }
        .onChange(of: showcase.triggerPreview) { _, trigger in
            if trigger && showcase.isActive, let url = inputURL {
                engine.generatePreview(inputURL: url,
                                       frameIndex: Int(previewFrameIndex),
                                       strength: strength,
                                       windowSize: Int32(windowSize),
                                       spatialStrength: spatialStrength)
            }
        }
        .onChange(of: showcase.triggerDenoise) { _, trigger in
            if trigger && showcase.isActive { triggerDenoise() }
        }
        .onChange(of: showHub) { _, hub in
            if showcase.isActive { showcase.hubVisible = hub }
        }
        .onChange(of: engine.state) { _, newState in
            guard showcase.isActive else { return }
            if case .ready = newState { showcase.analysisComplete = true }
            if case .done = newState { showcase.denoiseComplete = true }
        }
        .onChange(of: engine.previewImage) { _, img in
            if showcase.isActive && img != nil { showcase.previewComplete = true }
        }
    }

    // MARK: - Menu bar action wiring

    private func wireAppActions() {
        appActions.openFile = { showingFilePicker = true }
        appActions.startDenoise = { triggerDenoise() }
        appActions.addToQueue = { triggerAddToQueue() }
        appActions.generatePreview = {
            guard let url = inputURL else { return }
            engine.generatePreview(inputURL: url,
                                   frameIndex: Int(previewFrameIndex),
                                   strength: strength,
                                   windowSize: Int32(windowSize),
                                   spatialStrength: spatialStrength)
        }
        appActions.cancelProcessing = { engine.cancel() }
        appActions.hasInput = inputURL != nil
        appActions.canDenoise = inputURL != nil && license.canDenoise && !engine.isQueueRunning
        if case .processing = engine.state {
            appActions.canCancel = true
        } else {
            appActions.canCancel = false
        }
    }

    private func triggerDenoise() {
        guard license.canDenoise else { license.showActivation = true; return }
        guard let inURL = inputURL else { return }
        let outURL = outputURL ?? defaultOutputURL()
        let outFmt: Int32
        if inputIsCinemaDNG {
            outFmt = Int32(dngOutputFormat)
        } else if inputIsBRAW {
            outFmt = 3  // always BRAW output
        } else if inputIsR3D {
            outFmt = Int32(dngOutputFormat)  // 0=ProRes 4444, 4=EXR
        } else {
            outFmt = 0
        }
        if selectedRanges.isEmpty {
            engine.start(inputURL: inURL, outputURL: outURL,
                         strength: strength,
                         windowSize: Int32(windowSize),
                         spatialStrength: spatialStrength,
                         outputFormat: outFmt)
        } else if selectedRanges.count == 1 {
            let r = selectedRanges[0]
            engine.start(inputURL: inURL, outputURL: outURL,
                         strength: strength,
                         windowSize: Int32(windowSize),
                         spatialStrength: spatialStrength,
                         startFrame: Int32(r.start),
                         endFrame: Int32(r.end),
                         outputFormat: outFmt)
        } else {
            let sorted = selectedRanges.sorted { $0.start < $1.start }
            for (i, r) in sorted.enumerated() {
                let rangeURL = rangeOutputURL(base: outURL, index: i + 1, range: r)
                engine.addToQueue(inputURL: inURL, outputURL: rangeURL,
                                  strength: strength,
                                  windowSize: Int32(windowSize),
                                  spatialStrength: spatialStrength,
                                  startFrame: Int32(r.start),
                                  endFrame: Int32(r.end),
                                  outputFormat: outFmt)
            }
            engine.startQueueParallel()
        }
    }

    private func triggerAddToQueue() {
        guard let inURL = inputURL else { return }
        let outURL = outputURL ?? defaultOutputURL()
        let outFmt: Int32
        if inputIsCinemaDNG {
            outFmt = Int32(dngOutputFormat)
        } else if inputIsBRAW {
            outFmt = 3  // always BRAW output
        } else if inputIsR3D {
            outFmt = Int32(dngOutputFormat)  // 0=ProRes 4444, 4=EXR
        } else {
            outFmt = 0
        }
        if selectedRanges.isEmpty {
            engine.addToQueue(inputURL: inURL, outputURL: outURL,
                              strength: strength,
                              windowSize: Int32(windowSize),
                              spatialStrength: spatialStrength,
                              outputFormat: outFmt)
        } else {
            let sorted = selectedRanges.sorted { $0.start < $1.start }
            for (i, r) in sorted.enumerated() {
                let rangeURL = sorted.count == 1 ? outURL : rangeOutputURL(base: outURL, index: i + 1, range: r)
                engine.addToQueue(inputURL: inURL, outputURL: rangeURL,
                                  strength: strength,
                                  windowSize: Int32(windowSize),
                                  spatialStrength: spatialStrength,
                                  startFrame: Int32(r.start),
                                  endFrame: Int32(r.end),
                                  outputFormat: outFmt)
            }
        }
        inputURL = nil
        outputURL = nil
        engine.previewImage = nil
        engine.state = .idle
        rawProfileImage = nil
        noiseProfileMode = false
    }

    @ViewBuilder
    private var licenseChip: some View {
        if license.isLicensed {
            Label("Licensed", systemImage: "checkmark.seal.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else {
            Button(license.licenseStatusText) {
                license.showActivation = true
            }
            .buttonStyle(.plain)
            .font(.caption)
            .foregroundStyle(license.isTrialExpired ? .red : .secondary)
        }
    }

    // MARK: - Sidebar (left column)

    private var sidebar: some View {
        VStack(spacing: 12) {
            // File info (when loaded)
            if let url = inputURL {
                fileInfoRow(url: url)
            }

            // Motion + camera badges (when ready)
            switch engine.state {
            case .ready, .cancelled, .failed:
                motionBadge
                cameraBadge
            default:
                EmptyView()
            }

            // Denoise Settings
            sectionHeader("Denoise Settings", icon: "slider.horizontal.3", expanded: $denoiseSettingsExpanded)
            if denoiseSettingsExpanded {
                sliderControls
                    .padding(.top, 4)
            }

            // Ghosting indicator (only when footage is loaded and analyzed)
            switch engine.state {
            case .ready, .cancelled, .failed:
                if inputURL != nil { ghostingIndicator }
            default:
                EmptyView()
            }

            // Output
            sectionHeader("Output", icon: "folder", expanded: $outputExpanded)
            if outputExpanded {
                outputRow
                    .padding(.top, 4)
            }

            // LUT Preview
            sectionHeader("LUT Preview", icon: "camera.filters", expanded: $lutExpanded)
            if lutExpanded {
                lutControls
                    .padding(.top, 4)
            }

            // Action buttons (stacked vertically for sidebar)
            actionButtons

            Spacer()
        }
        .padding(16)
        .background(Color(nsColor: .windowBackgroundColor))
        .disabled({
            switch engine.state {
            case .analyzing, .processing: return true
            default: return false
            }
        }())
    }

    // MARK: - Section header

    private func sectionHeader(_ title: String, icon: String, expanded: Binding<Bool>) -> some View {
        Button {
            expanded.wrappedValue.toggle()
        } label: {
            HStack {
                Label(title, systemImage: icon)
                    .font(.subheadline.bold())
                    .foregroundStyle(.primary)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                    .rotationEffect(.degrees(expanded.wrappedValue ? 90 : 0))
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Right panel (main area)

    @ViewBuilder
    private var rightPanel: some View {
        switch engine.state {
        case .idle:
            dropZone
                .padding(24)
                .contentShape(Rectangle())
                .onDrop(of: [.fileURL], isTargeted: $isDragOver) { providers in
                    handleDrop(providers)
                }
                .onTapGesture {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [movType, .movie, dngType, brawType, crmType, ariType, mxfType, r3dType, nrawType]
                    panel.allowsMultipleSelection = false
                    panel.canChooseDirectories = true
                    if panel.runModal() == .OK, let url = panel.url {
                        setInput(url)
                    }
                }
        case .analyzing(let pct):
            analyzingView(pct: pct)
        case .ready, .cancelled, .failed:
            ScrollView {
                VStack(spacing: 16) {
                    if case .failed(let msg) = engine.state {
                        HStack {
                            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                            Text(msg).font(.callout)
                            Spacer()
                            Button("Dismiss") { engine.state = .ready }.buttonStyle(.plain)
                        }
                        .padding(10)
                        .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    }

                    previewSection

                    trimControls

                    Divider()

                    if !engine.queue.isEmpty || engine.isWatching {
                        queuePanel
                    }

                    watchFolderPanel
                }
                .padding(24)
            }
        case .processing(let pct, let frame, let total, let eta, let fps):
            processingView(pct: pct, frame: frame, total: total, eta: eta, fps: fps)
        case .done(let url):
            doneView(url: url)
        }
    }

    // MARK: - Analyzing view

    private func analyzingView(pct: Double) -> some View {
        VStack(spacing: 24) {
            Spacer()
            VStack(spacing: 12) {
                ProgressView()
                    .controlSize(.large)
                Text("Analyzing motion...")
                    .font(.headline)
                ProgressView(value: pct)
                    .progressViewStyle(.linear)
                    .frame(width: 200)
                Text("\(Int(pct * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button("Cancel") { engine.cancel(); engine.reset() }
                .buttonStyle(.bordered)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Motion badge

    private var motionBadge: some View {
        let avg = engine.motionAvg
        let level: String
        let color: Color
        if avg < 1.0 {
            level = "Low"; color = .green
        } else if avg < 4.0 {
            level = "Medium"; color = .orange
        } else {
            level = "High"; color = .red
        }

        return HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text("Motion: \(level)")
                .font(.subheadline.bold())
                .foregroundStyle(color)
            Text(String(format: "(%.1f px avg)", avg))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                setInput(inputURL!)
            } label: {
                Label("Re-analyze", systemImage: "arrow.clockwise")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Camera info badge

    @ViewBuilder
    private var cameraBadge: some View {
        if engine.cameraModel != nil || engine.detectedISO != nil {
            HStack(spacing: 6) {
                Image(systemName: "camera.fill")
                    .foregroundStyle(.secondary)
                    .font(.caption)

                if let profile = engine.matchedProfile {
                    Text(profile)
                        .font(.subheadline)
                } else if let camera = engine.cameraModel {
                    Text(camera)
                        .font(.subheadline)
                }

                if let iso = engine.detectedISO {
                    Text("ISO \(iso)")
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.12), in: Capsule())
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if let src = engine.calibrationSource {
                    Label(src, systemImage: "waveform.badge.checkmark")
                        .font(.caption2)
                        .foregroundStyle(.green)
                } else if let sigma = engine.suggestedSigma {
                    Text(String(format: "Suggested: %.1f", sigma))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .background(Color.blue.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: - Slider controls

    private var sliderControls: some View {
        VStack(spacing: 14) {
            // Preset picker
            VStack(alignment: .leading, spacing: 6) {
                Text("Preset").font(.subheadline)
                Picker("Preset", selection: $selectedPreset) {
                    ForEach(DenoisePreset.allCases) { preset in
                        Text(preset.rawValue).tag(preset)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .onChange(of: selectedPreset) { _, preset in
                    guard preset != .custom else { return }
                    strength = preset.strength
                    windowSize = preset.windowSize
                }
                Text(selectedPreset.hint)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .animation(.easeInOut(duration: 0.15), value: selectedPreset)
            }

            // Temporal strength
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Temporal Strength").font(.subheadline)
                        .help("Controls how aggressively the temporal filter reduces noise across frames. Higher values remove more noise but may soften fine detail.")
                    Spacer()
                    Text(String(format: "%.1f", strength))
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $strength, in: 0.5...2.0, step: 0.1)
                    .accessibilityLabel("Temporal Strength")
                    .accessibilityValue(String(format: "%.1f", strength))
                    .onChange(of: strength) { _, val in
                        if selectedPreset != .custom {
                            let match = DenoisePreset.matching(strength: val, windowSize: windowSize)
                            if match != selectedPreset { selectedPreset = match }
                        }
                    }
                HStack {
                    Text("More grain").font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                    Text("Smoother").font(.caption2).foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Window Size").font(.subheadline)
                        .help("Number of neighboring frames used for temporal denoising. Larger windows give stronger noise reduction but take longer to process.")
                    Spacer()
                    Text("\(Int(windowSize)) frames")
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $windowSize, in: 3...15, step: 2)
                    .accessibilityLabel("Window Size")
                    .accessibilityValue("\(Int(windowSize)) frames")
                    .onChange(of: windowSize) { _, val in
                        if selectedPreset != .custom {
                            let match = DenoisePreset.matching(strength: strength, windowSize: val)
                            if match != selectedPreset { selectedPreset = match }
                        }
                    }
                HStack {
                    Text("Faster").font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                    Text("Stronger denoise").font(.caption2).foregroundStyle(.secondary)
                }
            }

            Toggle(isOn: $engine.useCNNPostfilter) {
                HStack(spacing: 6) {
                    Image(systemName: "cpu")
                        .foregroundStyle(.orange)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("CNN Post-Filter")
                            .font(.subheadline)
                        Text("DnCNN refinement pass · ~40% slower")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .toggleStyle(.switch)
            .help("Apply a DnCNN neural network pass after temporal filtering for additional noise reduction. Increases processing time by ~40%.")

            // Subject protection
            Toggle(isOn: $engine.protectSubjects) {
                HStack(spacing: 6) {
                    Image(systemName: "person.fill")
                        .foregroundStyle(.blue)
                    Text("Protect Subjects")
                        .font(.subheadline)
                }
            }
            .toggleStyle(.switch)
            .help("Boost denoising on detected people for cleaner skin and hair")

            if engine.protectSubjects {
                Toggle(isOn: $engine.invertMask) {
                    HStack(spacing: 6) {
                        Image(systemName: "rectangle.2.swap")
                            .foregroundStyle(.secondary)
                        Text("Invert Mask")
                            .font(.subheadline)
                    }
                }
                .toggleStyle(.switch)
                .help("Invert: boost denoising on background instead of subjects")
                .padding(.leading, 16)
            }

            // Hot pixel / dark frame correction
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Hot Pixel Correction").font(.subheadline)
                    Spacer()
                }

                Toggle(isOn: $engine.autoDarkFrame) {
                    HStack(spacing: 6) {
                        Image(systemName: "sparkle")
                            .foregroundStyle(.purple)
                            .font(.caption)
                        Text("Auto-detect hot pixels")
                            .font(.callout)
                    }
                }
                .toggleStyle(.switch)
                .controlSize(.small)

                if engine.autoDarkFrame {
                    Text("Automatically detects and removes hot pixels from your footage")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if !engine.autoDarkFrame {
                    Text("Enable to detect and remove hot pixels from your footage")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Trim controls

    @ViewBuilder
    private var trimControls: some View {
        if frameCount > 1 {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: "scissors")
                        .foregroundStyle(.orange)
                        .font(.caption)
                    Text("Trim")
                        .font(.subheadline)
                }

                // Scrub preview thumbnail (driven by dragging ranges)
                if let thumb = engine.scrubThumbnail {
                    HStack {
                        Spacer()
                        Image(nsImage: thumb)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 160)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .strokeBorder(Color(nsColor: .separatorColor), lineWidth: 1)
                            )
                            .overlay(alignment: .bottomTrailing) {
                                Text("Frame \(Int(scrubFrame))")
                                    .font(.caption2.monospacedDigit())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
                                    .padding(4)
                            }
                        Spacer()
                    }
                }

                MultiRangeSlider(ranges: $selectedRanges, totalFrames: frameCount) { frame in
                    scrubFrame = Double(frame)
                    engine.scrubTo(frame: frame)
                }
            }
        }
    }

    // MARK: - Ghosting indicator

    private var ghostingIndicator: some View {
        let risk = ghostingRisk
        let label: String
        let color: Color
        let icon: String

        if risk < 0.3 {
            label = "Low ghosting risk"; color = .green; icon = "checkmark.circle.fill"
        } else if risk < 0.7 {
            label = "Moderate ghosting risk"; color = .orange; icon = "exclamationmark.triangle.fill"
        } else {
            label = "High ghosting risk — reduce window or lower strength"
            color = .red; icon = "xmark.circle.fill"
        }

        return HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(color)
            Text(label)
                .font(.callout)
                .foregroundStyle(color)
            Spacer()
        }
        .padding(10)
        .background(color.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    private var ghostingRisk: Double {
        let motion = max(Double(engine.motionAvg), 0.1)
        let winFactor = (windowSize - 1.0) / 14.0   // normalized to max window (15)
        let strFactor = Double(strength) / 1.5       // normalized to default strength
        return motion * winFactor * strFactor / 5.0
    }

    // MARK: - Preview section

    /// The displayed "after" image — with LUT applied if active.
    private var displayedDenoised: NSImage? {
        if lutEnabled, loadedLUT != nil, let applied = lutAppliedPreview {
            return applied
        }
        return engine.previewImage
    }

    private var previewSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Preview").font(.subheadline).foregroundStyle(.secondary)

                // Scope toggle
                Button {
                    scopeEnabled.toggle()
                } label: {
                    Image(systemName: scopeEnabled ? "waveform.circle.fill" : "waveform.circle")
                        .font(.callout)
                }
                .buttonStyle(.plain)
                .foregroundStyle(scopeEnabled ? Color.accentColor : Color.secondary)
                .help("Toggle video scope")
                .accessibilityLabel(scopeEnabled ? "Hide video scope" : "Show video scope")

                if scopeEnabled {
                    Picker("", selection: $scopeMode) {
                        ForEach(ScopeMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 150)
                    .controlSize(.mini)
                    .accessibilityLabel("Scope mode")
                }

                // Noise profile toggle — available as soon as a file is loaded
                if inputURL != nil {
                    Button {
                        noiseProfileMode.toggle()
                        if noiseProfileMode {
                            // If no raw frame is available yet, load one now
                            if engine.originalPreviewImage == nil && rawProfileImage == nil,
                               let url = inputURL {
                                Task {
                                    rawProfileImage = await DenoiseEngine.extractFrame(
                                        from: url, atIndex: Int(previewFrameIndex))
                                }
                            }
                        } else {
                            noiseProfileSelection = nil
                            noiseProfiler.reset()
                        }
                    } label: {
                        Image(systemName: noiseProfileMode ? "waveform.and.magnifyingglass.fill" : "waveform.and.magnifyingglass")
                            .font(.callout)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(noiseProfileMode ? Color.yellow : Color.secondary)
                    .help("Profile sensor noise from a flat region")
                }

                Spacer()
                if engine.isPreviewLoading {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Button {
                        guard let url = inputURL else { return }
                        engine.generatePreview(inputURL: url,
                                               frameIndex: Int(previewFrameIndex),
                                               strength: strength,
                                               windowSize: Int32(windowSize),
                                               spatialStrength: spatialStrength)
                    } label: {
                        Label("Generate", systemImage: "eye")
                            .font(.callout)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(inputURL == nil)
                }
            }

            if frameCount > 1 {
                HStack(spacing: 8) {
                    Text("Frame")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Slider(value: $previewFrameIndex,
                           in: 0...Double(max(frameCount - 1, 1)),
                           step: 1)
                    Text("\(Int(previewFrameIndex))/\(frameCount - 1)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 50, alignment: .trailing)
                }
            }

            if let denoised = displayedDenoised,
               let original = engine.originalPreviewImage {
                ZStack {
                    // In noise profile mode show undenoised original so the user
                    // can pick a visually flat patch from the raw noisy image.
                    if noiseProfileMode {
                        Image(nsImage: original)
                            .resizable()
                            .aspectRatio(CGFloat(original.size.width) / max(CGFloat(original.size.height), 1), contentMode: .fit)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } else {
                        CompareView(before: original, after: denoised)
                            .aspectRatio(CGFloat(denoised.size.width) / max(CGFloat(denoised.size.height), 1), contentMode: .fit)
                    }
                    if engine.protectSubjects, let overlay = engine.maskOverlayImage {
                        Image(nsImage: overlay)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .allowsHitTesting(false)
                    }
                    // Scope overlay
                    if scopeEnabled, let data = scopeData {
                        VStack {
                            Spacer()
                            ScopeView(data: data, mode: scopeMode)
                                .padding(8)
                        }
                        .allowsHitTesting(false)
                    }
                    // Noise profile selection overlay
                    if noiseProfileMode {
                        NoiseProfileSelectionOverlay(
                            imageSize: CGSize(width: videoWidth, height: videoHeight),
                            selectionRect: $noiseProfileSelection
                        ) { rect in
                            guard let url = inputURL else { return }
                            noiseProfiler.analyze(inputURL: url,
                                                  frameIndex: Int(previewFrameIndex),
                                                  patchRect: rect)
                        }
                        noiseProfileResultBadge
                    }
                }
            } else if noiseProfileMode, let raw = rawProfileImage {
                // No preview generated yet — show raw frame for patch selection only
                ZStack {
                    Image(nsImage: raw)
                        .resizable()
                        .aspectRatio(CGFloat(raw.size.width) / max(CGFloat(raw.size.height), 1), contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    NoiseProfileSelectionOverlay(
                        imageSize: CGSize(width: videoWidth, height: videoHeight),
                        selectionRect: $noiseProfileSelection
                    ) { rect in
                        guard let url = inputURL else { return }
                        noiseProfiler.analyze(inputURL: url,
                                              frameIndex: Int(previewFrameIndex),
                                              patchRect: rect)
                    }
                    noiseProfileResultBadge
                }
            } else if noiseProfileMode && rawProfileImage == nil && inputURL != nil {
                // Loading raw frame
                ProgressView("Loading frame…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let image = displayedDenoised {
                ZStack {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    if engine.protectSubjects, let overlay = engine.maskOverlayImage {
                        Image(nsImage: overlay)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .allowsHitTesting(false)
                    }
                    // Scope overlay
                    if scopeEnabled, let data = scopeData {
                        VStack {
                            Spacer()
                            ScopeView(data: data, mode: scopeMode)
                                .padding(8)
                        }
                        .allowsHitTesting(false)
                    }
                    // Noise profile selection overlay
                    if noiseProfileMode {
                        NoiseProfileSelectionOverlay(
                            imageSize: CGSize(width: videoWidth, height: videoHeight),
                            selectionRect: $noiseProfileSelection
                        ) { rect in
                            guard let url = inputURL else { return }
                            noiseProfiler.analyze(inputURL: url,
                                                  frameIndex: Int(previewFrameIndex),
                                                  patchRect: rect)
                        }
                        noiseProfileResultBadge
                    }
                }
            }
        }
    }

    // MARK: - Noise profile result badge

    @ViewBuilder
    private var noiseProfileResultBadge: some View {
        VStack {
            // Instruction hint at top when nothing is selected yet
            if !noiseProfiler.isAnalyzing && noiseProfiler.result == nil {
                Text("Drag to select a flat, textureless area")
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 6))
                    .foregroundStyle(.white.opacity(0.85))
                    .padding(.top, 12)
            }
            Spacer()
            if noiseProfiler.isAnalyzing {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Analyzing patch…")
                        .font(.caption)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.black.opacity(0.75), in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(.white)
                .padding(12)
            } else if let result = noiseProfiler.result {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Image(systemName: result.isValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundStyle(result.isValid ? .green : .yellow)
                        Text("Noise Profile")
                            .font(.caption.bold())
                            .foregroundStyle(.white)
                        Spacer()
                        Button {
                            if result.isValid {
                                engine.calibratedNoiseSigma  = result.sigma
                                engine.calibratedBlackLevel  = result.blackLevel
                                engine.calibratedShotGain    = result.shotGain
                                engine.calibratedReadNoise   = result.readNoise
                                engine.calibrationSource     = "Manual patch"
                                // Persist for this camera so it auto-applies next time
                                if let cam = engine.cameraModel {
                                    engine.saveCurrentCalibration(for: cam)
                                }
                            }
                            noiseProfileMode = false
                            noiseProfileSelection = nil
                        } label: {
                            Text(result.isValid ? "Apply" : "Dismiss")
                                .font(.caption.bold())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(result.isValid ? Color.yellow : Color.secondary)
                                .foregroundStyle(result.isValid ? .black : .white)
                                .clipShape(RoundedRectangle(cornerRadius: 5))
                        }
                        .buttonStyle(.plain)
                    }
                    if result.isValid {
                        Text(result.summaryLine)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.8))
                        HStack(spacing: 4) {
                            Circle()
                                .fill(result.qualityColor)
                                .frame(width: 6, height: 6)
                            Text(result.qualityHint)
                                .font(.caption2)
                                .foregroundStyle(result.qualityColor)
                        }
                        if engine.calibratedNoiseSigma != nil {
                            Text("✓ \(engine.calibrationSource ?? "VST noise model calibrated")")
                                .font(.caption2)
                                .foregroundStyle(.green.opacity(0.9))
                        }
                    } else {
                        Text(result.summaryLine)
                            .font(.caption2)
                            .foregroundStyle(.yellow.opacity(0.9))
                        Text("Tip: select flat sky, wall, or out-of-focus background")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.5))
                    }
                }
                .padding(12)
                .background(.black.opacity(0.82), in: RoundedRectangle(cornerRadius: 10))
                .padding(12)
            }
        }
        .allowsHitTesting(noiseProfiler.isAnalyzing || noiseProfiler.result != nil)
    }

    // MARK: - Drop zone

    private var dropZone: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 14)
                .fill(isDragOver
                      ? Color.accentColor.opacity(0.10)
                      : Color.secondary.opacity(0.05))
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(
                    isDragOver ? Color.accentColor : Color.secondary.opacity(0.35),
                    style: StrokeStyle(lineWidth: isDragOver ? 2 : 1.5, dash: [6, 4]))

            VStack(spacing: 10) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 40))
                    .foregroundStyle(isDragOver ? Color.accentColor : Color.secondary)
                    .scaleEffect(isDragOver ? 1.15 : 1.0)
                    .opacity(isDragOver ? 1.0 : 0.5)
                Text("Drop ProRes RAW file here")
                    .font(.headline)
                Text("or click to browse")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding()
        }
        .frame(height: 160)
        .scaleEffect(isDragOver ? 1.02 : 1.0)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: isDragOver)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Drop zone")
        .accessibilityHint("Drop a ProRes RAW MOV file here or click to browse")
    }

    // MARK: - File info

    private func fileInfoRow(url: URL) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "film").foregroundStyle(.secondary)
                Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("Change") { showingFilePicker = true }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .font(.callout)
            }
            .font(.callout)

            if videoWidth > 0 && videoHeight > 0 {
                HStack(spacing: 4) {
                    Text("\(videoWidth)\u{00D7}\(videoHeight)")
                    if frameCount > 0 {
                        Text("\u{00B7}")
                        Text("\(frameCount) frames")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Output row

    private var outputRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Output").font(.subheadline).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Text(outputURL?.lastPathComponent ?? suggestedOutputName())
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .font(.callout)
                    .foregroundStyle(outputURL == nil ? .secondary : .primary)
                Spacer()
                Button("Change...") { showingOutputPicker = true }
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.accentColor)
                    .font(.callout)
            }
            .padding(8)
            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))

            // CinemaDNG output format picker
            if inputIsCinemaDNG {
                Picker("Output Format", selection: $dngOutputFormat) {
                    Text("CinemaDNG").tag(0)
                    Text("ProRes RAW").tag(1)
                }
                .pickerStyle(.segmented)
                .controlSize(.small)
                .onChange(of: dngOutputFormat) { _ in
                    outputURL = nil  // reset so suggested name updates
                }
            }

            // R3D output format picker (RGB pipeline — no raw Bayer available)
            if inputIsR3D {
                Picker("Output Format", selection: $dngOutputFormat) {
                    Text("ProRes 4444 XQ").tag(0)
                    Text("EXR Sequence").tag(4)
                }
                .pickerStyle(.segmented)
                .controlSize(.small)
                .onChange(of: dngOutputFormat) { _ in
                    outputURL = nil
                }
            }

            // BRAW input always outputs BRAW (no format picker needed)

            // FCP library integration
            if !fcpDetector.detectedLibraries.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Toggle(isOn: $fcpDetector.enabled) {
                        HStack(spacing: 6) {
                            Image(systemName: "film.stack")
                                .foregroundStyle(.purple)
                                .font(.caption)
                            Text("Save to FCP Library")
                                .font(.callout)
                        }
                    }
                    .toggleStyle(.switch)
                    .controlSize(.small)

                    if fcpDetector.enabled {
                        if fcpDetector.detectedLibraries.count == 1,
                           let lib = fcpDetector.selectedLibrary {
                            HStack(spacing: 6) {
                                Text(lib.name)
                                    .font(.caption.bold())
                                    .foregroundStyle(.purple)
                                Spacer()
                            }
                        } else {
                            Picker("Library", selection: $fcpDetector.selectedLibrary) {
                                ForEach(fcpDetector.detectedLibraries) { lib in
                                    Text(lib.name).tag(Optional(lib))
                                }
                            }
                            .pickerStyle(.menu)
                            .controlSize(.small)
                        }

                        Text("Output saves next to the library. Use \"Import to Final Cut Pro\" when done.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(8)
                .background(Color.purple.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    // MARK: - Action buttons

    private var actionButtons: some View {
        VStack(spacing: 8) {
            Button {
                triggerDenoise()
            } label: {
                Label("Denoise", systemImage: "wand.and.sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(inputURL == nil || !license.canDenoise || engine.isQueueRunning)

            Button {
                triggerAddToQueue()
            } label: {
                Label("Add to Queue", systemImage: "plus.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .disabled(inputURL == nil || engine.isQueueRunning)
        }
    }

    // MARK: - Queue panel

    private var queuePanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Queue (\(engine.queue.count))")
                    .font(.headline)
                Spacer()
                if engine.isQueueRunning {
                    Button("Cancel All") { engine.cancelQueue() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                } else if engine.queue.contains(where: {
                    if case .pending = $0.status { return true }; return false
                }) {
                    Button("Start Queue") {
                        guard license.canDenoise else { license.showActivation = true; return }
                        engine.startQueue()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }

                if !engine.isQueueRunning {
                    Button("Clear") {
                        engine.queue.removeAll()
                    }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                }
            }

            ForEach(engine.queue) { item in
                queueItemRow(item)
            }
        }
    }

    private func queueItemRow(_ item: QueueItem) -> some View {
        HStack(spacing: 10) {
            switch item.status {
            case .pending:
                Image(systemName: "clock").foregroundStyle(.secondary)
            case .processing:
                ProgressView().controlSize(.small)
            case .done:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed:
                Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(item.inputURL.lastPathComponent)
                        .font(.callout).lineLimit(1)
                    if item.startFrame > 0 || item.endFrame > 0 {
                        Text("[\(item.startFrame)–\(item.endFrame)]")
                            .font(.caption2.monospacedDigit())
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color.orange.opacity(0.15), in: Capsule())
                            .foregroundStyle(.orange)
                    }
                }

                switch item.status {
                case .processing(let pct, let frame, let total, let eta, let fps):
                    ProgressView(value: pct)
                        .progressViewStyle(.linear)
                    HStack(spacing: 4) {
                        Text("Frame \(frame)/\(total)")
                        if fps > 0.01 {
                            Text(String(format: "\u{00B7} %.1f fps", fps))
                        }
                        Text("— \(eta)")
                    }
                    .font(.caption2).foregroundStyle(.secondary)
                case .failed(let msg):
                    Text(msg).font(.caption2).foregroundStyle(.red)
                case .done:
                    Text("Complete").font(.caption2).foregroundStyle(.green)
                case .pending:
                    Text("Waiting...").font(.caption2).foregroundStyle(.secondary)
                }
            }

            Spacer()

            if case .pending = item.status {
                Button { engine.removeFromQueue(id: item.id) } label: {
                    Image(systemName: "trash").foregroundStyle(.red)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(8)
        .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Watch folder panel

    private var watchFolderPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "folder.badge.gearshape")
                    .foregroundStyle(.secondary)
                Text("Watch Folder")
                    .font(.subheadline.bold())
                Spacer()

                if engine.isWatching {
                    Button("Stop") {
                        engine.setWatchFolder(nil)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button("Choose Folder...") {
                        showingFolderPicker = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            if let url = engine.watchFolderURL, engine.isWatching {
                HStack(spacing: 6) {
                    Circle()
                        .fill(.green)
                        .frame(width: 6, height: 6)
                    Text(url.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                }
                .padding(8)
                .background(Color.green.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))

                Text("New .mov files will be auto-queued with current settings.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text("Monitor a folder for new ProRes RAW files and auto-queue them.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .fileImporter(isPresented: $showingFolderPicker,
                      allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                engine.setWatchFolder(url,
                                      strength: strength,
                                      windowSize: Int32(windowSize),
                                      spatialStrength: spatialStrength)
            }
        }
    }

    // MARK: - Processing view

    private func processingView(pct: Double, frame: Int, total: Int, eta: String, fps: Double) -> some View {
        VStack(spacing: 24) {
            Spacer()

            // Circular progress ring
            ZStack {
                Circle()
                    .stroke(Color.accentColor.opacity(0.15), lineWidth: 8)
                Circle()
                    .trim(from: 0, to: pct)
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: pct)

                VStack(spacing: 2) {
                    Text("\(Int(pct * 100))%")
                        .font(.system(size: 28, weight: .semibold, design: .rounded).monospacedDigit())
                    if fps > 0.01 {
                        Text(String(format: "%.1f fps", fps))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(width: 120, height: 120)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Processing progress")
            .accessibilityValue("\(Int(pct * 100)) percent")

            // File name + frame info
            VStack(spacing: 6) {
                if let url = inputURL {
                    Text(url.deletingPathExtension().lastPathComponent)
                        .font(.headline)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                if total > 0 {
                    Text("Frame \(frame) of \(total)")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Starting...")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }

            // Linear progress bar + timing
            VStack(spacing: 6) {
                ProgressView(value: pct)
                    .progressViewStyle(.linear)
                    .tint(.accentColor)

                HStack {
                    // Elapsed time
                    if processingStartTime != nil {
                        Text(Self.formatDuration(processingElapsed))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(eta)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: 300)

            Button("Cancel") { engine.cancel() }
                .buttonStyle(.bordered)
                .controlSize(.large)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private static func formatDuration(_ seconds: TimeInterval) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s elapsed" }
        return "\(s / 60)m \(s % 60)s elapsed"
    }

    // MARK: - Done view

    private func doneView(url: URL) -> some View {
        VStack(spacing: 24) {
            Spacer()

            VStack(spacing: 12) {
                if showCheckmark {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 56))
                        .foregroundStyle(Color.green)
                        .transition(.scale.combined(with: .opacity))
                }

                Text("Done!")
                    .font(.title2.bold())
                Text(url.lastPathComponent)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)

                // File size + processing stats
                HStack(spacing: 8) {
                    if let size = Self.fileSize(url) {
                        Text(size)
                            .font(.caption.monospacedDigit())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.secondary.opacity(0.1), in: Capsule())
                    }

                    if let stats = lastProcessingStats {
                        Text("\(stats.frames) frames")
                            .font(.caption.monospacedDigit())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.secondary.opacity(0.1), in: Capsule())
                        Text(String(format: "%.1f fps", stats.fps))
                            .font(.caption.monospacedDigit())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.secondary.opacity(0.1), in: Capsule())
                        Text(Self.formatDuration(stats.elapsed).replacingOccurrences(of: " elapsed", with: ""))
                            .font(.caption.monospacedDigit())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.secondary.opacity(0.1), in: Capsule())
                    }
                }
                .foregroundStyle(.secondary)
            }

            VStack(spacing: 10) {
                HStack(spacing: 12) {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    Button {
                        engine.reset()
                        inputURL  = nil
                        outputURL = nil
                        showCheckmark = false
                        lastProcessingStats = nil
                    } label: {
                        Text("Denoise Another")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                }

                HStack(spacing: 12) {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(url.path, forType: .string)
                    } label: {
                        Label("Copy Path", systemImage: "doc.on.doc")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)

                    // Import to FCP button
                    if fcpDetector.enabled, fcpDetector.selectedLibrary != nil {
                        Button {
                            fcpDetector.importIntoFCP(fileURL: url)
                        } label: {
                            Label("Import to FCP", systemImage: "film.stack")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .tint(.purple)
                    }
                }
            }

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.6)) {
                showCheckmark = true
            }
            if autoRevealInFinder {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        }
    }

    private static func fileSize(_ url: URL) -> String? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let bytes = attrs[.size] as? Int64 else { return nil }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    // MARK: - Drag & drop

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }

        // Try modern URL loading first
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            _ = provider.loadObject(ofClass: URL.self) { url, error in
                guard let url = url else {
                    // Fallback: try raw data approach
                    provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                        if let data = item as? Data,
                           let fallbackURL = URL(dataRepresentation: data, relativeTo: nil) {
                            DispatchQueue.main.async { self.setInput(fallbackURL) }
                        }
                    }
                    return
                }
                DispatchQueue.main.async { self.setInput(url) }
            }
            return true
        }
        return false
    }

    private func setInput(_ url: URL) {
        // Validate file type: MOV, DNG, or directory (CinemaDNG folder)
        let ext = url.pathExtension.lowercased()
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)

        if !exists {
            inputError = "File not found."
            return
        }

        let validExts = ["mov", "dng", "braw", "crm", "ari", "mxf", "r3d", "nraw"]
        if !isDir.boolValue && !validExts.contains(ext) {
            inputError = "Unsupported format. Use ProRes RAW .mov, CinemaDNG (.dng), BRAW, Canon CRM, ARRIRAW (.ari), MXF, or RED R3D."
            return
        }

        inputURL  = url
        outputURL = nil
        previewFrameIndex = 0
        engine.previewImage = nil
        engine.originalPreviewImage = nil
        rawProfileImage = nil
        noiseProfileMode = false
        noiseProfileSelection = nil
        noiseProfiler.reset()

        // Reset trim ranges
        selectedRanges = []
        scrubFrame = 0

        // Detect format for UI display
        let formatCode = denoise_probe_format(url.path)
        let isCinemaDNG = (formatCode == 1)

        // Notify tab container of file load
        let formatColor: Color = {
            switch ext {
            case "mov":  return Color(red: 1.0, green: 0.58, blue: 0.0)  // ProRes orange
            case "braw": return Color(red: 0.61, green: 0.35, blue: 0.71) // BRAW purple
            case "ari", "mxf": return Color(red: 0.16, green: 0.50, blue: 0.73) // ARRI blue
            case "r3d":  return Color(red: 0.91, green: 0.30, blue: 0.24) // RED red
            case "dng":  return Color(red: 0.10, green: 0.74, blue: 0.61) // DNG teal
            case "crm":  return Color(red: 0.85, green: 0.11, blue: 0.14) // Canon red
            case "nraw": return Color(red: 0.98, green: 0.82, blue: 0.0)  // Nikon yellow
            default:     return .secondary
            }
        }()
        onFileLoaded?(url.lastPathComponent, formatColor)

        // Prepare scrub preview (CinemaDNG: skip AVAsset-based scrub for now)
        if !isCinemaDNG {
            engine.prepareScrub(url: url)
        }

        // Probe frame count
        let count = denoise_probe_frame_count(url.path)
        if count <= 0 {
            frameCount = 0
            if isCinemaDNG {
                inputError = "No .dng files found in this folder."
            } else {
                inputError = "Could not read frames from this file. Is it a ProRes RAW recording?"
            }
            return
        }
        frameCount = Int(count)

        // Probe dimensions
        var w: Int32 = 0, h: Int32 = 0
        if denoise_probe_dimensions(url.path, &w, &h) == 0 {
            videoWidth = Int(w)
            videoHeight = Int(h)
        } else {
            videoWidth = 0
            videoHeight = 0
        }

        // Short clip warning — clamp window size
        if frameCount < 3 {
            windowSize = Double(min(Int(windowSize), frameCount))
        }

        engine.reset()
        engine.analyzeMotion(inputURL: url)
    }

    /// Reset all session state when navigating back to the hub.
    private func resetSession() {
        engine.cancel()
        engine.reset()
        inputURL = nil
        outputURL = nil
        previewFrameIndex = 0
        frameCount = 0
        videoWidth = 0
        videoHeight = 0
        selectedRanges = []
        scrubFrame = 0
        lastProcessingStats = nil
        processingStartTime = nil
        processingElapsed = 0
        inputError = nil
        onFileLoaded?("New Session", .secondary)
    }

    // MARK: - LUT controls

    private var lutControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if let lut = loadedLUT {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(lut.title)
                            .font(.callout)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text("\(lut.type == .threeD ? "3D" : "1D") \u{2022} \(lut.size) pts")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("No LUT loaded")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Load\u{2026}") { showingLUTPicker = true }
                    .controlSize(.small)
                    .accessibilityLabel("Load LUT file")
            }

            if loadedLUT != nil {
                Toggle("Enable", isOn: $lutEnabled)
                    .toggleStyle(.switch)
                    .controlSize(.small)

                HStack {
                    Text("Intensity").font(.subheadline)
                    Spacer()
                    Text("\(Int(lutBlend * 100))%")
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $lutBlend, in: 0...1, step: 0.05)
                    .accessibilityLabel("LUT intensity")
                    .accessibilityValue("\(Int(lutBlend * 100)) percent")

                Button("Remove LUT") {
                    loadedLUT = nil
                    lutURL = nil
                    lutEnabled = false
                    lutAppliedPreview = nil
                    computeScopeData()
                }
                .font(.caption)
                .foregroundStyle(.red)
            }
        }
    }

    // MARK: - LUT application

    private func reapplyLUT() {
        guard lutEnabled, let lut = loadedLUT, let image = engine.previewImage else {
            lutAppliedPreview = nil
            return
        }
        let blend = lutBlend
        Task {
            let result = LUTProcessor.apply(lut: lut, to: image, blend: blend)
            lutAppliedPreview = result
        }
    }

    // MARK: - Scope data computation

    private func computeScopeData() {
        let image = displayedDenoised
        guard let image else { scopeData = nil; return }
        Task {
            let data = ScopeData.compute(from: image)
            scopeData = data
        }
    }

    // MARK: - Session persistence

    private func computeStateHash() -> Int {
        var hasher = Hasher()
        hasher.combine(inputURL)
        hasher.combine(outputURL)
        hasher.combine(strength)
        hasher.combine(windowSize)
        hasher.combine(spatialStrength)
        hasher.combine(engine.protectSubjects)
        hasher.combine(engine.invertMask)
        hasher.combine(engine.temporalFilterMode)
        hasher.combine(engine.useML)
        hasher.combine(dngOutputFormat)
        hasher.combine(brawOutputFormat)
        hasher.combine(lutURL)
        hasher.combine(lutBlend)
        hasher.combine(lutEnabled)
        hasher.combine(showHub)
        hasher.combine(frameCount)
        return hasher.finalize()
    }

    private func snapshotState() -> SessionSnapshot {
        SessionSnapshot(
            id: session.id.uuidString,
            label: session.label,
            formatColorHex: session.formatColor.toHex(),
            inputPath: inputURL?.path,
            outputPath: outputURL?.path,
            strength: strength,
            windowSize: windowSize,
            spatialStrength: spatialStrength,
            presetName: selectedPreset.rawValue,
            protectSubjects: engine.protectSubjects,
            invertMask: engine.invertMask,
            autoDarkFrame: engine.autoDarkFrame,
            temporalFilterMode: engine.temporalFilterMode,
            useML: engine.useML,
            dngOutputFormat: dngOutputFormat,
            brawOutputFormat: brawOutputFormat,
            selectedRanges: selectedRanges.map { CodableFrameRange(start: $0.start, end: $0.end) },
            lutPath: lutURL?.path,
            lutBlend: lutBlend,
            lutEnabled: lutEnabled,
            queueItems: engine.queue.map {
                CodableQueueItem(
                    inputPath: $0.inputURL.path,
                    outputPath: $0.outputURL.path,
                    strength: $0.strength,
                    windowSize: $0.windowSize,
                    spatialStrength: $0.spatialStrength,
                    useML: $0.useML,
                    startFrame: $0.startFrame,
                    endFrame: $0.endFrame,
                    outputFormat: $0.outputFormat
                )
            },
            timestamp: Date(),
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            frameCount: frameCount,
            showHub: showHub
        )
    }

    private func restoreState(from snapshot: SessionSnapshot) {
        strength = snapshot.strength
        windowSize = snapshot.windowSize
        spatialStrength = snapshot.spatialStrength
        selectedPreset = DenoisePreset(rawValue: snapshot.presetName) ?? .custom
        engine.protectSubjects = snapshot.protectSubjects
        engine.invertMask = snapshot.invertMask
        engine.autoDarkFrame = snapshot.autoDarkFrame
        engine.temporalFilterMode = snapshot.temporalFilterMode
        engine.useML = snapshot.useML
        dngOutputFormat = snapshot.dngOutputFormat
        brawOutputFormat = snapshot.brawOutputFormat
        selectedRanges = snapshot.selectedRanges.map { FrameRange(start: $0.start, end: $0.end) }
        showHub = snapshot.showHub
        lutBlend = snapshot.lutBlend
        lutEnabled = snapshot.lutEnabled

        // Restore LUT
        if let path = snapshot.lutPath {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: path) {
                lutURL = url
                loadedLUT = try? CubeLUTLoader.load(from: url)
            }
        }

        // Restore output path
        if let path = snapshot.outputPath {
            outputURL = URL(fileURLWithPath: path)
        }

        // Restore input file (triggers motion analysis)
        if let path = snapshot.inputPath,
           FileManager.default.fileExists(atPath: path) {
            setInput(URL(fileURLWithPath: path))
        }
    }

    // MARK: - Helpers

    private var inputIsCinemaDNG: Bool {
        guard let url = inputURL else { return false }
        return denoise_probe_format(url.path) == 1
    }

    private var inputIsBRAW: Bool {
        guard let url = inputURL else { return false }
        return denoise_probe_format(url.path) == 2
    }

    private var inputIsR3D: Bool {
        guard let url = inputURL else { return false }
        return denoise_probe_format(url.path) == 6
    }

    private func suggestedOutputName() -> String {
        if inputIsCinemaDNG {
            guard let name = inputURL?.lastPathComponent else {
                return dngOutputFormat == 1 ? "output_denoised.mov" : "output_denoised"
            }
            if dngOutputFormat == 1 {
                return "\(name)_denoised.mov"
            }
            return "\(name)_denoised"
        }
        if inputIsBRAW {
            guard let name = inputURL?.deletingPathExtension().lastPathComponent else {
                return "output_denoised.braw"
            }
            return "\(name)_denoised.braw"
        }
        if inputIsR3D {
            guard let name = inputURL?.deletingPathExtension().lastPathComponent else {
                return dngOutputFormat == 4 ? "output_denoised_exr" : "output_denoised.mov"
            }
            return dngOutputFormat == 4 ? "\(name)_denoised_exr" : "\(name)_denoised.mov"
        }
        guard let name = inputURL?.deletingPathExtension().lastPathComponent else {
            return "output_denoised.mov"
        }
        return "\(name)_denoised.mov"
    }

    private func rangeOutputURL(base: URL, index: Int, range: FrameRange) -> URL {
        let stem = base.deletingPathExtension().lastPathComponent
        let ext = base.pathExtension
        let name = "\(stem)_r\(index)_\(range.start)-\(range.end).\(ext)"
        return base.deletingLastPathComponent().appendingPathComponent(name)
    }

    private func defaultOutputURL() -> URL {
        // CinemaDNG: output next to input folder
        if inputIsCinemaDNG, let inURL = inputURL {
            let parent = inURL.deletingLastPathComponent()
            return parent.appendingPathComponent(suggestedOutputName())
        }

        let fallbackDir = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let baseDir: URL
        if !defaultOutputDirectory.isEmpty,
           FileManager.default.fileExists(atPath: defaultOutputDirectory) {
            baseDir = URL(fileURLWithPath: defaultOutputDirectory)
        } else {
            baseDir = fallbackDir
        }

        // If the input clip is inside an FCP library, save alongside the original media
        if let inURL = inputURL {
            return fcpDetector.outputURL(for: inURL, defaultDir: baseDir)
        }
        return baseDir.appendingPathComponent(suggestedOutputName())
    }
}

// LicenseView and EmptyDocument extracted to LicenseView.swift
