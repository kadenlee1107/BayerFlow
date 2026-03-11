import Foundation
import Combine
import AppKit
import AVFoundation
import UserNotifications
import Vision
import CoreVideo

// MARK: - Engine state

enum EngineState: Equatable {
    case idle
    case analyzing(progress: Double)
    case ready                      // analysis done, sliders visible
    case processing(progress: Double, frame: Int, total: Int, eta: String, fps: Double)
    case done(outputURL: URL)
    case failed(message: String)
    case cancelled
}

// MARK: - Queue item

struct QueueItem: Identifiable, Equatable {
    let id = UUID()
    let inputURL: URL
    var outputURL: URL
    var strength: Float
    var windowSize: Int32
    var spatialStrength: Float
    var useML: Bool = false
    var startFrame: Int32 = 0
    var endFrame: Int32 = -1   // -1 = all frames
    var outputFormat: Int32 = 0  // 0=auto, 1=force MOV, 2=force DNG
    var status: QueueItemStatus = .pending

    static func == (lhs: QueueItem, rhs: QueueItem) -> Bool {
        lhs.id == rhs.id && lhs.status == rhs.status
    }
}

enum QueueItemStatus: Equatable {
    case pending
    case processing(progress: Double, frame: Int, total: Int, eta: String, fps: Double)
    case done
    case failed(message: String)
}

// MARK: - Progress bridge (C callback → Swift)

private final class ProgressState: @unchecked Sendable {
    var current: Int32 = 0
    var total:   Int32 = 0
    var cancel:  Int32 = 0
    var startTime: Date = Date()
}

// MARK: - Engine

@MainActor
final class DenoiseEngine: ObservableObject {
    @Published var state: EngineState = .idle

    // Motion analysis results
    @Published var motionAvg: Float = 0
    @Published var motionMax: Float = 0

    // Preview
    @Published var previewImage: NSImage? = nil
    @Published var originalPreviewImage: NSImage? = nil
    @Published var maskOverlayImage: NSImage? = nil
    @Published var isPreviewLoading: Bool = false

    // Scrub thumbnail (lightweight, for trim timeline)
    @Published var scrubThumbnail: NSImage? = nil
    private var scrubTask: Task<Void, Never>?
    private var scrubGeneratorURL: URL?
    private var scrubGenerator: AVAssetImageGenerator?
    private var scrubFPS: Float = 30

    // Queue
    @Published var queue: [QueueItem] = []
    @Published var isQueueRunning: Bool = false

    // Watch folder
    @Published var watchFolderURL: URL? = nil
    @Published var isWatching: Bool = false

    // Camera metadata
    @Published var cameraModel: String? = nil
    @Published var detectedISO: Int? = nil
    @Published var suggestedSigma: Float? = nil
    @Published var matchedProfile: String? = nil

    // GPU / ML status
    let isGPUAvailable: Bool = metal_gpu_available() != 0
    let isMLAvailable: Bool = ml_denoiser_available() != 0

    // ML mode
    @Published var useML: Bool = false
    @Published var temporalFilterMode: Int32 = 2  // 0=NLM, 2=VST+Bilateral (default)
    @Published var useCNNPostfilter: Bool = false  // DnCNN post-filter (off by default)

    // Noise profiler calibration (nil/0 = use default S1M2 values)
    @Published var calibratedNoiseSigma: Float? = nil
    @Published var calibratedBlackLevel: Float = 0   // 0 = default (6032)
    @Published var calibratedShotGain: Float = 0     // 0 = default (180)
    @Published var calibratedReadNoise: Float = 0    // 0 = default (616)
    @Published var calibrationSource: String? = nil  // description shown in UI

    // MARK: - Camera noise profile persistence

    struct SavedNoiseProfile: Codable {
        var sigma: Float
        var blackLevel: Float
        var shotGain: Float
        var readNoise: Float
    }

    private static func profileKey(for camera: String) -> String {
        "noiseProfile_\(camera.replacingOccurrences(of: " ", with: "_"))"
    }

    static func loadSavedProfile(for camera: String) -> SavedNoiseProfile? {
        let key = profileKey(for: camera)
        guard let data = UserDefaults.standard.data(forKey: key),
              let profile = try? JSONDecoder().decode(SavedNoiseProfile.self, from: data)
        else { return nil }
        return profile
    }

    static func saveProfile(_ p: SavedNoiseProfile, for camera: String) {
        let key = profileKey(for: camera)
        if let data = try? JSONEncoder().encode(p) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    func applyNoiseProfile(_ p: SavedNoiseProfile, source: String) {
        calibratedNoiseSigma = p.sigma > 0 ? p.sigma : nil
        calibratedBlackLevel = p.blackLevel
        calibratedShotGain   = p.shotGain
        calibratedReadNoise  = p.readNoise
        calibrationSource    = source
    }

    func saveCurrentCalibration(for camera: String) {
        guard calibratedBlackLevel > 0 || calibratedShotGain > 0 else { return }
        let p = SavedNoiseProfile(sigma: calibratedNoiseSigma ?? 0,
                                  blackLevel: calibratedBlackLevel,
                                  shotGain: calibratedShotGain,
                                  readNoise: calibratedReadNoise)
        DenoiseEngine.saveProfile(p, for: camera)
    }

    // Subject protection
    @Published var protectSubjects: Bool = false
    @Published var invertMask: Bool = false

    // Dark frame
    @Published var darkFrameURL: URL? = nil
    @Published var autoDarkFrame: Bool = true  // auto-detect hot pixels

    // Hot pixel profile path (from app bundle)
    nonisolated private static let hotpixelProfilePath: String? = {
        Bundle.main.path(forResource: "S1M2_hotpixels", ofType: "bin")
    }()

    private var progressState: ProgressState?
    private var pollingTask: Task<Void, Never>?
    private var workTask: Task<Void, Never>?
    private var previewTask: Task<Void, Never>?
    private var folderWatcher: FolderWatcher?
    private var activeProgressStates: [ProgressState] = []

    // Watch folder default settings (captured when watch folder is set)
    private var watchStrength: Float = 1.5
    private var watchWindowSize: Int32 = 15
    private var watchSpatialStrength: Float = 0.0

    // MARK: - Error messages

    /// Map C DENOISE_ERR_* codes to user-friendly strings.
    nonisolated static func errorMessage(for code: Int32) -> String {
        switch code {
        case DENOISE_ERR_INPUT_OPEN:  return "Could not open input file."
        case DENOISE_ERR_OUTPUT_OPEN: return "Could not create output file."
        case DENOISE_ERR_ALLOC:       return "Out of memory."
        case DENOISE_ERR_ENCODE:      return "Encoding error during processing."
        case DENOISE_ERR_CANCELLED:   return "Cancelled."
        default:                      return "Unknown error (\(code))."
        }
    }

    // MARK: - Completion feedback

    /// Request notification permission (call once on first launch).
    static func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    /// Post a system notification and play a sound on processing completion.
    private func notifyCompletion(success: Bool, filename: String, itemCount: Int = 1) {
        let playSound = UserDefaults.standard.object(forKey: "playSoundOnCompletion") as? Bool ?? true
        let showNotif = UserDefaults.standard.object(forKey: "showNotificationOnCompletion") as? Bool ?? true

        // Sound
        if playSound {
            if success {
                NSSound(named: "Glass")?.play()
            } else {
                NSSound(named: "Basso")?.play()
            }
        }

        // Dock badge: clear it
        NSApp.dockTile.badgeLabel = nil

        // System notification
        guard showNotif else { return }
        let content = UNMutableNotificationContent()
        if success {
            content.title = "Denoise Complete"
            content.body = itemCount > 1
                ? "\(itemCount) files processed successfully."
                : "\(filename) is ready."
        } else {
            content.title = "Denoise Failed"
            content.body = "\(filename) could not be processed."
        }
        content.sound = nil  // we already played custom sound above

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil  // deliver immediately
        )
        UNUserNotificationCenter.current().add(request)
    }

    /// Update dock badge with progress percentage.
    func updateDockBadge(progress: Double) {
        let pct = Int(progress * 100)
        NSApp.dockTile.badgeLabel = pct > 0 && pct < 100 ? "\(pct)%" : nil
    }

    // MARK: - Motion analysis

    func analyzeMotion(inputURL: URL) {
        // Cancel any in-flight analysis (sets cancel flag so C returns promptly)
        progressState?.cancel = 1
        workTask?.cancel()
        pollingTask?.cancel()

        state = .analyzing(progress: 0)

        // Probe camera metadata in parallel with motion analysis
        probeCamera(inputURL: inputURL)

        let ps = ProgressState()
        progressState = ps
        let psPtr = Unmanaged.passRetained(ps).toOpaque()

        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                if let self {
                    let pct = Double(ps.current) / max(Double(ps.total), 1)
                    if case .analyzing = self.state {
                        self.state = .analyzing(progress: pct)
                    }
                }
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        let inputPath = inputURL.path

        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            let avg = UnsafeMutablePointer<Float>.allocate(capacity: 1)
            let mx  = UnsafeMutablePointer<Float>.allocate(capacity: 1)
            avg.pointee = 0
            mx.pointee  = 0
            defer { avg.deallocate(); mx.deallocate() }

            let result = analyze_motion(inputPath, avg, mx, { current, total, ctx in
                guard let ctx else { return 0 }
                let state = Unmanaged<ProgressState>.fromOpaque(ctx).takeUnretainedValue()
                state.current = current
                state.total   = total
                return state.cancel
            }, psPtr)

            Unmanaged<ProgressState>.fromOpaque(psPtr).release()

            let finalAvg = avg.pointee
            let finalMax = mx.pointee

            await MainActor.run { [weak self] in
                self?.pollingTask?.cancel()
                if result == DENOISE_OK {
                    self?.motionAvg = finalAvg
                    self?.motionMax = finalMax
                    self?.state = .ready
                } else {
                    self?.motionAvg = 0
                    self?.motionMax = 0
                    self?.state = .ready
                }
            }
        }
    }

    // MARK: - Preview

    func generatePreview(inputURL: URL, frameIndex: Int,
                         strength: Float, windowSize: Int32, spatialStrength: Float) {
        previewTask?.cancel()
        isPreviewLoading = true
        previewImage = nil
        originalPreviewImage = nil
        maskOverlayImage = nil

        // Extract original frame in parallel (for before/after comparison)
        let origInputURL = inputURL
        let origFrameIdx = frameIndex
        Task.detached(priority: .utility) { [weak self] in
            let image = await DenoiseEngine.extractFrame(from: origInputURL, atIndex: origFrameIdx)
            await MainActor.run { [weak self] in
                self?.originalPreviewImage = image
            }
        }

        let inputPath = inputURL.path
        let useMLMode = self.useML
        let useCNN = self.useCNNPostfilter
        let calibSigma = self.calibratedNoiseSigma ?? 0
        let calibBL = self.calibratedBlackLevel
        let calibSG = self.calibratedShotGain
        let calibRN = self.calibratedReadNoise
        let darkPath = self.darkFrameURL?.path
        let autoDF = self.autoDarkFrame
        let isoVal = self.detectedISO ?? 0
        let protectSub = self.protectSubjects
        let invertMsk = self.invertMask
        let profilePath = Self.hotpixelProfilePath
        let motionVal = self.motionAvg
        let tfMode = self.temporalFilterMode

        previewTask = Task.detached(priority: .userInitiated) {
            // Write denoised frame as a 1-frame ProRes RAW .mov to temp file
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("bayerflow_preview_\(UUID().uuidString).mov")
            let tempPath = tempURL.path
            defer { try? FileManager.default.removeItem(at: tempURL) }

            var cfg = DenoiseCConfig()
            cfg.window_size      = windowSize
            cfg.strength         = strength
            cfg.noise_sigma      = calibSigma
            cfg.black_level      = calibBL
            cfg.shot_gain        = calibSG
            cfg.read_noise       = calibRN
            cfg.spatial_strength = spatialStrength
            cfg.use_ml           = useMLMode ? 1 : 0
            cfg.use_cnn_postfilter = useCNN ? 1 : 0
            cfg.protect_subjects = protectSub ? 1 : 0
            cfg.invert_mask = invertMsk ? 1 : 0
            cfg.dark_frame_path  = nil
            cfg.auto_dark_frame  = (darkPath == nil && autoDF) ? 1 : 0
            cfg.hotpixel_profile_path = nil
            cfg.detected_iso     = Int32(isoVal)
            cfg.motion_avg       = motionVal
            cfg.temporal_filter_mode = tfMode

            let result: Int32 = DenoiseEngine.withOptionalCStrings(
                darkPath, profilePath
            ) { darkCStr, profileCStr in
                cfg.dark_frame_path = darkCStr
                cfg.hotpixel_profile_path = profileCStr
                return denoise_preview_frame(inputPath, Int32(frameIndex), &cfg, tempPath)
            }

            guard result == DENOISE_OK else {
                let msg = DenoiseEngine.errorMessage(for: result)
                debugLog("preview: denoise_preview_frame failed — \(msg)")
                ErrorLogger.shared.error("Preview failed (code \(result)): \(msg)")
                await MainActor.run { [weak self] in
                    self?.isPreviewLoading = false
                    self?.state = .failed(message: "Preview failed: \(msg)")
                }
                return
            }

            // Check the temp file exists and has content
            let attrs = try? FileManager.default.attributesOfItem(atPath: tempPath)
            let fileSize = attrs?[.size] as? Int ?? 0
            debugLog("preview: temp file \(tempPath) — \(fileSize) bytes")

            // Use AVAssetImageGenerator to decode the ProRes RAW frame
            // This goes through Apple's ProRes RAW decoder — exact same look as final output
            let image = await DenoiseEngine.extractFrame(from: tempURL)
            debugLog("preview: extracted image = \(image != nil ? "\(Int(image!.size.width))x\(Int(image!.size.height))" : "nil")")

            await MainActor.run { [weak self] in
                self?.previewImage = image
                self?.isPreviewLoading = false
            }

            // Generate mask overlay if subject protection is on
            if protectSub, let image = image {
                let overlay = await DenoiseEngine.generateMaskOverlay(from: image, invert: invertMsk)
                await MainActor.run { [weak self] in
                    self?.maskOverlayImage = overlay
                }
            }
        }
    }

    nonisolated static func extractFrame(from movURL: URL, atIndex frameIndex: Int = 0) async -> NSImage? {
        let asset = AVURLAsset(url: movURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = .zero

        // Compute time for requested frame index
        var requestTime = CMTime.zero
        if frameIndex > 0 {
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                if let track = tracks.first {
                    let fps = try await track.load(.nominalFrameRate)
                    if fps > 0 {
                        requestTime = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps))
                    }
                }
            } catch {
                debugLog("preview: could not determine fps, using frame 0")
            }
        }

        do {
            let (cgImage, _) = try await generator.image(at: requestTime)
            return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        } catch {
            debugLog("preview: AVAssetImageGenerator error — \(error)")
            return nil
        }
    }

    /// Generate a colored mask overlay from an NSImage using Vision person segmentation.
    /// Returns a semi-transparent image: colored where mask is active, clear elsewhere.
    nonisolated private static func generateMaskOverlay(from image: NSImage, invert: Bool) async -> NSImage? {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }

        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let result = request.results?.first else { return nil }
        let segBuffer = result.pixelBuffer

        CVPixelBufferLockBaseAddress(segBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(segBuffer, .readOnly) }

        let segW = CVPixelBufferGetWidth(segBuffer)
        let segH = CVPixelBufferGetHeight(segBuffer)
        let segBPR = CVPixelBufferGetBytesPerRow(segBuffer)
        guard let segBase = CVPixelBufferGetBaseAddress(segBuffer) else { return nil }
        let segPtr = segBase.assumingMemoryBound(to: UInt8.self)

        // Create RGBA overlay at image size
        let imgW = cgImage.width
        let imgH = cgImage.height
        var pixels = [UInt8](repeating: 0, count: imgW * imgH * 4)

        for y in 0..<imgH {
            let segY = min(y * segH / imgH, segH - 1)
            for x in 0..<imgW {
                let segX = min(x * segW / imgW, segW - 1)
                var alpha = Float(segPtr[segY * segBPR + segX]) / 255.0
                if invert { alpha = 1.0 - alpha }

                let px = (y * imgW + x) * 4
                // Blue tint for active mask regions
                pixels[px + 0] = 50   // R
                pixels[px + 1] = 120  // G
                pixels[px + 2] = 255  // B
                pixels[px + 3] = UInt8(alpha * 100)  // A — semi-transparent
            }
        }

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: &pixels, width: imgW, height: imgH,
                                 bitsPerComponent: 8, bytesPerRow: imgW * 4,
                                 space: colorSpace,
                                 bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let overlayImage = ctx.makeImage() else { return nil }

        return NSImage(cgImage: overlayImage, size: NSSize(width: imgW, height: imgH))
    }

    // MARK: - Scrub thumbnail

    /// Prepare the scrub generator for a new input file.
    func prepareScrub(url: URL) {
        scrubGeneratorURL = url
        scrubGenerator = nil
        scrubThumbnail = nil
        scrubFPS = 30

        Task {
            let asset = AVURLAsset(url: url)
            let gen = AVAssetImageGenerator(asset: asset)
            gen.appliesPreferredTrackTransform = true
            gen.maximumSize = CGSize(width: 480, height: 480)
            // Allow some tolerance for faster seeking
            gen.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 30)
            gen.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 30)

            if let track = try? await asset.loadTracks(withMediaType: .video).first {
                let fps = (try? await track.load(.nominalFrameRate)) ?? 30
                await MainActor.run { self.scrubFPS = fps }
            }
            await MainActor.run { self.scrubGenerator = gen }
        }
    }

    /// Generate a scrub thumbnail for the given frame (debounced).
    func scrubTo(frame: Int) {
        scrubTask?.cancel()
        scrubTask = Task {
            try? await Task.sleep(nanoseconds: 60_000_000) // 60ms debounce
            guard !Task.isCancelled, let gen = scrubGenerator else { return }

            let time = CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(scrubFPS))
            if let (cgImage, _) = try? await gen.image(at: time) {
                let img = NSImage(cgImage: cgImage,
                                  size: NSSize(width: cgImage.width, height: cgImage.height))
                await MainActor.run {
                    guard !Task.isCancelled else { return }
                    self.scrubThumbnail = img
                }
            }
        }
    }

    // MARK: - Denoise

    func start(inputURL: URL, outputURL: URL,
               strength: Float, windowSize: Int32, spatialStrength: Float,
               startFrame: Int32 = 0, endFrame: Int32 = -1,
               outputFormat: Int32 = 0) {
        state = .processing(progress: 0, frame: 0, total: 0, eta: "—", fps: 0)

        let ps = ProgressState()
        ps.startTime = Date()
        progressState = ps
        let psPtr = Unmanaged.passRetained(ps).toOpaque()

        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                if let self {
                    let cur   = Int(ps.current)
                    let tot   = Int(ps.total)
                    let pct   = tot > 0 ? Double(cur) / Double(tot) : 0
                    let elapsed = Date().timeIntervalSince(ps.startTime)
                    let fps   = cur > 0 && elapsed > 0 ? Double(cur) / elapsed : 0
                    let eta   = self.etaString(current: cur, total: tot, start: ps.startTime)
                    if case .processing = self.state {
                        self.state = .processing(progress: pct, frame: cur, total: tot, eta: eta, fps: fps)
                        self.updateDockBadge(progress: pct)
                    }
                }
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        let inputPath  = inputURL.path
        let outputPath = outputURL.path
        let darkPath = self.darkFrameURL?.path
        let autoDF = self.autoDarkFrame
        let isoVal = self.detectedISO ?? 0
        let protectSub = self.protectSubjects
        let invertMsk = self.invertMask
        let profilePath = Self.hotpixelProfilePath
        let trimStart = startFrame
        let trimEnd = endFrame

        let motionVal = self.motionAvg
        let tfMode = self.temporalFilterMode
        let useCNN = self.useCNNPostfilter
        let calibSigmaStart = self.calibratedNoiseSigma ?? 0
        let calibBLStart = self.calibratedBlackLevel
        let calibSGStart = self.calibratedShotGain
        let calibRNStart = self.calibratedReadNoise

        var cfg = DenoiseCConfig()
        cfg.window_size      = useML ? 5 : windowSize  // FastDVDnet always uses 5 frames
        cfg.strength         = strength
        cfg.noise_sigma      = calibSigmaStart
        cfg.black_level      = calibBLStart
        cfg.shot_gain        = calibSGStart
        cfg.read_noise       = calibRNStart
        cfg.spatial_strength = spatialStrength
        cfg.use_ml           = useML ? 1 : 0
        cfg.use_cnn_postfilter = useCNN ? 1 : 0
            cfg.protect_subjects = protectSub ? 1 : 0
            cfg.invert_mask = invertMsk ? 1 : 0
        cfg.dark_frame_path  = nil
        cfg.auto_dark_frame  = (darkPath == nil && autoDF) ? 1 : 0
        cfg.hotpixel_profile_path = nil
        cfg.detected_iso     = Int32(isoVal)
        cfg.motion_avg       = motionVal
        cfg.temporal_filter_mode = tfMode
        cfg.start_frame      = startFrame
        cfg.end_frame        = endFrame
        cfg.output_format    = outputFormat
        cfg.collect_training_data = UserDefaults.standard.bool(forKey: "trainingDataConsent") ? 1 : 0

        let outputIsBRAW = (outputFormat == 3) || (outputFormat == 0 && denoise_probe_format(inputPath) == 2)
        let outputIsMOV = !outputIsBRAW && ((outputFormat == 1) || denoise_probe_format(inputPath) != 1)

        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            let result: Int32 = DenoiseEngine.withOptionalCStrings(
                darkPath, profilePath
            ) { darkCStr, profileCStr in
                cfg.dark_frame_path = darkCStr
                cfg.hotpixel_profile_path = profileCStr
                return denoise_file(inputPath, outputPath, &cfg, { current, total, ctx in
                    guard let ctx else { return 0 }
                    let state = Unmanaged<ProgressState>.fromOpaque(ctx).takeUnretainedValue()
                    state.current = current
                    state.total   = total
                    return state.cancel
                }, psPtr)
            }

            Unmanaged<ProgressState>.fromOpaque(psPtr).release()

            // Remux audio/timecode from source into output (MOV output only)
            if result == DENOISE_OK && outputIsMOV && denoise_probe_format(inputPath) != 1 {
                await DenoiseEngine.remuxWithSourceTracks(
                    videoOnlyURL: URL(fileURLWithPath: outputPath),
                    sourceURL: URL(fileURLWithPath: inputPath),
                    startFrame: trimStart,
                    endFrame: trimEnd
                )
            }

            await MainActor.run { [weak self] in
                self?.pollingTask?.cancel()
                let fname = outputURL.lastPathComponent
                switch result {
                case DENOISE_OK:
                    self?.state = .done(outputURL: outputURL)
                    self?.notifyCompletion(success: true, filename: fname)
                    TrainingDataManager.shared.flush()
                case DENOISE_ERR_CANCELLED:
                    self?.state = .cancelled
                default:
                    let errMsg = DenoiseEngine.errorMessage(for: result)
                    ErrorLogger.shared.error("Denoise failed (code \(result)): \(errMsg) — \(fname)")
                    self?.state = .failed(message: errMsg)
                    self?.notifyCompletion(success: false, filename: fname)
                }
            }
        }
    }

    func cancel() {
        progressState?.cancel = 1
        for ps in activeProgressStates { ps.cancel = 1 }
    }

    func reset() {
        workTask?.cancel()
        pollingTask?.cancel()
        previewTask?.cancel()
        progressState?.cancel = 1
        motionAvg = 0
        motionMax = 0
        previewImage = nil
        originalPreviewImage = nil
        maskOverlayImage = nil
        isPreviewLoading = false
        cameraModel = nil
        detectedISO = nil
        suggestedSigma = nil
        matchedProfile = nil
        calibratedNoiseSigma = nil
        calibratedBlackLevel = 0
        calibratedShotGain = 0
        calibratedReadNoise = 0
        calibrationSource = nil
        darkFrameURL = nil
        state = .idle
    }

    // MARK: - Camera metadata probe

    func probeCamera(inputURL: URL) {
        let path = inputURL.path
        Task.detached(priority: .utility) { [weak self] in
            var cameraStr = [CChar](repeating: 0, count: 256)
            var iso: Int32 = 0

            let result = denoise_probe_camera(path, &cameraStr, 256, &iso)

            guard result == 0 else { return }

            let camera = String(cString: cameraStr)
            let isoVal = Int(iso)
            let match = NoiseProfiles.match(camera: camera, iso: isoVal)

            await MainActor.run { [weak self] in
                guard let self else { return }
                self.cameraModel = camera.isEmpty ? nil : camera
                self.detectedISO = isoVal > 0 ? isoVal : nil
                if let match {
                    self.suggestedSigma = match.sigma
                    self.matchedProfile = match.profile.displayName
                } else {
                    self.suggestedSigma = nil
                    self.matchedProfile = nil
                }

                // Auto-apply saved noise profile for this camera if available
                if !camera.isEmpty,
                   let saved = DenoiseEngine.loadSavedProfile(for: camera) {
                    self.applyNoiseProfile(saved, source: "Saved · \(camera)")
                }
            }
        }
    }

    // MARK: - Watch folder

    func setWatchFolder(_ url: URL?,
                        strength: Float = 1.0,
                        windowSize: Int32 = 9,
                        spatialStrength: Float = 0.0) {
        folderWatcher?.stop()
        folderWatcher = nil
        watchFolderURL = url
        watchStrength = strength
        watchWindowSize = windowSize
        watchSpatialStrength = spatialStrength

        guard let url else {
            isWatching = false
            return
        }

        let watcher = FolderWatcher(url: url)
        watcher.onNewFile = { [weak self] fileURL in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let outURL = self.autoOutputURL(for: fileURL)
                self.addToQueue(inputURL: fileURL, outputURL: outURL,
                               strength: self.watchStrength,
                               windowSize: self.watchWindowSize,
                               spatialStrength: self.watchSpatialStrength,
                               useML: self.useML)
                if !self.isQueueRunning {
                    self.startQueue()
                }
            }
        }
        watcher.start()
        folderWatcher = watcher
        isWatching = true
    }

    func updateWatchSettings(strength: Float, windowSize: Int32, spatialStrength: Float) {
        watchStrength = strength
        watchWindowSize = windowSize
        watchSpatialStrength = spatialStrength
    }

    private func autoOutputURL(for inputURL: URL) -> URL {
        let dir = inputURL.deletingLastPathComponent()
        let name = inputURL.deletingPathExtension().lastPathComponent
        let isBRAW = denoise_probe_format(inputURL.path) == 2
        let ext = isBRAW ? "braw" : "mov"
        return dir.appendingPathComponent("\(name)_denoised.\(ext)")
    }

    // MARK: - Queue

    func addToQueue(inputURL: URL, outputURL: URL,
                    strength: Float, windowSize: Int32, spatialStrength: Float,
                    useML: Bool = false,
                    startFrame: Int32 = 0, endFrame: Int32 = -1,
                    outputFormat: Int32 = 0) {
        let item = QueueItem(inputURL: inputURL, outputURL: outputURL,
                             strength: strength, windowSize: windowSize,
                             spatialStrength: spatialStrength, useML: useML,
                             startFrame: startFrame, endFrame: endFrame,
                             outputFormat: outputFormat)
        queue.append(item)
    }

    func removeFromQueue(id: UUID) {
        queue.removeAll { $0.id == id }
    }

    func startQueue() {
        guard !isQueueRunning else { return }
        isQueueRunning = true
        processNextQueueItem()
    }

    func cancelQueue() {
        progressState?.cancel = 1
        for ps in activeProgressStates { ps.cancel = 1 }
        isQueueRunning = false
    }

    // MARK: - Parallel queue processing

    func startQueueParallel() {
        guard !isQueueRunning else { return }
        isQueueRunning = true

        // Gather all pending items
        let pendingIds = queue.compactMap { item -> UUID? in
            if case .pending = item.status { return item.id }
            return nil
        }
        guard !pendingIds.isEmpty else {
            isQueueRunning = false
            return
        }

        // Snapshot item configs and create per-item progress states (on MainActor)
        let darkPath = self.darkFrameURL?.path
        let autoDF = self.autoDarkFrame
        let isoVal = Int32(self.detectedISO ?? 0)
        let protectSub = self.protectSubjects
        let invertMsk = self.invertMask
        let profilePath = Self.hotpixelProfilePath
        let motionVal = self.motionAvg
        let tfMode = self.temporalFilterMode
        let useCNNBatch = self.useCNNPostfilter

        struct WorkItem {
            let id: UUID
            let inputPath: String
            let outputPath: String
            let windowSize: Int32
            let strength: Float
            let spatialStrength: Float
            let useML: Int32
            let useCNN: Int32
            let startFrame: Int32
            let endFrame: Int32
            let outputFormat: Int32
            let ps: ProgressState
        }

        var workItems: [WorkItem] = []
        for id in pendingIds {
            guard let idx = queue.firstIndex(where: { $0.id == id }) else { continue }
            queue[idx].status = .processing(progress: 0, frame: 0, total: 0, eta: "—", fps: 0)
            let item = queue[idx]
            let ps = ProgressState()
            ps.startTime = Date()
            workItems.append(WorkItem(
                id: item.id,
                inputPath: item.inputURL.path,
                outputPath: item.outputURL.path,
                windowSize: item.useML ? 5 : item.windowSize,
                strength: item.strength,
                spatialStrength: item.spatialStrength,
                useML: item.useML ? 1 : 0,
                useCNN: useCNNBatch ? 1 : 0,
                startFrame: item.startFrame,
                endFrame: item.endFrame,
                outputFormat: item.outputFormat,
                ps: ps
            ))
        }

        activeProgressStates = workItems.map { $0.ps }

        // Single polling task updates progress for all items
        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { break }
                for wi in workItems {
                    let cur = Int(wi.ps.current)
                    let tot = Int(wi.ps.total)
                    let pct = tot > 0 ? Double(cur) / Double(tot) : 0
                    let elapsed = Date().timeIntervalSince(wi.ps.startTime)
                    let fps = cur > 0 && elapsed > 0 ? Double(cur) / elapsed : 0
                    let eta = self.etaString(current: cur, total: tot, start: wi.ps.startTime)
                    if let idx = self.queue.firstIndex(where: { $0.id == wi.id }) {
                        self.queue[idx].status = .processing(
                            progress: pct, frame: cur, total: tot, eta: eta, fps: fps)
                    }
                }
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        // Launch all items in parallel
        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            await withTaskGroup(of: (UUID, Int32).self) { group in
                for wi in workItems {
                    let psPtr = Unmanaged.passRetained(wi.ps).toOpaque()

                    group.addTask {
                        var cfg = DenoiseCConfig()
                        cfg.window_size      = wi.windowSize
                        cfg.strength         = wi.strength
                        cfg.noise_sigma      = 0
                        cfg.spatial_strength = wi.spatialStrength
                        cfg.use_ml           = wi.useML
                        cfg.use_cnn_postfilter = wi.useCNN
            cfg.protect_subjects = protectSub ? 1 : 0
            cfg.invert_mask = invertMsk ? 1 : 0
                        cfg.dark_frame_path  = nil
                        cfg.auto_dark_frame  = (darkPath == nil && autoDF) ? 1 : 0
                        cfg.hotpixel_profile_path = nil
                        cfg.detected_iso     = isoVal
                        cfg.motion_avg       = motionVal
                        cfg.temporal_filter_mode = tfMode
                        cfg.start_frame      = wi.startFrame
                        cfg.end_frame        = wi.endFrame
                        cfg.output_format    = wi.outputFormat
                        cfg.collect_training_data = UserDefaults.standard.bool(forKey: "trainingDataConsent") ? 1 : 0

                        let wiOutputIsBRAW = (wi.outputFormat == 3) || (wi.outputFormat == 0 && denoise_probe_format(wi.inputPath) == 2)
                        let wiOutputIsMOV = !wiOutputIsBRAW && ((wi.outputFormat == 1) || denoise_probe_format(wi.inputPath) != 1)

                        let result: Int32 = DenoiseEngine.withOptionalCStrings(
                            darkPath, profilePath
                        ) { darkCStr, profileCStr in
                            cfg.dark_frame_path = darkCStr
                            cfg.hotpixel_profile_path = profileCStr
                            return denoise_file(wi.inputPath, wi.outputPath, &cfg, { current, total, ctx in
                                guard let ctx else { return 0 }
                                let state = Unmanaged<ProgressState>.fromOpaque(ctx).takeUnretainedValue()
                                state.current = current
                                state.total   = total
                                return state.cancel
                            }, psPtr)
                        }

                        Unmanaged<ProgressState>.fromOpaque(psPtr).release()

                        if result == DENOISE_OK && wiOutputIsMOV && denoise_probe_format(wi.inputPath) != 1 {
                            await DenoiseEngine.remuxWithSourceTracks(
                                videoOnlyURL: URL(fileURLWithPath: wi.outputPath),
                                sourceURL: URL(fileURLWithPath: wi.inputPath),
                                startFrame: wi.startFrame,
                                endFrame: wi.endFrame
                            )
                        }

                        return (wi.id, result)
                    }
                }

                // Collect results as each worker finishes
                for await (itemId, result) in group {
                    await MainActor.run { [weak self] in
                        guard let self else { return }
                        if let idx = self.queue.firstIndex(where: { $0.id == itemId }) {
                            switch result {
                            case DENOISE_OK:
                                self.queue[idx].status = .done
                            default:
                                let errMsg = DenoiseEngine.errorMessage(for: result)
                                ErrorLogger.shared.error("Queue item failed (code \(result)): \(errMsg) — \(self.queue[idx].inputURL.lastPathComponent)")
                                self.queue[idx].status = .failed(message: errMsg)
                            }
                        }
                    }
                }
            }

            await MainActor.run { [weak self] in
                self?.pollingTask?.cancel()
                self?.activeProgressStates.removeAll()
                self?.isQueueRunning = false
            }
        }
    }

    private func processNextQueueItem() {
        guard isQueueRunning else { return }

        guard let index = queue.firstIndex(where: {
            if case .pending = $0.status { return true }
            return false
        }) else {
            isQueueRunning = false
            let doneCount = queue.filter { if case .done = $0.status { return true }; return false }.count
            let failCount = queue.filter { if case .failed = $0.status { return true }; return false }.count
            if doneCount + failCount > 0 {
                notifyCompletion(success: failCount == 0, filename: "Queue", itemCount: doneCount)
            }
            return
        }

        queue[index].status = .processing(progress: 0, frame: 0, total: 0, eta: "—", fps: 0)

        let item = queue[index]
        let itemId = item.id

        let ps = ProgressState()
        ps.startTime = Date()
        progressState = ps
        let psPtr = Unmanaged.passRetained(ps).toOpaque()

        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { break }
                let cur = Int(ps.current)
                let tot = Int(ps.total)
                let pct = tot > 0 ? Double(cur) / Double(tot) : 0
                let elapsed = Date().timeIntervalSince(ps.startTime)
                let fps = cur > 0 && elapsed > 0 ? Double(cur) / elapsed : 0
                let eta = self.etaString(current: cur, total: tot, start: ps.startTime)
                if let idx = self.queue.firstIndex(where: { $0.id == itemId }) {
                    self.queue[idx].status = .processing(
                        progress: pct, frame: cur, total: tot, eta: eta, fps: fps)
                }
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        let inputPath  = item.inputURL.path
        let outputPath = item.outputURL.path
        let darkPath = self.darkFrameURL?.path
        let autoDF = self.autoDarkFrame
        let isoVal = self.detectedISO ?? 0
        let profilePath = Self.hotpixelProfilePath
        let itemTrimStart = item.startFrame
        let itemTrimEnd = item.endFrame

        let motionVal = self.motionAvg
        let tfMode = self.temporalFilterMode
        let protectSub = self.protectSubjects
        let invertMsk = self.invertMask
        let useCNNWatch = self.useCNNPostfilter

        var cfg = DenoiseCConfig()
        cfg.window_size      = item.useML ? 5 : item.windowSize
        cfg.strength         = item.strength
        cfg.noise_sigma      = 0
        cfg.spatial_strength = item.spatialStrength
        cfg.use_ml           = item.useML ? 1 : 0
        cfg.use_cnn_postfilter = useCNNWatch ? 1 : 0
        cfg.protect_subjects = protectSub ? 1 : 0
        cfg.invert_mask = invertMsk ? 1 : 0
        cfg.dark_frame_path  = nil
        cfg.auto_dark_frame  = (darkPath == nil && autoDF) ? 1 : 0
        cfg.hotpixel_profile_path = nil
        cfg.detected_iso     = Int32(isoVal)
        cfg.motion_avg       = motionVal
        cfg.temporal_filter_mode = tfMode
        cfg.start_frame      = item.startFrame
        cfg.end_frame        = item.endFrame
        cfg.output_format    = item.outputFormat
        cfg.collect_training_data = UserDefaults.standard.bool(forKey: "trainingDataConsent") ? 1 : 0

        let seqOutputIsBRAW = (item.outputFormat == 3) || (item.outputFormat == 0 && denoise_probe_format(inputPath) == 2)
        let seqOutputIsMOV = !seqOutputIsBRAW && ((item.outputFormat == 1) || denoise_probe_format(inputPath) != 1)

        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            let result: Int32 = DenoiseEngine.withOptionalCStrings(
                darkPath, profilePath
            ) { darkCStr, profileCStr in
                cfg.dark_frame_path = darkCStr
                cfg.hotpixel_profile_path = profileCStr
                return denoise_file(inputPath, outputPath, &cfg, { current, total, ctx in
                    guard let ctx else { return 0 }
                    let state = Unmanaged<ProgressState>.fromOpaque(ctx).takeUnretainedValue()
                    state.current = current
                    state.total   = total
                    return state.cancel
                }, psPtr)
            }

            Unmanaged<ProgressState>.fromOpaque(psPtr).release()

            // Remux audio/timecode from source into output (MOV only)
            if result == DENOISE_OK && seqOutputIsMOV && denoise_probe_format(inputPath) != 1 {
                await DenoiseEngine.remuxWithSourceTracks(
                    videoOnlyURL: URL(fileURLWithPath: outputPath),
                    sourceURL: URL(fileURLWithPath: inputPath),
                    startFrame: itemTrimStart,
                    endFrame: itemTrimEnd
                )
            }

            await MainActor.run { [weak self] in
                self?.pollingTask?.cancel()
                guard let self else { return }
                if let idx = self.queue.firstIndex(where: { $0.id == itemId }) {
                    switch result {
                    case DENOISE_OK:
                        self.queue[idx].status = .done
                    default:
                        self.queue[idx].status = .failed(message: DenoiseEngine.errorMessage(for: result))
                    }
                }
                self.processNextQueueItem()
            }
        }
    }

    // MARK: - ETA

    private func etaString(current: Int, total: Int, start: Date) -> String {
        guard current > 2, total > 0 else { return "calculating..." }
        let elapsed = Date().timeIntervalSince(start)
        let fps     = Double(current) / elapsed
        let remaining = Double(total - current) / fps
        if remaining < 60 { return "\(Int(remaining))s remaining" }
        let mins = Int(remaining) / 60
        let secs = Int(remaining) % 60
        return "\(mins)m \(secs)s remaining"
    }

    // MARK: - Auto-chunked parallel denoise

    /// Split a full clip into N chunks, denoise in parallel, then concatenate into one output file.
    func startAutoChunked(inputURL: URL, outputURL: URL,
                          totalFrames: Int,
                          strength: Float, windowSize: Int32, spatialStrength: Float) {
        guard totalFrames > 0 else { return }

        // Determine optimal chunk count
        let cores = ProcessInfo.processInfo.activeProcessorCount
        let maxChunks = max(2, min(cores / 3, 4))
        let minFramesPerChunk = Int(windowSize) * 3
        let numChunks = max(1, min(maxChunks, totalFrames / max(minFramesPerChunk, 1)))

        if numChunks <= 1 {
            // Too few frames to benefit from chunking
            start(inputURL: inputURL, outputURL: outputURL,
                  strength: strength, windowSize: windowSize,
                  spatialStrength: spatialStrength)
            return
        }

        state = .processing(progress: 0, frame: 0, total: 0, eta: "—", fps: 0)

        let framesPerChunk = totalFrames / numChunks

        struct ChunkInfo {
            let index: Int
            let startFrame: Int32
            let endFrame: Int32
            let tempURL: URL
            let ps: ProgressState
        }

        var chunks: [ChunkInfo] = []
        for i in 0..<numChunks {
            let start = Int32(i * framesPerChunk)
            let end = (i == numChunks - 1) ? Int32(totalFrames) : Int32((i + 1) * framesPerChunk)
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("bayerflow_chunk\(i)_\(UUID().uuidString).mov")
            let ps = ProgressState()
            ps.startTime = Date()
            chunks.append(ChunkInfo(index: i, startFrame: start, endFrame: end,
                                    tempURL: tempURL, ps: ps))
        }

        activeProgressStates = chunks.map { $0.ps }

        // Aggregate progress polling
        let overallStart = Date()
        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { break }
                var totalCur = 0
                var totalTot = 0
                for chunk in chunks {
                    totalCur += Int(chunk.ps.current)
                    totalTot += Int(chunk.ps.total)
                }
                let pct = totalTot > 0 ? Double(totalCur) / Double(totalTot) : 0
                let elapsed = Date().timeIntervalSince(overallStart)
                let fps = totalCur > 0 && elapsed > 0 ? Double(totalCur) / elapsed : 0
                let eta = self.etaString(current: totalCur, total: totalTot, start: overallStart)
                if case .processing = self.state {
                    self.state = .processing(progress: pct, frame: totalCur, total: totalTot, eta: eta, fps: fps)
                }
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }

        let inputPath = inputURL.path
        let darkPath = self.darkFrameURL?.path
        let autoDF = self.autoDarkFrame
        let isoVal = Int32(self.detectedISO ?? 0)
        let profilePath = Self.hotpixelProfilePath
        let motionVal = self.motionAvg
        let tfMode = self.temporalFilterMode
        let protectSub = self.protectSubjects
        let invertMsk = self.invertMask
        let useMLMode: Int32 = self.useML ? 1 : 0
        let useCNNChunk: Int32 = self.useCNNPostfilter ? 1 : 0
        let effectiveWindowSize = self.useML ? Int32(5) : windowSize

        debugLog("auto-chunk: splitting \(totalFrames) frames into \(numChunks) chunks of ~\(framesPerChunk) frames each")

        workTask = Task.detached(priority: .userInitiated) { [weak self] in
            var cancelled = false
            var anyFailed = false

            await withTaskGroup(of: (Int, Int32).self) { group in
                for chunk in chunks {
                    let psPtr = Unmanaged.passRetained(chunk.ps).toOpaque()
                    let chunkOutputPath = chunk.tempURL.path

                    group.addTask {
                        var cfg = DenoiseCConfig()
                        cfg.window_size      = effectiveWindowSize
                        cfg.strength         = strength
                        cfg.noise_sigma      = 0
                        cfg.spatial_strength = spatialStrength
                        cfg.use_ml           = useMLMode
                        cfg.use_cnn_postfilter = useCNNChunk
                        cfg.protect_subjects = protectSub ? 1 : 0
                        cfg.invert_mask = invertMsk ? 1 : 0
                        cfg.dark_frame_path  = nil
                        cfg.auto_dark_frame  = (darkPath == nil && autoDF) ? 1 : 0
                        cfg.hotpixel_profile_path = nil
                        cfg.detected_iso     = isoVal
                        cfg.motion_avg       = motionVal
                        cfg.temporal_filter_mode = tfMode
                        cfg.start_frame      = chunk.startFrame
                        cfg.end_frame        = chunk.endFrame
                        cfg.output_format    = 0  // chunks always auto
                        cfg.collect_training_data = UserDefaults.standard.bool(forKey: "trainingDataConsent") ? 1 : 0

                        let result: Int32 = DenoiseEngine.withOptionalCStrings(
                            darkPath, profilePath
                        ) { darkCStr, profileCStr in
                            cfg.dark_frame_path = darkCStr
                            cfg.hotpixel_profile_path = profileCStr
                            return denoise_file(inputPath, chunkOutputPath, &cfg, { current, total, ctx in
                                guard let ctx else { return 0 }
                                let state = Unmanaged<ProgressState>.fromOpaque(ctx).takeUnretainedValue()
                                state.current = current
                                state.total   = total
                                return state.cancel
                            }, psPtr)
                        }

                        Unmanaged<ProgressState>.fromOpaque(psPtr).release()
                        return (chunk.index, result)
                    }
                }

                for await (idx, result) in group {
                    debugLog("auto-chunk: chunk \(idx) finished with code \(result)")
                    if result == DENOISE_ERR_CANCELLED { cancelled = true }
                    else if result != DENOISE_OK { anyFailed = true }
                }
            }

            var finalState: EngineState

            if cancelled {
                finalState = .cancelled
            } else if anyFailed {
                finalState = .failed(message: "One or more chunks failed to process.")
            } else {
                // Concatenate chunks in order
                let chunkURLs = chunks.sorted { $0.index < $1.index }.map { $0.tempURL }
                let outputFileURL = outputURL
                let success = await DenoiseEngine.concatenateChunks(chunkURLs, to: outputFileURL)

                if success {
                    // Remux audio/timecode from source (MOV only, not BRAW)
                    if denoise_probe_format(inputPath) != 1 && denoise_probe_format(inputPath) != 2 {
                        await DenoiseEngine.remuxWithSourceTracks(
                            videoOnlyURL: outputFileURL,
                            sourceURL: URL(fileURLWithPath: inputPath),
                            startFrame: 0,
                            endFrame: -1
                        )
                    }
                    finalState = .done(outputURL: outputFileURL)
                } else {
                    finalState = .failed(message: "Failed to concatenate chunks.")
                }
            }

            // Clean up temp chunk files
            for chunk in chunks {
                try? FileManager.default.removeItem(at: chunk.tempURL)
            }

            let capturedState = finalState
            await MainActor.run { [weak self] in
                self?.pollingTask?.cancel()
                self?.activeProgressStates.removeAll()
                self?.state = capturedState
                if case .done(let url) = capturedState {
                    self?.notifyCompletion(success: true, filename: url.lastPathComponent)
                } else if case .failed = capturedState {
                    self?.notifyCompletion(success: false, filename: "Queue")
                }
            }
        }
    }

    // MARK: - Chunk concatenation

    /// Concatenate multiple ProRes RAW MOV chunks into a single output file.
    nonisolated private static func concatenateChunks(_ chunkURLs: [URL], to outputURL: URL) async -> Bool {
        let composition = AVMutableComposition()

        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            debugLog("concat: could not create video track")
            return false
        }

        var insertTime = CMTime.zero

        for (i, chunkURL) in chunkURLs.enumerated() {
            let asset = AVURLAsset(url: chunkURL)
            guard let track = try? await asset.loadTracks(withMediaType: .video).first,
                  let duration = try? await asset.load(.duration) else {
                debugLog("concat: could not load chunk \(i)")
                return false
            }

            do {
                try videoTrack.insertTimeRange(
                    CMTimeRange(start: .zero, duration: duration),
                    of: track, at: insertTime)
                insertTime = CMTimeAdd(insertTime, duration)
                debugLog("concat: added chunk \(i) (\(String(format: "%.2f", CMTimeGetSeconds(duration)))s)")
            } catch {
                debugLog("concat: failed to insert chunk \(i) — \(error)")
                return false
            }
        }

        guard let session = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetPassthrough
        ) else {
            debugLog("concat: could not create export session")
            return false
        }

        // Remove existing output if present
        try? FileManager.default.removeItem(at: outputURL)

        do {
            try await session.export(to: outputURL, as: .mov)
        } catch {
            debugLog("concat: export failed — \(error.localizedDescription)")
            return false
        }

        debugLog("concat: successfully merged \(chunkURLs.count) chunks → \(outputURL.lastPathComponent)")
        return true
    }

    // MARK: - Audio/timecode remux

    /// After denoise produces a video-only MOV, remux in audio and timecode tracks from the source.
    nonisolated private static func remuxWithSourceTracks(
        videoOnlyURL: URL,
        sourceURL: URL,
        startFrame: Int32,
        endFrame: Int32
    ) async {
        let sourceAsset = AVURLAsset(url: sourceURL)
        let videoAsset = AVURLAsset(url: videoOnlyURL)

        // Check if source has audio or timecode tracks
        let audioTracks: [AVAssetTrack]
        let timecodeTracks: [AVAssetTrack]
        do {
            audioTracks = try await sourceAsset.loadTracks(withMediaType: .audio)
            timecodeTracks = try await sourceAsset.loadTracks(withMediaType: .timecode)
        } catch {
            debugLog("remux: could not load source tracks — \(error)")
            return
        }

        guard !audioTracks.isEmpty || !timecodeTracks.isEmpty else {
            debugLog("remux: source has no audio or timecode tracks, skipping")
            return
        }

        // Get video track and duration from denoised output
        guard let videoTrack = try? await videoAsset.loadTracks(withMediaType: .video).first,
              let videoDuration = try? await videoAsset.load(.duration) else {
            debugLog("remux: could not read denoised video track")
            return
        }

        // Get source FPS for computing trim time range
        let sourceFPS: Float
        if let sourceVideoTrack = try? await sourceAsset.loadTracks(withMediaType: .video).first {
            sourceFPS = (try? await sourceVideoTrack.load(.nominalFrameRate)) ?? 30
        } else {
            sourceFPS = 30
        }

        // Source time range — maps to the portion of the source we denoised
        let sourceStart: CMTime
        if startFrame > 0 {
            sourceStart = CMTime(value: CMTimeValue(startFrame), timescale: CMTimeScale(sourceFPS))
        } else {
            sourceStart = .zero
        }
        let sourceTimeRange = CMTimeRange(start: sourceStart, duration: videoDuration)

        // Build composition
        let composition = AVMutableComposition()

        // Video from denoised output (full track)
        guard let videoCompTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            debugLog("remux: could not add video track to composition")
            return
        }
        do {
            try videoCompTrack.insertTimeRange(
                CMTimeRange(start: .zero, duration: videoDuration),
                of: videoTrack, at: .zero)
        } catch {
            debugLog("remux: failed to insert video — \(error)")
            return
        }

        // Audio from source (trimmed to matching range)
        if let audioTrack = audioTracks.first {
            if let audioCompTrack = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) {
                do {
                    try audioCompTrack.insertTimeRange(sourceTimeRange, of: audioTrack, at: .zero)
                    debugLog("remux: added audio track")
                } catch {
                    debugLog("remux: failed to insert audio — \(error)")
                }
            }
        }

        // Timecode from source (trimmed to matching range)
        if let timecodeTrack = timecodeTracks.first {
            if let tcCompTrack = composition.addMutableTrack(
                withMediaType: .timecode,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) {
                do {
                    try tcCompTrack.insertTimeRange(sourceTimeRange, of: timecodeTrack, at: .zero)
                    debugLog("remux: added timecode track")
                } catch {
                    debugLog("remux: failed to insert timecode — \(error)")
                }
            }
        }

        // Export with passthrough (no re-encoding)
        guard let session = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetPassthrough
        ) else {
            debugLog("remux: could not create export session")
            return
        }

        let tempURL = videoOnlyURL.deletingLastPathComponent()
            .appendingPathComponent("_remux_\(UUID().uuidString).mov")

        do {
            try await session.export(to: tempURL, as: .mov)
        } catch {
            debugLog("remux: export failed — \(error.localizedDescription)")
            try? FileManager.default.removeItem(at: tempURL)
            return
        }

        // Replace video-only output with remuxed version
        do {
            try FileManager.default.removeItem(at: videoOnlyURL)
            try FileManager.default.moveItem(at: tempURL, to: videoOnlyURL)
            debugLog("remux: successfully added audio/timecode to output")
        } catch {
            debugLog("remux: failed to replace output — \(error)")
            try? FileManager.default.removeItem(at: tempURL)
        }
    }

    // MARK: - C string helpers

    /// Call body with two optional C strings, keeping them alive for the duration.
    nonisolated static func withOptionalCStrings<R>(
        _ a: String?, _ b: String?,
        body: (UnsafePointer<CChar>?, UnsafePointer<CChar>?) -> R
    ) -> R {
        if let a {
            return a.withCString { aCStr in
                if let b {
                    return b.withCString { bCStr in body(aCStr, bCStr) }
                } else {
                    return body(aCStr, nil)
                }
            }
        } else {
            if let b {
                return b.withCString { bCStr in body(nil, bCStr) }
            } else {
                return body(nil, nil)
            }
        }
    }
}
