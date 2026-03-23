import Foundation
import AppKit

/// Receives denoise requests from the FCP Workflow Extension via DistributedNotificationCenter,
/// runs the denoise engine, and posts progress back.
@MainActor
final class FCPExtensionBridge {
    static let shared = FCPExtensionBridge()

    private let requestName  = Notification.Name("com.bayerflow.app.fcpDenoise")
    private let progressName = Notification.Name("com.bayerflow.app.fcpProgress")

    private var observer: NSObjectProtocol?

    func startListening() {
        // Handle --fcp-denoise CLI args (launched by the extension)
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "--fcp-denoise"), idx + 1 < args.count {
            let clipPaths = Array(args[(idx + 1)...])
            ErrorLogger.shared.log("FCP Extension: launched with \(clipPaths.count) clips via CLI")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                self?.processClips(clipPaths, config: [:])
            }
        }

        // Listen for distributed notifications
        observer = DistributedNotificationCenter.default().addObserver(
            forName: requestName,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self,
                  let info = notification.userInfo as? [String: Any],
                  let clips = info["clips"] as? [String] else { return }
            ErrorLogger.shared.log("FCP Extension: received denoise request for \(clips.count) clips")
            self.processClips(clips, config: info)
        }
    }

    private func processClips(_ clipPaths: [String], config: [String: Any]) {
        let strength    = Float(config["strength"] as? Double ?? 1.5)
        let windowSize  = Int32(config["windowSize"] as? Int ?? 15)
        let spatial     = Float(config["spatialStrength"] as? Double ?? 0.0)
        let useCNN      = config["useCNN"] as? Bool ?? true
        let outputFmt   = Int32(config["outputFormat"] as? Int ?? 0)

        for clipPath in clipPaths {
            let inputURL = URL(fileURLWithPath: clipPath)
            let baseName = inputURL.deletingPathExtension().lastPathComponent
            let ext = inputURL.pathExtension
            let outputPath = inputURL.deletingLastPathComponent()
                .appendingPathComponent("\(baseName)_denoised.\(ext)").path
            let inputPath = inputURL.path

            ErrorLogger.shared.log("FCP Extension: denoising \(inputURL.lastPathComponent)")

            let pName = self.progressName

            Task.detached {
                var cfg = DenoiseCConfig()
                cfg.strength = strength
                cfg.window_size = windowSize
                cfg.spatial_strength = spatial
                cfg.use_cnn_postfilter = useCNN ? 1 : 0
                cfg.output_format = outputFmt
                cfg.temporal_filter_mode = 2  // VST+Bilateral
                cfg.auto_dark_frame = 1
                cfg.start_frame = 0
                cfg.end_frame = -1

                let result = inputPath.withCString { inC in
                    outputPath.withCString { outC in
                        withUnsafePointer(to: &cfg) { cfgPtr in
                            denoise_file(inC, outC, cfgPtr, { current, total, ctx in
                                // Progress callback — post notification
                                let info: [String: Any] = [
                                    "progress": Double(current) / max(Double(total), 1.0),
                                    "frame": Int(current),
                                    "total": Int(total)
                                ]
                                DistributedNotificationCenter.default().postNotificationName(
                                    Notification.Name("com.bayerflow.app.fcpProgress"),
                                    object: nil,
                                    userInfo: info,
                                    deliverImmediately: true
                                )
                                return 0  // continue
                            }, nil)
                        }
                    }
                }

                DistributedNotificationCenter.default().postNotificationName(
                    pName, object: nil,
                    userInfo: [
                        "done": true,
                        "output": outputPath,
                        "success": result == 0
                    ] as [String: Any],
                    deliverImmediately: true
                )
            }
        }
    }

    deinit {
        if let observer = observer {
            DistributedNotificationCenter.default().removeObserver(observer)
        }
    }
}
