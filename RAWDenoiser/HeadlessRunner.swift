import Foundation
import AppKit

/// Headless CLI mode for automated denoising.
/// Usage: BayerFlow --headless --input path.MOV --output path.MOV [--frames N] [--window N] [--strength F]
enum HeadlessRunner {

    // C callback needs a non-capturing function — use a simple counter via context
    private final class ProgressCtx {
        var lastFrame: Int32 = -1
    }

    static func run() -> Never {
        // Hide dock icon & windows
        NSApplication.shared.setActivationPolicy(.prohibited)

        let args = ProcessInfo.processInfo.arguments
        guard let inputIdx = args.firstIndex(of: "--input"),
              inputIdx + 1 < args.count else {
            fputs("ERROR: --input <path> required\n", stderr)
            exit(1)
        }
        let inputPath = args[inputIdx + 1]

        let inputFormatCode = denoise_probe_format(inputPath)
        let isCinemaDNG = (inputFormatCode == 1)
        let isBRAW = (inputFormatCode == 2)
        let isCRM = (inputFormatCode == 3)
        let isARRIRAW = (inputFormatCode == 4)
        let isMXF = (inputFormatCode == 5)
        let isR3D = (inputFormatCode == 6)
        let isCineform = (inputFormatCode == 7)

        // Determine output path (may depend on --output-format parsed later,
        // so we use a closure to defer)
        let explicitOutput: String?
        if let outIdx = args.firstIndex(of: "--output"), outIdx + 1 < args.count {
            explicitOutput = args[outIdx + 1]
        } else {
            explicitOutput = nil
        }

        var endFrame: Int32 = -1
        if let fIdx = args.firstIndex(of: "--frames"), fIdx + 1 < args.count,
           let n = Int32(args[fIdx + 1]) {
            endFrame = n
        }

        var windowSize: Int32 = 15
        if let wIdx = args.firstIndex(of: "--window"), wIdx + 1 < args.count,
           let n = Int32(args[wIdx + 1]) {
            windowSize = n
        }

        var strength: Float = 1.5
        if let sIdx = args.firstIndex(of: "--strength"), sIdx + 1 < args.count,
           let f = Float(args[sIdx + 1]) {
            strength = f
        }

        var temporalMode: Int32 = 2  // VST+Bilateral default
        if let tmIdx = args.firstIndex(of: "--temporal-mode"), tmIdx + 1 < args.count,
           let m = Int32(args[tmIdx + 1]) {
            temporalMode = m
        }

        // Output format: 0=auto, 1=force MOV (ProRes RAW), 2=force DNG, 3=force BRAW, 4=EXR sequence, 5=CineForm
        var outputFormat: Int32 = 0
        if let ofIdx = args.firstIndex(of: "--output-format"), ofIdx + 1 < args.count {
            let fmt = args[ofIdx + 1].lowercased()
            if fmt == "mov" || fmt == "prores" { outputFormat = 1 }
            else if fmt == "dng" || fmt == "cinemadng" { outputFormat = 2 }
            else if fmt == "braw" { outputFormat = 3 }
            else if fmt == "exr" { outputFormat = 4 }
            else if fmt == "cfhd" || fmt == "cineform" { outputFormat = 5 }
        }

        // Auto-set output format for BRAW input when not explicitly specified
        if outputFormat == 0 && isBRAW {
            outputFormat = 3
        }

        // Resolve output path
        let outputPath: String
        if let explicit = explicitOutput {
            outputPath = explicit
        } else {
            let url = URL(fileURLWithPath: inputPath)
            let dir = url.deletingLastPathComponent().path
            let stem = isCinemaDNG ? url.lastPathComponent : url.deletingPathExtension().lastPathComponent
            if outputFormat == 3 || (outputFormat == 0 && isBRAW) {
                outputPath = "\(dir)/\(stem)_headless.braw"
            } else if outputFormat == 4 {
                outputPath = "\(dir)/\(stem)_denoised_exr"
            } else if outputFormat == 5 {
                outputPath = "\(dir)/\(stem)_headless.mov"
            } else if (outputFormat == 1) || !isCinemaDNG {
                outputPath = "\(dir)/\(stem)_headless.mov"
            } else {
                outputPath = "\(dir)/\(stem)_denoised"
            }
        }

        fputs("=== BayerFlow Headless Mode ===\n", stderr)
        fputs("  Input:    \(inputPath)\n", stderr)
        fputs("  Output:   \(outputPath)\n", stderr)
        fputs("  Frames:   \(endFrame > 0 ? "\(endFrame)" : "all")\n", stderr)
        fputs("  Window:   \(windowSize)\n", stderr)
        fputs("  Strength: \(strength)\n", stderr)
        fputs("  TF Mode:  \(temporalMode) (\(temporalMode == 2 ? "VST+Bilateral" : "NLM"))\n", stderr)
        let inputFmtName = isR3D ? "RED R3D" : isMXF ? "MXF/ARRIRAW" : isARRIRAW ? "ARRIRAW" : isCRM ? "Canon CRM" : isBRAW ? "BRAW" : isCineform ? "GoPro CineForm" : isCinemaDNG ? "CinemaDNG" : "ProRes RAW MOV"
        fputs("  InFmt:    \(inputFmtName)\n", stderr)
        if outputFormat != 0 {
            let fmtName = outputFormat == 1 ? "ProRes RAW MOV" : outputFormat == 3 ? "BRAW" : outputFormat == 4 ? "EXR Sequence (16-bit half-float)" : outputFormat == 5 ? "CineForm (CFHD)" : "CinemaDNG"
            fputs("  OutFmt:   \(fmtName)\n", stderr)
        } else if isR3D {
            fputs("  OutFmt:   ProRes 4444 (RGB input — no raw Bayer available)\n", stderr)
        } else if isCRM || isARRIRAW || isMXF {
            fputs("  OutFmt:   ProRes RAW MOV (auto)\n", stderr)
        }
        fputs("  GPU:      \(MetalTemporalFilter.shared != nil ? "Metal" : "CPU fallback")\n", stderr)
        fputs("-------------------------------\n", stderr)

        // Probe camera ISO for hot pixel profile
        var detectedISO: Int32 = 0
        var cameraModel = [CChar](repeating: 0, count: 256)
        if denoise_probe_camera(inputPath, &cameraModel, 256, &detectedISO) == 0 {
            let model = String(cString: cameraModel)
            fputs("  Camera:   \(model.isEmpty ? "unknown" : model)\n", stderr)
            fputs("  ISO:      \(detectedISO > 0 ? "\(detectedISO)" : "unknown")\n", stderr)
        }

        // Look for bundled hot pixel profile
        let hotpixelProfilePath = Bundle.main.path(forResource: "S1M2_hotpixels", ofType: "bin")

        var cfg = DenoiseCConfig()
        cfg.window_size = windowSize
        cfg.strength = strength
        cfg.noise_sigma = 0          // auto
        cfg.spatial_strength = 0
        cfg.use_ml = 0
        cfg.use_cnn_postfilter = args.contains("--cnn") ? 1 : 0
        cfg.dark_frame_path = nil
        cfg.auto_dark_frame = 1      // auto-detect hot pixels
        cfg.hotpixel_profile_path = nil
        cfg.detected_iso = detectedISO
        cfg.motion_avg = 0
        cfg.temporal_filter_mode = temporalMode
        cfg.start_frame = 0
        cfg.end_frame = endFrame
        cfg.output_format = outputFormat
        cfg.collect_training_data = args.contains("--contribute-data") ? 1 : 0

        // Hot pixel info
        if hotpixelProfilePath != nil && detectedISO > 0 {
            fputs("  HotPixel: profile + auto-detect (ISO \(detectedISO))\n", stderr)
        } else {
            fputs("  HotPixel: auto-detect from input\n", stderr)
        }

        // --sweep-techniques: run technique comparison sweep instead of denoising
        if args.contains("--sweep-techniques") {
            fputs("Running technique sweep...\n", stderr)
            let sweepStart = Int32(cfg.start_frame)
            let sweepResult = denoise_technique_sweep(inputPath, sweepStart > 0 ? sweepStart : 30, &cfg)
            if sweepResult > 0 {
                fputs("Technique sweep complete: \(sweepResult) techniques tested\n", stderr)
                fputs("Output: /tmp/bayerflow_technique_sweep/\n", stderr)
                exit(0)
            } else {
                fputs("Technique sweep FAILED with code \(sweepResult)\n", stderr)
                exit(1)
            }
        }

        let startTime = Date()

        // Wrap denoise_file in withCString closures so profile path pointer stays valid
        let result: Int32
        if let profilePath = hotpixelProfilePath, detectedISO > 0 {
            result = profilePath.withCString { profileCStr in
                cfg.hotpixel_profile_path = profileCStr
                return denoise_file(inputPath, outputPath, &cfg, nil, nil)
            }
        } else {
            result = denoise_file(inputPath, outputPath, &cfg, nil, nil)
        }

        let elapsed = Date().timeIntervalSince(startTime)
        fputs("\n", stderr)

        if result == DENOISE_OK {
            fputs("DONE in \(String(format: "%.1f", elapsed))s → \(outputPath)\n", stderr)
            exit(0)
        } else {
            fputs("FAILED with code \(result) after \(String(format: "%.1f", elapsed))s\n", stderr)
            exit(Int32(result))
        }
    }
}
