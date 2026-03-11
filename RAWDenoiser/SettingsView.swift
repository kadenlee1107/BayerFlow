import SwiftUI
import UniformTypeIdentifiers
import Metal

struct SettingsView: View {
    @AppStorage("defaultOutputDirectory") private var defaultOutputDirectory = ""
    @AppStorage("autoRevealInFinder") private var autoRevealInFinder = false
    @AppStorage("rememberSettings") private var rememberSettings = true
    @AppStorage("playSoundOnCompletion") private var playSoundOnCompletion = true
    @AppStorage("showNotificationOnCompletion") private var showNotification = true
    @AppStorage("defaultTemporalMode") private var defaultTemporalMode = 0
    @AppStorage("defaultWindowSize") private var defaultWindowSize = 15
    @AppStorage("trainingDataConsent") private var trainingDataConsent = false

    @State private var showingFolderPicker = false

    private var gpuName: String {
        MTLCreateSystemDefaultDevice()?.name ?? "Not available"
    }

    private var appVersion: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "\(version) (\(build))"
    }

    var body: some View {
        Form {
            Section("Output") {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Default output directory")
                        if defaultOutputDirectory.isEmpty {
                            Text("~/Movies")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            Text(defaultOutputDirectory)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.head)
                        }
                    }
                    Spacer()
                    if !defaultOutputDirectory.isEmpty {
                        Button("Reset") {
                            defaultOutputDirectory = ""
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    }
                    Button("Choose...") {
                        showingFolderPicker = true
                    }
                    .controlSize(.small)
                }

                Toggle("Auto-reveal output in Finder", isOn: $autoRevealInFinder)
            }

            Section("Notifications") {
                Toggle("Play sound on completion", isOn: $playSoundOnCompletion)
                Toggle("Show system notification on completion", isOn: $showNotification)
            }

            Section("Processing") {
                Picker("Default temporal mode", selection: $defaultTemporalMode) {
                    Text("VST + Bilateral").tag(0)
                    Text("NLM").tag(1)
                }
                .pickerStyle(.menu)

                Picker("Default window size", selection: $defaultWindowSize) {
                    Text("3 frames").tag(3)
                    Text("5 frames").tag(5)
                    Text("7 frames").tag(7)
                    Text("9 frames").tag(9)
                    Text("11 frames").tag(11)
                    Text("13 frames").tag(13)
                    Text("15 frames").tag(15)
                }
                .pickerStyle(.menu)
            }

            Section("General") {
                Toggle("Remember slider settings between sessions", isOn: $rememberSettings)

                Button("Reset All Settings to Defaults") {
                    defaultOutputDirectory = ""
                    autoRevealInFinder = false
                    rememberSettings = true
                    playSoundOnCompletion = true
                    showNotification = true
                    defaultTemporalMode = 0
                    defaultWindowSize = 15
                }
                .foregroundStyle(.red)
                .font(.callout)
            }

            Section("Training Data") {
                Toggle("Contribute anonymous patches to improve denoising", isOn: $trainingDataConsent)
                    .onChange(of: trainingDataConsent) { _, newValue in
                        if !newValue {
                            TrainingDataManager.shared.deleteAllData()
                        }
                    }
                Text("Small pixel patches are collected during processing and uploaded in the background. No personal information is included.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if trainingDataConsent {
                    let stats = TrainingDataManager.shared.getStats()
                    LabeledContent("Patches contributed", value: "\(stats.patchesContributed)")
                    LabeledContent("Pending upload", value: String(format: "%.1f MB", stats.pendingUploadMB))
                }
            }

            Section("Command Line") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("BayerFlow can be run headless from the terminal for batch processing and automation.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text("""
                    BayerFlow --headless --input <path> [options]
                      --output <path>        Output file path
                      --frames <N>           Process first N frames
                      --window <N>           Temporal window size (3-15)
                      --strength <F>         Denoise strength (0.5-2.0)
                      --temporal-mode <N>    0=VST+Bilateral, 1=NLM
                      --output-format <fmt>  mov, dng, braw, exr, cineform
                      --contribute-data      Upload anonymous training patches
                    """)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                }
            }

            Section("About") {
                LabeledContent("Version", value: appVersion)
                LabeledContent("GPU", value: gpuName)
                LabeledContent("Log file") {
                    Button(ErrorLogger.shared.logFilePath) {
                        NSWorkspace.shared.selectFile(ErrorLogger.shared.logFilePath,
                                                      inFileViewerRootedAtPath: "")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                }
                LabeledContent("Support") {
                    Link("bayerflow.com", destination: URL(string: "https://bayerflow.com")!)
                        .foregroundStyle(.blue)
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 450, height: 620)
        .fileImporter(isPresented: $showingFolderPicker,
                      allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                defaultOutputDirectory = url.path
            }
        }
    }
}
