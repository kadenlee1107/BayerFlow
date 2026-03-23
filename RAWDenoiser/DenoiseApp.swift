import SwiftUI
import UniformTypeIdentifiers

// MARK: - FocusedValues for menu → view communication

struct AppActionsKey: FocusedValueKey {
    typealias Value = AppActions
}

struct TabActionsKey: FocusedValueKey {
    typealias Value = TabActions
}

extension FocusedValues {
    var appActions: AppActions? {
        get { self[AppActionsKey.self] }
        set { self[AppActionsKey.self] = newValue }
    }
    var tabActions: TabActions? {
        get { self[TabActionsKey.self] }
        set { self[TabActionsKey.self] = newValue }
    }
}

/// Actions for tab management, routed via FocusedValue.
@MainActor
final class TabActions {
    var newTab: (() -> Void)?
    var closeTab: (() -> Void)?
}

/// Actions that menu commands can trigger on the active window.
@MainActor
final class AppActions {
    var openFile: (() -> Void)?
    var startDenoise: (() -> Void)?
    var addToQueue: (() -> Void)?
    var generatePreview: (() -> Void)?
    var cancelProcessing: (() -> Void)?

    var canDenoise: Bool = false
    var canCancel: Bool = false
    var hasInput: Bool = false
}

// MARK: - App

@main
struct RAWDenoiserApp: App {
    @StateObject private var license = LicenseManager()
    @StateObject private var showcase = ShowcaseController()
    @State private var showSplash = true
    @State private var showTrainingConsent = false
    @State private var showOnboarding = false
    @FocusedValue(\.appActions) private var actions
    @FocusedValue(\.tabActions) private var tabActions

    init() {
        if ProcessInfo.processInfo.arguments.contains("--headless") {
            HeadlessRunner.run()  // never returns
        }
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        ErrorLogger.shared.log("BayerFlow \(version) launched")
        DenoiseEngine.requestNotificationPermission()
        // Retry pending training data uploads on launch
        if UserDefaults.standard.bool(forKey: "trainingDataConsent") {
            TrainingDataUploader.shared.uploadPendingBatches()
        }
        // Listen for FCP Workflow Extension denoise requests
        FCPExtensionBridge.shared.startListening()
    }

    var body: some Scene {
        WindowGroup {
            ZStack {
                if !showSplash {
                    TabContainerView()
                        .environmentObject(license)
                        .environmentObject(showcase)
                        .transition(.opacity)
                }

                if showSplash {
                    SplashView {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            showSplash = false
                        }
                        // Auto-start showcase if launched with --showcase flag
                        if let path = showcase.pendingFilePath {
                            showcase.pendingFilePath = nil
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                                showcase.start(filePath: path)
                            }
                        }
                        // Show onboarding on first launch, then training consent
                        else if !UserDefaults.standard.bool(forKey: "onboardingShown") {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                showOnboarding = true
                                UserDefaults.standard.set(true, forKey: "onboardingShown")
                            }
                        } else if !UserDefaults.standard.bool(forKey: "trainingDataConsentShown") {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                showTrainingConsent = true
                                UserDefaults.standard.set(true, forKey: "trainingDataConsentShown")
                            }
                        }
                    }
                    .transition(.opacity)
                }
            }
            .sheet(isPresented: $showOnboarding, onDismiss: {
                // After onboarding, show training consent if not yet shown
                if !UserDefaults.standard.bool(forKey: "trainingDataConsentShown") {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        showTrainingConsent = true
                        UserDefaults.standard.set(true, forKey: "trainingDataConsentShown")
                    }
                }
            }) {
                OnboardingView(isPresented: $showOnboarding)
            }
            .sheet(isPresented: $showTrainingConsent) {
                TrainingConsentView(isPresented: $showTrainingConsent)
            }
        }
        .windowResizability(.contentSize)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Tab") {
                    tabActions?.newTab?()
                }
                .keyboardShortcut("T", modifiers: .command)

                Button("Close Tab") {
                    tabActions?.closeTab?()
                }
                .keyboardShortcut("W", modifiers: .command)

                Divider()

                Button("Open…") {
                    actions?.openFile?()
                }
                .keyboardShortcut("O", modifiers: .command)
                .disabled(actions?.hasInput == true && actions?.canCancel == true)
            }

            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") {
                    UpdateChecker.shared.checkForUpdates()
                }

                Divider()

                Button("Activate License…") {
                    license.showActivation = true
                }
                .keyboardShortcut("L", modifiers: [.command, .shift])
            }

            CommandMenu("Process") {
                Button("Denoise") {
                    actions?.startDenoise?()
                }
                .keyboardShortcut("D", modifiers: .command)
                .disabled(actions?.canDenoise != true)

                Button("Add to Queue") {
                    actions?.addToQueue?()
                }
                .keyboardShortcut("D", modifiers: [.command, .shift])
                .disabled(actions?.hasInput != true)

                Divider()

                Button("Generate Preview") {
                    actions?.generatePreview?()
                }
                .keyboardShortcut("P", modifiers: .command)
                .disabled(actions?.hasInput != true)

                Divider()

                Button("Cancel") {
                    actions?.cancelProcessing?()
                }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(actions?.canCancel != true)

                Divider()

                Button(showcase.isActive ? "Stop Showcase" : "Showcase Demo\u{2026}") {
                    if showcase.isActive {
                        showcase.stop()
                    } else {
                        let panel = NSOpenPanel()
                        panel.allowedContentTypes = [.movie]
                        panel.message = "Select a ProRes RAW file for the showcase demo"
                        panel.prompt = "Use for Demo"
                        if panel.runModal() == .OK, let url = panel.url {
                            showcase.start(filePath: url.path)
                        }
                    }
                }
            }
        }

        Settings {
            SettingsView()
        }
    }
}
