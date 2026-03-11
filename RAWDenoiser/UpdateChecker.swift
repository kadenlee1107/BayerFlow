import AppKit

/// Lightweight manual update checker.
/// Fetches a JSON file from the website to compare versions.
/// Format: { "version": "1.1", "url": "https://bayerflow.com/download" }
@MainActor
final class UpdateChecker {
    static let shared = UpdateChecker()
    private static let versionURL = URL(string: "https://bayerflow.com/version.json")!

    private var isChecking = false

    func checkForUpdates() {
        guard !isChecking else { return }
        isChecking = true

        Task {
            defer { isChecking = false }

            do {
                let (data, _) = try await URLSession.shared.data(from: Self.versionURL)
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let latestVersion = json["version"] as? String,
                      let downloadURL = json["url"] as? String else {
                    showAlert(title: "Update Check Failed",
                              message: "Could not read version information from the server.")
                    return
                }

                let currentVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0"

                if compareVersions(current: currentVersion, latest: latestVersion) == .orderedAscending {
                    let alert = NSAlert()
                    alert.messageText = "Update Available"
                    alert.informativeText = "BayerFlow \(latestVersion) is available. You are running \(currentVersion)."
                    alert.addButton(withTitle: "Download")
                    alert.addButton(withTitle: "Later")
                    alert.alertStyle = .informational

                    if alert.runModal() == .alertFirstButtonReturn {
                        if let url = URL(string: downloadURL) {
                            NSWorkspace.shared.open(url)
                        }
                    }
                } else {
                    showAlert(title: "You're Up to Date",
                              message: "BayerFlow \(currentVersion) is the latest version.")
                }
            } catch {
                showAlert(title: "Update Check Failed",
                          message: "Could not connect to the update server. Check your internet connection.")
            }
        }
    }

    private func compareVersions(current: String, latest: String) -> ComparisonResult {
        return current.compare(latest, options: .numeric)
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.alertStyle = .informational
        alert.runModal()
    }
}
