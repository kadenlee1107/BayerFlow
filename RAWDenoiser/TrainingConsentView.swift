/*
 * Training Data Consent View
 *
 * Opt-in consent dialog shown once when the feature becomes available.
 * Explains what data is collected and why, and lets the user choose
 * to contribute or opt out.
 */

import SwiftUI

struct TrainingConsentView: View {
    @Binding var isPresented: Bool
    @AppStorage("trainingDataConsent") private var consent = false

    var body: some View {
        VStack(spacing: 20) {
            // Header
            Image(systemName: "brain.head.profile")
                .font(.system(size: 48))
                .foregroundStyle(.tint)

            Text("Help Improve BayerFlow Denoising")
                .font(.title2)
                .fontWeight(.semibold)

            // Explanation
            VStack(alignment: .leading, spacing: 12) {
                Text("BayerFlow can collect small anonymous pixel patches during processing to help train better denoising models.")
                    .foregroundStyle(.secondary)

                Divider()

                Label("What we collect", systemImage: "checkmark.circle.fill")
                    .fontWeight(.medium)
                Text("Small 256\u{00D7}256 pixel patches (noisy + denoised pairs), noise level, and ISO. No full frames, no filenames, no personal information.")
                    .foregroundStyle(.secondary)
                    .font(.callout)

                Label("What we don't collect", systemImage: "xmark.circle.fill")
                    .fontWeight(.medium)
                    .foregroundStyle(.red)
                Text("No full images, no file paths, no GPS data, no camera model name (hashed only), no personally identifiable information.")
                    .foregroundStyle(.secondary)
                    .font(.callout)

                Label("How it works", systemImage: "arrow.up.circle.fill")
                    .fontWeight(.medium)
                Text("Patches are saved locally and uploaded in the background when you're online. You can disable this at any time in Settings.")
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }
            .frame(maxWidth: 420)

            // Buttons
            HStack(spacing: 16) {
                Button("No Thanks") {
                    consent = false
                    isPresented = false
                }
                .buttonStyle(.bordered)

                Button("Contribute") {
                    consent = true
                    isPresented = false
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.top, 4)
        }
        .padding(32)
        .frame(width: 500)
    }
}
