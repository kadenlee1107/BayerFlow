import SwiftUI

struct OnboardingView: View {
    @Binding var isPresented: Bool

    private let steps: [(icon: String, title: String, description: String)] = [
        ("film", "Load Your Footage",
         "Drag & drop a RAW video file or click the format orb to browse. Supports ProRes RAW, BRAW, ARRIRAW, RED R3D, CinemaDNG, Canon CRM, and more."),
        ("slider.horizontal.3", "Adjust Settings",
         "Use the sidebar to tune temporal strength, window size, and spatial filtering. Generate a preview to check quality before processing."),
        ("bolt.fill", "Denoise",
         "Hit Denoise (⌘D) to process. BayerFlow uses GPU-accelerated temporal filtering with optical flow — your output keeps the original RAW format."),
    ]

    var body: some View {
        VStack(spacing: 24) {
            Text("Welcome to BayerFlow")
                .font(.title.bold())

            Text("Professional RAW video denoising")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 20) {
                ForEach(Array(steps.enumerated()), id: \.offset) { _, step in
                    HStack(alignment: .top, spacing: 14) {
                        Image(systemName: step.icon)
                            .font(.title2)
                            .foregroundStyle(.orange)
                            .frame(width: 32)

                        VStack(alignment: .leading, spacing: 4) {
                            Text(step.title)
                                .font(.headline)
                            Text(step.description)
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
            .padding(.horizontal, 8)

            Button("Get Started") {
                isPresented = false
            }
            .keyboardShortcut(.defaultAction)
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .controlSize(.large)
        }
        .padding(32)
        .frame(width: 440)
    }
}
