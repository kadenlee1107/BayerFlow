import SwiftUI

struct SplashView: View {
    @State private var progress: Double = 0
    @State private var statusText: String = "Initializing…"
    @State private var opacity: Double = 1

    let onFinished: () -> Void

    var body: some View {
        ZStack {
            // Dark gradient background
            LinearGradient(
                colors: [Color(white: 0.08), Color(white: 0.12)],
                startPoint: .top, endPoint: .bottom
            )

            VStack(spacing: 0) {
                Spacer()

                // Logo (includes "BayerFlow" text)
                Image("SplashLogo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 200)
                    .shadow(color: .black.opacity(0.4), radius: 16, y: 8)

                Text("ProRes RAW Denoiser")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))
                    .padding(.top, 8)

                Spacer()

                // Loading bar + status
                VStack(spacing: 8) {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                        .tint(.orange.opacity(0.8))
                        .frame(width: 260)

                    Text(statusText)
                        .font(.system(size: 11, weight: .regular))
                        .foregroundStyle(.white.opacity(0.4))
                }
                .padding(.bottom, 32)
            }
        }
        .frame(width: 520, height: 360)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .opacity(opacity)
        .task {
            await runStartup()
        }
    }

    private func runStartup() async {
        let start = Date()

        // Step 1: License check
        statusText = "Checking license…"
        withAnimation(.linear(duration: 0.4)) { progress = 0.3 }
        try? await Task.sleep(nanoseconds: 400_000_000)

        // Step 2: Loading components
        statusText = "Loading components…"
        withAnimation(.linear(duration: 0.5)) { progress = 0.6 }
        try? await Task.sleep(nanoseconds: 500_000_000)

        // Step 3: Preparing workspace
        statusText = "Preparing workspace…"
        withAnimation(.linear(duration: 0.4)) { progress = 0.9 }
        try? await Task.sleep(nanoseconds: 400_000_000)

        // Ensure minimum 2 seconds total
        let elapsed = Date().timeIntervalSince(start)
        if elapsed < 2.0 {
            let remaining = UInt64((2.0 - elapsed) * 1_000_000_000)
            try? await Task.sleep(nanoseconds: remaining)
        }

        // Finish
        statusText = "Ready"
        withAnimation(.linear(duration: 0.2)) { progress = 1.0 }
        try? await Task.sleep(nanoseconds: 200_000_000)

        // Fade out
        withAnimation(.easeOut(duration: 0.3)) { opacity = 0 }
        try? await Task.sleep(nanoseconds: 350_000_000)

        onFinished()
    }
}
