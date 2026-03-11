import SwiftUI
import UniformTypeIdentifiers

// MARK: - License activation sheet

struct LicenseView: View {
    @EnvironmentObject private var license: LicenseManager
    @State private var email = ""
    @State private var key   = ""

    var body: some View {
        VStack(spacing: 20) {
            VStack(spacing: 4) {
                Text("Activate BayerFlow")
                    .font(.title3.bold())
                Text("Enter the email address and license key from your purchase confirmation.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Email").font(.subheadline).foregroundStyle(.secondary)
                TextField("you@example.com", text: $email)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("License key").font(.subheadline).foregroundStyle(.secondary)
                TextField("Paste license key here", text: $key, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...4)
            }

            if let err = license.activationError {
                HStack {
                    Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.red)
                    Text(err).font(.callout)
                }
                .padding(10)
                .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }

            HStack(spacing: 12) {
                Button("Cancel") { license.showActivation = false }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Activate") {
                    license.activate(email: email, licenseKey: key)
                }
                .buttonStyle(.borderedProminent)
                .disabled(email.isEmpty || key.isEmpty)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 420)
        .fixedSize()
    }
}

// MARK: - Minimal FileDocument for fileExporter

struct EmptyDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.movie] }
    init() {}
    init(configuration: ReadConfiguration) throws {}
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data())
    }
}
