import Foundation
import SwiftUI

// MARK: - Codable session snapshot

struct CodableFrameRange: Codable {
    var start: Int
    var end: Int
}

struct CodableQueueItem: Codable {
    var inputPath: String
    var outputPath: String
    var strength: Float
    var windowSize: Int32
    var spatialStrength: Float
    var useML: Bool
    var startFrame: Int32
    var endFrame: Int32
    var outputFormat: Int32
}

struct SessionSnapshot: Codable, Identifiable {
    let id: String          // session UUID string
    var label: String
    var formatColorHex: String

    // File paths
    var inputPath: String?
    var outputPath: String?

    // Denoise settings
    var strength: Float
    var windowSize: Double
    var spatialStrength: Float
    var presetName: String

    // Engine flags
    var protectSubjects: Bool
    var invertMask: Bool
    var autoDarkFrame: Bool
    var temporalFilterMode: Int32
    var useML: Bool

    // Format-specific output
    var dngOutputFormat: Int
    var brawOutputFormat: Int

    // Trim ranges
    var selectedRanges: [CodableFrameRange]

    // LUT state
    var lutPath: String?
    var lutBlend: Float
    var lutEnabled: Bool

    // Queue
    var queueItems: [CodableQueueItem]

    // Metadata
    var timestamp: Date
    var videoWidth: Int
    var videoHeight: Int
    var frameCount: Int

    // Hub state
    var showHub: Bool
}

// MARK: - Color ↔ hex helpers

extension Color {
    func toHex() -> String {
        guard let c = NSColor(self).usingColorSpace(.sRGB) else { return "808080" }
        let r = Int(c.redComponent * 255)
        let g = Int(c.greenComponent * 255)
        let b = Int(c.blueComponent * 255)
        return String(format: "%02X%02X%02X", r, g, b)
    }

    static func fromHex(_ hex: String) -> Color {
        var h = hex
        if h.hasPrefix("#") { h.removeFirst() }
        guard h.count == 6,
              let val = UInt64(h, radix: 16) else { return .secondary }
        let r = Double((val >> 16) & 0xFF) / 255
        let g = Double((val >> 8) & 0xFF) / 255
        let b = Double(val & 0xFF) / 255
        return Color(red: r, green: g, blue: b)
    }
}

// MARK: - Persistence manager

final class SessionPersistenceManager {
    static let shared = SessionPersistenceManager()

    private let sessionsDir: URL

    private init() {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        sessionsDir = appSupport
            .appendingPathComponent("BayerFlow/Sessions", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: sessionsDir, withIntermediateDirectories: true
        )
    }

    func save(_ snapshot: SessionSnapshot) {
        let fileURL = sessionsDir.appendingPathComponent("\(snapshot.id).json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(snapshot) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }

    func loadAll() -> [SessionSnapshot] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: sessionsDir, includingPropertiesForKeys: nil
        ) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return files.compactMap { url -> SessionSnapshot? in
            guard url.pathExtension == "json",
                  let data = try? Data(contentsOf: url) else { return nil }
            return try? decoder.decode(SessionSnapshot.self, from: data)
        }
        .sorted { $0.timestamp > $1.timestamp }
    }

    func delete(id: String) {
        let fileURL = sessionsDir.appendingPathComponent("\(id).json")
        try? FileManager.default.removeItem(at: fileURL)
    }

    func deleteAll() {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: sessionsDir, includingPropertiesForKeys: nil
        ) else { return }
        for file in files where file.pathExtension == "json" {
            try? FileManager.default.removeItem(at: file)
        }
    }
}
