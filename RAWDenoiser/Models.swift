import SwiftUI

// MARK: - Denoise Presets

enum DenoisePreset: String, CaseIterable, Identifiable {
    case light    = "Light"
    case standard = "Standard"
    case strong   = "Strong"
    case custom   = "Custom"

    var id: String { rawValue }

    var strength: Float {
        switch self {
        case .light:    return 0.8
        case .standard: return 1.2
        case .strong:   return 1.5
        case .custom:   return 1.5
        }
    }

    var windowSize: Double {
        switch self {
        case .light:    return 7
        case .standard: return 11
        case .strong:   return 15
        case .custom:   return 15
        }
    }

    var hint: String {
        switch self {
        case .light:    return "Preserves maximum detail"
        case .standard: return "Balanced noise reduction"
        case .strong:   return "Maximum noise reduction"
        case .custom:   return "Manual slider control"
        }
    }

    /// Returns the preset matching given values, or .custom if none match.
    static func matching(strength: Float, windowSize: Double) -> DenoisePreset {
        for p in [DenoisePreset.light, .standard, .strong] {
            if abs(p.strength - strength) < 0.05 && abs(p.windowSize - windowSize) < 0.5 {
                return p
            }
        }
        return .custom
    }
}

// MARK: - Format Hub

struct FormatItem: Identifiable {
    let id = UUID()
    let name: String
    let initials: String
    let color: Color
    let isActive: Bool
    let imageName: String
    var needsSDK: Bool = false
}

let rawFormats: [FormatItem] = [
    FormatItem(name: "ProRes RAW", initials: "PR", color: Color(red: 1.0, green: 0.58, blue: 0.0), isActive: true, imageName: "LogoProRes"),
    FormatItem(name: "Blackmagic RAW", initials: "BM", color: Color(red: 0.61, green: 0.35, blue: 0.71), isActive: true, imageName: "LogoBlackmagic"),
    FormatItem(name: "ARRIRAW", initials: "AR", color: Color(red: 0.16, green: 0.50, blue: 0.73), isActive: true, imageName: "LogoARRI"),
    FormatItem(name: "RED R3D", initials: "R3D", color: Color(red: 0.91, green: 0.30, blue: 0.24), isActive: true, imageName: "LogoRED", needsSDK: true),
    FormatItem(name: "CinemaDNG", initials: "DNG", color: Color(red: 0.10, green: 0.74, blue: 0.61), isActive: true, imageName: "LogoCinemaDNG"),
    FormatItem(name: "Canon CRM", initials: "CRM", color: Color(red: 0.85, green: 0.11, blue: 0.14), isActive: true, imageName: "LogoCanon"),
    FormatItem(name: "Nikon N-RAW", initials: "NR", color: Color(red: 0.98, green: 0.82, blue: 0.0), isActive: true, imageName: "LogoNikon", needsSDK: true),
    FormatItem(name: "GoPro RAW", initials: "GP", color: Color(red: 0.0, green: 0.60, blue: 0.87), isActive: true, imageName: "LogoGoPro"),
    FormatItem(name: "Z CAM ZRAW", initials: "ZR", color: Color(red: 0.20, green: 0.20, blue: 0.20), isActive: true, imageName: "LogoZCam"),
]
