import Foundation
import simd

// MARK: - Cube LUT model

struct CubeLUT {
    enum LUTType { case oneD, threeD }
    let type: LUTType
    let size: Int           // 1D: entry count, 3D: grid size per axis
    let domainMin: SIMD3<Float>
    let domainMax: SIMD3<Float>
    let data: [SIMD3<Float>]  // flat RGB triplets (R-fastest for 3D)
    let title: String
}

// MARK: - Parser

enum CubeLUTError: LocalizedError {
    case noSizeDeclaration
    case sizeMismatch(expected: Int, got: Int)
    case emptyFile

    var errorDescription: String? {
        switch self {
        case .noSizeDeclaration: return "No LUT_1D_SIZE or LUT_3D_SIZE found"
        case .sizeMismatch(let exp, let got): return "Expected \(exp) entries, got \(got)"
        case .emptyFile: return "File is empty"
        }
    }
}

enum CubeLUTLoader {

    static func load(from url: URL) throws -> CubeLUT {
        _ = url.startAccessingSecurityScopedResource()
        defer { url.stopAccessingSecurityScopedResource() }

        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)
        guard !lines.isEmpty else { throw CubeLUTError.emptyFile }

        var title = url.deletingPathExtension().lastPathComponent
        var size1D: Int?
        var size3D: Int?
        var domainMin = SIMD3<Float>(0, 0, 0)
        var domainMax = SIMD3<Float>(1, 1, 1)
        var entries: [SIMD3<Float>] = []

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }

            if trimmed.hasPrefix("TITLE") {
                // Extract quoted title: TITLE "My LUT"
                if let first = trimmed.firstIndex(of: "\""),
                   let last = trimmed.lastIndex(of: "\""), first != last {
                    title = String(trimmed[trimmed.index(after: first)..<last])
                }
                continue
            }
            if trimmed.hasPrefix("LUT_1D_SIZE") {
                size1D = Int(trimmed.split(separator: " ").last ?? "")
                continue
            }
            if trimmed.hasPrefix("LUT_3D_SIZE") {
                size3D = Int(trimmed.split(separator: " ").last ?? "")
                continue
            }
            if trimmed.hasPrefix("DOMAIN_MIN") {
                let vals = trimmed.split(separator: " ").dropFirst().compactMap { Float($0) }
                if vals.count == 3 { domainMin = SIMD3<Float>(vals[0], vals[1], vals[2]) }
                continue
            }
            if trimmed.hasPrefix("DOMAIN_MAX") {
                let vals = trimmed.split(separator: " ").dropFirst().compactMap { Float($0) }
                if vals.count == 3 { domainMax = SIMD3<Float>(vals[0], vals[1], vals[2]) }
                continue
            }

            // Try to parse as "R G B" float triplet
            let parts = trimmed.split(separator: " ").compactMap { Float($0) }
            if parts.count >= 3 {
                entries.append(SIMD3<Float>(parts[0], parts[1], parts[2]))
            }
        }

        if let s = size3D {
            let expected = s * s * s
            guard entries.count == expected else {
                throw CubeLUTError.sizeMismatch(expected: expected, got: entries.count)
            }
            return CubeLUT(type: .threeD, size: s,
                          domainMin: domainMin, domainMax: domainMax,
                          data: entries, title: title)
        } else if let s = size1D {
            guard entries.count == s else {
                throw CubeLUTError.sizeMismatch(expected: s, got: entries.count)
            }
            return CubeLUT(type: .oneD, size: s,
                          domainMin: domainMin, domainMax: domainMax,
                          data: entries, title: title)
        } else {
            throw CubeLUTError.noSizeDeclaration
        }
    }
}
