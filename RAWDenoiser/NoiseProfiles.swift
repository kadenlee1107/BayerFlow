import Foundation

nonisolated struct NoiseProfile: Sendable {
    let cameraPattern: String   // regex matching camera model string
    let displayName: String
    let isoSigmaMap: [(iso: Int, sigma: Float)]  // sorted by ISO ascending

    /// Linear interpolation between calibrated ISO/sigma pairs.
    func sigma(forISO iso: Int) -> Float {
        guard !isoSigmaMap.isEmpty else { return 0 }
        let isoF = Float(iso)

        // Below minimum
        if isoF <= Float(isoSigmaMap.first!.iso) {
            return isoSigmaMap.first!.sigma
        }
        // Above maximum
        if isoF >= Float(isoSigmaMap.last!.iso) {
            return isoSigmaMap.last!.sigma
        }

        // Find bracketing pair and interpolate
        for i in 0..<(isoSigmaMap.count - 1) {
            let lo = isoSigmaMap[i]
            let hi = isoSigmaMap[i + 1]
            if isoF >= Float(lo.iso) && isoF <= Float(hi.iso) {
                let t = (isoF - Float(lo.iso)) / (Float(hi.iso) - Float(lo.iso))
                return lo.sigma + t * (hi.sigma - lo.sigma)
            }
        }

        return isoSigmaMap.last!.sigma
    }
}

nonisolated enum NoiseProfiles {
    // Camera noise profile database.
    // Sigma values are in 16-bit pixel units for the green channel noise floor.
    // These are conservative starting estimates — refine with real footage.
    static let database: [NoiseProfile] = [

        // Nikon Z-series (ProRes RAW via Atomos Ninja V / Ninja V+)
        NoiseProfile(
            cameraPattern: "(?i)nikon.*z\\s*6",
            displayName: "Nikon Z6 / Z6 II / Z6 III",
            isoSigmaMap: [
                (100, 0.8), (200, 1.0), (400, 1.5), (800, 2.5),
                (1600, 4.0), (3200, 6.0), (6400, 9.0), (12800, 14.0), (25600, 22.0)
            ]
        ),
        NoiseProfile(
            cameraPattern: "(?i)nikon.*z\\s*(8|9)",
            displayName: "Nikon Z8 / Z9",
            isoSigmaMap: [
                (64, 0.5), (100, 0.7), (200, 0.9), (400, 1.3), (800, 2.2),
                (1600, 3.5), (3200, 5.5), (6400, 8.0), (12800, 12.0), (25600, 18.0)
            ]
        ),
        NoiseProfile(
            cameraPattern: "(?i)nikon.*z\\s*5",
            displayName: "Nikon Z5",
            isoSigmaMap: [
                (100, 0.9), (400, 1.6), (800, 2.8), (1600, 4.5),
                (3200, 7.0), (6400, 10.0), (12800, 16.0)
            ]
        ),

        // Canon (ProRes RAW via Atomos)
        NoiseProfile(
            cameraPattern: "(?i)canon.*r5\\s*c",
            displayName: "Canon EOS R5 C",
            isoSigmaMap: [
                (100, 0.6), (400, 1.2), (800, 2.0), (1600, 3.2),
                (3200, 5.0), (6400, 7.5), (12800, 11.0), (25600, 17.0)
            ]
        ),
        NoiseProfile(
            cameraPattern: "(?i)canon.*r5",
            displayName: "Canon EOS R5",
            isoSigmaMap: [
                (100, 0.7), (400, 1.3), (800, 2.2), (1600, 3.5),
                (3200, 5.5), (6400, 8.0), (12800, 12.0)
            ]
        ),
        NoiseProfile(
            cameraPattern: "(?i)canon.*r3",
            displayName: "Canon EOS R3",
            isoSigmaMap: [
                (100, 0.7), (400, 1.2), (800, 2.0), (1600, 3.0),
                (3200, 4.8), (6400, 7.0), (12800, 10.0), (25600, 15.0)
            ]
        ),

        // Panasonic (ProRes RAW via Atomos)
        NoiseProfile(
            cameraPattern: "(?i)panasonic.*(s1h|s5|s5ii|gh6|gh7)",
            displayName: "Panasonic S1H / S5 / GH6",
            isoSigmaMap: [
                (100, 0.8), (400, 1.5), (800, 2.5), (1600, 4.0),
                (3200, 6.5), (6400, 9.5), (12800, 14.0)
            ]
        ),

        // Sony (ProRes RAW via Atomos)
        NoiseProfile(
            cameraPattern: "(?i)sony.*(a7s|fx3|fx6|fx9)",
            displayName: "Sony A7S III / FX3 / FX6",
            isoSigmaMap: [
                (100, 0.5), (400, 0.9), (800, 1.5), (1600, 2.5),
                (3200, 4.0), (6400, 6.0), (12800, 9.0), (25600, 13.0), (51200, 20.0)
            ]
        ),

        // RED (ProRes RAW out)
        NoiseProfile(
            cameraPattern: "(?i)red.*(komodo|raptor|v-raptor|dsmc3)",
            displayName: "RED Komodo / V-Raptor",
            isoSigmaMap: [
                (250, 0.6), (800, 1.5), (1600, 2.8), (3200, 4.5),
                (6400, 7.0), (12800, 10.0)
            ]
        ),
    ]

    /// Find the best matching profile for a camera model string and ISO.
    /// Returns nil if no profile matches.
    static func match(camera: String, iso: Int) -> (profile: NoiseProfile, sigma: Float)? {
        for profile in database {
            guard let regex = try? NSRegularExpression(pattern: profile.cameraPattern) else { continue }
            let range = NSRange(camera.startIndex..., in: camera)
            if regex.firstMatch(in: camera, range: range) != nil {
                let sigma = profile.sigma(forISO: iso)
                return (profile, sigma)
            }
        }
        return nil
    }
}
