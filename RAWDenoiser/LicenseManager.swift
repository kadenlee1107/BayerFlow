import Foundation
import Combine
import CryptoKit
import Security

// MARK: - License Manager

@MainActor
final class LicenseManager: ObservableObject {
    @Published var isLicensed: Bool = false
    @Published var trialDaysRemaining: Int = 14
    @Published var showActivation: Bool = false
    @Published var activationError: String? = nil

    // Ed25519 public key for license validation.
    // Private key: save securely, NEVER commit. See licensing/generate_keypair.swift.
    private static let publicKeyHex = "e2c8b6a800342c7633dc086c9dbb80bc7f25a309a5b17f09652c18baf0e1fcf0"

    private static let keychainService = "com.rawdenoiser.license"
    private static let keychainAccount = "licensekey"
    private static let emailDefaultsKey = "rawdenoiser_email"
    private static let firstLaunchKey   = "rawdenoiser_first_launch"

    init() {
        recordFirstLaunchIfNeeded()
        checkLicense()
    }

    // MARK: - Public interface

    var canDenoise: Bool { isLicensed || trialDaysRemaining > 0 }

    var licenseStatusText: String {
        if isLicensed { return "Licensed" }
        if trialDaysRemaining > 0 { return "Trial — \(trialDaysRemaining) day\(trialDaysRemaining == 1 ? "" : "s") left" }
        return "Trial expired"
    }

    var isTrialExpired: Bool { !isLicensed && trialDaysRemaining <= 0 }

    func activate(email: String, licenseKey: String) {
        activationError = nil
        guard validate(licenseKey: licenseKey, email: email) else {
            activationError = "Invalid license key. Check your email and key, or contact support."
            return
        }
        keychainSave(licenseKey)
        UserDefaults.standard.set(email.lowercased().trimmingCharacters(in: .whitespaces),
                                  forKey: Self.emailDefaultsKey)
        isLicensed = true
        showActivation = false
    }

    // MARK: - Internal

    private func recordFirstLaunchIfNeeded() {
        if UserDefaults.standard.object(forKey: Self.firstLaunchKey) == nil {
            UserDefaults.standard.set(Date(), forKey: Self.firstLaunchKey)
        }
    }

    private func checkLicense() {
        // App Store builds: presence of receipt file means Apple validated the purchase.
        let receiptPath = Bundle.main.bundleURL
            .appendingPathComponent("Contents/_MASReceipt/receipt").path
        if FileManager.default.fileExists(atPath: receiptPath) {
            isLicensed = true
            return
        }

        // Direct sale: stored license key in Keychain
        if let key   = keychainLoad(),
           let email = UserDefaults.standard.string(forKey: Self.emailDefaultsKey),
           validate(licenseKey: key, email: email) {
            isLicensed = true
            return
        }

        // Trial countdown
        if let first = UserDefaults.standard.object(forKey: Self.firstLaunchKey) as? Date {
            let days = Calendar.current.dateComponents([.day], from: first, to: Date()).day ?? 0
            trialDaysRemaining = max(0, 14 - days)
        }
    }

    private func validate(licenseKey: String, email: String) -> Bool {
        let trimmedKey   = licenseKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedEmail = email.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        guard let keyData = Data(base64Encoded: trimmedKey),
              keyData.count >= 64,
              let emailData = trimmedEmail.data(using: .utf8)
        else { return false }

        let signatureBytes = keyData.prefix(64)

        do {
            let pubKeyData = Data(hexString: Self.publicKeyHex)!
            let publicKey  = try Curve25519.Signing.PublicKey(rawRepresentation: pubKeyData)
            return publicKey.isValidSignature(signatureBytes, for: emailData)
        } catch {
            return false
        }
    }

    // MARK: - Keychain

    private func keychainSave(_ value: String) {
        guard let data = value.data(using: .utf8) else { return }
        let attrs: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrService: Self.keychainService,
            kSecAttrAccount: Self.keychainAccount,
            kSecValueData:   data,
        ]
        SecItemDelete(attrs as CFDictionary)
        SecItemAdd(attrs as CFDictionary, nil)
    }

    private func keychainLoad() -> String? {
        let query: [CFString: Any] = [
            kSecClass:       kSecClassGenericPassword,
            kSecAttrService: Self.keychainService,
            kSecAttrAccount: Self.keychainAccount,
            kSecReturnData:  kCFBooleanTrue!,
            kSecMatchLimit:  kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - Helpers

extension Data {
    init?(hexString: String) {
        let hex = hexString.replacingOccurrences(of: " ", with: "")
        guard hex.count % 2 == 0 else { return nil }
        var data = Data(capacity: hex.count / 2)
        var idx = hex.startIndex
        while idx < hex.endIndex {
            let nextIdx = hex.index(idx, offsetBy: 2)
            guard let byte = UInt8(hex[idx..<nextIdx], radix: 16) else { return nil }
            data.append(byte)
            idx = nextIdx
        }
        self = data
    }
}
