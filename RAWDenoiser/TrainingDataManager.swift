/*
 * Training Data Manager
 *
 * Receives patches from C bridge, serializes them into compressed batch files
 * (.bfpatch), and manages local disk storage. Triggers upload when batches
 * are ready.
 *
 * Storage: ~/Library/Application Support/BayerFlow/training_data/
 * Format: BFPT binary with LZ4-compressed patch data
 */

import Foundation
import Compression

class TrainingDataManager {
    static let shared = TrainingDataManager()

    private let storageDir: URL
    private let maxStorageMB: Int = 500
    private let batchThreshold = 20  // patches per batch

    private var currentBatch: [TrainingPatch] = []
    private var totalPatchesContributed: Int {
        get { UserDefaults.standard.integer(forKey: "trainingPatchesContributed") }
        set { UserDefaults.standard.set(newValue, forKey: "trainingPatchesContributed") }
    }

    private let queue = DispatchQueue(label: "com.bayerflow.training-data", qos: .utility)

    private init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        storageDir = appSupport.appendingPathComponent("BayerFlow/training_data", isDirectory: true)
        try? FileManager.default.createDirectory(at: storageDir, withIntermediateDirectories: true)
    }

    /// Called from C bridge on the denoising thread. Copies into queue for async processing.
    func submitPatch(_ patch: TrainingPatch) {
        queue.async { [weak self] in
            guard let self else { return }

            // Redundant Swift-side validation (C side validates first, this is a safety net)
            guard self.validatePatch(patch) else { return }

            self.currentBatch.append(patch)

            if self.currentBatch.count >= self.batchThreshold {
                self.finalizeBatch()
            }
        }
    }

    /// Validate patch data before serializing to disk.
    /// Redundant safety net — C side validates first, but this catches anything it misses.
    private func validatePatch(_ patch: TrainingPatch) -> Bool {
        let total = patch.patchW * patch.patchH
        guard total > 0 else { return false }
        guard patch.noisyData.count == total * 2,
              patch.denoisedData.count == total * 2 else { return false }

        let clampLow: UInt16 = 64       // 4 << 4
        let clampHigh: UInt16 = 65456   // 4091 << 4

        var noisyClamped = 0
        var denoisedClamped = 0
        var identical = 0

        patch.noisyData.withUnsafeBytes { noisyBuf in
            patch.denoisedData.withUnsafeBytes { denoisedBuf in
                let noisy = noisyBuf.bindMemory(to: UInt16.self)
                let denoised = denoisedBuf.bindMemory(to: UInt16.self)

                for i in 0..<total {
                    let n = noisy[i], d = denoised[i]
                    if n == clampLow || n == clampHigh { noisyClamped += 1 }
                    if d == clampLow || d == clampHigh { denoisedClamped += 1 }
                    if n == d { identical += 1 }
                }
            }
        }

        // Reject if >1% pixels at clamp bounds (decoder corruption)
        if noisyClamped > total / 100 || denoisedClamped > total / 100 { return false }

        // Reject if noisy == denoised for >95% of pixels (denoiser did nothing)
        if identical > total * 95 / 100 { return false }

        // Reject all-same-value (stuck sensor/buffer)
        let firstNoisy = patch.noisyData.withUnsafeBytes { $0.load(as: UInt16.self) }
        let allSame = patch.noisyData.withUnsafeBytes { buf -> Bool in
            let pixels = buf.bindMemory(to: UInt16.self)
            return pixels.allSatisfy { $0 == firstNoisy }
        }
        if allSame { return false }

        return true
    }

    /// Flush remaining patches into a batch file (called when processing ends).
    func flush() {
        queue.async { [weak self] in
            guard let self, !self.currentBatch.isEmpty else { return }
            self.finalizeBatch()
        }
    }

    /// Get stats for the Settings UI.
    func getStats() -> (patchesContributed: Int, pendingUploadMB: Double) {
        let patches = totalPatchesContributed
        let pendingBytes = pendingBatchBytes()
        return (patches, Double(pendingBytes) / (1024 * 1024))
    }

    // MARK: - Private

    private func finalizeBatch() {
        guard !currentBatch.isEmpty else { return }

        let patches = currentBatch
        currentBatch = []

        let batchID = UUID().uuidString.prefix(12)
        let filename = "batch_\(batchID).bfpatch"
        let fileURL = storageDir.appendingPathComponent(filename)

        do {
            let data = try serializeBatch(patches)
            try data.write(to: fileURL)

            totalPatchesContributed += patches.count

            // Update manifest
            updateManifest(filename: filename, patchCount: patches.count, status: "pending")

            fputs("TrainingDataManager: wrote batch \(filename) — \(patches.count) patches, \(data.count / 1024) KB\n", stderr)

            // Trigger upload
            TrainingDataUploader.shared.uploadPendingBatches()

            // Enforce disk quota
            enforceQuota()

        } catch {
            fputs("TrainingDataManager: batch write failed: \(error)\n", stderr)
        }
    }

    /// Serialize patches into the BFPT binary format with LZ4 compression.
    private func serializeBatch(_ patches: [TrainingPatch]) throws -> Data {
        var data = Data()

        // Header (32 bytes)
        data.append(contentsOf: [0x42, 0x46, 0x50, 0x54])  // "BFPT" magic
        data.appendLE(UInt16(1))                             // version
        data.appendLE(UInt16(patches.count))                 // count
        data.append(Data(count: 24))                         // reserved

        for patch in patches {
            // Per-patch metadata (40 bytes)
            data.appendLE(UInt16(patch.patchW))
            data.appendLE(UInt16(patch.patchH))
            data.appendLE(UInt16(patch.frameW))
            data.appendLE(UInt16(patch.frameH))
            data.appendLE(UInt16(patch.patchX))
            data.appendLE(UInt16(patch.patchY))
            data.appendLE(UInt32(patch.frameIdx))
            data.appendLE(patch.noiseSigma)
            data.appendLE(patch.flowMag)
            data.appendLE(UInt32(patch.iso))
            data.appendLE(fnv1aHash(patch.cameraModel))  // camera hash

            // LZ4 compress noisy and denoised data
            let noisyCompressed = lz4Compress(patch.noisyData) ?? patch.noisyData
            let denoisedCompressed = lz4Compress(patch.denoisedData) ?? patch.denoisedData

            data.appendLE(UInt32(noisyCompressed.count))
            data.appendLE(UInt32(denoisedCompressed.count))
            data.append(noisyCompressed)
            data.append(denoisedCompressed)
        }

        return data
    }

    /// FNV-1a hash for camera model anonymization.
    private func fnv1aHash(_ string: String) -> UInt32 {
        var hash: UInt32 = 2166136261
        for byte in string.utf8 {
            hash ^= UInt32(byte)
            hash &*= 16777619
        }
        return hash
    }

    /// LZ4 compress data using Apple's Compression framework.
    private func lz4Compress(_ input: Data) -> Data? {
        let srcSize = input.count
        let dstCapacity = srcSize  // worst case = same size (random data)
        let dstBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: dstCapacity + 4)
        defer { dstBuffer.deallocate() }

        // Store original size as LE uint32 prefix
        var origSize = UInt32(srcSize).littleEndian
        memcpy(dstBuffer, &origSize, 4)

        let compressedSize = input.withUnsafeBytes { srcBuf -> Int in
            guard let srcPtr = srcBuf.baseAddress else { return 0 }
            return compression_encode_buffer(
                dstBuffer.advanced(by: 4),
                dstCapacity,
                srcPtr.assumingMemoryBound(to: UInt8.self),
                srcSize,
                nil,
                COMPRESSION_LZ4
            )
        }

        guard compressedSize > 0 else { return nil }
        return Data(bytes: dstBuffer, count: compressedSize + 4)
    }

    // MARK: - Manifest

    private var manifestURL: URL { storageDir.appendingPathComponent("manifest.json") }

    private func loadManifest() -> [[String: Any]] {
        guard let data = try? Data(contentsOf: manifestURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }
        return json
    }

    private func saveManifest(_ entries: [[String: Any]]) {
        guard let data = try? JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted) else { return }
        try? data.write(to: manifestURL)
    }

    private func updateManifest(filename: String, patchCount: Int, status: String) {
        var manifest = loadManifest()
        let entry: [String: Any] = [
            "filename": filename,
            "patch_count": patchCount,
            "status": status,
            "created_at": ISO8601DateFormatter().string(from: Date())
        ]
        manifest.append(entry)
        saveManifest(manifest)
    }

    func markBatchUploaded(_ filename: String) {
        queue.async { [weak self] in
            guard let self else { return }
            var manifest = self.loadManifest()
            for i in 0..<manifest.count {
                if manifest[i]["filename"] as? String == filename {
                    manifest[i]["status"] = "uploaded"
                }
            }
            self.saveManifest(manifest)
        }
    }

    func pendingBatchFiles() -> [URL] {
        let manifest = loadManifest()
        return manifest
            .filter { ($0["status"] as? String) == "pending" }
            .compactMap { entry -> URL? in
                guard let filename = entry["filename"] as? String else { return nil }
                let url = storageDir.appendingPathComponent(filename)
                return FileManager.default.fileExists(atPath: url.path) ? url : nil
            }
    }

    private func pendingBatchBytes() -> Int {
        var total = 0
        for url in pendingBatchFiles() {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
               let size = attrs[.size] as? Int {
                total += size
            }
        }
        return total
    }

    /// Delete oldest uploaded batches to stay under disk quota.
    private func enforceQuota() {
        var manifest = loadManifest()
        let maxBytes = maxStorageMB * 1024 * 1024

        // Calculate total storage
        var totalBytes = 0
        for entry in manifest {
            guard let filename = entry["filename"] as? String else { continue }
            let url = storageDir.appendingPathComponent(filename)
            if let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int {
                totalBytes += size
            }
        }

        // Delete oldest uploaded batches first
        if totalBytes > maxBytes {
            let uploadedIndices = manifest.enumerated()
                .filter { ($0.element["status"] as? String) == "uploaded" }
                .map { $0.offset }
                .reversed()

            var toRemove: [Int] = []
            for idx in uploadedIndices {
                guard totalBytes > maxBytes else { break }
                if let filename = manifest[idx]["filename"] as? String {
                    let url = storageDir.appendingPathComponent(filename)
                    if let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int {
                        try? FileManager.default.removeItem(at: url)
                        totalBytes -= size
                        toRemove.append(idx)
                    }
                }
            }

            for idx in toRemove.sorted().reversed() {
                manifest.remove(at: idx)
            }
            saveManifest(manifest)
        }
    }

    /// Delete all local training data (when user opts out).
    func deleteAllData() {
        queue.async { [weak self] in
            guard let self else { return }
            self.currentBatch = []
            try? FileManager.default.removeItem(at: self.storageDir)
            try? FileManager.default.createDirectory(at: self.storageDir, withIntermediateDirectories: true)
            fputs("TrainingDataManager: all local training data deleted\n", stderr)
        }
    }
}

// MARK: - Data Helpers

private extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: MemoryLayout<T>.size))
    }

    mutating func appendLE(_ value: Float) {
        var v = value
        append(Data(bytes: &v, count: 4))
    }
}
