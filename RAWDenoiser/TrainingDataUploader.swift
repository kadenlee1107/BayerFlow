/*
 * Training Data Uploader
 *
 * Background URLSession-based uploader for .bfpatch training data files.
 * Handles offline queuing, retry, and configurable endpoint.
 *
 * Upload flow:
 * 1. POST /upload-request → get presigned URL + batch_id
 * 2. PUT batch file to presigned URL
 * 3. POST /upload-complete with batch_id
 * 4. Mark batch as uploaded in manifest, delete local file
 */

import Foundation

class TrainingDataUploader: NSObject {
    static let shared = TrainingDataUploader()

    private var isUploading = false
    private let queue = DispatchQueue(label: "com.bayerflow.training-upload", qos: .utility)

    /// Configurable endpoint URL (stored in UserDefaults for dev/testing).
    var endpointBase: String {
        UserDefaults.standard.string(forKey: "trainingDataEndpoint")
            ?? "https://bayerflow-training-api.bayerflow.workers.dev/v1/training"
    }

    /// Anonymous device ID — random UUID generated once, NOT hardware ID.
    private var deviceID: String {
        if let existing = UserDefaults.standard.string(forKey: "trainingDeviceID") {
            return existing
        }
        let newID = UUID().uuidString
        UserDefaults.standard.set(newID, forKey: "trainingDeviceID")
        return newID
    }

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 600
        config.httpMaximumConnectionsPerHost = 1
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()

    /// Upload any pending batch files. Called after batch finalization and on app launch.
    func uploadPendingBatches() {
        queue.async { [weak self] in
            guard let self, !self.isUploading else { return }

            let pending = TrainingDataManager.shared.pendingBatchFiles()
            guard !pending.isEmpty else { return }

            self.isUploading = true
            fputs("TrainingDataUploader: \(pending.count) pending batches to upload\n", stderr)

            self.uploadNext(from: pending, index: 0)
        }
    }

    private func uploadNext(from files: [URL], index: Int) {
        guard index < files.count else {
            isUploading = false
            return
        }

        let fileURL = files[index]
        let filename = fileURL.lastPathComponent

        uploadBatch(fileURL: fileURL) { [weak self] success in
            if success {
                TrainingDataManager.shared.markBatchUploaded(filename)
                fputs("TrainingDataUploader: uploaded \(filename)\n", stderr)
            } else {
                fputs("TrainingDataUploader: upload failed for \(filename), will retry later\n", stderr)
            }

            // Continue with next file regardless of success
            self?.queue.asyncAfter(deadline: .now() + 1.0) {
                self?.uploadNext(from: files, index: index + 1)
            }
        }
    }

    /// Upload a single batch file via the presigned URL flow.
    private func uploadBatch(fileURL: URL, completion: @escaping (Bool) -> Void) {
        guard let batchData = try? Data(contentsOf: fileURL) else {
            completion(false)
            return
        }

        let filename = fileURL.lastPathComponent

        // Step 1: Request presigned upload URL
        requestPresignedURL(batchSize: batchData.count, filename: filename) { [weak self] result in
            guard let self else { completion(false); return }

            switch result {
            case .success(let uploadInfo):
                // Step 2: PUT batch data to presigned URL
                self.putData(batchData, to: uploadInfo.presignedURL) { putSuccess in
                    guard putSuccess else { completion(false); return }

                    // Step 3: Notify server of completion
                    self.notifyUploadComplete(batchID: uploadInfo.batchID) { _ in
                        completion(true)
                    }
                }

            case .failure(let error):
                fputs("TrainingDataUploader: presigned URL request failed: \(error.localizedDescription)\n", stderr)
                completion(false)
            }
        }
    }

    // MARK: - HTTP Helpers

    private struct UploadInfo {
        let presignedURL: URL
        let batchID: String
    }

    private func requestPresignedURL(batchSize: Int, filename: String,
                                      completion: @escaping (Result<UploadInfo, Error>) -> Void) {
        guard let url = URL(string: "\(endpointBase)/upload-request") else {
            completion(.failure(URLError(.badURL)))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "batch_size": batchSize,
            "filename": filename,
            "device_id": deviceID,
            "app_version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        session.dataTask(with: request) { data, response, error in
            guard let data, error == nil,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let urlStr = json["presigned_url"] as? String,
                  let presignedURL = URL(string: urlStr),
                  let batchID = json["batch_id"] as? String else {
                completion(.failure(error ?? URLError(.badServerResponse)))
                return
            }

            completion(.success(UploadInfo(presignedURL: presignedURL, batchID: batchID)))
        }.resume()
    }

    private func putData(_ data: Data, to url: URL, completion: @escaping (Bool) -> Void) {
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.httpBody = data

        session.dataTask(with: request) { _, response, error in
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            completion(error == nil && (200..<300).contains(status))
        }.resume()
    }

    private func notifyUploadComplete(batchID: String, completion: @escaping (Bool) -> Void) {
        guard let url = URL(string: "\(endpointBase)/upload-complete") else {
            completion(false)
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: [
            "batch_id": batchID,
            "device_id": deviceID
        ])

        session.dataTask(with: request) { _, response, error in
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            completion(error == nil && (200..<300).contains(status))
        }.resume()
    }
}
