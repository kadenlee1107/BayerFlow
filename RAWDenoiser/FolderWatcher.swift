import Foundation

nonisolated final class FolderWatcher: @unchecked Sendable {
    let url: URL
    nonisolated(unsafe) var onNewFile: ((URL) -> Void)?

    private var source: DispatchSourceFileSystemObject?
    private var fileDescriptor: Int32 = -1
    private var knownFiles: Set<String> = []
    private var debounceItem: DispatchWorkItem?
    private let watchQueue = DispatchQueue(label: "com.bayerflow.folderwatcher", qos: .utility)

    init(url: URL) {
        self.url = url
    }

    deinit {
        stop()
    }

    func start() {
        stop()

        // Snapshot existing files so we only trigger on NEW ones
        knownFiles = scanMovFiles()

        fileDescriptor = open(url.path, O_EVTONLY)
        guard fileDescriptor >= 0 else { return }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: .write,
            queue: watchQueue
        )

        source.setEventHandler { [weak self] in
            self?.handleDirectoryChange()
        }

        source.setCancelHandler { [weak self] in
            guard let self else { return }
            if self.fileDescriptor >= 0 {
                close(self.fileDescriptor)
                self.fileDescriptor = -1
            }
        }

        self.source = source
        source.resume()
    }

    func stop() {
        debounceItem?.cancel()
        source?.cancel()
        source = nil
    }

    private func handleDirectoryChange() {
        // Debounce: wait 2 seconds for file copy to finish
        debounceItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            self?.checkForNewFiles()
        }
        debounceItem = item
        watchQueue.asyncAfter(deadline: .now() + 2.0, execute: item)
    }

    private func checkForNewFiles() {
        let currentFiles = scanMovFiles()
        let newFiles = currentFiles.subtracting(knownFiles)

        for filename in newFiles {
            let fileURL = url.appendingPathComponent(filename)

            // Wait for file size to stabilize (still being copied?)
            guard isFileStable(fileURL) else { continue }

            knownFiles.insert(filename)
            onNewFile?(fileURL)
        }

        // Update known files (in case files were removed)
        knownFiles = currentFiles
    }

    private static let supportedExtensions: Set<String> = [
        "mov", "braw", "dng", "ari", "r3d", "crm", "mxf", "nraw"
    ]

    private func scanMovFiles() -> Set<String> {
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: url.path) else {
            return []
        }
        return Set(contents.filter { name in
            let ext = (name as NSString).pathExtension.lowercased()
            return Self.supportedExtensions.contains(ext)
        })
    }

    private func isFileStable(_ fileURL: URL) -> Bool {
        guard let attrs1 = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
              let size1 = attrs1[.size] as? Int else { return false }

        Thread.sleep(forTimeInterval: 1.0)

        guard let attrs2 = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
              let size2 = attrs2[.size] as? Int else { return false }

        return size1 == size2 && size1 > 0
    }
}
