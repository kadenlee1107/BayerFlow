import Foundation

/// Lightweight file-based error logger.
/// Writes to ~/Library/Logs/BayerFlow/bayerflow.log with automatic rotation.
final class ErrorLogger {
    static let shared = ErrorLogger()

    private let logDir: URL
    private let logFile: URL
    private let maxLogSize = 5 * 1024 * 1024 // 5 MB
    private let queue = DispatchQueue(label: "com.bayerflow.logger", qos: .utility)
    private let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return f
    }()

    private init() {
        let logsDir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Logs")
            .appendingPathComponent("BayerFlow")
        self.logDir = logsDir
        self.logFile = logsDir.appendingPathComponent("bayerflow.log")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
    }

    func log(_ message: String, level: Level = .info, file: String = #fileID, line: Int = #line) {
        queue.async { [self] in
            rotateIfNeeded()
            let timestamp = dateFormatter.string(from: Date())
            let entry = "[\(timestamp)] [\(level.rawValue)] \(file):\(line) — \(message)\n"
            if FileManager.default.fileExists(atPath: logFile.path) {
                if let handle = try? FileHandle(forWritingTo: logFile) {
                    handle.seekToEndOfFile()
                    handle.write(Data(entry.utf8))
                    handle.closeFile()
                }
            } else {
                try? Data(entry.utf8).write(to: logFile, options: .atomic)
            }
        }
    }

    func error(_ message: String, file: String = #fileID, line: Int = #line) {
        log(message, level: .error, file: file, line: line)
    }

    func warning(_ message: String, file: String = #fileID, line: Int = #line) {
        log(message, level: .warning, file: file, line: line)
    }

    var logFilePath: String { logFile.path }

    enum Level: String {
        case info = "INFO"
        case warning = "WARN"
        case error = "ERROR"
    }

    private func rotateIfNeeded() {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: logFile.path),
              let size = attrs[.size] as? Int, size > maxLogSize else { return }
        let rotated = logDir.appendingPathComponent("bayerflow.1.log")
        try? FileManager.default.removeItem(at: rotated)
        try? FileManager.default.moveItem(at: logFile, to: rotated)
    }
}
