import Foundation
import Combine

/// Detects Final Cut Pro library locations and offers to save denoised output
/// directly into a library's media folder so FCP discovers it automatically.
///
/// FCP libraries are `.fcpbundle` packages typically stored in `~/Movies/`.
/// Inside each library, media lives under:
///   `LibraryName.fcpbundle/<UUID>/<UUID>/Original Media/`
///
/// Workflow: user denoises a raw clip → output goes into the selected FCP
/// library's media folder → FCP sees it in its browser without manual import.
final class FCPLibraryDetector: ObservableObject {

    struct FCPLibrary: Identifiable, Hashable {
        let id = UUID()
        let name: String         // Display name (e.g. "My Project")
        let url: URL             // Path to the .fcpbundle
        let mediaFolders: [URL]  // All "Original Media" folders found inside
    }

    @Published var detectedLibraries: [FCPLibrary] = []
    @Published var selectedLibrary: FCPLibrary? = nil
    @Published var enabled: Bool = false

    private let fileManager = FileManager.default

    // MARK: - Scan for FCP libraries

    /// Scans known locations for FCP library bundles.
    /// Auto-selects the first library found. Call on app launch.
    func scan() {
        var libraries: [FCPLibrary] = []

        // 1. Check ~/Movies/ (default FCP location)
        if let moviesDir = fileManager.urls(for: .moviesDirectory, in: .userDomainMask).first {
            libraries.append(contentsOf: findLibraries(in: moviesDir))
        }

        // 2. Check ~/Desktop/ (common alternative)
        if let desktopDir = fileManager.urls(for: .desktopDirectory, in: .userDomainMask).first {
            libraries.append(contentsOf: findLibraries(in: desktopDir))
        }

        // 3. Check ~/Documents/
        if let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            libraries.append(contentsOf: findLibraries(in: docsDir))
        }

        // 4. Use Spotlight to find any .fcpbundle files elsewhere
        libraries.append(contentsOf: spotlightSearch())

        // Deduplicate by URL
        var seen = Set<String>()
        detectedLibraries = libraries.filter { lib in
            let path = lib.url.path
            if seen.contains(path) { return false }
            seen.insert(path)
            return true
        }

        // Auto-select the first library with media folders
        if let first = detectedLibraries.first(where: { !$0.mediaFolders.isEmpty })
            ?? detectedLibraries.first {
            selectedLibrary = first
            enabled = true
        }
    }

    // MARK: - Output URL

    /// Build an output URL for a denoised clip.
    /// Saves to the same directory as the selected FCP library (e.g. ~/Movies/)
    /// so it's easy for the user to import into FCP.
    func outputURL(for inputURL: URL, defaultDir: URL) -> URL {
        let baseName = inputURL.deletingPathExtension().lastPathComponent
        let isBRAW = denoise_probe_format(inputURL.path) == 2
        let ext = isBRAW ? "braw" : "mov"
        let outputName = "\(baseName)_denoised.\(ext)"

        if enabled, let lib = selectedLibrary {
            // Save next to the library bundle, not inside it
            let libraryDir = lib.url.deletingLastPathComponent()
            return libraryDir.appendingPathComponent(outputName)
        }
        return defaultDir.appendingPathComponent(outputName)
    }

    /// Import a file into FCP by asking FCP to open it (triggers import dialog).
    func importIntoFCP(fileURL: URL) {
        let script = """
        tell application "Final Cut Pro"
            activate
            open POSIX file "\(fileURL.path)"
        end tell
        """
        if let appleScript = NSAppleScript(source: script) {
            var error: NSDictionary?
            appleScript.executeAndReturnError(&error)
        }
    }

    // MARK: - Private helpers

    private func findLibraries(in directory: URL) -> [FCPLibrary] {
        var results: [FCPLibrary] = []

        guard let contents = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return results }

        for item in contents where item.pathExtension == "fcpbundle" {
            results.append(buildLibrary(from: item))
        }

        return results
    }

    private func buildLibrary(from bundleURL: URL) -> FCPLibrary {
        let name = bundleURL.deletingPathExtension().lastPathComponent
        let mediaFolders = findMediaFolders(in: bundleURL)
        return FCPLibrary(name: name, url: bundleURL, mediaFolders: mediaFolders)
    }

    /// Recursively finds "Original Media" folders inside an FCP bundle.
    private func findMediaFolders(in bundleURL: URL) -> [URL] {
        var mediaFolders: [URL] = []

        guard let enumerator = fileManager.enumerator(
            at: bundleURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles],
            errorHandler: nil
        ) else { return mediaFolders }

        // Limit search depth to avoid traversing huge media trees
        var visited = 0
        let maxVisit = 500

        while let url = enumerator.nextObject() as? URL {
            visited += 1
            if visited > maxVisit { break }

            if url.lastPathComponent == "Original Media" {
                var isDir: ObjCBool = false
                if fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                    mediaFolders.append(url)
                }
            }
        }

        return mediaFolders
    }

    /// Use Spotlight to find .fcpbundle files anywhere on disk.
    private func spotlightSearch() -> [FCPLibrary] {
        var results: [FCPLibrary] = []

        // Synchronous Spotlight search using MDQuery
        let queryString = "kMDItemFSName == '*.fcpbundle' && kMDItemContentTypeTree == 'com.apple.package'"
        guard let query = MDQueryCreate(kCFAllocatorDefault, queryString as CFString, nil, nil) else {
            return results
        }

        MDQuerySetSearchScope(query, [kMDQueryScopeHome] as CFArray, 0)

        if MDQueryExecute(query, CFOptionFlags(kMDQuerySynchronous.rawValue)) {
            let count = MDQueryGetResultCount(query)
            for i in 0..<count {
                guard let rawPtr = MDQueryGetResultAtIndex(query, i) else { continue }
                let item = Unmanaged<MDItem>.fromOpaque(rawPtr).takeUnretainedValue()
                if let path = MDItemCopyAttribute(item, kMDItemPath) as? String {
                    let url = URL(fileURLWithPath: path)
                    if url.pathExtension == "fcpbundle" {
                        results.append(buildLibrary(from: url))
                    }
                }
            }
        }

        return results
    }
}
