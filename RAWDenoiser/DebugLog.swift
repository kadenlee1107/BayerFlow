import Foundation

/// Debug-only logging. Compiles to nothing in Release builds.
@inline(__always)
nonisolated func debugLog(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    #if DEBUG
    let output = items.map { "\($0)" }.joined(separator: separator)
    Swift.print(output, terminator: terminator)
    #endif
}
