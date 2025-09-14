// Author: DocuRay Team | Version: v0.1.2 | Modified: 2025-09-14 | Purpose: Concurrent file system scanner for FastName indexing
import Foundation

public struct ScanOptions {
    public let roots: [String]
    public let excludePatterns: [String]
    public let followSymlinks: Bool
    public let maxDepth: Int?

    public init(roots: [String], excludePatterns: [String] = [], followSymlinks: Bool = false, maxDepth: Int? = nil) {
        self.roots = roots; self.excludePatterns = excludePatterns
        self.followSymlinks = followSymlinks; self.maxDepth = maxDepth
    }
}

public class Scanner {
    private let options: ScanOptions
    private let store: SQLiteStore
    private let fileManager = FileManager.default
    private let queue = DispatchQueue(label: "scanner", attributes: .concurrent)
    private let group = DispatchGroup()

    public init(options: ScanOptions, store: SQLiteStore) {
        self.options = options; self.store = store
    }

    public func scanAll() async throws -> ScanResult {
        var totalFiles = 0
        var totalDirs = 0
        let startTime = CFAbsoluteTimeGetCurrent()

        for root in options.roots {
            try await scanDirectory(path: root, depth: 0, totalFiles: &totalFiles, totalDirs: &totalDirs)
        }

        let duration = CFAbsoluteTimeGetCurrent() - startTime
        return ScanResult(filesScanned: totalFiles, dirsScanned: totalDirs, duration: duration)
    }

    private func scanDirectory(path: String, depth: Int, totalFiles: inout Int, totalDirs: inout Int) async throws {
        if let maxDepth = options.maxDepth, depth >= maxDepth { return }
        if shouldExclude(path: path) { return }

        let url = URL(fileURLWithPath: path)

        // Check if path exists
        guard fileManager.fileExists(atPath: path) else {
            throw ScannerError.rootDirectoryNotFound(path)
        }

        do {
            let contents = try fileManager.contentsOfDirectory(atPath: path)

            for item in contents {
                let itemPath = path + "/" + item
                let itemURL = URL(fileURLWithPath: itemPath)

                if shouldExclude(path: itemPath) { continue }

                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: itemPath, isDirectory: &isDirectory) else { continue }

                // Handle symlinks
                let resourceValues = try? itemURL.resourceValues(forKeys: [URLResourceKey.isSymbolicLinkKey])
                let isSymlink = resourceValues?.isSymbolicLink ?? false
                if !options.followSymlinks && isSymlink { continue }

                // Get file attributes
                let attrs = try? fileManager.attributesOfItem(atPath: itemPath)
                let size = attrs?[.size] as? Int64 ?? 0
                let modTime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0

                let record = FileRecord(
                    path: itemPath,
                    size: size,
                    modTime: Int64(modTime),
                    isDir: isDirectory.boolValue
                )

                try store.upsertFile(record)

                if isDirectory.boolValue {
                    totalDirs += 1
                    // Recursively scan subdirectory
                    try await scanDirectory(path: itemPath, depth: depth + 1, totalFiles: &totalFiles, totalDirs: &totalDirs)
                } else {
                    totalFiles += 1
                }
            }
        } catch {
            throw ScannerError.cannotEnumerateDirectory(path)
        }
    }

    private func shouldExclude(path: String) -> Bool {
        let fileName = URL(fileURLWithPath: path).lastPathComponent
        return options.excludePatterns.contains { pattern in
            fileName.contains(pattern) || path.contains(pattern)
        }
    }
}

public struct ScanResult {
    public let filesScanned: Int
    public let dirsScanned: Int
    public let duration: Double

    public init(filesScanned: Int, dirsScanned: Int, duration: Double) {
        self.filesScanned = filesScanned; self.dirsScanned = dirsScanned; self.duration = duration
    }
}

public enum ScannerError: Error {
    case cannotEnumerateDirectory(String)
    case rootDirectoryNotFound(String)
}