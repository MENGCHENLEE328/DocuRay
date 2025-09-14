// Author: DocuRay Team | Version: v0.1.3 | Modified: 2025-09-14 | Purpose: In-memory fast name search index for FastName
import Foundation

public struct SearchResult {
    public let path: String
    public let score: Double
    public let isDir: Bool

    public init(path: String, score: Double, isDir: Bool) {
        self.path = path; self.score = score; self.isDir = isDir
    }
}

public struct SearchOptions {
    public let maxResults: Int
    public let extensions: [String]
    public let rootPaths: [String]
    public let caseSensitive: Bool

    public init(maxResults: Int = 50, extensions: [String] = [], rootPaths: [String] = [], caseSensitive: Bool = false) {
        self.maxResults = maxResults; self.extensions = extensions
        self.rootPaths = rootPaths; self.caseSensitive = caseSensitive
    }
}

public class NameIndex {
    private var fileRecords: [FileRecord] = []
    private var nameCache: [String: [Int]] = [:]  // filename -> indices mapping
    private let store: SQLiteStore
    private var isLoaded: Bool = false

    public init(store: SQLiteStore) {
        self.store = store
    }

    public func loadFromStore() throws {
        // Load all file records from SQLite store
        fileRecords.removeAll()
        nameCache.removeAll()

        let records = try store.getAllFiles()

        for record in records {
            addRecord(record)
        }

        isLoaded = true
    }

    public func search(query: String, options: SearchOptions = SearchOptions()) -> [SearchResult] {
        guard isLoaded else { return [] }
        guard !query.isEmpty else { return [] }

        let normalizedQuery = options.caseSensitive ? query : query.lowercased()
        var results: [SearchResult] = []

        // Search through cached filenames
        for (i, record) in fileRecords.enumerated() {
            let filename = URL(fileURLWithPath: record.path).lastPathComponent
            let normalizedFilename = options.caseSensitive ? filename : filename.lowercased()

            // Skip if doesn't match extension filter
            if !options.extensions.isEmpty {
                let fileExt = URL(fileURLWithPath: record.path).pathExtension.lowercased()
                if !options.extensions.contains(where: { ext in
                    let normalizedExt = ext.lowercased().replacingOccurrences(of: ".", with: "")
                    return normalizedExt == fileExt
                }) {
                    continue
                }
            }

            // Skip if doesn't match root path filter
            if !options.rootPaths.isEmpty {
                let matchesRoot = options.rootPaths.contains { rootPath in
                    record.path.hasPrefix(rootPath)
                }
                if !matchesRoot { continue }
            }

            // Calculate match score
            if let score = calculateScore(filename: normalizedFilename, query: normalizedQuery) {
                results.append(SearchResult(
                    path: record.path,
                    score: score,
                    isDir: record.isDir
                ))
            }

            // Limit results for performance
            if results.count >= options.maxResults * 2 { break }
        }

        // Sort by score (higher is better) and limit results
        results.sort { $0.score > $1.score }
        return Array(results.prefix(options.maxResults))
    }

    private func calculateScore(filename: String, query: String) -> Double? {
        // Exact match gets highest score
        if filename == query { return 1000.0 }

        // Prefix match gets high score
        if filename.hasPrefix(query) { return 800.0 + Double(query.count) / Double(filename.count) * 100 }

        // Substring match gets medium score
        if filename.contains(query) {
            let position = filename.range(of: query)?.lowerBound.utf16Offset(in: filename) ?? filename.count
            let positionScore = max(0, 100 - Double(position))  // Earlier matches score higher
            let lengthScore = Double(query.count) / Double(filename.count) * 100
            return 400.0 + positionScore + lengthScore
        }

        // Fuzzy match for individual characters
        let fuzzyScore = calculateFuzzyScore(filename: filename, query: query)
        return fuzzyScore > 0 ? fuzzyScore : nil
    }

    private func calculateFuzzyScore(filename: String, query: String) -> Double {
        let filenameChars = Array(filename)
        let queryChars = Array(query)
        var matches = 0
        var queryIndex = 0

        for char in filenameChars {
            if queryIndex < queryChars.count && char == queryChars[queryIndex] {
                matches += 1
                queryIndex += 1
            }
        }

        if matches == queryChars.count {
            return 100.0 * Double(matches) / Double(filenameChars.count)
        }
        return 0
    }

    public func addRecord(_ record: FileRecord) {
        guard !fileRecords.contains(where: { $0.path == record.path }) else { return }

        fileRecords.append(record)
        let filename = URL(fileURLWithPath: record.path).lastPathComponent.lowercased()

        if nameCache[filename] == nil {
            nameCache[filename] = []
        }
        nameCache[filename]?.append(fileRecords.count - 1)
    }

    public func removeRecord(path: String) {
        if let index = fileRecords.firstIndex(where: { $0.path == path }) {
            let record = fileRecords[index]
            fileRecords.remove(at: index)

            // Update cache
            let filename = URL(fileURLWithPath: record.path).lastPathComponent.lowercased()
            nameCache[filename]?.removeAll { $0 == index }
            if nameCache[filename]?.isEmpty == true {
                nameCache.removeValue(forKey: filename)
            }

            // Update indices in cache (shift down after removal)
            for (key, indices) in nameCache {
                nameCache[key] = indices.map { $0 > index ? $0 - 1 : $0 }
            }
        }
    }

    public func getRecordCount() -> Int {
        return fileRecords.count
    }

    public func markAsLoaded() {
        isLoaded = true
    }
}

public enum NameIndexError: Error {
    case notImplemented(String)
    case loadFailed(String)
}