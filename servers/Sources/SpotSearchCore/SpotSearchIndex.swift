// SpotSearch main index interface (per-volume + in-memory) // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

public final class SpotSearchIndex { // Facade for name/path search
    private let prefix = PrefixIndex()
    private let fuzzy = BKTree()
    private let scorer = Ranking()
    private let watcher = FSEventsWatcher()

    public init() { } // Init without side effects

    // MARK: - Mutation
    public func add(path: String, weight: Double = 1.0) { // Add a single file path
        prefix.insert(path: path)
        fuzzy.insert(term: (path as NSString).lastPathComponent)
        // weight currently unused; to integrate into scorer
    }

    public func addAll(paths: [String]) { // Batch add
        for p in paths { add(path: p) }
    }

    // MARK: - Query
    public func search(query: String, limit: Int = SpotSearchConfig.maxResults) -> [SearchHit] { // Prefix→fuzzy cascade
        let prefixHits = prefix.lookup(query: query, limit: limit)
        if prefixHits.count >= limit { return scorer.rank(prefixHits, query: query, limit: limit) }
        let remaining = max(0, limit - prefixHits.count)
        let fuzzyHits = fuzzy.lookup(term: query, maxDistance: 2, limit: remaining)
        let merged = prefixHits + fuzzyHits
        return scorer.rank(merged, query: query, limit: limit)
    }

    // MARK: - Incremental updates
    public func startWatching(paths: [String]) { // Start FSEvents watcher
        watcher.start(paths: paths) { [weak self] event in
            switch event.kind {
            case .created, .renamed:
                self?.add(path: event.path)
            case .deleted:
                self?.prefix.remove(path: event.path)
                self?.fuzzy.remove(term: (event.path as NSString).lastPathComponent)
            case .modified:
                // noop for name index
                break
            }
        }
    }
}

public struct SearchHit { // Single line hit
    public let path: String
    public let score: Double
}

