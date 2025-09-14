// Prefix index for fast prefix/substring match // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

final class PrefixIndex { // Minimal in-memory index (hash buckets)
    private var byPrefix: [String: [String]] = [:]
    private let queue = DispatchQueue(label: "spot.prefix.index", qos: SpotSearchConfig.lowPriorityQoS)

    func insert(path: String) { // Tokenize into prefixes
        let name = (path as NSString).lastPathComponent.lowercased()
        queue.sync {
            for i in 1...min(6, name.count) { // first N prefixes for speed
                let key = String(name.prefix(i))
                byPrefix[key, default: []].append(path)
            }
        }
    }

    func remove(path: String) { // Lazy removal (best-effort)
        let name = (path as NSString).lastPathComponent.lowercased()
        queue.sync {
            for i in 1...min(6, name.count) {
                let key = String(name.prefix(i))
                if var arr = byPrefix[key] { byPrefix[key] = arr.filter { $0 != path } }
            }
        }
    }

    func lookup(query: String, limit: Int) -> [SearchHit] { // Fast prefix candidates
        let q = query.lowercased()
        var paths: [String] = []
        queue.sync { paths = byPrefix[String(q.prefix(min(6, q.count)))] ?? [] }
        let sliced = paths.prefix(limit)
        return sliced.map { SearchHit(path: $0, score: 1.0) }
    }
}

