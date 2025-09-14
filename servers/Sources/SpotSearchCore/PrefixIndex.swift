// Prefix index for fast prefix/substring match // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

final class PrefixIndex { // Minimal in-memory index (hash buckets + 3-gram)
    private var byPrefix: [String: [String]] = [:]
    private var byTrigram: [String: [String]] = [:]
    private let queue = DispatchQueue(label: "spot.prefix.index", qos: SpotSearchConfig.lowPriorityQoS)

    func insert(path: String) { // Tokenize into prefixes
        let name = (path as NSString).lastPathComponent.lowercased()
        queue.sync {
            for i in 1...min(6, name.count) { // first N prefixes for speed
                let key = String(name.prefix(i))
                byPrefix[key, default: []].append(path)
            }
            // index 3-grams for substring lookup
            if name.count >= 3 {
                let chars = Array(name)
                for i in 0..<(chars.count - 2) {
                    let key = String(chars[i..<(i+3)])
                    byTrigram[key, default: []].append(path)
                }
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
            if name.count >= 3 {
                let chars = Array(name)
                for i in 0..<(chars.count - 2) {
                    let key = String(chars[i..<(i+3)])
                    if var arr = byTrigram[key] { byTrigram[key] = arr.filter { $0 != path } }
                }
            }
        }
    }

    func lookup(query: String, limit: Int) -> [SearchHit] { // Prefix + substring candidates
        let q = query.lowercased()
        var paths: [String] = []
        queue.sync {
            if q.count >= 3, let list = byTrigram[String(q.prefix(3))] { paths = list }
            let pfx = byPrefix[String(q.prefix(min(6, q.count)))] ?? []
            paths = pfx + paths
        }
        if paths.isEmpty { return [] }
        // de-dup keeping order
        var seen = Set<String>()
        let unique = paths.filter { seen.insert($0).inserted }
        return Array(unique.prefix(limit)).map { SearchHit(path: $0, score: 1.0) }
    }
}
