// Heuristic ranking for name/path search // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

final class Ranking { // Simple heuristics (to be extended)
    func rank(_ hits: [SearchHit], query: String, limit: Int) -> [SearchHit] { // Score and sort
        let q = query.lowercased()
        let scored = hits.map { hit -> SearchHit in
            let name = (hit.path as NSString).lastPathComponent.lowercased()
            var s = hit.score
            if name == q { s += 2.0 } // exact file name
            if name.hasPrefix(q) { s += 1.0 } // prefix match
            if name.contains(q) { s += 0.5 } // substring
            return SearchHit(path: hit.path, score: s)
        }
        return Array(scored.sorted { $0.score > $1.score }.prefix(limit))
    }
}

