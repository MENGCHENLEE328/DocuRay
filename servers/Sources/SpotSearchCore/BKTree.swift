// BK-tree for fuzzy matching (edit distance) // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

final class BKTree { // Minimal stub (linear until replaced)
    private var termToPaths: [String: Set<String>] = [:] // term -> paths
    private let queue = DispatchQueue(label: "spot.bktree", qos: SpotSearchConfig.lowPriorityQoS)

    func insert(term: String, path: String) { // map filename to path set
        queue.sync { termToPaths[term.lowercased(), default: []].insert(path) }
    }
    func remove(term: String) { queue.sync { termToPaths.removeValue(forKey: term.lowercased()) } }

    func lookup(term: String, maxDistance: Int, limit: Int) -> [SearchHit] { // Linear fallback
        let q = term.lowercased()
        var out: [SearchHit] = []
        queue.sync {
            for (t, paths) in termToPaths {
                let d = BKTree.editDistance(q, t)
                if d <= maxDistance {
                    for p in paths {
                        out.append(SearchHit(path: p, score: 1.0 / Double(1 + d)))
                        if out.count >= limit { break }
                    }
                }
                if out.count >= limit { break }
            }
        }
        return out
    }

    static func editDistance(_ a: String, _ b: String) -> Int { // Classic DP
        let aa = Array(a), bb = Array(b)
        var dp = Array(repeating: Array(repeating: 0, count: bb.count + 1), count: aa.count + 1)
        for i in 0...aa.count { dp[i][0] = i }
        for j in 0...bb.count { dp[0][j] = j }
        for i in 1...aa.count {
            for j in 1...bb.count {
                let cost = (aa[i-1] == bb[j-1]) ? 0 : 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
            }
        }
        return dp[aa.count][bb.count]
    }
}
