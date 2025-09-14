// SpotSearch config and constants // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

public struct SpotSearchConfig { // Centralized tunables
    public static let maxResults: Int = 100
    public static let maxQueueDepth: Int = 10_000
    public static let flushBatchSize: Int = 2_000
    public static let lowPriorityQoS: DispatchQoS = .utility
}
