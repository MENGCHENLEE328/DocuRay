// FSEvents watcher wrapper (stub) // Author: Team DocuRay | Generated: module bootstrap | Version: 0.1.0 | Modified: 2025-09-14

import Foundation

final class FSEventsWatcher { // Placeholder to be replaced by real FSEvents stream
    enum Kind { case created, deleted, modified, renamed }
    struct Event { let path: String; let kind: Kind }

    private var handler: ((Event) -> Void)?
    private var running = false

    func start(paths: [String], onEvent: @escaping (Event) -> Void) { // Start stream
        self.handler = onEvent
        self.running = true
        // TODO: integrate real FSEvents with lastEventId persistence
    }

    func stop() { running = false; handler = nil }
}

