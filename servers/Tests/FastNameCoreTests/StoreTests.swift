// Author: DocuRay Team | Version: v0.1.1 | Modified: 2025-09-14 | Purpose: SQLiteStore TDD tests
import XCTest
@testable import FastNameCore

final class StoreTests: XCTestCase {
    private var tempDir: URL!
    private var dbPath: String!

    override func setUp() {
        super.setUp()
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try! FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        dbPath = tempDir.appendingPathComponent("test.db").path
    }

    override func tearDown() {
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        super.tearDown()
    }

    func test_create_tables_on_init() throws {
        let store = try SQLiteStore(path: dbPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dbPath), "Database file should exist after init")
    }

    func test_upsert_and_get_file_record() throws {
        let store = try SQLiteStore(path: dbPath)
        let record = FileRecord(path: "/test/file.txt", size: 1024, modTime: 1694691234, isDir: false)

        try store.upsertFile(record)
        let retrieved = try store.getFile(path: "/test/file.txt")

        XCTAssertNotNil(retrieved, "Should retrieve inserted record")
        XCTAssertEqual(retrieved?.path, "/test/file.txt")
        XCTAssertEqual(retrieved?.size, 1024)
        XCTAssertEqual(retrieved?.modTime, 1694691234)
        XCTAssertEqual(retrieved?.isDir, false)
    }

    func test_get_nonexistent_file_returns_nil() throws {
        let store = try SQLiteStore(path: dbPath)
        let retrieved = try store.getFile(path: "/nonexistent/file.txt")
        XCTAssertNil(retrieved, "Should return nil for nonexistent file")
    }

    func test_batch_insert_performance() throws {
        let store = try SQLiteStore(path: dbPath)
        let records = (0..<10000).map { i in
            FileRecord(path: "/test/file\(i).txt", size: Int64(i), modTime: 1694691234, isDir: false)
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        try store.batchInsert(records)
        let endTime = CFAbsoluteTimeGetCurrent()

        let duration = endTime - startTime
        XCTAssertLessThan(duration, 5.0, "Batch insert of 10k records should complete within 5 seconds, took \(duration)s")

        // Verify some records were inserted
        let first = try store.getFile(path: "/test/file0.txt")
        let last = try store.getFile(path: "/test/file9999.txt")
        XCTAssertNotNil(first)
        XCTAssertNotNil(last)
        XCTAssertEqual(last?.size, 9999)
    }

    func test_upsert_updates_existing_record() throws {
        let store = try SQLiteStore(path: dbPath)
        let original = FileRecord(path: "/test/file.txt", size: 1024, modTime: 1694691234, isDir: false)
        let updated = FileRecord(path: "/test/file.txt", size: 2048, modTime: 1694691500, isDir: false)

        try store.upsertFile(original)
        try store.upsertFile(updated)

        let retrieved = try store.getFile(path: "/test/file.txt")
        XCTAssertEqual(retrieved?.size, 2048, "Size should be updated")
        XCTAssertEqual(retrieved?.modTime, 1694691500, "ModTime should be updated")
    }
}