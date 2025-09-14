// Author: DocuRay Team | Version: v0.1.3 | Modified: 2025-09-14 | Purpose: NameIndex TDD tests
import XCTest
@testable import FastNameCore

final class NameIndexTests: XCTestCase {
    private var tempDir: URL!
    private var dbPath: String!
    private var store: SQLiteStore!
    private var index: NameIndex!

    override func setUp() async throws {
        try await super.setUp()
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        dbPath = tempDir.appendingPathComponent("index_test.db").path
        store = try SQLiteStore(path: dbPath)
        index = NameIndex(store: store)

        // Populate store with test data
        try populateTestData()
    }

    override func tearDown() async throws {
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    private func populateTestData() throws {
        let testRecords = [
            FileRecord(path: "/project/src/main.swift", size: 1024, modTime: 1694691234, isDir: false),
            FileRecord(path: "/project/src/utils.swift", size: 512, modTime: 1694691235, isDir: false),
            FileRecord(path: "/project/tests/MainTests.swift", size: 800, modTime: 1694691236, isDir: false),
            FileRecord(path: "/project/docs/README.md", size: 2048, modTime: 1694691237, isDir: false),
            FileRecord(path: "/project/node_modules/package.json", size: 256, modTime: 1694691238, isDir: false),
            FileRecord(path: "/project/build", size: 0, modTime: 1694691239, isDir: true),
            FileRecord(path: "/project/src/controllers/UserController.swift", size: 1536, modTime: 1694691240, isDir: false),
            FileRecord(path: "/project/src/models/User.swift", size: 768, modTime: 1694691241, isDir: false),
            FileRecord(path: "/home/user/documents/file.txt", size: 128, modTime: 1694691242, isDir: false),
            FileRecord(path: "/home/user/downloads/image.png", size: 4096, modTime: 1694691243, isDir: false)
        ]

        try store.batchInsert(testRecords)
    }

    func test_load_from_store() throws {
        XCTAssertEqual(index.getRecordCount(), 0, "Index should be empty before loading")

        try index.loadFromStore()

        XCTAssertEqual(index.getRecordCount(), 10, "Index should contain all test records after loading")
    }

    func test_search_exact_filename_match() throws {
        try index.loadFromStore()

        let results = index.search(query: "main.swift")

        XCTAssertGreaterThan(results.count, 0, "Should find exact filename match")
        XCTAssertEqual(results.first?.path, "/project/src/main.swift", "Should return exact match first")
        XCTAssertGreaterThan(results.first?.score ?? 0, 900, "Exact match should have very high score")
    }

    func test_search_prefix_match() throws {
        try index.loadFromStore()

        let results = index.search(query: "main")

        let exactMatch = results.first { $0.path.contains("main.swift") }
        XCTAssertNotNil(exactMatch, "Should find prefix match")
        XCTAssertGreaterThan(exactMatch?.score ?? 0, 700, "Prefix match should have high score")
    }

    func test_search_substring_match() throws {
        try index.loadFromStore()

        let results = index.search(query: "swift")

        let swiftFiles = results.filter { $0.path.contains("swift") }
        XCTAssertGreaterThan(swiftFiles.count, 0, "Should find files containing 'swift'")

        // Should find multiple Swift files
        let mainSwift = swiftFiles.first { $0.path.contains("main.swift") }
        let utilsSwift = swiftFiles.first { $0.path.contains("utils.swift") }
        XCTAssertNotNil(mainSwift, "Should find main.swift")
        XCTAssertNotNil(utilsSwift, "Should find utils.swift")
    }

    func test_search_with_extension_filter() throws {
        try index.loadFromStore()

        let options = SearchOptions(extensions: [".swift"])
        let results = index.search(query: "User", options: options)

        XCTAssertGreaterThan(results.count, 0, "Should find Swift files with 'User'")

        // All results should be .swift files
        for result in results {
            XCTAssertTrue(result.path.hasSuffix(".swift"), "All results should be Swift files: \(result.path)")
        }
    }

    func test_search_with_root_path_filter() throws {
        try index.loadFromStore()

        let options = SearchOptions(rootPaths: ["/project/src"])
        let results = index.search(query: "swift", options: options)

        XCTAssertGreaterThan(results.count, 0, "Should find files in /project/src")

        // All results should be under /project/src
        for result in results {
            XCTAssertTrue(result.path.hasPrefix("/project/src"), "All results should be under /project/src: \(result.path)")
        }

        // Should not find files outside /project/src
        let homeFiles = results.filter { $0.path.hasPrefix("/home") }
        XCTAssertEqual(homeFiles.count, 0, "Should not find files under /home")
    }

    func test_search_max_results_limit() throws {
        try index.loadFromStore()

        let options = SearchOptions(maxResults: 3)
        let results = index.search(query: "swift", options: options)

        XCTAssertLessThanOrEqual(results.count, 3, "Should respect maxResults limit")
    }

    func test_search_case_sensitivity() throws {
        try index.loadFromStore()

        // Case insensitive (default)
        let resultsInsensitive = index.search(query: "SWIFT")
        XCTAssertGreaterThan(resultsInsensitive.count, 0, "Should find files with case insensitive search")

        // Case sensitive
        let optionsSensitive = SearchOptions(caseSensitive: true)
        let resultsSensitive = index.search(query: "SWIFT", options: optionsSensitive)
        XCTAssertEqual(resultsSensitive.count, 0, "Should not find files with case sensitive search for 'SWIFT'")

        let resultsSensitiveCorrect = index.search(query: "swift", options: optionsSensitive)
        XCTAssertGreaterThan(resultsSensitiveCorrect.count, 0, "Should find files with case sensitive search for 'swift'")
    }

    func test_search_results_sorted_by_score() throws {
        try index.loadFromStore()

        let results = index.search(query: "main")

        // Results should be sorted by score (descending)
        for i in 1..<results.count {
            XCTAssertGreaterThanOrEqual(results[i-1].score, results[i].score,
                "Results should be sorted by score: \(results[i-1].score) >= \(results[i].score)")
        }
    }

    func test_add_and_remove_records() throws {
        try index.loadFromStore()

        let initialCount = index.getRecordCount()

        // Add new record
        let newRecord = FileRecord(path: "/new/file.txt", size: 100, modTime: 1694691250, isDir: false)
        index.addRecord(newRecord)

        XCTAssertEqual(index.getRecordCount(), initialCount + 1, "Should have one more record after adding")

        // Should be able to search for new record
        let searchResults = index.search(query: "file.txt")
        let foundNew = searchResults.contains { $0.path == "/new/file.txt" }
        XCTAssertTrue(foundNew, "Should find newly added record")

        // Remove record
        index.removeRecord(path: "/new/file.txt")
        XCTAssertEqual(index.getRecordCount(), initialCount, "Should have original count after removing")

        // Should not find removed record
        let searchAfterRemove = index.search(query: "file.txt")
        let foundAfterRemove = searchAfterRemove.contains { $0.path == "/new/file.txt" }
        XCTAssertFalse(foundAfterRemove, "Should not find removed record")
    }

    func test_search_performance_large_dataset() throws {
        // Create a large dataset
        var largeDataset: [FileRecord] = []
        for i in 0..<10000 {
            largeDataset.append(FileRecord(
                path: "/large/dataset/file\(i).swift",
                size: Int64(i),
                modTime: 1694691234 + Int64(i),
                isDir: false
            ))
        }

        try store.batchInsert(largeDataset)
        try index.loadFromStore()

        let startTime = CFAbsoluteTimeGetCurrent()
        let results = index.search(query: "file5000")
        let duration = CFAbsoluteTimeGetCurrent() - startTime

        XCTAssertLessThan(duration, 0.1, "Search should complete within 100ms for 10k records, took \(duration)s")
        XCTAssertGreaterThan(results.count, 0, "Should find results in large dataset")

        // Should find the exact match
        let exactMatch = results.first { $0.path.contains("file5000.swift") }
        XCTAssertNotNil(exactMatch, "Should find exact match in large dataset")
    }
}