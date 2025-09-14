// Author: DocuRay Team | Version: v0.1.2 | Modified: 2025-09-14 | Purpose: Scanner TDD tests
import XCTest
@testable import FastNameCore

final class ScannerTests: XCTestCase {
    private var tempDir: URL!
    private var dbPath: String!
    private var store: SQLiteStore!

    override func setUp() async throws {
        try await super.setUp()
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        dbPath = tempDir.appendingPathComponent("scanner_test.db").path
        store = try SQLiteStore(path: dbPath)

        // Create test directory structure
        try createTestStructure()
    }

    override func tearDown() async throws {
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    private func createTestStructure() throws {
        let fm = FileManager.default

        // Create directories
        try fm.createDirectory(at: tempDir.appendingPathComponent("subdir1"), withIntermediateDirectories: true)
        try fm.createDirectory(at: tempDir.appendingPathComponent("subdir2"), withIntermediateDirectories: true)
        try fm.createDirectory(at: tempDir.appendingPathComponent("node_modules"), withIntermediateDirectories: true)

        // Create files
        try "test content 1".write(to: tempDir.appendingPathComponent("file1.txt"), atomically: true, encoding: .utf8)
        try "test content 2".write(to: tempDir.appendingPathComponent("subdir1/file2.log"), atomically: true, encoding: .utf8)
        try "test content 3".write(to: tempDir.appendingPathComponent("subdir2/file3.json"), atomically: true, encoding: .utf8)
        try "ignored".write(to: tempDir.appendingPathComponent("node_modules/package.json"), atomically: true, encoding: .utf8)

        // Create symlink if possible
        let symlinkPath = tempDir.appendingPathComponent("symlink_to_file1")
        let targetPath = tempDir.appendingPathComponent("file1.txt")
        try? fm.createSymbolicLink(at: symlinkPath, withDestinationURL: targetPath)
    }

    func test_scan_empty_directory() async throws {
        let emptyDir = tempDir.appendingPathComponent("empty")
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        let options = ScanOptions(roots: [emptyDir.path])
        let scanner = Scanner(options: options, store: store)

        let result = try await scanner.scanAll()

        XCTAssertEqual(result.filesScanned, 0, "Should scan 0 files in empty directory")
        XCTAssertEqual(result.dirsScanned, 0, "Should scan 0 subdirectories in empty directory")
        XCTAssertGreaterThan(result.duration, 0, "Duration should be positive")
    }

    func test_scan_basic_structure() async throws {
        let options = ScanOptions(roots: [tempDir.path])
        let scanner = Scanner(options: options, store: store)

        let result = try await scanner.scanAll()

        // Should find files but exclude node_modules by default
        XCTAssertGreaterThan(result.filesScanned, 0, "Should scan some files")
        XCTAssertGreaterThan(result.dirsScanned, 0, "Should scan some directories")

        // Verify files are stored in database
        let file1 = try store.getFile(path: tempDir.appendingPathComponent("file1.txt").path)
        XCTAssertNotNil(file1, "file1.txt should be stored in database")
        XCTAssertEqual(file1?.isDir, false, "file1.txt should be marked as file, not directory")
        XCTAssertGreaterThan(file1?.size ?? 0, 0, "file1.txt should have non-zero size")
    }

    func test_exclude_patterns() async throws {
        let options = ScanOptions(
            roots: [tempDir.path],
            excludePatterns: ["node_modules", ".log"]
        )
        let scanner = Scanner(options: options, store: store)

        let result = try await scanner.scanAll()

        // Verify excluded files are not in database
        let nodeModulesFile = try store.getFile(path: tempDir.appendingPathComponent("node_modules/package.json").path)
        let logFile = try store.getFile(path: tempDir.appendingPathComponent("subdir1/file2.log").path)

        XCTAssertNil(nodeModulesFile, "node_modules files should be excluded")
        XCTAssertNil(logFile, ".log files should be excluded")

        // Verify non-excluded files are present
        let jsonFile = try store.getFile(path: tempDir.appendingPathComponent("subdir2/file3.json").path)
        XCTAssertNotNil(jsonFile, "JSON files should not be excluded")
    }

    func test_symlink_handling() async throws {
        // Test without following symlinks (default)
        let optionsNoFollow = ScanOptions(roots: [tempDir.path], followSymlinks: false)
        let scannerNoFollow = Scanner(options: optionsNoFollow, store: store)

        let resultNoFollow = try await scannerNoFollow.scanAll()

        let symlinkPath = tempDir.appendingPathComponent("symlink_to_file1").path
        let symlink = try store.getFile(path: symlinkPath)

        // With followSymlinks=false, symlink should not be scanned
        XCTAssertNil(symlink, "Symlinks should not be followed by default")

        // Test with following symlinks
        let optionsFollow = ScanOptions(roots: [tempDir.path], followSymlinks: true)
        let scannerFollow = Scanner(options: optionsFollow, store: store)

        let resultFollow = try await scannerFollow.scanAll()

        // With followSymlinks=true, more files might be found (symlink targets)
        // This is implementation dependent, just verify it doesn't crash
        XCTAssertGreaterThanOrEqual(resultFollow.filesScanned, resultNoFollow.filesScanned)
    }

    func test_max_depth_limit() async throws {
        let options = ScanOptions(
            roots: [tempDir.path],
            maxDepth: 1  // Only scan root level
        )
        let scanner = Scanner(options: options, store: store)

        let result = try await scanner.scanAll()

        // Should find root level files but not subdirectory files
        let rootFile = try store.getFile(path: tempDir.appendingPathComponent("file1.txt").path)
        let subFile = try store.getFile(path: tempDir.appendingPathComponent("subdir1/file2.log").path)

        XCTAssertNotNil(rootFile, "Root level files should be scanned")
        XCTAssertNil(subFile, "Subdirectory files should not be scanned with maxDepth=1")
    }

    func test_nonexistent_root_throws_error() async throws {
        let options = ScanOptions(roots: ["/nonexistent/path"])
        let scanner = Scanner(options: options, store: store)

        do {
            _ = try await scanner.scanAll()
            XCTFail("Should throw error for nonexistent root directory")
        } catch {
            // Expected to throw error
            XCTAssertTrue(error is ScannerError, "Should throw ScannerError")
        }
    }
}