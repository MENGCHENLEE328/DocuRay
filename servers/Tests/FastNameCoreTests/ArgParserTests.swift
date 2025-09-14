import XCTest
@testable import FastNameCore

// Author: DocuRay Team | Version: v0.1.0 | Purpose: T1 tests for ArgParser

final class ArgParserTests: XCTestCase {
    func test_help_command_parsed() throws {
        let p = ArgParser()
        let cmd = try p.parse(["--help"])
        if case .help = cmd { /* ok */ } else { XCTFail("expected help") }
        XCTAssertTrue(p.helpText().contains("fastname"))
    }

    func test_search_command_parsed() throws {
        let p = ArgParser()
        let cmd = try p.parse(["search", "hello", "--top", "10", "--ext", "md,txt"]) 
        if case let .search(q, top, exts, _) = cmd {
            XCTAssertEqual(q, "hello")
            XCTAssertEqual(top, 10)
            XCTAssertEqual(exts, ["md", "txt"])
        } else { XCTFail("expected search command") }
    }
}

