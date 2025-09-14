// SpotSearchCore unit tests (RED first) // Author: Team DocuRay | Generated: TDD | Version: 0.1.0 | Modified: 2025-09-14

import XCTest
@testable import SpotSearchCore

final class SpotSearchCoreTests: XCTestCase {
    func testExactAndPrefixOrdering() {
        let idx = SpotSearchIndex()
        idx.addAll(paths: [
            "/proj/docs/Report_Q3_2024.pdf",
            "/proj/docs/Report_Q2_2024.pdf",
            "/proj/readme.md"
        ])

        let hits = idx.search(query: "Report_Q3_2024")
        XCTAssertGreaterThan(hits.count, 0)
        XCTAssertEqual((hits.first?.path as NSString?)?.lastPathComponent, "Report_Q3_2024.pdf")
    }

    func testSubstringMatchReturnsResult() {
        let idx = SpotSearchIndex()
        idx.addAll(paths: [
            "/a/b/AnnualFinancialStatement2024.xlsx",
            "/a/b/notes.txt"
        ])
        // interior substring, not a prefix of name
        let hits = idx.search(query: "FinancialStatement")
        XCTAssertTrue(hits.contains(where: { $0.path.hasSuffix("AnnualFinancialStatement2024.xlsx") }))
    }

    func testFuzzyMatchWithinTwoEdits() {
        let idx = SpotSearchIndex()
        idx.addAll(paths: [
            "/x/y/architecture.md",
            "/x/y/archive.zip"
        ])
        // two edits from "architecture" -> "architechtur"
        let hits = idx.search(query: "architechtur")
        XCTAssertTrue(hits.contains(where: { ($0.path as NSString).lastPathComponent == "architecture.md" }))
    }
}

