// Author: DocuRay Team | Version: v0.1.1 | Modified: 2025-09-14 | Purpose: SQLite persistence layer for FastName indexing
import Foundation
import SQLite3

public struct FileRecord {
    public let path: String
    public let size: Int64
    public let modTime: Int64
    public let isDir: Bool

    public init(path: String, size: Int64, modTime: Int64, isDir: Bool) {
        self.path = path; self.size = size; self.modTime = modTime; self.isDir = isDir
    }
}

public class SQLiteStore {
    private var db: OpaquePointer?
    private let dbPath: String

    public init(path: String) throws {
        self.dbPath = path
        try openDatabase()
        try createTables()
        try enableWAL()
    }

    deinit { if db != nil { sqlite3_close(db) } }

    private func openDatabase() throws {
        guard sqlite3_open(dbPath, &db) == SQLITE_OK else {
            throw SQLiteError.cannotOpenDatabase(sqlite3_errmsg(db).map(String.init(cString:)) ?? "Unknown error")
        }
    }

    private func createTables() throws {
        let metaSQL = """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        let filesSQL = """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mod_time INTEGER NOT NULL,
                is_dir INTEGER NOT NULL
            )
        """
        try execute(metaSQL)
        try execute(filesSQL)
    }

    private func enableWAL() throws {
        try execute("PRAGMA journal_mode=WAL")
    }

    private func execute(_ sql: String) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else {
            throw SQLiteError.executionFailed(sqlite3_errmsg(db).map(String.init(cString:)) ?? "Unknown error")
        }
    }

    public func upsertFile(_ record: FileRecord) throws {
        let sql = "INSERT OR REPLACE INTO files (path, size, mod_time, is_dir) VALUES (?, ?, ?, ?)"
        var stmt: OpaquePointer?
        defer { if stmt != nil { sqlite3_finalize(stmt) } }

        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw SQLiteError.prepareFailed(sqlite3_errmsg(db).map(String.init(cString:)) ?? "Unknown error")
        }

        sqlite3_bind_text(stmt, 1, record.path, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        sqlite3_bind_int64(stmt, 2, record.size)
        sqlite3_bind_int64(stmt, 3, record.modTime)
        sqlite3_bind_int(stmt, 4, record.isDir ? 1 : 0)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw SQLiteError.executionFailed(sqlite3_errmsg(db).map(String.init(cString:)) ?? "Unknown error")
        }
    }

    public func getFile(path: String) throws -> FileRecord? {
        let sql = "SELECT path, size, mod_time, is_dir FROM files WHERE path = ?"
        var stmt: OpaquePointer?
        defer { if stmt != nil { sqlite3_finalize(stmt) } }

        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw SQLiteError.prepareFailed(sqlite3_errmsg(db).map(String.init(cString:)) ?? "Unknown error")
        }

        sqlite3_bind_text(stmt, 1, path, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))

        if sqlite3_step(stmt) == SQLITE_ROW {
            return FileRecord(
                path: String(cString: sqlite3_column_text(stmt, 0)),
                size: sqlite3_column_int64(stmt, 1),
                modTime: sqlite3_column_int64(stmt, 2),
                isDir: sqlite3_column_int(stmt, 3) == 1
            )
        }
        return nil
    }

    public func batchInsert(_ records: [FileRecord]) throws {
        try execute("BEGIN TRANSACTION")
        defer { try? execute("COMMIT") }

        for record in records {
            try upsertFile(record)
        }
    }
}

public enum SQLiteError: Error {
    case cannotOpenDatabase(String)
    case executionFailed(String)
    case prepareFailed(String)
}