// Author: DocuRay Team | Version: v0.1.0 | Purpose: CLI arg parsing core (Phase-1, no MCP/LLM)

public enum Command {
    case help
    case index(roots: [String], excludes: [String], followSymlinks: Bool)
    case search(query: String, top: Int, exts: [String], roots: [String])
    case open(path: String, mode: OpenMode, editor: String?, line: Int?)
    case status
}

public enum OpenMode: String { case reveal, open, editor }

public struct ArgParser {
    public init() {}

    public func parse(_ args: [String]) throws -> Command {
        var it = args.makeIterator()
        guard let first = it.next() else { return .help }
        switch first {
        case "--help", "help":
            return .help
        case "index":
            var roots: [String] = []
            var excludes: [String] = []
            var follow = false
            while let a = it.next() {
                if a == "--roots", let v = it.next() { roots = v.split(separator: ",").map(String.init) }
                else if a == "--exclude", let v = it.next() { excludes = v.split(separator: ",").map(String.init) }
                else if a == "--follow-symlinks" { follow = true }
            }
            return .index(roots: roots, excludes: excludes, followSymlinks: follow)
        case "search":
            guard let q = it.next() else { return .help }
            var top = 50
            var exts: [String] = []
            var roots: [String] = []
            while let a = it.next() {
                if a == "--top", let v = it.next(), let n = Int(v) { top = n }
                else if a == "--ext", let v = it.next() { exts = v.split(separator: ",").map(String.init) }
                else if a == "--root", let v = it.next() { roots.append(v) }
            }
            return .search(query: q, top: top, exts: exts, roots: roots)
        case "open":
            var path: String? = nil
            var mode: OpenMode = .open
            var editor: String? = nil
            var line: Int? = nil
            while let a = it.next() {
                if a == "--path", let v = it.next() { path = v }
                else if a == "--mode", let v = it.next(), let m = OpenMode(rawValue: v) { mode = m }
                else if a == "--editor", let v = it.next() { editor = v }
                else if a == "--line", let v = it.next() { line = Int(v) }
            }
            guard let p = path else { return .help }
            return .open(path: p, mode: mode, editor: editor, line: line)
        case "status":
            return .status
        default:
            return .help
        }
    }

    public func helpText() -> String {
        return """
        fastname - offline ultra-fast filename search (Phase-1)
          Commands:
            help                      Show help
            index --roots <a,b>       Build index from roots
                  [--exclude <p,q>]  Exclude patterns
                  [--follow-symlinks]
            search <query>            Search by name
                  [--top N] [--ext a,b] [--root r]
            open --path <abs>         Open or reveal target
                  [--mode open|reveal|editor] [--editor code] [--line 123]
            status                    Show index status
        """
    }
}

