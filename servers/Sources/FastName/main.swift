import Foundation
import FastNameCore

// Author: DocuRay Team | Version: v0.1.0 | Purpose: CLI entry for FastName (Phase-1)

let parser = ArgParser()

func run() -> Int32 {
    var argv = CommandLine.arguments
    _ = argv.removeFirst() // drop executable
    do {
        let cmd = try parser.parse(argv)
        switch cmd {
        case .help:
            print(parser.helpText())
            return 0
        case .status:
            print("status: TODO (files=0, snapshot=0)")
            return 0
        case let .search(query, top, exts, roots):
            print("search q=\(query) top=\(top) exts=\(exts) roots=\(roots)")
            return 0
        case let .index(roots, excludes, follow):
            print("index roots=\(roots) exclude=\(excludes) follow=\(follow)")
            return 0
        case let .open(path, mode, editor, line):
            print("open path=\(path) mode=\(mode.rawValue) editor=\(editor ?? "-") line=\(line?.description ?? "-")")
            return 0
        }
    } catch {
        fputs("error: \(error)\n", stderr)
        return 2
    }
}

exit(run())
