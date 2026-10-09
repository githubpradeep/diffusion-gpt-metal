import Foundation

/// A word list stored as one token per line. Ids 0..3 are pad, start, end, and unknown.
struct WordList {
    let tokens: [String]
    let stoi: [String: UInt32]
    let pad: UInt32 = 0
    let bos: UInt32 = 1
    let eos: UInt32 = 2
    let unk: UInt32 = 3

    static func load(_ path: String) throws -> WordList {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        let tokens = text.split(separator: "\n", omittingEmptySubsequences: false).map { line in
            String(line).replacingOccurrences(of: "\r", with: "")
        }.filter { !$0.isEmpty }
        precondition(tokens.count > 4, "vocab \(path) is too small")
        var stoi: [String: UInt32] = [:]
        for (i, tok) in tokens.enumerated() {
            stoi[tok] = UInt32(i)
        }
        return WordList(tokens: tokens, stoi: stoi)
    }

    func decode(_ ids: [UInt32]) -> String {
        let skip: Set<UInt32> = [pad, bos, eos, unk]
        let punct: Set<String> = ["।", "?", "!", ",", ".", ":", ";"]
        var parts: [String] = []
        for id in ids {
            if id >= UInt32(tokens.count) || skip.contains(id) { continue }
            let tok = tokens[Int(id)]
            if punct.contains(tok), !parts.isEmpty {
                parts[parts.count - 1] += tok
            } else {
                parts.append(tok)
            }
        }
        return parts.joined(separator: " ")
    }

    func encode(_ text: String) -> (ids: [UInt32], unknown: [String]) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return ([], []) }
        var spaced = trimmed
        for mark in ["।", "?", "!", ",", ".", ":", ";"] {
            spaced = spaced.replacingOccurrences(of: mark, with: " \(mark) ")
        }
        let pieces = spaced.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        var ids: [UInt32] = [bos]
        var unknown: [String] = []
        for piece in pieces {
            if let id = stoi[piece] {
                ids.append(id)
            } else {
                unknown.append(piece)
            }
        }
        return (ids, unknown)
    }
}
