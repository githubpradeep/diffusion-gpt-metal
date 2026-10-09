import Foundation

// WordLevel vocabulary from raincandy-u/MacroStories (378 tokens).
// The G-dot prefix (U+0120) is the space marker used by the tokenizer.

enum MacroVocab {
    static let pad: UInt32 = 0
    static let bos: UInt32 = 1
    static let eos: UInt32 = 2
    static let unk: UInt32 = 3
    static let person1: UInt32 = 4
    static let person2: UInt32 = 5
    static let person3: UInt32 = 6

    static let tokens: [String] = [
        "<|pad|>",
        "<|start_story|>",
        "<|end_story|>",
        "<|unk|>",
        "[PERSON_NAME_1]",
        "[PERSON_NAME_2]",
        "[PERSON_NAME_3]",
        "_",
        ".",
        "\u{0120}the",
        "NAME",
        "PERSON",
        "[",
        "\u{0120}",
        "]",
        "1",
        "\u{0120}and",
        "\u{0120}a",
        "\u{0120}They",
        "\u{0120}to",
        "\u{0120}The",
        "\u{0120}was",
        "2",
        "\u{0120}in",
        "\u{0120}ball",
        "\u{0120}red",
        "\u{0120}of",
        "\u{0120}flower",
        "\u{0120}looked",
        "\u{0120}at",
        "\u{0120}water",
        "\u{0120}kite",
        "\u{0120}walked",
        ",",
        "\u{0120}bucket",
        "\u{0120}small",
        "\u{0120}stick",
        "\u{0120}bench",
        "\u{0120}into",
        "\u{0120}it",
        "\u{0120}bread",
        "\u{0120}wanted",
        "\u{0120}smiled",
        "\u{0120}dry",
        "\u{0120}grabbed",
        "\u{0120}reached",
        "\u{0120}cup",
        "\u{0120}book",
        "\u{0120}toward",
        "\u{0120}on",
        "\u{0120}birdhouse",
        "\u{0120}again",
        "\u{0120}across",
        "\u{0120}board",
        "].",
        "\u{0120}for",
        "\u{0120}paper",
        "\u{0120}path",
        "\u{0120}carefully",
        "\u{0120}basket",
        "\u{0120}held",
        "\u{0120}said",
        "\u{0120}moved",
        "\u{0120}village",
        "\u{0120}under",
        "\u{0120}over",
        "\u{0120}onto",
        "\u{0120}had",
        "\u{0120}placed",
        "\u{0120}saw",
        "\u{0120}yard",
        "\u{0120}from",
        "\u{0120}stones",
        "\u{0120}grass",
        "\u{0120}cloth",
        "\u{0120}happy",
        "\u{0120}large",
        "\u{0120}garden",
        "\u{0120}patch",
        "I",
        "\u{0120}mud",
        "\u{0120}gap",
        "\u{0120}wooden",
        "\u{0120}several",
        "\u{0120}rolled",
        "\u{0120}stone",
        "\u{0120}tear",
        "\u{0120}bright",
        "\u{0120}very",
        "\u{0120}flew",
        "\u{0120}poured",
        "\u{0120}glue",
        "\u{0120}heavy",
        "\u{0120}It",
        "\u{0120}nails",
        "\u{0120}soil",
        "\"",
        "\u{0120}firmly",
        "\u{0120}fresh",
        "\u{0120}home",
        "\u{0120}stepped",
        "\u{0120}out",
        "\u{0120}dirt",
        "\u{0120}near",
        "\u{0120}steady",
        "\u{0120}tightly",
        "!\"",
        "\u{0120}knelt",
        "\u{0120}with",
        "\u{0120}house",
        "\u{0120}were",
        "\u{0120}ground",
        "\u{0120}Finally",
        "\u{0120}felt",
        "\u{0120}went",
        "\u{0120}back",
        "\u{0120}pushed",
        "\u{0120}trips",
        "\u{0120}because",
        "\u{0120}pressed",
        "\u{0120}too",
        "\u{0120}around",
        "\u{0120}got",
        "\u{0120}gripped",
        "\u{0120}ran",
        "\u{0120}up",
        ",\"",
        "The",
        "\u{0120}one",
        "\u{0120}sun",
        "\u{0120}arrived",
        "\u{0120}but",
        "\u{0120}long",
        "\u{0120}noticed",
        "\u{0120}finished",
        "\u{0120}through",
        "\u{0120}carried",
        "\u{0120}tree",
        "\u{0120}park",
        "\u{0120}enough",
        "\u{0120}dipped",
        "\u{0120}safely",
        "\u{0120}wood",
        "\u{0120}\"",
        "\u{0120}bakery",
        "\u{0120}found",
        "\u{0120}sky",
        "\u{0120}high",
        "\u{0120}made",
        "\u{0120}together",
        "\u{0120}sat",
        "\u{0120}hole",
        "\u{0120}muddy",
        "\u{0120}dark",
        "\u{0120}door",
        "\u{0120}kept",
        "\u{0120}bag",
        "\u{0120}edge",
        "\u{0120}school",
        "\u{0120}their",
        "\u{0120}flat",
        "\u{0120}piece",
        "\u{0120}tapped",
        "\u{0120}want",
        "\u{0120}watched",
        "\u{0120}not",
        "\u{0120}some",
        "\u{0120}wind",
        "\u{0120}full",
        "\u{0120}petals",
        "\u{0120}balanced",
        "\u{0120}build",
        "\u{0120}received",
        "\u{0120}thirsty",
        "\u{0120}worked",
        "\u{0120}fix",
        "\u{0120}down",
        "\u{0120}fly",
        "\u{0120}sad",
        "\u{0120}city",
        "\u{0120}help",
        "\u{0120}sturdy",
        "\u{0120}took",
        "\u{0120}hammer",
        "\u{0120}reach",
        "\u{0120}thick",
        "\u{0120}this",
        "\u{0120}bottom",
        "\u{0120}first",
        "\u{0120}space",
        "\u{0120}until",
        "\u{0120}waited",
        "\u{0120}wrapped",
        "\u{0120}along",
        "\u{0120}many",
        "\u{0120}side",
        "\u{0120}nearby",
        "\u{0120}return",
        "\u{0120}far",
        "\u{0120}plant",
        "\u{0120}sighed",
        "\u{0120}warm",
        "\u{0120}clean",
        "\u{0120}handed",
        "\u{0120}middle",
        "\u{0120}away",
        "\u{0120}falling",
        "\u{0120}inside",
        "\u{0120}spotted",
        "\u{0120}tried",
        "\u{0120}stopped",
        "\u{0120}slowly",
        "\u{0120}string",
        "\u{0120}woods",
        "\u{0120}against",
        "\u{0120}colorful",
        "\u{0120}other",
        "\u{0120}blue",
        "\u{0120}by",
        "\u{0120}A",
        "\u{0120}lift",
        "\u{0120}pieces",
        "\u{0120}work",
        "\u{0120}fall",
        "\u{0120}picked",
        "\u{0120}However",
        "\u{0120}forward",
        "\u{0120}hands",
        "\u{0120}brought",
        "\u{0120}loaves",
        "\u{0120}them",
        "\u{0120}they",
        "\u{0120}another",
        "\u{0120}can",
        "\u{0120}carry",
        "\u{0120}stayed",
        "\u{0120}returned",
        "\u{0120}shouted",
        "\u{0120}strong",
        "\u{0120}ready",
        "\u{0120}time",
        "\u{0120}birds",
        "\u{0120}bring",
        "\u{0120}narrow",
        "\u{0120}good",
        "\u{0120}keep",
        "\u{0120}needed",
        "\u{0120}next",
        "\u{0120}opened",
        "\u{0120}did",
        "\u{0120}put",
        "You",
        "\u{0120}forth",
        "\u{0120}more",
        "\u{0120}stood",
        "\u{0120}both",
        "\u{0120}fence",
        "\u{0120}would",
        "\u{0120}could",
        "\u{0120}perfectly",
        "\u{0120}secure",
        "\u{0120}Then",
        "\u{0120}added",
        "\u{0120}deep",
        "\u{0120}loaf",
        "\u{0120}my",
        "\u{0120}friend",
        "\u{0120}Suddenly",
        "\u{0120}grip",
        "\u{0120}step",
        "\u{0120}play",
        "\u{0120}repeated",
        "\u{0120}you",
        "\u{0120}caught",
        "\u{0120}fixed",
        "\u{0120}times",
        "\u{0120}way",
        "\u{0120}bird",
        "\u{0120}then",
        "\u{0120}tight",
        "\u{0120}torn",
        "'s",
        "\u{0120}get",
        "\u{0120}nail",
        "\u{0120}open",
        "\u{0120}After",
        "\u{0120}cups",
        "\u{0120}fell",
        "\u{0120}have",
        "\u{0120}hold",
        "\u{0120}lifted",
        "\u{0120}make",
        "A",
        "We",
        "\u{0120}\"[",
        "\u{0120}empty",
        "\u{0120}stuck",
        "\u{0120}well",
        "\u{0120}filled",
        "\u{0120}helped",
        "\u{0120}stay",
        "\u{0120}done",
        "\u{0120}fabric",
        "\u{0120}little",
        "\u{0120}new",
        "\u{0120}place",
        "\u{0120}proud",
        "\u{0120}quickly",
        "\u{0120}balance",
        "\u{0120}cheered",
        "\u{0120}spread",
        "\u{0120}steps",
        "\u{0120}gave",
        "\u{0120}that",
        "\u{0120}there",
        "],",
        "\u{0120}after",
        "\u{0120}between",
        "\u{0120}brown",
        "\u{0120}edges",
        "\u{0120}now",
        "\u{0120}see",
        "\u{0120}while",
        "\u{0120}as",
        "\u{0120}do",
        "\u{0120}drop",
        "\u{0120}is",
        "\u{0120}longer",
        "\u{0120}move",
        "\u{0120}no",
        "\u{0120}still",
        "It",
        "\u{0120}I",
        "\u{0120}find",
        "\u{0120}its",
        "\u{0120}me",
        "\u{0120}need",
        "\u{0120}roll",
        "\u{0120}when",
        "!",
        ".\"",
        "?\"",
        "Yes",
        "\u{0120}finally",
        "\u{0120}give",
        "\u{0120}your",
        "Look",
        "\u{0120}dinner",
        "\u{0120}every",
        "\u{0120}go",
        "\u{0120}lying",
        "\u{0120}stand",
        "\u{0120}walk",
        "Can",
        "Is",
        "Where",
        "\u{0120}But",
        "\u{0120}Its",
        "\u{0120}Many",
        "\u{0120}Now",
        "\u{0120}One",
        "\u{0120}Slowly",
        "\u{0120}There",
        "\u{0120}This",
        "\u{0120}am",
        "\u{0120}are",
        "\u{0120}be",
        "\u{0120}before",
        "\u{0120}began",
        "\u{0120}look",
        "\u{0120}much",
        "\u{0120}push",
        "\u{0120}smile",
        "\u{0120}so",
        "\u{0120}tap",
        "\u{0120}themselves",
        "\u{0120}watered",
        "\u{0120}works",
    ]

    static let stoi: [String: UInt32] = {
        var map: [String: UInt32] = [:]
        map.reserveCapacity(tokens.count)
        for (i, tok) in tokens.enumerated() { map[tok] = UInt32(i) }
        return map
    }()

    static func id(_ token: String) -> UInt32 {
        guard let v = stoi[token] else { fatalError("missing MacroStories token \(token)") }
        return v
    }

    static func piece(_ token: String) -> String {
        switch token {
        case "<|pad|>", "<|start_story|>", "<|end_story|>", "<|unk|>":
            return ""
        case "[PERSON_NAME_1]": return "Alex"
        case "[PERSON_NAME_2]": return "Robin"
        case "[PERSON_NAME_3]": return "Casey"
        default:
            if token.hasPrefix("\u{0120}") {
                return " " + String(token.dropFirst())
            }
            return token
        }
    }

    static func decode(_ ids: [UInt32]) -> String {
        var text = ""
        for id in ids {
            let i = Int(id)
            if i >= 0 && i < tokens.count { text += piece(tokens[i]) }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Encode a story opening the way the training stories are tokenized.
    /// `unknown` lists words that are not in the 378-word list.
    static func encodePrompt(_ raw: String) -> (ids: [UInt32], unknown: [String]) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return ([bos], []) }
        let text = prefixSpace(trimmed) ? " " + trimmed : trimmed
        var ids: [UInt32] = [bos]
        var unknown: [String] = []
        var seen = Set<String>()
        for piece in gpt2Pieces(text) {
            if let name = personID(piece) {
                if piece.hasPrefix(" ") { ids.append(id("\u{0120}")) }
                ids.append(name)
                continue
            }
            let key = piece.replacingOccurrences(of: " ", with: "\u{0120}")
            if let token = stoi[key] {
                ids.append(token)
            } else if seen.insert(piece).inserted {
                unknown.append(piece.trimmingCharacters(in: .whitespaces))
            }
        }
        return (ids, unknown)
    }

    private static func prefixSpace(_ text: String) -> Bool {
        if text.hasPrefix("\"") { return false }
        guard text.first == "I" else { return true }
        if text.count == 1 { return false }
        let second = text[text.index(after: text.startIndex)]
        return second.isLetter
    }

    private static func personID(_ piece: String) -> UInt32? {
        let word = piece.hasPrefix(" ") ? String(piece.dropFirst()) : piece
        switch word {
        case "Alex": return person1
        case "Robin": return person2
        case "Casey": return person3
        default: return nil
        }
    }

    private static func gpt2Pieces(_ text: String) -> [String] {
        let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let re = try! NSRegularExpression(pattern: pattern)
        let ns = text as NSString
        let full = NSRange(location: 0, length: ns.length)
        var pieces: [String] = []
        var cursor = 0
        for match in re.matches(in: text, range: full) {
            if match.range.location > cursor {
                pieces.append(ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)))
            }
            pieces.append(ns.substring(with: match.range))
            cursor = match.range.location + match.range.length
        }
        if cursor < ns.length { pieces.append(ns.substring(from: cursor)) }
        return pieces
    }
}
