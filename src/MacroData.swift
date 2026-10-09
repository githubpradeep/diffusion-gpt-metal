import Foundation

// Closed-vocabulary children's stories in the MacroStories word list.
// The published training set was not released. This generator stays inside
// those 378 tokens, but crosses plots, actors, places, and actions so a
// sample is not a copy of one memorized script.

private let macroSpace = "\u{0120}"

private struct StoryWriter {
    var ids: [UInt32]
    let limit: Int

    init(limit: Int) {
        ids = [MacroVocab.bos]
        self.limit = limit
    }

    func fits(_ n: Int) -> Bool { ids.count + n + 1 <= limit }

    mutating func add(_ token: String) {
        ids.append(MacroVocab.id(token))
    }

    mutating func add(_ tokens: [String]) {
        for token in tokens { add(token) }
    }

    mutating func sentence(_ tokens: [String]) {
        var tokens = tokens
        if ids.count > 1, tokens.first == "\"" {
            tokens.insert(macroSpace, at: 0)
        }
        guard fits(tokens.count) else { return }
        add(tokens)
    }

    func finish() -> [UInt32] {
        var out = ids
        out.append(MacroVocab.eos)
        return out
    }
}

private func sp(_ word: String) -> String { macroSpace + word }

private func person(_ n: Int) -> String {
    switch n {
    case 2: return "[PERSON_NAME_2]"
    case 3: return "[PERSON_NAME_3]"
    default: return "[PERSON_NAME_1]"
    }
}

/// The tokenizer keeps the space before a person name as its own token.
private func name(_ n: Int) -> [String] { [macroSpace, person(n)] }

private func pick<T>(_ rng: inout RNG, _ items: [T]) -> T {
    items[rng.below(items.count)]
}

private func shuffled<T>(_ rng: inout RNG, _ items: [T]) -> [T] {
    var copy = items
    if copy.count < 2 { return copy }
    for i in stride(from: copy.count - 1, through: 1, by: -1) {
        copy.swapAt(i, rng.below(i + 1))
    }
    return copy
}

private func placePhrase(_ rng: inout RNG) -> [String] {
    let choices: [[String]] = [
        [sp("in"), sp("the"), sp("village"), sp("garden")],
        [sp("in"), sp("the"), sp("yard")],
        [sp("in"), sp("the"), sp("park")],
        [sp("near"), sp("the"), sp("path")],
        [sp("by"), sp("the"), sp("bench")],
        [sp("in"), sp("the"), sp("woods")],
        [sp("near"), sp("the"), sp("house")],
        [sp("by"), sp("the"), sp("tree")],
        [sp("in"), sp("the"), sp("school"), sp("yard")],
        [sp("on"), sp("the"), sp("grass")],
        [sp("near"), sp("the"), sp("door")],
        [sp("along"), sp("the"), sp("fence")],
        [sp("under"), sp("the"), sp("tree")],
        [sp("across"), sp("the"), sp("yard")],
    ]
    return choices[rng.below(choices.count)]
}

private func adjective(_ rng: inout RNG) -> String {
    let words = ["small", "large", "little", "red", "bright", "blue", "brown", "wooden", "new", "fresh"]
    return words[rng.below(words.count)]
}

private func manner(_ rng: inout RNG) -> String {
    let words = ["carefully", "slowly", "quickly", "firmly", "again"]
    return words[rng.below(words.count)]
}

private struct Scene {
    var hero: Int
    var friend: Int
    var adj: String
    var how: String
    var place: [String]
    var thing: String
    var hold: Bool
}

/// One opening, two actions, one ending. The same plot does not repeat one script.
private func assemble(_ rng: inout RNG, _ scene: Scene, setup: [[String]], acts: [[String]], end: [[String]]) -> [[String]] {
    let pool = shuffled(&rng, acts)
    let n = min(2, pool.count)
    var middle = Array(pool.prefix(n))
    if rng.below(2) == 0, !middle.isEmpty {
        middle.insert(helpLine(&rng, scene), at: rng.below(middle.count + 1))
    }
    return [pick(&rng, setup)] + middle + [pick(&rng, end)]
}

/// A short fixed story used to prove the backward pass can memorize one sequence.
func macroShortStory() -> [UInt32] {
    var w = StoryWriter(limit: 80)
    w.add(["\"", "I", sp("want"), sp("to"), sp("help"), sp("this"), sp("flower"), ",\""])
    w.add(name(1) + [sp("said"), "."])
    w.add([sp("The"), sp("flower"), sp("was"), sp("dry"), sp("in"), sp("the"), sp("yard"), "."])
    w.add(name(1) + [sp("wanted"), sp("to"), sp("water"), sp("the"), sp("flower"), "."])
    w.add(name(1) + [sp("dipped"), sp("the"), sp("cup"), sp("into"), sp("the"), sp("bucket"), "."])
    w.add(name(1) + [sp("poured"), sp("the"), sp("water"), sp("onto"), sp("the"), sp("soil"), "."])
    w.add([sp("Finally"), ",", sp("the"), sp("flower"), sp("got"), sp("enough"), sp("water"), "."])
    w.add(name(1) + [sp("smiled"), "."])
    return w.finish()
}

func macroStory(_ rng: inout RNG, maxTokens: Int) -> [UInt32] {
    let scene = Scene(
        hero: 1,
        friend: 2 + rng.below(2),
        adj: adjective(&rng),
        how: manner(&rng),
        place: placePhrase(&rng),
        thing: "",
        hold: true
    )
    let plot = rng.below(16)
    let lines = storyLines(&rng, scene, plot: plot)
    var writer = StoryWriter(limit: maxTokens)
    for line in lines {
        writer.sentence(line)
    }
    return writer.finish()
}

private func storyLines(_ rng: inout RNG, _ base: Scene, plot: Int) -> [[String]] {
    var scene = base
    while scene.friend == scene.hero { scene.friend = 1 + rng.below(3) }
    switch plot {
    case 0: return waterLines(&rng, &scene)
    case 1: return mendLines(&rng, &scene)
    case 2: return buildLines(&rng, &scene)
    case 3: return carryLines(&rng, &scene)
    case 4: return mudLines(&rng, &scene)
    case 5: return flyLines(&rng, &scene)
    case 6: return doorLines(&rng, &scene)
    case 7: return bucketLines(&rng, &scene)
    case 8: return birdLines(&rng, &scene)
    case 9: return playLines(&rng, &scene)
    case 10: return pathLines(&rng, &scene)
    case 11: return cleanLines(&rng, &scene)
    case 12: return findLines(&rng, &scene)
    case 13: return birdFellLines(&rng, &scene)
    case 14: return visitLines(&rng, &scene)
    default: return petalLines(&rng, &scene)
    }
}

private func quote(_ scene: Scene, _ verb: String, _ thing: String) -> [String] {
    ["\"", "I", sp("want"), sp("to"), sp(verb), sp("this"), sp(thing), ",\""] + name(scene.hero) + [sp("said"), "."]
}

private func sawState(_ scene: Scene, _ state: String) -> [String] {
    [sp("The"), sp(scene.adj), sp(scene.thing), sp("was"), sp(state)] + scene.place + ["."]
}

private func wanted(_ scene: Scene, _ verb: String) -> [String] {
    name(scene.hero) + [sp("wanted"), sp("to"), sp(verb), sp("the"), sp(scene.thing), "."]
}

private func helpLine(_ rng: inout RNG, _ scene: Scene) -> [String] {
    var choices: [[String]] = [
        name(scene.friend) + [sp("helped"), "."],
        name(scene.friend) + [sp("watched"), "."],
        ["\"", "Can", sp("you"), sp("help"), sp("me"), "?\"",] + name(scene.hero) + [sp("said"), "."],
    ]
    if scene.hold {
        choices.append(name(scene.friend) + [sp("held"), sp("the"), sp(scene.thing), sp("steady"), "."])
    }
    return choices[rng.below(choices.count)]
}

private func smile(_ scene: Scene, _ because: [String]) -> [String] {
    if because.isEmpty {
        if scene.hero % 2 == 0 {
            return name(scene.hero) + [sp("was"), sp("happy"), "."]
        }
        return name(scene.hero) + [sp(scene.hero == 1 ? "smiled" : "cheered"), "."]
    }
    return name(scene.hero) + [sp("smiled"), sp("because")] + because
}

private func waterLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["flower", "plant", "grass", "tree"])
    let setup = [
        quote(scene, rng.below(2) == 0 ? "help" : "water", scene.thing),
        sawState(scene, rng.below(2) == 0 ? "dry" : "thirsty"),
        wanted(scene, "water"),
    ]
    let acts = [
        name(scene.hero) + [sp("poured"), sp("the"), sp("water"), sp("onto"), sp("the"), sp("soil"), "."],
        name(scene.hero) + [sp("grabbed"), sp("a"), sp("large"), sp("bucket"), ",", sp("but"), sp("the"), sp("bucket"), sp("was"), sp("too"), sp("heavy"), "."],
        name(scene.hero) + [sp("looked"), sp("at"), sp("the"), sp("small"), sp("cup"), sp("on"), sp("the"), sp("bench"), "."],
        name(scene.hero) + [sp("dipped"), sp("the"), sp("cup"), sp("into"), sp("the"), sp("bucket"), "."],
        name(scene.hero) + [sp("walked"), sp("to"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("carried"), sp("the"), sp("cup"), sp(scene.how), "."],
        name(scene.hero) + [sp("knelt"), sp("by"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("went"), sp("back"), sp("for"), sp("more"), sp("water"), "."],
        name(scene.hero) + [sp("stepped"), sp("toward"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("noticed"), sp("the"), sp("dry"), sp("soil"), "."],
        name(scene.hero) + [sp("returned"), sp("with"), sp("more"), sp("water"), "."],
        name(scene.hero) + [sp("sat"), sp("by"), sp("the"), sp(scene.thing), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp(scene.thing), sp("got"), sp("enough"), sp("water"), "."],
        smile(scene, [sp("the"), sp(scene.thing), sp("was"), sp("not"), sp("dry"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func mendLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["kite", "book", "cloth", "bag", "paper"])
    let patch = pick(&rng, ["cloth", "paper"])
    let setup = [
        quote(scene, "fix", scene.thing),
        sawState(scene, "torn"),
        wanted(scene, "fix"),
    ]
    let acts = [
        name(scene.hero) + [sp("pressed"), sp("the"), sp(patch), sp("onto"), sp("the"), sp("tear"), "."],
        name(scene.hero) + [sp("saw"), sp("a"), sp("long"), sp("tear"), sp("in"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("found"), sp("the"), sp("glue"), sp("and"), sp("a"), sp("piece"), sp("of"), sp(patch), "."],
        name(scene.hero) + [sp("added"), sp("glue"), sp(scene.how), "."],
        name(scene.hero) + [sp("held"), sp("the"), sp(scene.thing), sp("until"), sp("the"), sp("glue"), sp("was"), sp("dry"), "."],
        name(scene.hero) + [sp("wrapped"), sp("string"), sp("around"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("worked"), sp(scene.how), "."],
        name(scene.hero) + [sp("spread"), sp("the"), sp("glue"), sp("on"), sp("the"), sp("tear"), "."],
        name(scene.hero) + [sp("took"), sp("a"), sp("piece"), sp("of"), sp("fabric"), "."],
        name(scene.hero) + [sp("put"), sp("the"), sp(patch), sp("over"), sp("the"), sp("tear"), "."],
    ]
    var end = [
        [sp("Finally"), ",", sp("the"), sp(scene.thing), sp("was"), sp("fixed"), "."],
        smile(scene, [sp("the"), sp(scene.thing), sp("was"), sp("not"), sp("torn"), sp("now"), "."]),
    ]
    if scene.thing == "kite" && rng.below(2) == 0 {
        end.insert([sp("The"), sp("kite"), sp("flew"), sp("in"), sp("the"), sp("sky"), "."], at: 1)
    }
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func buildLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["birdhouse", "fence", "board"])
    let setup = [
        quote(scene, "fix", scene.thing),
        sawState(scene, "falling"),
        name(scene.hero) + [sp("saw"), sp("a"), sp("gap"), sp("in"), sp("the"), sp("wood"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("pressed"), sp("the"), sp("nail"), sp("into"), sp("the"), sp("wood"), "."],
        name(scene.hero) + [sp("grabbed"), sp("a"), sp("large"), sp("board"), ",", sp("but"), sp("it"), sp("was"), sp("too"), sp("heavy"), "."],
        name(scene.hero) + [sp("picked"), sp("up"), sp("the"), sp("hammer"), sp("and"), sp("a"), sp("nail"), "."],
        name(scene.hero) + [sp("placed"), sp("the"), sp("board"), sp("against"), sp("the"), sp("gap"), "."],
        name(scene.hero) + [sp("tapped"), sp("the"), sp("hammer"), sp(scene.how), "."],
        name(scene.hero) + [sp("held"), sp("the"), sp(scene.thing), sp("steady"), "."],
        name(scene.hero) + [sp("noticed"), sp("a"), sp("hole"), sp("in"), sp("the"), sp("wood"), "."],
        name(scene.hero) + [sp("took"), sp("the"), sp("hammer"), sp("from"), sp("the"), sp("bench"), "."],
        name(scene.hero) + [sp("stood"), sp("by"), sp("the"), sp(scene.thing), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp(scene.thing), sp("was"), sp("sturdy"), "."],
        smile(scene, [sp("the"), sp(scene.thing), sp("did"), sp("not"), sp("fall"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func carryLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["bread", "loaf"])
    let setup = [
        ["\"", "I", sp("need"), sp("to"), sp("bring"), sp("this"), sp(scene.thing), sp("home"), ",\""] + name(scene.hero) + [sp("said"), "."],
        [sp("The"), sp(scene.adj), sp(scene.thing), sp("was"), sp("at"), sp("the"), sp("bakery"), "."],
        wanted(scene, "carry"),
    ]
    let acts = [
        name(scene.hero) + [sp("placed"), sp("the"), sp(scene.thing), sp("into"), sp("the"), sp("basket"), "."],
        name(scene.hero) + [sp("grabbed"), sp("a"), sp("large"), sp("bag"), ",", sp("but"), sp("the"), sp("bag"), sp("was"), sp("too"), sp("heavy"), "."],
        name(scene.hero) + [sp("looked"), sp("at"), sp("the"), sp("small"), sp("basket"), "."],
        name(scene.hero) + [sp("carried"), sp("the"), sp("basket"), sp(scene.how)] + scene.place + ["."],
        name(scene.hero) + [sp("walked"), sp("along"), sp("the"), sp("path"), "."],
        name(scene.friend) + [sp("carried"), sp("the"), sp("basket"), "."],
        name(scene.hero) + [sp("walked"), sp("back"), sp("to"), sp("the"), sp("house"), "."],
        name(scene.hero) + [sp("held"), sp("the"), sp("basket"), sp("with"), sp("both"), sp("hands"), "."],
        name(scene.friend) + [sp("waited"), sp("by"), sp("the"), sp("door"), "."],
    ]
    let end = [
        [sp("Finally"), ","] + name(scene.hero) + [sp("brought"), sp("the"), sp(scene.thing), sp("home"), "."],
        smile(scene, [sp("the"), sp(scene.thing), sp("was"), sp("home"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func mudLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["ball", "cup", "stick", "book"])
    let setup = [
        quote(scene, "get", scene.thing),
        [sp("The"), sp(scene.adj), sp(scene.thing), sp("was"), sp("stuck"), sp("in"), sp("the"), sp("mud"), "."],
        wanted(scene, "get"),
    ]
    let acts = [
        name(scene.hero) + [sp("pushed"), sp("the"), sp(scene.thing), sp("out"), sp("of"), sp("the"), sp("mud"), "."],
        name(scene.hero) + [sp("grabbed"), sp("the"), sp(scene.thing), ",", sp("but"), sp("it"), sp("was"), sp("too"), sp("muddy"), "."],
        name(scene.hero) + [sp("looked"), sp("at"), sp("a"), sp("stick"), sp("near"), sp("the"), sp("path"), "."],
        name(scene.hero) + [sp("pushed"), sp("the"), sp("stick"), sp("under"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("lifted"), sp("the"), sp(scene.thing), sp(scene.how), "."],
        name(scene.hero) + [sp("worked"), sp(scene.how), "."],
        name(scene.hero) + [sp("stepped"), sp("into"), sp("the"), sp("mud"), "."],
        name(scene.hero) + [sp("reached"), sp("into"), sp("the"), sp("mud"), "."],
        name(scene.friend) + [sp("found"), sp("a"), sp("stick"), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp(scene.thing), sp("was"), sp("out"), sp("of"), sp("the"), sp("mud"), "."],
        smile(scene, [sp("the"), sp(scene.thing), sp("was"), sp("not"), sp("stuck"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func flyLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "kite"
    let setup = [
        quote(scene, "fly", "kite"),
        [sp("The"), sp(scene.adj), sp("kite"), sp("was")] + scene.place + ["."],
        [sp("The"), sp("wind"), sp("was"), sp("strong"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("held"), sp("the"), sp("string"), sp(scene.how), "."],
        name(scene.hero) + [sp("ran"), sp("across"), sp("the"), sp("yard"), "."],
        name(scene.hero) + [sp("looked"), sp("up"), sp("at"), sp("the"), sp("sky"), "."],
        name(scene.friend) + [sp("held"), sp("the"), sp("kite"), sp("high"), "."],
        name(scene.hero) + [sp("walked"), sp("into"), sp("the"), sp("wind"), "."],
        [sp("The"), sp("kite"), sp("flew"), sp("over"), sp("the"), sp("tree"), "."],
        name(scene.hero) + [sp("stood"), sp("on"), sp("the"), sp("grass"), "."],
        name(scene.hero) + [sp("waited"), sp("for"), sp("the"), sp("wind"), "."],
    ]
    let end = [
        [sp("The"), sp("kite"), sp("flew"), sp("high"), sp("in"), sp("the"), sp("sky"), "."],
        smile(scene, [sp("the"), sp("kite"), sp("could"), sp("fly"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func doorLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "door"
    let setup = [
        quote(scene, "open", "door"),
        [sp("The"), sp(scene.adj), sp("door"), sp("was"), sp("stuck")] + scene.place + ["."],
        wanted(scene, "open"),
    ]
    let acts = [
        name(scene.hero) + [sp("pushed"), sp("the"), sp("door"), ",", sp("but"), sp("it"), sp("did"), sp("not"), sp("open"), "."],
        name(scene.friend) + [sp("pushed"), sp("the"), sp("door"), "."],
        name(scene.hero) + [sp("gripped"), sp("the"), sp("door"), sp(scene.how), "."],
        name(scene.hero) + [sp("tried"), sp("again"), "."],
        [sp("Suddenly")] + name(scene.hero) + [sp("moved"), sp("the"), sp("door"), "."],
        name(scene.hero) + [sp("took"), sp("a"), sp("step"), sp("back"), "."],
        name(scene.hero) + [sp("stood"), sp("by"), sp("the"), sp("door"), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp("door"), sp("opened"), "."],
        smile(scene, [sp("the"), sp("door"), sp("was"), sp("open"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func bucketLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["bucket", "cup"])
    let setup = [
        quote(scene, "help", scene.thing),
        [sp("The"), sp(scene.thing), sp("was"), sp("empty")] + scene.place + ["."],
        name(scene.hero) + [sp("needed"), sp("water"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("poured"), sp("the"), sp("water"), sp("into"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("walked"), sp("to"), sp("the"), sp("water"), "."],
        name(scene.hero) + [sp("dipped"), sp("the"), sp("cup"), sp("into"), sp("the"), sp("water"), "."],
        name(scene.hero) + [sp("carried"), sp("the"), sp(scene.thing), sp(scene.how), "."],
        name(scene.hero) + [sp("went"), sp("back"), sp("for"), sp("more"), "."],
        name(scene.hero) + [sp("stood"), sp("by"), sp("the"), sp("water"), "."],
        name(scene.hero) + [sp("put"), sp("the"), sp("cup"), sp("down"), "."],
        name(scene.hero) + [sp("returned"), sp("with"), sp("the"), sp(scene.thing), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp(scene.thing), sp("was"), sp("full"), "."],
        smile(scene, [sp("there"), sp("was"), sp("enough"), sp("water"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func birdLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "birdhouse"
    let setup = [
        [sp("The"), sp("birds"), sp("had"), sp("no"), sp("home")] + scene.place + ["."],
        quote(scene, "build", "birdhouse"),
        name(scene.hero) + [sp("wanted"), sp("a"), sp("home"), sp("for"), sp("the"), sp("birds"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("pressed"), sp("a"), sp("nail"), sp("into"), sp("the"), sp("wood"), "."],
        name(scene.hero) + [sp("found"), sp("wood"), sp("and"), sp("nails"), "."],
        name(scene.hero) + [sp("picked"), sp("up"), sp("the"), sp("hammer"), "."],
        name(scene.hero) + [sp("tapped"), sp("the"), sp("hammer"), sp(scene.how), "."],
        name(scene.friend) + [sp("held"), sp("the"), sp("wood"), sp("steady"), "."],
        name(scene.hero) + [sp("placed"), sp("the"), sp("birdhouse"), sp("by"), sp("the"), sp("tree"), "."],
        name(scene.hero) + [sp("put"), sp("the"), sp("birdhouse"), sp("up"), sp("in"), sp("the"), sp("tree"), "."],
        name(scene.hero) + [sp("made"), sp("a"), sp("small"), sp("door"), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp("birdhouse"), sp("was"), sp("ready"), "."],
        [sp("The"), sp("birds"), sp("stayed"), sp("in"), sp("their"), sp("new"), sp("home"), "."],
        smile(scene, []),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func playLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "ball"
    let setup = [
        quote(scene, "play", "ball"),
        [sp("The"), sp(scene.adj), sp("ball"), sp("was")] + scene.place + ["."],
        name(scene.hero) + [sp("wanted"), sp("to"), sp("play"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("rolled"), sp("the"), sp("ball"), "."],
        name(scene.friend) + [sp("caught"), sp("the"), sp("ball"), "."],
        name(scene.hero) + [sp("ran"), sp("after"), sp("the"), sp("ball"), "."],
        [sp("The"), sp("ball"), sp("rolled"), sp("across"), sp("the"), sp("grass"), "."],
        name(scene.hero) + [sp("kept"), sp("the"), sp("ball"), sp("up"), "."],
        [sp("They"), sp("ran"), sp("together"), "."],
        [sp("The"), sp("ball"), sp("fell"), sp("near"), sp("the"), sp("bench"), "."],
        name(scene.hero) + [sp("gave"), sp("the"), sp("ball"), sp("away"), "."],
        name(scene.hero) + [sp("put"), sp("the"), sp("ball"), sp("down"), "."],
        [sp("They"), sp("sat"), sp("on"), sp("the"), sp("grass"), "."],
    ]
    let end = [
        [sp("They"), sp("had"), sp("a"), sp("good"), sp("time"), "."],
        [sp("They"), sp("were"), sp("happy"), "."],
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func pathLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "path"
    scene.hold = false
    let setup = [
        [sp("There"), sp("was"), sp("a"), sp("deep"), sp("gap"), sp("in"), sp("the"), sp("path"), "."],
        quote(scene, "fix", "path"),
        name(scene.hero) + [sp("could"), sp("not"), sp("walk"), sp("across"), sp("the"), sp("gap"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("placed"), sp("the"), sp("stone"), sp("into"), sp("the"), sp("gap"), "."],
        name(scene.hero) + [sp("looked"), sp("at"), sp("the"), sp("stones"), sp("by"), sp("the"), sp("path"), "."],
        name(scene.hero) + [sp("picked"), sp("up"), sp("a"), sp("large"), sp("stone"), "."],
        name(scene.hero) + [sp("pushed"), sp("the"), sp("stone"), sp(scene.how), "."],
        name(scene.friend) + [sp("carried"), sp("another"), sp("stone"), "."],
        name(scene.hero) + [sp("filled"), sp("the"), sp("gap"), sp("with"), sp("stones"), "."],
        name(scene.hero) + [sp("stepped"), sp("over"), sp("the"), sp("gap"), "."],
        name(scene.hero) + [sp("noticed"), sp("the"), sp("deep"), sp("hole"), "."],
        [sp("The"), sp("stones"), sp("made"), sp("a"), sp("flat"), sp("path"), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp("gap"), sp("was"), sp("full"), "."],
        name(scene.hero) + [sp("walked"), sp("across"), sp("the"), sp("stones"), "."],
        smile(scene, [sp("the"), sp("path"), sp("was"), sp("good"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func cleanLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["cloth", "cup", "ball", "bench"])
    let setup = [
        quote(scene, "clean", scene.thing),
        [sp("The"), sp(scene.thing), sp("was"), sp("muddy")] + scene.place + ["."],
        wanted(scene, "clean"),
    ]
    let acts = [
        name(scene.hero) + [sp("poured"), sp("water"), sp("onto"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("got"), sp("a"), sp("cup"), sp("of"), sp("water"), "."],
        name(scene.hero) + [sp("worked"), sp(scene.how), "."],
        name(scene.friend) + [sp("brought"), sp("more"), sp("water"), "."],
        name(scene.hero) + [sp("put"), sp("the"), sp(scene.thing), sp("in"), sp("the"), sp("water"), "."],
        name(scene.hero) + [sp("took"), sp("the"), sp(scene.thing), sp("out"), "."],
        name(scene.hero) + [sp("placed"), sp("the"), sp(scene.thing), sp("on"), sp("the"), sp("bench"), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp(scene.thing), sp("was"), sp("clean"), "."],
        smile(scene, [sp("the"), sp(scene.thing), sp("was"), sp("not"), sp("muddy"), sp("now"), "."]),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func findLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = pick(&rng, ["book", "ball", "cup", "stick"])
    let setup = [
        name(scene.hero) + [sp("could"), sp("not"), sp("find"), sp("the"), sp(scene.thing), "."],
        [sp("The"), sp(scene.adj), sp(scene.thing), sp("was"), sp("lying")] + scene.place + ["."],
        quote(scene, "get", scene.thing),
    ]
    let acts = [
        name(scene.hero) + [sp("looked"), sp("under"), sp("the"), sp("bench"), "."],
        name(scene.hero) + [sp("looked"), sp("along"), sp("the"), sp("path"), "."],
        name(scene.friend) + [sp("shouted"), "."],
        name(scene.friend) + [sp("found"), sp("the"), sp(scene.thing), "."],
        name(scene.hero) + [sp("spotted"), sp("the"), sp(scene.thing), sp("by"), sp("the"), sp("tree"), "."],
        name(scene.hero) + [sp("reached"), sp("under"), sp("the"), sp("bench"), "."],
        name(scene.hero) + [sp("picked"), sp("the"), sp(scene.thing), sp("up"), "."],
    ]
    let end = [
        [sp("Finally"), ","] + name(scene.hero) + [sp("found"), sp("the"), sp(scene.thing), "."],
        smile(scene, []),
        [sp("They"), sp("were"), sp("happy"), "."],
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func birdFellLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "bird"
    let setup = [
        [sp("The"), sp("bird"), sp("fell"), sp("from"), sp("the"), sp("tree"), "."],
        [sp("The"), sp(scene.adj), sp("bird"), sp("was"), sp("on"), sp("the"), sp("ground"), "."],
        name(scene.hero) + [sp("wanted"), sp("to"), sp("help"), sp("the"), sp("bird"), "."],
    ]
    let acts = [
        name(scene.hero) + [sp("knelt"), sp("by"), sp("the"), sp("bird"), "."],
        name(scene.hero) + [sp("lifted"), sp("the"), sp("bird"), sp(scene.how), "."],
        name(scene.hero) + [sp("carried"), sp("the"), sp("bird"), sp("to"), sp("the"), sp("birdhouse"), "."],
        name(scene.hero) + [sp("placed"), sp("the"), sp("bird"), sp("inside"), sp("the"), sp("birdhouse"), "."],
        name(scene.hero) + [sp("waited"), sp("by"), sp("the"), sp("tree"), "."],
        name(scene.friend) + [sp("brought"), sp("a"), sp("small"), sp("cup"), "."],
    ]
    let end = [
        [sp("The"), sp("bird"), sp("stayed"), sp("in"), sp("the"), sp("birdhouse"), "."],
        [sp("Finally"), ",", sp("the"), sp("bird"), sp("was"), sp("well"), "."],
        smile(scene, []),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func visitLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "friend"
    scene.hold = false
    let setup = [
        name(scene.friend) + [sp("arrived"), sp("at"), sp("the"), sp("house"), "."],
        name(scene.hero) + [sp("waited"), sp("by"), sp("the"), sp("door"), "."],
        name(scene.hero) + [sp("wanted"), sp("to"), sp("walk"), "."],
    ]
    let acts = [
        [sp("They"), sp("walked"), sp("to"), sp("the"), sp("park"), "."],
        [sp("They"), sp("sat"), sp("on"), sp("the"), sp("bench"), "."],
        [sp("They"), sp("sat"), sp("under"), sp("the"), sp("tree"), "."],
        name(scene.friend) + [sp("noticed"), sp("the"), sp("flower"), "."],
        [sp("They"), sp("walked"), sp("along"), sp("the"), sp("path"), "."],
        [sp("The"), sp("sun"), sp("was"), sp("warm"), "."],
        [sp("They"), sp("looked"), sp("at"), sp("the"), sp("garden"), "."],
    ]
    let end = [
        [sp("They"), sp("had"), sp("a"), sp("good"), sp("time"), "."],
        [sp("They"), sp("were"), sp("happy"), "."],
        smile(scene, []),
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

private func petalLines(_ rng: inout RNG, _ scene: inout Scene) -> [[String]] {
    scene.thing = "petals"
    scene.hold = false
    let setup = [
        [sp("The"), sp("petals"), sp("fell"), sp("from"), sp("the"), sp("flower"), "."],
        [sp("The"), sp(scene.adj), sp("petals"), sp("were"), sp("on"), sp("the"), sp("grass"), "."],
        name(scene.hero) + [sp("noticed"), sp("the"), sp("petals")] + scene.place + ["."],
    ]
    let acts = [
        name(scene.hero) + [sp("picked"), sp("up"), sp("the"), sp("petals"), "."],
        name(scene.hero) + [sp("placed"), sp("the"), sp("petals"), sp("in"), sp("a"), sp("cup"), "."],
        name(scene.hero) + [sp("carried"), sp("the"), sp("petals"), sp("to"), sp("the"), sp("house"), "."],
        name(scene.friend) + [sp("picked"), sp("up"), sp("the"), sp("petals"), "."],
        name(scene.hero) + [sp("put"), sp("the"), sp("petals"), sp("on"), sp("the"), sp("bench"), "."],
        [sp("The"), sp("petals"), sp("were"), sp("bright"), "."],
    ]
    let end = [
        [sp("Finally"), ",", sp("the"), sp("petals"), sp("were"), sp("in"), sp("the"), sp("cup"), "."],
        smile(scene, [sp("the"), sp("flower"), sp("was"), sp("still"), sp("bright"), "."]),
        [sp("The"), sp("garden"), sp("looked"), sp("bright"), "."],
    ]
    return assemble(&rng, scene, setup: setup, acts: acts, end: end)
}

struct MacroSplit {
    var train: [[UInt32]]
    var val: [[UInt32]]
}

func loadMacroCorpus(_ path: String) -> MacroSplit {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), data.count >= 12 else {
        fputs("missing story corpus \(path)\n", stderr)
        fputs("sample the teacher with: .venv/bin/python scripts/distill_stories.py\n", stderr)
        exit(1)
    }
    let magic = data.prefix(4)
    guard magic.elementsEqual([0x4D, 0x44, 0x53, 0x54]) else {
        fputs("\(path) is not a story corpus\n", stderr)
        exit(1)
    }
    var offset = 4
    func u32() -> UInt32 {
        precondition(offset + 4 <= data.count, "truncated story corpus")
        let value = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
        }
        offset += 4
        return value
    }
    let version = u32()
    precondition(version == 1, "unsupported story corpus version \(version)")
    let count = Int(u32())
    var train: [[UInt32]] = []
    var val: [[UInt32]] = []
    train.reserveCapacity(count)
    for index in 0..<count {
        let length = Int(u32())
        precondition(offset + length * 4 <= data.count, "truncated story")
        var story = [UInt32]()
        story.reserveCapacity(length)
        for _ in 0..<length { story.append(u32()) }
        if index % 20 == 0 { val.append(story) } else { train.append(story) }
    }
    if train.isEmpty { train = val }
    if val.isEmpty { val = train }
    precondition(!train.isEmpty, "story corpus was empty")
    return MacroSplit(train: train, val: val)
}

func macroCorpusBatch(_ stories: [[UInt32]], batch: Int, block: Int, rng: inout RNG) -> (x: [UInt32], y: [UInt32], valid: Int) {
    var x = [UInt32](repeating: MacroVocab.pad, count: batch * block)
    var y = [UInt32](repeating: MacroVocab.pad, count: batch * block)
    var valid = 0
    for b in 0..<batch {
        let story = stories[rng.below(stories.count)]
        let usable = story.count - 1
        if usable <= 0 { continue }
        if usable <= block {
            for t in 0..<usable {
                x[b * block + t] = story[t]
                y[b * block + t] = story[t + 1]
                valid += 1
            }
        } else {
            let start = rng.below(usable - block + 1)
            for t in 0..<block {
                x[b * block + t] = story[start + t]
                y[b * block + t] = story[start + t + 1]
                valid += 1
            }
        }
    }
    precondition(valid > 0, "story batch was empty")
    return (x, y, valid)
}

func macroBatch(batch: Int, block: Int, rng: inout RNG) -> (x: [UInt32], y: [UInt32], valid: Int) {
    var x = [UInt32](repeating: MacroVocab.pad, count: batch * block)
    var y = [UInt32](repeating: MacroVocab.pad, count: batch * block)
    var valid = 0
    for b in 0..<batch {
        let story = macroStory(&rng, maxTokens: block + 1)
        let last = story.count - 1
        for t in 0..<block where t < last {
            x[b * block + t] = story[t]
            y[b * block + t] = story[t + 1]
            valid += 1
        }
    }
    precondition(valid > 0, "story batch was empty")
    return (x, y, valid)
}
