import Foundation

func emitToolLine(_ object: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(object),
          let data = try? JSONSerialization.data(withJSONObject: object) else {
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

/// One prompt per line on stdin, one JSON line on stdout. The model stays loaded.
func runToolRepl(gpu: GPU, args: Args) throws {
    let words = try WordList.load(args.str("vocab", "vocab/tools.txt"))
    let ckpt = args.str("ckpt", "build/tools.ckpt")
    let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: ckpt), batch: 1, seed: 1)
    let seed = args.flags["seed"].flatMap(UInt64.init) ?? 1
    var rng = RNG(seed)
    let tokens = args.int("tokens", 48)
    let temp = args.float("temp", 0)
    let topP = args.float("top-p", 1)
    while let line = readLine(strippingNewline: true) {
        let prompt = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if prompt.isEmpty { continue }
        let encoded = words.encode(prompt)
        if !encoded.unknown.isEmpty {
            emitToolLine(["error": "unknown", "unknown": encoded.unknown])
            continue
        }
        let text = generateMacro(model: model, tokens: tokens, temp: temp, topP: topP,
                                 rng: &rng, prompt: encoded.ids, words: words)
        emitToolLine(["text": text])
    }
}

func reportToolEval(gpu: GPU, args: Args, words: WordList) throws {
    let ckpt = args.str("ckpt", "build/tools.ckpt")
    let dataPath = args.str("data", "data/tools/calls.bin")
    let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: ckpt), batch: 1, seed: 1)
    let corpus = loadMacroCorpus(dataPath)
    let result = evalToolCalls(model: model, stories: corpus.val, words: words)
    print("held-out exact \(result.hit)/\(result.n)")
    for miss in result.misses {
        print("---\n\(miss)")
    }
}

func macroSampleSeed() -> UInt64 {
    let ticks = UInt64(Date().timeIntervalSince1970 * 1_000_000)
    return ticks == 0 ? 1 : ticks
}

let args = Args.parse()
switch args.command {
case "help", "-h", "--help":
    printHelp()
case "check":
    let gpu = GPU()
    exit(runChecks(gpu) ? 0 : 1)
case "overfit":
    let gpu = GPU()
    let bits = args.int("bits", 8)
    let steps = args.int("steps", 250)
    print("overfit one batch  bits \(bits)  steps \(steps)")
    let result = memorize(gpu: gpu, bits: bits, layers: 2, heads: 4, embd: 32,
                          batch: 4, block: 32, steps: steps, lr: 3e-3, seed: 1, verbose: true)
    print(String(format: "initial %.4f  final %.4f  acc %.1f%%", result.initial, result.final, result.acc * 100))
    if !(result.final.isFinite && result.final < result.initial * 0.5) {
        fputs("overfit did not bring the loss down. The backward pass is not learning.\n", stderr)
        exit(1)
    }
case "train":
    let gpu = GPU()
    do {
        try trainShakespeare(gpu: gpu, args: args)
    } catch {
        fputs("train failed: \(error)\n", stderr)
        exit(1)
    }
case "stories-check":
    let gpu = GPU()
    var report = Report()
    print("MacroStories gradient check")
    macroGradCheck(gpu: gpu, &report)
    print("MacroStories one-story memorization")
    let macro = macroMemorize(gpu: gpu, steps: 80, lr: 0.05)
    let drop = macro.initial - macro.final
    report.expect(macro.final.isFinite && drop > 2,
                  String(format: "loss %.3f -> %.3f", macro.initial, macro.final))
    let story = macroShortStory()
    print(MacroVocab.decode(story))
    if report.failed != 0 {
        fputs("stories-check failed\n", stderr)
        exit(1)
    }
    print("stories-check passed")
case "stories":
    let gpu = GPU()
    do {
        try trainMacro(gpu: gpu, args: args)
    } catch {
        fputs("stories failed: \(error)\n", stderr)
        exit(1)
    }
case "stories-hi":
    let gpu = GPU()
    do {
        let words = try WordList.load(args.str("vocab", "vocab/hindi.txt"))
        try trainMacro(gpu: gpu, args: args, words: words,
                       dataDefault: "data/hindi-stories.bin",
                       ckptDefault: "build/hindi.ckpt",
                       title: "Hindi stories",
                       blockDefault: 256)
    } catch {
        fputs("stories-hi failed: \(error)\n", stderr)
        exit(1)
    }
case "stories-hi-sample":
    let gpu = GPU()
    let ckpt = args.str("ckpt", "build/hindi.ckpt")
    do {
        let words = try WordList.load(args.str("vocab", "vocab/hindi.txt"))
        let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: ckpt), batch: 1, seed: 1)
        let seed = args.flags["seed"].flatMap(UInt64.init) ?? macroSampleSeed()
        var rng = RNG(seed)
        let encoded = words.encode(args.str("prompt", ""))
        if !encoded.unknown.isEmpty {
            fputs("prompt has words outside the Hindi list: \(encoded.unknown.joined(separator: ", "))\n", stderr)
            exit(1)
        }
        fputs("seed \(seed)\n", stderr)
        let text = generateMacro(model: model, tokens: args.int("tokens", 320),
                                 temp: args.float("temp", 0.7), topP: args.float("top-p", 0.9),
                                 rng: &rng, prompt: encoded.ids, words: words)
        print(text)
    } catch {
        fputs("stories-hi-sample failed: \(error)\n", stderr)
        exit(1)
    }
case "tools":
    let gpu = GPU()
    do {
        let words = try WordList.load(args.str("vocab", "vocab/tools.txt"))
        try trainMacro(gpu: gpu, args: args, words: words,
                       dataDefault: "data/tools/calls.bin",
                       ckptDefault: "build/tools.ckpt",
                       title: "Tool calls",
                       blockDefault: 64,
                       embdDefault: 256,
                       intermediateDefault: 758,
                       headsDefault: 4,
                       sampleTokens: 48,
                       tempDefault: 0)
        try reportToolEval(gpu: gpu, args: args, words: words)
    } catch {
        fputs("tools failed: \(error)\n", stderr)
        exit(1)
    }
case "tools-repl":
    let gpu = GPU()
    do {
        try runToolRepl(gpu: gpu, args: args)
    } catch {
        fputs("tools-repl failed: \(error)\n", stderr)
        exit(1)
    }
case "tools-sample":
    let gpu = GPU()
    let ckpt = args.str("ckpt", "build/tools.ckpt")
    do {
        let words = try WordList.load(args.str("vocab", "vocab/tools.txt"))
        let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: ckpt), batch: 1, seed: 1)
        let seed = args.flags["seed"].flatMap(UInt64.init) ?? macroSampleSeed()
        var rng = RNG(seed)
        let encoded = words.encode(args.str("prompt", ""))
        if !encoded.unknown.isEmpty {
            fputs("prompt has words outside the tool list: \(encoded.unknown.joined(separator: ", "))\n", stderr)
            exit(1)
        }
        fputs("seed \(seed)\n", stderr)
        let text = generateMacro(model: model, tokens: args.int("tokens", 48),
                                 temp: args.float("temp", 0), topP: args.float("top-p", 1),
                                 rng: &rng, prompt: encoded.ids, words: words)
        print(text)
    } catch {
        fputs("tools-sample failed: \(error)\n", stderr)
        exit(1)
    }
case "tools-eval":
    let gpu = GPU()
    do {
        let words = try WordList.load(args.str("vocab", "vocab/tools.txt"))
        try reportToolEval(gpu: gpu, args: args, words: words)
    } catch {
        fputs("tools-eval failed: \(error)\n", stderr)
        exit(1)
    }
case "stories-sample":
    let gpu = GPU()
    let ckpt = args.str("ckpt", "build/macro.ckpt")
    do {
        let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: ckpt), batch: 1, seed: 1)
        let seed = args.flags["seed"].flatMap(UInt64.init) ?? macroSampleSeed()
        var rng = RNG(seed)
        let encoded = MacroVocab.encodePrompt(args.str("prompt", ""))
        if !encoded.unknown.isEmpty {
            fputs("prompt has words outside the 378-word list: \(encoded.unknown.joined(separator: ", "))\n", stderr)
            exit(1)
        }
        fputs("seed \(seed)\n", stderr)
        let text = generateMacro(model: model, tokens: args.int("tokens", 640),
                                 temp: args.float("temp", 0.5), topP: args.float("top-p", 0.9),
                                 rng: &rng, prompt: encoded.ids)
        print(text)
    } catch {
        fputs("stories-sample failed: \(error)\n", stderr)
        exit(1)
    }
case "sample":
    let gpu = GPU()
    let ckpt = args.str("ckpt", "build/model.ckpt")
    do {
        let model = try GPT.load(gpu: gpu, url: URL(fileURLWithPath: ckpt), batch: 1, seed: 1)
        var rng = RNG(UInt64(args.int("seed", 1)))
        let text = generate(model: model, prompt: args.str("prompt", "ROMEO:\n"),
                            tokens: args.int("tokens", 200), temp: args.float("temp", 0.8), rng: &rng)
        print(text)
    } catch {
        fputs("sample failed: \(error)\n", stderr)
        exit(1)
    }
default:
    fputs("unknown command '\(args.command)'\n", stderr)
    printHelp()
    exit(1)
}
