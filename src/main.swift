import Foundation

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
