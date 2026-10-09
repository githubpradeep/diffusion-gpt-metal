import Darwin
import Foundation

struct Dataset {
    let ids: [UInt32]
    let itos: [UInt8]
    let stoi: [UInt8: UInt32]
    let trainEnd: Int

    func batch(batch: Int, block: Int, train: Bool, rng: inout RNG) -> (x: [UInt32], y: [UInt32]) {
        let lo = train ? 0 : trainEnd
        let hi = train ? trainEnd : ids.count
        let span = hi - lo - block
        precondition(span > 0, "not enough tokens for a batch")
        var x = [UInt32](repeating: 0, count: batch * block)
        var y = [UInt32](repeating: 0, count: batch * block)
        for b in 0..<batch {
            let start = lo + rng.below(span)
            for t in 0..<block {
                x[b * block + t] = ids[start + t]
                y[b * block + t] = ids[start + t + 1]
            }
        }
        return (x, y)
    }
}

func loadShakespeare(_ path: String) throws -> Dataset {
    if !FileManager.default.fileExists(atPath: path) {
        let dir = (path as NSString).deletingLastPathComponent
        if !dir.isEmpty && dir != "." {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
        let url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        fputs("downloading Tiny Shakespeare\n", stderr)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
        p.arguments = ["-fL", "--retry", "3", "-o", path, url]
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 || !FileManager.default.fileExists(atPath: path) {
            throw NSError(domain: "gpt", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "failed to download Tiny Shakespeare from \(url)",
            ])
        }
    }
    let raw = try Data(contentsOf: URL(fileURLWithPath: path))
    precondition(!raw.isEmpty, "dataset is empty")
    var seen = Set<UInt8>()
    for b in raw { seen.insert(b) }
    let itos = seen.sorted()
    var stoi: [UInt8: UInt32] = [:]
    for (i, b) in itos.enumerated() { stoi[b] = UInt32(i) }
    let ids = raw.map { stoi[$0]! }
    let trainEnd = max(ids.count * 9 / 10, 1)
    return Dataset(ids: ids, itos: itos, stoi: stoi, trainEnd: trainEnd)
}

struct MemResult {
    var initial: Float
    var final: Float
    var acc: Float
}

/// Train on a single repeated batch until the backward pass can memorize it.
func memorize(gpu: GPU, bits: Int, layers: Int, heads: Int, embd: Int, batch: Int, block: Int,
              steps: Int, lr: Float, seed: UInt64, verbose: Bool) -> MemResult {
    let phrase = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n"
    var text = phrase
    while text.utf8.count < batch * block + 8 { text += phrase }
    let bytes = Array(text.utf8)
    var seen = Set<UInt8>()
    for b in bytes { seen.insert(b) }
    let itos = seen.sorted()
    var stoi: [UInt8: UInt32] = [:]
    for (i, b) in itos.enumerated() { stoi[b] = UInt32(i) }
    let ids = bytes.map { stoi[$0]! }
    var x = [UInt32](repeating: 0, count: batch * block)
    var y = [UInt32](repeating: 0, count: batch * block)
    for b in 0..<batch {
        for t in 0..<block {
            let i = b * block + t
            x[b * block + t] = ids[i]
            y[b * block + t] = ids[i + 1]
        }
    }
    let cfg = GPTConfig(vocab: itos.count, block: block, batch: batch, layers: layers,
                        heads: heads, embd: embd, bits: bits)
    let model = GPT(gpu: gpu, cfg: cfg, itos: itos, seed: seed)
    var initial: Float = 0
    var final: Float = 0
    for step in 1...steps {
        let loss = model.step(tokenIds: x, targetIds: y, lr: lr, wd: 0, clip: 0, stepIndex: step)
        if step == 1 { initial = loss }
        final = loss
        if verbose && (step == 1 || step % 25 == 0 || step == steps) {
            print(String(format: "  step %4d  loss %.4f", step, loss))
            fflush(stdout)
        }
    }
    let held = model.lossOnly(tokenIds: x, targetIds: y, batch: batch, block: block)
    let acc = model.tokenAccuracy(targets: y)
    if verbose {
        print(String(format: "  held-out forward loss %.4f  token acc %.1f%%", held, acc * 100))
    }
    return MemResult(initial: initial, final: final, acc: acc)
}

func trainShakespeare(gpu: GPU, args: Args) throws {
    let dataPath = args.str("data", "data/input.txt")
    let data = try loadShakespeare(dataPath)
    let bits = args.int("bits", 8)
    let batch = args.int("batch", 16)
    let block = args.int("block", 64)
    let layers = args.int("layers", 4)
    let heads = args.int("heads", 4)
    let embd = args.int("embd", 128)
    let steps = args.int("steps", 800)
    let lr = args.float("lr", 1e-3)
    let wd = args.float("wd", 0.1)
    let clip = args.float("clip", 1)
    let seed = UInt64(args.int("seed", 1337))
    let ckptPath = args.str("ckpt", "build/model.ckpt")
    let cfg = GPTConfig(vocab: data.itos.count, block: block, batch: batch, layers: layers,
                        heads: heads, embd: embd, bits: bits)
    cfg.validate()
    precondition(data.trainEnd > block + 1, "training split is shorter than the context")
    let model = GPT(gpu: gpu, cfg: cfg, itos: data.itos, seed: seed)
    let qCount = model.parameterCount
    print("device \(gpu.device.name)")
    print("vocab \(cfg.vocab)  params \(qCount)  bits \(bits)  block \(block)  batch \(batch)")
    print(String(format: "layers %d  heads %d  embd %d  steps %d  lr %.2e", layers, heads, embd, steps, lr))
    print(String(format: "uniform-guess loss %.3f", log(Float(cfg.vocab))))
    var rng = RNG(seed &+ 17)
    let t0 = CFAbsoluteTimeGetCurrent()
    for step in 1...steps {
        let sched = learningRate(step: step, total: steps, base: lr, warmup: min(100, steps / 10))
        let (x, y) = data.batch(batch: batch, block: block, train: true, rng: &rng)
        let loss = model.step(tokenIds: x, targetIds: y, lr: sched, wd: wd, clip: clip, stepIndex: step)
        if !loss.isFinite {
            fputs("loss became non-finite. Try --bits 8 or a smaller --lr.\n", stderr)
            exit(1)
        }
        if step == 1 || step % 20 == 0 || step == steps {
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let toks = Double(step * batch * block) / max(dt, 1e-6)
            var line = String(format: "step %4d/%d  loss %.4f  lr %.2e  %.0f tok/s", step, steps, loss, sched, toks)
            if step % 100 == 0 && data.ids.count - data.trainEnd > block + 1 {
                let (vx, vy) = data.batch(batch: batch, block: block, train: false, rng: &rng)
                let vloss = model.lossOnly(tokenIds: vx, targetIds: vy, batch: batch, block: block)
                line += String(format: "  val %.4f", vloss)
            }
            print(line)
            fflush(stdout)
        }
        if step % 200 == 0 || step == steps {
            let sample = generate(model: model, prompt: "ROMEO:", tokens: 80, temp: 0.8, rng: &rng)
            print("---\n\(sample)\n---")
            fflush(stdout)
        }
    }
    let url = URL(fileURLWithPath: ckptPath)
    let dir = url.deletingLastPathComponent().path
    if dir != "" && dir != "." {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }
    try model.save(url)
    print("wrote \(ckptPath)")
}

func generate(model: GPT, prompt: String, tokens: Int, temp: Float, rng: inout RNG) -> String {
    var ids: [UInt32] = []
    for b in prompt.utf8 {
        if let id = model.stoi[b] { ids.append(id) }
    }
    if ids.isEmpty { ids = [0] }
    for _ in 0..<tokens {
        let window = Array(ids.suffix(model.cfg.block))
        let logits = model.nextLogits(window)
        ids.append(sampleToken(logits, temp: temp, rng: &rng))
    }
    return decode(ids, itos: model.itos)
}

func sampleToken(_ logits: [Float], temp: Float, rng: inout RNG) -> UInt32 {
    if temp <= 0 {
        var best = 0
        var bestV = -Float.infinity
        for i in 0..<logits.count where logits[i] > bestV {
            bestV = logits[i]
            best = i
        }
        return UInt32(best)
    }
    var m = -Float.infinity
    for z in logits { m = max(m, z) }
    let weights = logits.map { exp(($0 - m) / temp) }
    let sum = weights.reduce(0, +)
    var u = rng.uniform() * sum
    for i in 0..<weights.count {
        u -= weights[i]
        if u <= 0 { return UInt32(i) }
    }
    return UInt32(weights.count - 1)
}

func decode(_ ids: [UInt32], itos: [UInt8]) -> String {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(ids.count)
    for id in ids where Int(id) < itos.count { bytes.append(itos[Int(id)]) }
    return String(decoding: bytes, as: UTF8.self)
}

func learningRate(step: Int, total: Int, base: Float, warmup: Int) -> Float {
    if warmup > 0 && step < warmup {
        return base * Float(step) / Float(warmup)
    }
    let span = max(total - warmup, 1)
    let t = min(Float(step - warmup) / Float(span), 1)
    let coeff = 0.5 * (1 + cos(Float.pi * t))
    return max(base * 0.1, base * coeff)
}

struct Args {
    var command: String
    var flags: [String: String]

    func str(_ k: String, _ d: String) -> String { flags[k] ?? d }
    func int(_ k: String, _ d: Int) -> Int { flags[k].flatMap(Int.init) ?? d }
    func float(_ k: String, _ d: Float) -> Float { flags[k].flatMap(Float.init) ?? d }

    static func parse() -> Args {
        var command = "help"
        var flags: [String: String] = [:]
        var i = 1
        let argv = CommandLine.arguments
        if argv.count > 1 && !argv[1].hasPrefix("--") {
            command = argv[1]
            i = 2
        }
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if i + 1 < argv.count && !argv[i + 1].hasPrefix("--") {
                    flags[key] = argv[i + 1]
                    i += 2
                } else {
                    flags[key] = "1"
                    i += 1
                }
            } else {
                i += 1
            }
        }
        return Args(command: command, flags: flags)
    }
}

func printHelp() {
    print("""
    transformer-metal — GPT from scratch on Tiny Shakespeare, Metal int4/int8 kernels

    usage:
      gpt check
      gpt overfit [--bits 8] [--steps 250]
      gpt train [--bits 8] [--steps 800] [--batch 16] [--block 64] [--layers 4] [--heads 4] [--embd 128] [--lr 1e-3]
      gpt sample [--ckpt build/model.ckpt] [--prompt ROMEO:] [--tokens 200] [--temp 0.8]
      gpt stories [--steps 4000] [--batch 32] [--block 512] [--lr 0.003] [--seed 456] [--data data/macro-distill.bin]
      gpt stories-sample [--ckpt build/macro.ckpt] [--prompt "The bird fell"] [--tokens 640] [--temp 0.5] [--top-p 0.9] [--seed N]

    --bits 4, 8, or 32. 8-bit is the default. 32 keeps every GEMM in fp32.
    Weights are group-quantized (group 8) only for the forward GEMM. Gradients
    and Adam stay in fp32, then the master weights are re-quantized. That is
    straight-through quantized training, which is what actually converges.
    """)
}
