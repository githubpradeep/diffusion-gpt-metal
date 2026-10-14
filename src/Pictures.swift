import Darwin
import Foundation

/// Named 24×24 icons shared by the autoregressive painter and the diffusion transformer.
enum Pictures {
    static let side = 24
    static let patch = 4
    static let colors = 32
    static let holdout = 8
    static let pixels = side * side
    static let color0: UInt32 = 4
    static let name0: UInt32 = 4 + 32
    static let seqLen = 1 + pixels + 1
    static let blank = [230, 223, 212]
    static let paintStride = 48

    static let tokensDefault = "data/images/tokens.bin"
    static let rgbDefault = "data/images/rgb.bin"
    static let vocabDefault = "vocab/images.txt"
    static let paletteDefault = "data/images/palette.json"
    static let arCkptDefault = "build/images-ar.ckpt"
    static let ditCkptDefault = "build/images-dit.ckpt"
}

struct IconRGB {
    var name: Int
    var rgb: [Float]
}

struct IconFile {
    var side: Int
    var icons: [IconRGB]
}

func requirePicture(_ path: String) {
    if !FileManager.default.fileExists(atPath: path) {
        fputs("missing \(path)\nrun: make pictures\n", stderr)
        exit(1)
    }
}

func loadPalette(_ path: String) throws -> [[Int]] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let obj = try JSONSerialization.jsonObject(with: data)
    guard let dict = obj as? [String: Any], let rows = dict["colors"] as? [Any] else {
        throw NSError(domain: "pictures", code: 1, userInfo: [NSLocalizedDescriptionKey: "palette \(path) has no colors"])
    }
    var colors: [[Int]] = []
    for row in rows {
        let nums = row as? [Any] ?? []
        let rgb = nums.prefix(3).map { n -> Int in
            if let i = n as? Int { return i }
            if let d = n as? Double { return Int(d.rounded()) }
            return 0
        }
        if rgb.count == 3 { colors.append(rgb) }
    }
    if colors.count != Pictures.colors {
        throw NSError(domain: "pictures", code: 2, userInfo: [NSLocalizedDescriptionKey: "palette has \(colors.count) colors"])
    }
    return colors
}

func loadIconRGB(_ path: String) -> [IconRGB] {
    loadIconFile(path).icons
}

func loadIconFile(_ path: String) -> IconFile {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), data.count >= 20 else {
        fputs("missing icon rgb \(path)\nrun: make pictures\n", stderr)
        exit(1)
    }
    var offset = 0
    func take(_ n: Int) -> Data {
        precondition(offset + n <= data.count, "truncated icon rgb")
        let slice = data.subdata(in: offset..<(offset + n))
        offset += n
        return slice
    }
    func u32() -> UInt32 {
        take(4).withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
    }
    let magic = take(4)
    precondition(Array(magic) == [0x49, 0x52, 0x47, 0x42], "\(path) is not an icon rgb file")
    let version = u32()
    precondition(version == 1, "unsupported icon rgb version \(version)")
    let count = Int(u32())
    let side = Int(u32())
    let channels = Int(u32())
    precondition(side > 0 && channels == 3, "icon rgb is \(side)x\(side) x\(channels)")
    let span = side * side * channels
    var icons: [IconRGB] = []
    icons.reserveCapacity(count)
    for _ in 0..<count {
        let name = Int(u32())
        let bytes = take(span * MemoryLayout<Float>.size)
        var rgb = [Float](repeating: 0, count: span)
        _ = rgb.withUnsafeMutableBytes { raw in
            bytes.copyBytes(to: raw)
        }
        icons.append(IconRGB(name: name, rgb: rgb))
    }
    precondition(offset == data.count, "icon rgb length does not match the header")
    return IconFile(side: side, icons: icons)
}

func iconNameIndex(_ words: WordList, _ name: String) -> Int? {
    let key = name.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let id = words.stoi[key], id >= Pictures.name0 else { return nil }
    return Int(id - Pictures.name0)
}

func colorBan(vocab: Int, allowEnd: Bool, end: UInt32) -> Set<Int> {
    var ban = Set<Int>()
    let lo = Int(Pictures.color0)
    let hi = Int(Pictures.name0)
    for i in 0..<vocab where i < lo || i >= hi {
        if allowEnd && UInt32(i) == end { continue }
        ban.insert(i)
    }
    return ban
}

func paintColors(_ ids: [UInt32], palette: [[Int]]) -> [Int] {
    var out: [Int] = []
    out.reserveCapacity(Pictures.pixels * 3)
    for i in 0..<Pictures.pixels {
        if i < ids.count {
            let k = Int(ids[i]) - Int(Pictures.color0)
            if k >= 0 && k < palette.count {
                out.append(palette[k][0])
                out.append(palette[k][1])
                out.append(palette[k][2])
                continue
            }
        }
        out.append(contentsOf: Pictures.blank)
    }
    return out
}

func rgbBytes(_ values: [Float]) -> [Int] {
    values.map { v in
        let u = min(max((v + 1) * 0.5, 0), 1)
        return Int((u * 255).rounded())
    }
}

func imagePayload(kind: String, name: String, params: Int, frames: [[Int]], loops: [[Int]],
                    side: Int = Pictures.side) -> [String: Any] {
    [
        "kind": kind,
        "name": name,
        "width": side,
        "height": side,
        "params": params,
        "frames": frames,
        "loops": loops,
    ]
}

func emitImage(_ object: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(object),
          let data = try? JSONSerialization.data(withJSONObject: object) else {
        fputs("could not encode the image sample\n", stderr)
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

private func imageMacroConfig(words: WordList, batch: Int, args: Args, block: Int? = nil,
                              loops: Int? = nil, embd: Int? = nil, intermediate: Int? = nil,
                              heads: Int? = nil) -> MacroConfig {
    var cfg = MacroConfig(vocab: words.tokens.count,
                          block: block ?? args.int("block", Pictures.seqLen),
                          batch: batch,
                          loops: loops ?? args.int("loops", 4))
    cfg.embd = embd ?? args.int("embd", 64)
    cfg.intermediate = intermediate ?? args.int("intermediate", 128)
    cfg.heads = heads ?? args.int("heads", 4)
    return cfg
}

func trainImageAR(gpu: GPU, args: Args, vocabDefault: String = Pictures.vocabDefault,
                  dataDefault: String = Pictures.tokensDefault,
                  ckptDefault: String = Pictures.arCkptDefault,
                  blockDefault: Int = Pictures.seqLen,
                  title: String = "autoregressive icons") throws {
    let vocabPath = args.str("vocab", vocabDefault)
    let dataPath = args.str("data", dataDefault)
    requirePicture(vocabPath)
    requirePicture(dataPath)
    let words = try WordList.load(vocabPath)
    let batch = args.int("batch", 16)
    let steps = args.int("steps", 2000)
    let lr = args.float("lr", 0.003)
    let wd = args.float("wd", 0)
    let clip = args.float("clip", 0)
    let seed = UInt64(args.int("seed", 456))
    let ckptPath = args.str("ckpt", ckptDefault)
    let cfg = imageMacroConfig(words: words, batch: batch, args: args, block: args.int("block", blockDefault))
    cfg.validate()
    let model = Macro(gpu: gpu, cfg: cfg, seed: seed)
    print("device \(gpu.device.name)")
    print("\(title)  params \(model.parameterCount)  vocab \(cfg.vocab)  embd \(cfg.embd)  ffn \(cfg.intermediate)")
    print("loops \(cfg.loops)  heads \(cfg.heads)/\(cfg.kvHeads)  block \(cfg.block)  batch \(batch)  steps \(steps)")
    print(String(format: "uniform-guess loss %.3f", log(Float(cfg.vocab))))
    // holdout 1 puts every icon in the training rows. A larger stride keeps
    // every Nth icon out, which is how a name on the page can miss its picture.
    let holdout = args.int("holdout", 1)
    let corpus = loadMacroCorpus(dataPath, holdout: holdout)
    let name = words.tokens[Int(corpus.train[0][0])]
    print("corpus \(dataPath)  train \(corpus.train.count)  val \(corpus.val.count)  first \(name)")
    var rng = RNG(UInt64(args.int("data-seed", 42)))
    var valRNG = RNG(7)
    let t0 = CFAbsoluteTimeGetCurrent()
    for step in 1...steps {
        let sched = macroLearningRate(step: step, total: steps, base: lr)
        let (x, y, valid) = macroCorpusBatch(corpus.train, batch: batch, block: cfg.block, rng: &rng)
        let loss = model.step(tokenIds: x, targetIds: y, valid: valid, lr: sched, wd: wd, clip: clip, stepIndex: step)
        if !loss.isFinite {
            fputs("loss became non-finite.\n", stderr)
            exit(1)
        }
        if step == 1 || step % 20 == 0 || step == steps {
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let toks = Double(step * batch * cfg.block) / max(dt, 1e-6)
            var line = String(format: "step %4d/%d  loss %.4f  lr %.2e  %.0f tok/s", step, steps, loss, sched, toks)
            if step % 100 == 0 || step == steps {
                let (vx, vy, vvalid) = macroCorpusBatch(corpus.val, batch: batch, block: cfg.block, rng: &valRNG)
                let vloss = model.lossOnly(tokenIds: vx, targetIds: vy, valid: vvalid)
                line += String(format: "  val %.4f", vloss)
            }
            print(line)
            fflush(stdout)
        }
    }
    try writeFile(ckptPath) { try model.save($0) }
    print("wrote \(ckptPath)")
}

func drawImageAR(model: Macro, words: WordList, palette: [[Int]], name: String,
                 temp: Float, topP: Float, rng: inout RNG) -> [String: Any] {
    guard let index = iconNameIndex(words, name) else {
        return ["error": "unknown", "unknown": [name]]
    }
    let nameID = Pictures.name0 + UInt32(index)
    var ids: [UInt32] = [nameID]
    var colors: [UInt32] = []
    var frames: [[Int]] = [paintColors([], palette: palette)]
    let ban = colorBan(vocab: model.cfg.vocab, allowEnd: true, end: words.eos)
    for _ in 0..<Pictures.pixels {
        let window = Array(ids.suffix(model.cfg.block))
        let next = sampleNucleus(model.nextLogits(window), temp: temp, topP: topP, rng: &rng, ban: ban)
        if next == words.eos { break }
        ids.append(next)
        colors.append(next)
        if colors.count % Pictures.paintStride == 0 {
            frames.append(paintColors(colors, palette: palette))
        }
    }
    if !colors.isEmpty && colors.count % Pictures.paintStride != 0 {
        frames.append(paintColors(colors, palette: palette))
    }
    let lensBan = colorBan(vocab: model.cfg.vocab, allowEnd: false, end: words.eos)
    let cells = model.lensTopK(Array(ids.suffix(model.cfg.block)), k: 1, ban: lensBan)
    var loops: [[Int]] = []
    for pass in 1...model.cfg.loops {
        var guessed: [UInt32] = []
        let limit = min(Pictures.pixels, cells.count)
        for i in 0..<limit {
            if pass < cells[i].count, let top = cells[i][pass].first {
                guessed.append(UInt32(top.0))
            }
        }
        loops.append(paintColors(guessed, palette: palette))
    }
    let label = words.tokens[Int(nameID)]
    return imagePayload(kind: "ar", name: label, params: model.parameterCount, frames: frames, loops: loops)
}

func runImageARRepl(gpu: GPU, args: Args) throws {
    let words = try WordList.load(args.str("vocab", Pictures.vocabDefault))
    let palette = try loadPalette(args.str("palette", Pictures.paletteDefault))
    let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: args.str("ckpt", Pictures.arCkptDefault)),
                               batch: 1, seed: 1)
    var rng = RNG(args.flags["seed"].flatMap(UInt64.init) ?? 1)
    let temp = args.float("temp", 0)
    let topP = args.float("top-p", 1)
    while let line = readLine(strippingNewline: true) {
        let name = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { continue }
        emitImage(drawImageAR(model: model, words: words, palette: palette, name: name,
                              temp: temp, topP: topP, rng: &rng))
    }
}

func runImageChecks(gpu: GPU) throws -> Bool {
    requirePicture(Pictures.vocabDefault)
    requirePicture(Pictures.tokensDefault)
    requirePicture(Pictures.rgbDefault)
    let words = try WordList.load(Pictures.vocabDefault)
    let corpus = loadMacroCorpus(Pictures.tokensDefault, holdout: Pictures.holdout)
    let icons = loadIconRGB(Pictures.rgbDefault)
    var report = Report()
    let story = corpus.train[0]
    let nameIndex = Int(story[0] - Pictures.name0)
    let icon = icons[nameIndex]
    let label = words.tokens[Int(story[0])]
    report.expect(story.count == Pictures.seqLen && icon.name == nameIndex && icon.rgb.count == Pictures.pixels * 3,
                  "icon \(label) is \(story.count) tokens and \(icon.rgb.count / 3) pixels")
    let round = unpatchImage(patchImage(icon.rgb, side: Pictures.side, patch: Pictures.patch),
                             side: Pictures.side, patch: Pictures.patch)
    let roundErr = zip(round, icon.rgb).map { abs($0 - $1) }.max() ?? 0
    report.expect(roundErr < 1e-5, "patch round-trip")

    print("autoregressive memorization of \(label)")
    let ar = memorizeIconAR(gpu: gpu, words: words, story: story)
    let drop = ar.initial - ar.final
    report.expect(ar.final.isFinite && drop > 1.5,
                  String(format: "ar loss %.3f -> %.3f (drop %.3f)", ar.initial, ar.final, drop))

    print("diffusion gradient check")
    ditGradCheck(gpu: gpu, icon: icon, &report)
    print("diffusion memorization of \(label)")
    let dit = ditMemorize(gpu: gpu, icon: icon)
    report.expect(dit.final.isFinite && dit.final < dit.initial * 0.5,
                  String(format: "dit x0 %.3f -> %.3f", dit.initial, dit.final))
    report.expect(dit.mae < dit.gray * 0.55,
                  String(format: "dit sample mae %.3f  gray %.3f", dit.mae, dit.gray))
    if report.failed != 0 {
        fputs("images-check failed\n", stderr)
        return false
    }
    print("images-check passed")
    return true
}

private struct IconMem {
    var initial: Float
    var final: Float
}

private func memorizeIconAR(gpu: GPU, words: WordList, story: [UInt32]) -> IconMem {
    let batch = 2
    let steps = 40
    let block = Pictures.seqLen
    let cfg = imageMacroConfig(words: words, batch: batch, args: Args(command: "", flags: [:]),
                               block: block, loops: 4, embd: 64, intermediate: 128, heads: 4)
    cfg.validate()
    let model = Macro(gpu: gpu, cfg: cfg, seed: 3)
    var rng = RNG(1)
    var initial: Float = 0
    var final: Float = 0
    for step in 1...steps {
        let (x, y, valid) = macroCorpusBatch([story], batch: batch, block: block, rng: &rng)
        let loss = model.step(tokenIds: x, targetIds: y, valid: valid, lr: 0.02, wd: 0, clip: 0, stepIndex: step)
        if step == 1 { initial = loss }
        final = loss
    }
    return IconMem(initial: initial, final: final)
}

enum PhotoAR {
    static let tokens = "data/photos/tokens.bin"
    static let vocab = "vocab/photos-ar.txt"
    static let codebook = "data/photos/codebook.bin"
    static let ckpt = "build/photos-ar.ckpt"
    static let side = 128
    static let patch = 16
    static let grid = side / patch
    static let patches = grid * grid
    static let seqLen = 1 + patches + 1
}

struct PatchBook {
    var side: Int
    var patch: Int
    var name0: Int
    var rgb: [[UInt8]]
}

func loadPatchBook(_ path: String) -> PatchBook {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), data.count >= 24 else {
        fputs("missing patch book \(path)\nrun: python3 scripts/make_photo_tokens.py\n", stderr)
        exit(1)
    }
    var offset = 0
    func take(_ n: Int) -> Data {
        precondition(offset + n <= data.count, "truncated patch book")
        let slice = data.subdata(in: offset..<(offset + n))
        offset += n
        return slice
    }
    func u32() -> Int { Int(take(4).withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }) }
    let magic = take(4)
    precondition(Array(magic) == [0x50, 0x43, 0x42, 0x4B], "\(path) is not a patch book")
    let version = u32()
    precondition(version == 1, "unsupported patch book \(version)")
    let side = u32()
    let patch = u32()
    let count = u32()
    let name0 = u32()
    let span = patch * patch * 3
    var rgb: [[UInt8]] = []
    rgb.reserveCapacity(count)
    for _ in 0..<count {
        rgb.append(Array(take(span)))
    }
    precondition(offset == data.count, "patch book length does not match the header")
    return PatchBook(side: side, patch: patch, name0: name0, rgb: rgb)
}

func trainPhotoAR(gpu: GPU, args: Args) throws {
    try trainImageAR(gpu: gpu, args: args, vocabDefault: PhotoAR.vocab, dataDefault: PhotoAR.tokens,
                     ckptDefault: PhotoAR.ckpt, blockDefault: PhotoAR.seqLen, title: "autoregressive photos")
}

private func paintPatches(_ ids: [UInt32], book: PatchBook) -> [Int] {
    let g = book.side / book.patch
    let span = book.patch * book.patch * 3
    var canvas = [UInt8](repeating: 214, count: book.side * book.side * 3)
    for (n, id) in ids.enumerated() where n < g * g {
        let index = Int(id) - 4
        guard index >= 0 && index < book.rgb.count else { continue }
        let py = n / g
        let px = n % g
        let patch = book.rgb[index]
        guard patch.count == span else { continue }
        for iy in 0..<book.patch {
            for ix in 0..<book.patch {
                let dst = ((py * book.patch + iy) * book.side + (px * book.patch + ix)) * 3
                let src = (iy * book.patch + ix) * 3
                canvas[dst] = patch[src]
                canvas[dst + 1] = patch[src + 1]
                canvas[dst + 2] = patch[src + 2]
            }
        }
    }
    return canvas.map { Int($0) }
}

func drawPhotoAR(model: Macro, words: WordList, book: PatchBook, name: String,
                 temp: Float, topP: Float, rng: inout RNG) -> [String: Any] {
    guard let id = words.stoi[name.trimmingCharacters(in: .whitespacesAndNewlines)], id >= UInt32(book.name0) else {
        return ["error": "unknown", "unknown": [name]]
    }
    var ids: [UInt32] = [id]
    var patches: [UInt32] = []
    var frames: [[Int]] = [paintPatches([], book: book)]
    var ban = Set<Int>()
    for token in 0..<model.cfg.vocab where token < 4 || token >= book.name0 {
        ban.insert(token)
    }
    for _ in 0..<PhotoAR.patches {
        let window = Array(ids.suffix(model.cfg.block))
        let next = sampleNucleus(model.nextLogits(window), temp: temp, topP: topP, rng: &rng, ban: ban)
        ids.append(next)
        patches.append(next)
        frames.append(paintPatches(patches, book: book))
    }
    let cells = model.lensTopK(Array(ids.suffix(model.cfg.block)), k: 1, ban: ban)
    var loops: [[Int]] = []
    for pass in 1...model.cfg.loops {
        var guessed: [UInt32] = []
        for i in 0..<min(PhotoAR.patches, cells.count) {
            if pass < cells[i].count, let top = cells[i][pass].first {
                guessed.append(UInt32(top.0))
            }
        }
        loops.append(paintPatches(guessed, book: book))
    }
    return imagePayload(kind: "ar", name: words.tokens[Int(id)], params: model.parameterCount,
                        frames: frames, loops: loops, side: book.side)
}

func runPhotoARRepl(gpu: GPU, args: Args) throws {
    let words = try WordList.load(args.str("vocab", PhotoAR.vocab))
    let book = loadPatchBook(args.str("codebook", PhotoAR.codebook))
    let model = try Macro.load(gpu: gpu, url: URL(fileURLWithPath: args.str("ckpt", PhotoAR.ckpt)),
                               batch: 1, seed: 1)
    var rng = RNG(args.flags["seed"].flatMap(UInt64.init) ?? 1)
    let temp = args.float("temp", 0)
    let topP = args.float("top-p", 1)
    while let line = readLine(strippingNewline: true) {
        let name = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { continue }
        emitImage(drawPhotoAR(model: model, words: words, book: book, name: name,
                              temp: temp, topP: topP, rng: &rng))
    }
}

private func writeFile(_ path: String, _ body: (URL) throws -> Void) throws {
    let url = URL(fileURLWithPath: path)
    let dir = url.deletingLastPathComponent().path
    if dir != "" && dir != "." {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }
    try body(url)
}
