import Foundation
import Metal

/// One fp32 master tensor, plus the int4/int8 copy used by forward GEMMs.
final class Param {
    let name: String
    let rows: Int
    let cols: Int
    let quantized: Bool
    let decay: Bool
    let w: Buf
    let g: Buf
    let m: Buf
    let v: Buf
    var q: RawBuf?
    var scale: Buf?
    var count: Int { rows * cols }

    init(device: MTLDevice, name: String, rows: Int, cols: Int, quantized: Bool, decay: Bool, bits: Int, qGroup: Int) {
        self.name = name
        self.rows = rows
        self.cols = cols
        self.quantized = quantized
        self.decay = decay
        w = Buf(device, count: rows * cols)
        g = Buf(device, count: rows * cols)
        m = Buf(device, count: rows * cols)
        v = Buf(device, count: rows * cols)
        if quantized {
            let groups = cols / qGroup
            scale = Buf(device, count: rows * groups)
            let nbytes = bits == 4 ? rows * cols / 2 : rows * cols
            q = RawBuf(device, bytes: nbytes)
        }
    }
}

/// Saved activations for one pre-norm block.
final class Block {
    let ln1g, ln1b, wq, bq, wk, bk, wv, bv, wo, bo: Param
    let ln2g, ln2b, wfc, bfc, wproj, bproj: Param
    let ln1, mean1, rstd1, q, k, v, av, xMid, ln2, mean2, rstd2, h, g, probs: Buf

    init(device: MTLDevice, index: Int, cfg: GPTConfig, make: (String, Int, Int, Bool, Bool) -> Param) {
        let C = cfg.embd
        let F = cfg.mlp
        let bt = cfg.batch * cfg.block
        ln1g = make("b\(index).ln1.g", C, 1, false, false)
        ln1b = make("b\(index).ln1.b", C, 1, false, false)
        wq = make("b\(index).wq", C, C, cfg.bits != 32, true)
        bq = make("b\(index).bq", C, 1, false, false)
        wk = make("b\(index).wk", C, C, cfg.bits != 32, true)
        bk = make("b\(index).bk", C, 1, false, false)
        wv = make("b\(index).wv", C, C, cfg.bits != 32, true)
        bv = make("b\(index).bv", C, 1, false, false)
        wo = make("b\(index).wo", C, C, cfg.bits != 32, true)
        bo = make("b\(index).bo", C, 1, false, false)
        ln2g = make("b\(index).ln2.g", C, 1, false, false)
        ln2b = make("b\(index).ln2.b", C, 1, false, false)
        wfc = make("b\(index).fc", F, C, cfg.bits != 32, true)
        bfc = make("b\(index).fc.b", F, 1, false, false)
        wproj = make("b\(index).proj", C, F, cfg.bits != 32, true)
        bproj = make("b\(index).proj.b", C, 1, false, false)

        ln1 = Buf(device, count: bt * C)
        mean1 = Buf(device, count: bt)
        rstd1 = Buf(device, count: bt)
        q = Buf(device, count: bt * C)
        k = Buf(device, count: bt * C)
        v = Buf(device, count: bt * C)
        av = Buf(device, count: bt * C)
        xMid = Buf(device, count: bt * C)
        ln2 = Buf(device, count: bt * C)
        mean2 = Buf(device, count: bt)
        rstd2 = Buf(device, count: bt)
        h = Buf(device, count: bt * F)
        g = Buf(device, count: bt * F)
        probs = Buf(device, count: cfg.batch * cfg.heads * cfg.block * cfg.block)
    }
}

final class GPT {
    let gpu: GPU
    let cfg: GPTConfig
    var itos: [UInt8]
    var stoi: [UInt8: UInt32]
    private(set) var params: [Param] = []
    private var blocks: [Block] = []
    private let wte: Param
    private let wpe: Param
    private let lnfg: Param
    private let lnfb: Param
    private var xs: [Buf] = []
    private let lnF, meanF, rstdF, logits, dlogits, nll, lossSum, sumsq: Buf
    private let scores, dprobs, gx, t1, t2, dq, dk, dv, dav, gg, gh: Buf
    private let tokens: UBuf
    private let targets: UBuf
    private(set) var B = 1
    private(set) var T = 1

    var parameterCount: Int { params.reduce(0) { $0 + $1.count } }

    init(gpu: GPU, cfg: GPTConfig, itos: [UInt8], seed: UInt64, quantizeNow: Bool = true) {
        cfg.validate()
        precondition(itos.count == cfg.vocab)
        self.gpu = gpu
        self.cfg = cfg
        self.itos = itos
        var map: [UInt8: UInt32] = [:]
        for (i, b) in itos.enumerated() { map[b] = UInt32(i) }
        stoi = map

        let device = gpu.device
        let C = cfg.embd
        let maxBT = cfg.batch * cfg.block
        var built: [Param] = []
        func make(_ name: String, _ rows: Int, _ cols: Int, _ quant: Bool, _ decay: Bool) -> Param {
            let p = Param(device: device, name: name, rows: rows, cols: cols,
                          quantized: quant, decay: decay, bits: cfg.bits, qGroup: cfg.qGroup)
            built.append(p)
            return p
        }

        let tokEmb = make("wte", cfg.vocab, C, cfg.bits != 32, true)
        let posEmb = make("wpe", cfg.block, C, false, true)
        var blockList: [Block] = []
        for i in 0..<cfg.layers {
            blockList.append(Block(device: device, index: i, cfg: cfg, make: make))
        }
        let fgn = make("ln_f.g", C, 1, false, false)
        let fbn = make("ln_f.b", C, 1, false, false)
        wte = tokEmb
        wpe = posEmb
        blocks = blockList
        params = built
        lnfg = fgn
        lnfb = fbn

        var stream: [Buf] = []
        for _ in 0...cfg.layers { stream.append(Buf(device, count: maxBT * C)) }
        xs = stream
        lnF = Buf(device, count: maxBT * C)
        meanF = Buf(device, count: maxBT)
        rstdF = Buf(device, count: maxBT)
        logits = Buf(device, count: maxBT * cfg.vocab)
        dlogits = Buf(device, count: maxBT * cfg.vocab)
        nll = Buf(device, count: maxBT)
        lossSum = Buf(device, count: 1)
        sumsq = Buf(device, count: max(built.count, 1))
        let attn = cfg.batch * cfg.heads * cfg.block * cfg.block
        scores = Buf(device, count: attn)
        dprobs = Buf(device, count: attn)
        gx = Buf(device, count: maxBT * C)
        t1 = Buf(device, count: maxBT * max(C, cfg.mlp))
        t2 = Buf(device, count: maxBT * C)
        dq = Buf(device, count: maxBT * C)
        dk = Buf(device, count: maxBT * C)
        dv = Buf(device, count: maxBT * C)
        dav = Buf(device, count: maxBT * C)
        gg = Buf(device, count: maxBT * cfg.mlp)
        gh = Buf(device, count: maxBT * cfg.mlp)
        tokens = UBuf(device, count: maxBT)
        targets = UBuf(device, count: maxBT)

        var rng = RNG(seed)
        let base: Float = 0.02
        let resid = 0.02 / sqrt(2 * Float(cfg.layers))
        for p in params where p.cols > 1 && p.name != "wpe" && !p.name.hasSuffix(".g") {
            let s: Float = (p.name.hasSuffix(".wo") || p.name.hasSuffix(".proj")) ? resid : base
            fillNormal(p.w, std: s, rng: &rng)
        }
        fillNormal(wpe.w, std: base, rng: &rng)
        for p in params where p.name.hasSuffix(".g") {
            for i in 0..<p.count { p.w.p[i] = 1 }
        }
        if quantizeNow && cfg.bits != 32 {
            gpu.begin()
            quantizeAll()
            gpu.end()
        }
    }

    func step(tokenIds: [UInt32], targetIds: [UInt32], lr: Float, wd: Float, clip: Float,
              stepIndex: Int, b1: Float = 0.9, b2: Float = 0.95, adamEps: Float = 1e-8) -> Float {
        let M = cfg.batch * cfg.block
        precondition(tokenIds.count == M && targetIds.count == M)
        tokens.upload(tokenIds)
        targets.upload(targetIds)
        B = cfg.batch
        T = cfg.block

        gpu.begin()
        zeroGrads()
        forward(withLoss: true)
        backward()
        for (i, p) in params.enumerated() {
            gpu.reduceSumsq(p.g, into: sumsq, n: p.count, slot: i)
        }
        gpu.end()

        let loss = lossSum.p[0] / Float(M)
        var ss: Float = 0
        for i in 0..<params.count { ss += sumsq.p[i] }
        let norm = sqrt(ss)
        let invBC1 = 1 / (1 - pow(b1, Float(stepIndex)))
        let invBC2 = 1 / (1 - pow(b2, Float(stepIndex)))

        gpu.begin()
        if clip > 0, norm > clip {
            let s = clip / (norm + 1e-6)
            for p in params { gpu.scale(p.g, n: p.count, by: s) }
        }
        for p in params {
            gpu.adam(w: p.w, g: p.g, m: p.m, v: p.v, n: p.count, lr: lr, b1: b1, b2: b2,
                     eps: adamEps, wd: p.decay ? wd : 0, invBC1: invBC1, invBC2: invBC2)
        }
        if cfg.bits != 32 { quantizeAll() }
        gpu.end()
        return loss
    }

    /// Last-position logits for a packed batch of shape [1, T].
    func nextLogits(_ tokenIds: [UInt32]) -> [Float] {
        precondition(!tokenIds.isEmpty && tokenIds.count <= cfg.block)
        B = 1
        T = tokenIds.count
        tokens.upload(tokenIds)
        gpu.begin()
        forward(withLoss: false)
        gpu.end()
        let V = cfg.vocab
        let row = (T - 1) * V
        return Array(UnsafeBufferPointer(start: logits.p.advanced(by: row), count: V))
    }

    /// Teacher-forced argmax accuracy on the current logits. Call after a forward.
    func tokenAccuracy(targets ys: [UInt32]) -> Float {
        let V = cfg.vocab
        var correct = 0
        for i in 0..<ys.count {
            var best = 0
            var bestV = -Float.infinity
            let row = i * V
            for v in 0..<V {
                let z = logits.p[row + v]
                if z > bestV { bestV = z; best = v }
            }
            if UInt32(best) == ys[i] { correct += 1 }
        }
        return Float(correct) / Float(ys.count)
    }

    func lossOnly(tokenIds: [UInt32], targetIds: [UInt32], batch: Int, block: Int) -> Float {
        precondition(tokenIds.count == batch * block)
        B = batch
        T = block
        tokens.upload(tokenIds)
        targets.upload(targetIds)
        gpu.begin()
        forward(withLoss: true)
        gpu.end()
        return lossSum.p[0] / Float(batch * block)
    }

    func save(_ url: URL) throws {
        var data = Data()
        data.append(contentsOf: [0x54, 0x47, 0x4D, 0x54]) // TGMT
        appendU32(1, &data)
        appendU32(UInt32(cfg.bits), &data)
        appendU32(UInt32(cfg.vocab), &data)
        appendU32(UInt32(cfg.block), &data)
        appendU32(UInt32(cfg.layers), &data)
        appendU32(UInt32(cfg.heads), &data)
        appendU32(UInt32(cfg.embd), &data)
        appendU32(UInt32(cfg.qGroup), &data)
        data.append(contentsOf: itos)
        for p in params {
            data.append(Data(bytes: p.w.p, count: p.count * MemoryLayout<Float>.size))
        }
        try data.write(to: url)
    }

    static func load(gpu: GPU, url: URL, batch: Int, seed: UInt64) throws -> GPT {
        let data = try Data(contentsOf: url)
        var off = 0
        func take(_ n: Int) -> Data {
            let s = data.subdata(in: off..<(off + n))
            off += n
            return s
        }
        func u32() -> UInt32 {
            let d = take(4)
            return d.withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        }
        let magic = take(4)
        precondition(Array(magic) == [0x54, 0x47, 0x4D, 0x54], "not a transformer-metal checkpoint")
        let version = u32()
        precondition(version == 1, "unsupported checkpoint version \(version)")
        let bits = Int(u32())
        let vocab = Int(u32())
        let block = Int(u32())
        let layers = Int(u32())
        let heads = Int(u32())
        let embd = Int(u32())
        let qGroup = Int(u32())
        let itos = Array(take(vocab))
        let cfg = GPTConfig(vocab: vocab, block: block, batch: batch, layers: layers,
                            heads: heads, embd: embd, bits: bits, qGroup: qGroup)
        let model = GPT(gpu: gpu, cfg: cfg, itos: itos, seed: seed, quantizeNow: false)
        for p in model.params {
            let nbytes = p.count * MemoryLayout<Float>.size
            let slice = take(nbytes)
            slice.withUnsafeBytes { raw in
                p.w.p.update(from: raw.bindMemory(to: Float.self).baseAddress!, count: p.count)
            }
        }
        precondition(off == data.count, "checkpoint length does not match the model")
        if bits != 32 {
            gpu.begin()
            model.quantizeAll()
            gpu.end()
        }
        return model
    }

    // MARK: - forward / backward

    private func forward(withLoss: Bool) {
        let M = B * T
        let C = cfg.embd
        let F = cfg.mlp
        let H = cfg.heads
        let D = cfg.headDim
        let V = cfg.vocab
        gpu.embed(bits: cfg.bits, tokens: tokens, wte: wte.w, q: wte.q, scale: wte.scale,
                  wpe: wpe.w, y: xs[0], B: B, T: T, C: C, qGroup: cfg.qGroup)
        for (li, layer) in blocks.enumerated() {
            gpu.lnFwd(x: xs[li], y: layer.ln1, mean: layer.mean1, rstd: layer.rstd1,
                      gamma: layer.ln1g.w, beta: layer.ln1b.w, rows: M, cols: C, eps: cfg.eps)
            linear(layer.wq, layer.bq, layer.ln1, layer.q, M: M, N: C, K: C)
            linear(layer.wk, layer.bk, layer.ln1, layer.k, M: M, N: C, K: C)
            linear(layer.wv, layer.bv, layer.ln1, layer.v, M: M, N: C, K: C)
            gpu.attnFwd(q: layer.q, k: layer.k, v: layer.v, scores: scores, probs: layer.probs,
                        av: layer.av, B: B, T: T, H: H, D: D)
            linear(layer.wo, layer.bo, layer.av, t1, M: M, N: C, K: C)
            gpu.add(xs[li], t1, layer.xMid, n: M * C)
            gpu.lnFwd(x: layer.xMid, y: layer.ln2, mean: layer.mean2, rstd: layer.rstd2,
                      gamma: layer.ln2g.w, beta: layer.ln2b.w, rows: M, cols: C, eps: cfg.eps)
            linear(layer.wfc, layer.bfc, layer.ln2, layer.h, M: M, N: F, K: C)
            gpu.geluFwd(layer.h, layer.g, n: M * F)
            linear(layer.wproj, layer.bproj, layer.g, t1, M: M, N: C, K: F)
            gpu.add(layer.xMid, t1, xs[li + 1], n: M * C)
        }
        gpu.lnFwd(x: xs[cfg.layers], y: lnF, mean: meanF, rstd: rstdF,
                  gamma: lnfg.w, beta: lnfb.w, rows: M, cols: C, eps: cfg.eps)
        gpu.gemmNT(bits: cfg.bits, w: wte.w, q: wte.q, scale: wte.scale, x: lnF, bias: nil,
                   y: logits, M: M, N: V, K: C, qGroup: cfg.qGroup)
        if withLoss {
            gpu.loss(logits: logits, targets: targets, nll: nll, dlogits: dlogits, BT: M, V: V)
            gpu.reduceSum(nll, into: lossSum, n: M, slot: 0)
        }
    }

    private func backward() {
        let M = B * T
        let C = cfg.embd
        let F = cfg.mlp
        let H = cfg.heads
        let D = cfg.headDim
        let V = cfg.vocab
        let nC = M * C

        // Tied head: dW overwrites wte.g, then the input embedding adds into it.
        gpu.gemmDW(dy: dlogits, x: lnF, dw: wte.g, M: M, N: V, K: C)
        gpu.gemmDX(bits: cfg.bits, w: wte.w, q: wte.q, scale: wte.scale, dy: dlogits, dx: t1,
                   M: M, N: V, K: C, qGroup: cfg.qGroup)
        gpu.lnBwd(x: xs[cfg.layers], mean: meanF, rstd: rstdF, gamma: lnfg.w, beta: lnfb.w,
                  dy: t1, dx: gx, dgamma: lnfg.g, dbeta: lnfb.g, rows: M, cols: C, eps: cfg.eps)

        for li in stride(from: blocks.count - 1, through: 0, by: -1) {
            let layer = blocks[li]
            // gx is d(x next). MLP branch consumes it, then LayerNorm adds in.
            gpu.gemmDW(dy: gx, x: layer.g, dw: layer.wproj.g, M: M, N: C, K: F)
            gpu.gemmDX(bits: cfg.bits, w: layer.wproj.w, q: layer.wproj.q, scale: layer.wproj.scale,
                       dy: gx, dx: gg, M: M, N: C, K: F, qGroup: cfg.qGroup)
            gpu.biasGrad(dy: gx, db: layer.bproj.g, M: M, N: C)
            gpu.geluBwd(x: layer.h, dy: gg, dx: gh, n: M * F)
            gpu.gemmDW(dy: gh, x: layer.ln2, dw: layer.wfc.g, M: M, N: F, K: C)
            gpu.gemmDX(bits: cfg.bits, w: layer.wfc.w, q: layer.wfc.q, scale: layer.wfc.scale,
                       dy: gh, dx: t1, M: M, N: F, K: C, qGroup: cfg.qGroup)
            gpu.biasGrad(dy: gh, db: layer.bfc.g, M: M, N: F)
            gpu.lnBwd(x: layer.xMid, mean: layer.mean2, rstd: layer.rstd2, gamma: layer.ln2g.w,
                      beta: layer.ln2b.w, dy: t1, dx: t2, dgamma: layer.ln2g.g, dbeta: layer.ln2b.g,
                      rows: M, cols: C, eps: cfg.eps)
            gpu.acc(gx, t2, n: nC)

            gpu.gemmDW(dy: gx, x: layer.av, dw: layer.wo.g, M: M, N: C, K: C)
            gpu.gemmDX(bits: cfg.bits, w: layer.wo.w, q: layer.wo.q, scale: layer.wo.scale,
                       dy: gx, dx: dav, M: M, N: C, K: C, qGroup: cfg.qGroup)
            gpu.biasGrad(dy: gx, db: layer.bo.g, M: M, N: C)
            gpu.attnBwd(q: layer.q, k: layer.k, v: layer.v, probs: layer.probs, dout: dav,
                        dq: dq, dk: dk, dv: dv, dprobs: dprobs, dscores: scores,
                        B: B, T: T, H: H, D: D)

            gpu.gemmDW(dy: dq, x: layer.ln1, dw: layer.wq.g, M: M, N: C, K: C)
            gpu.gemmDX(bits: cfg.bits, w: layer.wq.w, q: layer.wq.q, scale: layer.wq.scale,
                       dy: dq, dx: t1, M: M, N: C, K: C, qGroup: cfg.qGroup)
            gpu.biasGrad(dy: dq, db: layer.bq.g, M: M, N: C)

            gpu.gemmDW(dy: dk, x: layer.ln1, dw: layer.wk.g, M: M, N: C, K: C)
            gpu.gemmDX(bits: cfg.bits, w: layer.wk.w, q: layer.wk.q, scale: layer.wk.scale,
                       dy: dk, dx: t2, M: M, N: C, K: C, qGroup: cfg.qGroup)
            gpu.acc(t1, t2, n: nC)
            gpu.biasGrad(dy: dk, db: layer.bk.g, M: M, N: C)

            gpu.gemmDW(dy: dv, x: layer.ln1, dw: layer.wv.g, M: M, N: C, K: C)
            gpu.gemmDX(bits: cfg.bits, w: layer.wv.w, q: layer.wv.q, scale: layer.wv.scale,
                       dy: dv, dx: t2, M: M, N: C, K: C, qGroup: cfg.qGroup)
            gpu.acc(t1, t2, n: nC)
            gpu.biasGrad(dy: dv, db: layer.bv.g, M: M, N: C)

            gpu.lnBwd(x: xs[li], mean: layer.mean1, rstd: layer.rstd1, gamma: layer.ln1g.w,
                      beta: layer.ln1b.w, dy: t1, dx: t2, dgamma: layer.ln1g.g, dbeta: layer.ln1b.g,
                      rows: M, cols: C, eps: cfg.eps)
            gpu.acc(gx, t2, n: nC)
        }

        gpu.posBwd(dx: gx, dwp: wpe.g, B: B, T: T, C: C)
        gpu.embedBwd(tokens: tokens, dx: gx, dwe: wte.g, B: B, T: T, C: C, V: V)
    }

    private func linear(_ w: Param, _ bias: Param, _ x: Buf, _ y: Buf, M: Int, N: Int, K: Int) {
        gpu.gemmNT(bits: cfg.bits, w: w.w, q: w.q, scale: w.scale, x: x, bias: bias.w,
                   y: y, M: M, N: N, K: K, qGroup: cfg.qGroup)
    }

    private func zeroGrads() {
        for p in params { gpu.zero(p.g, n: p.count) }
    }

    fileprivate func quantizeAll() {
        guard cfg.bits != 32 else { return }
        for p in params where p.quantized {
            gpu.quantize(w: p.w, q: p.q!, scale: p.scale!, rows: p.rows, cols: p.cols,
                         bits: cfg.bits, qGroup: cfg.qGroup)
        }
    }
}

func fillNormal(_ buf: Buf, std: Float, rng: inout RNG) {
    for i in 0..<buf.count { buf.p[i] = rng.normal() * std }
}

private func appendU32(_ v: UInt32, _ data: inout Data) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}
