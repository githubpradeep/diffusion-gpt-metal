import Darwin
import Foundation
import Metal

/// MacroStories: one decoder block applied `loops` times with shared weights.
/// Matches raincandy-u/MacroStories (Ouro student): 32-d, SwiGLU 48, 2 query
/// heads, 1 KV head, RoPE, sandwich RMSNorm, value residual, tied embeddings.
/// The early-exit gate is allocated so the count is 19,969, and it is inactive:
/// loss is next-token cross-entropy on the last loop only.
struct MacroConfig {
    var vocab = MacroVocab.tokens.count
    var block: Int
    var batch: Int
    var loops = 4
    var heads = 2
    var kvHeads = 1
    var embd = 32
    var intermediate = 48
    var eps: Float = 1e-6
    var ropeTheta: Float = 10000
    var valueMix: Float = 0.5

    var headDim: Int { embd / heads }
    var kvDim: Int { kvHeads * headDim }

    func validate() {
        precondition(vocab > 4, "vocab is too small")
        precondition(batch > 0 && block > 1 && loops > 0)
        precondition(heads > 0 && kvHeads == 1, "this build repeats a single KV head")
        precondition(embd % heads == 0)
        precondition(headDim % 2 == 0)
        precondition(intermediate > 0)
    }
}

private final class LoopActs {
    let r1, r1b, r2, r2b, rf: Buf
    let ln1, q, k, v, probs, av, attn, mid, ln2, gate, up, sw, mlp, pre: Buf

    init(device: MTLDevice, batch: Int, block: Int, cfg: MacroConfig) {
        let m = batch * block
        let c = cfg.embd
        let f = cfg.intermediate
        let attnN = batch * cfg.heads * block * block
        r1 = Buf(device, count: m)
        r1b = Buf(device, count: m)
        r2 = Buf(device, count: m)
        r2b = Buf(device, count: m)
        rf = Buf(device, count: m)
        ln1 = Buf(device, count: m * c)
        q = Buf(device, count: m * c)
        k = Buf(device, count: m * c)
        v = Buf(device, count: m * c)
        probs = Buf(device, count: attnN)
        av = Buf(device, count: m * c)
        attn = Buf(device, count: m * c)
        mid = Buf(device, count: m * c)
        ln2 = Buf(device, count: m * c)
        gate = Buf(device, count: m * f)
        up = Buf(device, count: m * f)
        sw = Buf(device, count: m * f)
        mlp = Buf(device, count: m * c)
        pre = Buf(device, count: m * c)
    }
}

final class Macro {
    let gpu: GPU
    let cfg: MacroConfig
    private(set) var params: [Param] = []
    private var muon: [Param] = []
    private var adam: [Param] = []
    private let wte: Param
    private let wq, wk, wv, wo: Param
    private let wgate, wup, wdown: Param
    private let n1, n1b, n2, n2b, nf: Param
    private let exitW, exitB: Param
    private var loops: [LoopActs] = []
    private var xs: [Buf] = []
    private let zeroPos: Buf
    private let cosTab, sinTab: Buf
    private let qPre, kPre, kRot, vPre, vMix, vFirst: Buf
    private let scores, dprobs, logits, dlogits, nll, lossSum, sumsq: Buf
    private let gx, t1, t2, dln, dq, dk, dv, dk1, dv1, dvPre, dFirst, dav, dsw, dg, du: Buf
    private let tokens: UBuf
    private let targets: UBuf
    private(set) var B = 1
    private(set) var T = 1

    var parameterCount: Int { params.reduce(0) { $0 + $1.count } }

    init(gpu: GPU, cfg: MacroConfig, seed: UInt64) {
        cfg.validate()
        self.gpu = gpu
        self.cfg = cfg
        let device = gpu.device
        let C = cfg.embd
        let F = cfg.intermediate
        let KV = cfg.kvDim
        let maxM = cfg.batch * cfg.block

        var built: [Param] = []
        var muonP: [Param] = []
        var adamP: [Param] = []
        func make(_ name: String, _ rows: Int, _ cols: Int, _ kind: Int) -> Param {
            // kind: 0 muon, 1 adam with decay, 2 adam without decay, 3 frozen
            let p = Param(device: device, name: name, rows: rows, cols: cols,
                          quantized: false, decay: kind == 1, bits: 32, qGroup: 8)
            built.append(p)
            if kind == 0 { muonP.append(p) }
            if kind == 1 || kind == 2 { adamP.append(p) }
            return p
        }

        wte = make("wte", cfg.vocab, C, 1)
        wq = make("wq", C, C, 0)
        wk = make("wk", KV, C, 0)
        wv = make("wv", KV, C, 0)
        wo = make("wo", C, C, 0)
        wgate = make("wgate", F, C, 0)
        wup = make("wup", F, C, 0)
        wdown = make("wdown", C, F, 0)
        n1 = make("n1", C, 1, 2)
        n1b = make("n1b", C, 1, 2)
        n2 = make("n2", C, 1, 2)
        n2b = make("n2b", C, 1, 2)
        nf = make("nf", C, 1, 2)
        exitW = make("exit.w", 1, C, 3)
        exitB = make("exit.b", 1, 1, 3)
        params = built
        muon = muonP
        adam = adamP

        loops = (0..<cfg.loops).map { _ in LoopActs(device: device, batch: cfg.batch, block: cfg.block, cfg: cfg) }
        xs = (0...cfg.loops).map { _ in Buf(device, count: maxM * C) }
        zeroPos = Buf(device, count: cfg.block * C)
        let tables = ropeTables(maxT: cfg.block, dim: cfg.headDim, theta: cfg.ropeTheta)
        cosTab = Buf(device, count: tables.cos.count)
        sinTab = Buf(device, count: tables.sin.count)
        cosTab.upload(tables.cos)
        sinTab.upload(tables.sin)

        qPre = Buf(device, count: maxM * C)
        kPre = Buf(device, count: maxM * KV)
        kRot = Buf(device, count: maxM * KV)
        vPre = Buf(device, count: maxM * KV)
        vMix = Buf(device, count: maxM * KV)
        vFirst = Buf(device, count: maxM * KV)
        let attnN = cfg.batch * cfg.heads * cfg.block * cfg.block
        scores = Buf(device, count: attnN)
        dprobs = Buf(device, count: attnN)
        logits = Buf(device, count: maxM * cfg.vocab)
        dlogits = Buf(device, count: maxM * cfg.vocab)
        nll = Buf(device, count: maxM)
        lossSum = Buf(device, count: 1)
        sumsq = Buf(device, count: max(muon.count + adam.count, 1))
        gx = Buf(device, count: maxM * C)
        t1 = Buf(device, count: maxM * max(C, F))
        t2 = Buf(device, count: maxM * max(C, F))
        dln = Buf(device, count: maxM * C)
        dq = Buf(device, count: maxM * C)
        dk = Buf(device, count: maxM * C)
        dv = Buf(device, count: maxM * C)
        dk1 = Buf(device, count: maxM * KV)
        dv1 = Buf(device, count: maxM * KV)
        dvPre = Buf(device, count: maxM * KV)
        dFirst = Buf(device, count: maxM * KV)
        dav = Buf(device, count: maxM * C)
        dsw = Buf(device, count: maxM * F)
        dg = Buf(device, count: maxM * F)
        du = Buf(device, count: maxM * F)
        tokens = UBuf(device, count: maxM)
        targets = UBuf(device, count: maxM)

        var rng = RNG(seed)
        let std: Float = 0.02
        for p in [wq, wk, wv, wo, wgate, wup, wdown] {
            fillNormal(p.w, std: std, rng: &rng)
        }
        fillNormal(wte.w, std: std, rng: &rng)
        for c in 0..<C { wte.w.p[c] = 0 }
        for p in [n1, n1b, n2, n2b, nf] {
            for i in 0..<p.count { p.w.p[i] = 1 }
        }
        fillNormal(exitW.w, std: std, rng: &rng)
        exitB.w.p[0] = 0
        if cfg.vocab == 378 && cfg.embd == 32 && cfg.intermediate == 48 && cfg.heads == 2 {
            precondition(parameterCount == 19969, "expected 19969 parameters, got \(parameterCount)")
        }
    }

    func step(tokenIds: [UInt32], targetIds: [UInt32], valid: Int, lr: Float, wd: Float, clip: Float,
              stepIndex: Int, b1: Float = 0.9, b2: Float = 0.95, adamEps: Float = 1e-8,
              muonBeta: Float = 0.95) -> Float {
        let loss = lossAndGrad(tokenIds: tokenIds, targetIds: targetIds, valid: valid)
        let trainable = muon + adam
        gpu.begin()
        for (i, p) in trainable.enumerated() {
            gpu.reduceSumsq(p.g, into: sumsq, n: p.count, slot: i)
        }
        gpu.end()
        var ss: Float = 0
        for i in 0..<trainable.count { ss += sumsq.p[i] }
        let norm = sqrt(ss)
        if clip > 0, norm > clip {
            let s = clip / (norm + 1e-6)
            gpu.begin()
            for p in trainable { gpu.scale(p.g, n: p.count, by: s) }
            gpu.end()
        }
        for p in muon {
            muonApply(p, lr: lr, beta: muonBeta, wd: 0)
        }
        let C = cfg.embd
        for c in 0..<C { wte.g.p[c] = 0 }
        let invBC1 = 1 / (1 - pow(b1, Float(stepIndex)))
        let invBC2 = 1 / (1 - pow(b2, Float(stepIndex)))
        gpu.begin()
        for p in adam {
            gpu.adam(w: p.w, g: p.g, m: p.m, v: p.v, n: p.count, lr: lr, b1: b1, b2: b2,
                     eps: adamEps, wd: p.decay ? wd : 0, invBC1: invBC1, invBC2: invBC2)
        }
        gpu.end()
        for c in 0..<C {
            wte.w.p[c] = 0
            wte.m.p[c] = 0
            wte.v.p[c] = 0
        }
        return loss
    }

    func lossAndGrad(tokenIds: [UInt32], targetIds: [UInt32], valid: Int) -> Float {
        bind(tokenIds, targetIds)
        let inv = valid > 0 ? 1 / Float(valid) : 0
        gpu.begin()
        zeroGrads()
        forward(withLoss: true, invCount: inv)
        backward()
        gpu.end()
        return lossSum.p[0] * inv
    }

    func lossOnly(tokenIds: [UInt32], targetIds: [UInt32], valid: Int) -> Float {
        bind(tokenIds, targetIds)
        let inv = valid > 0 ? 1 / Float(valid) : 0
        gpu.begin()
        forward(withLoss: true, invCount: inv)
        gpu.end()
        return lossSum.p[0] * inv
    }

    func nextLogits(_ tokenIds: [UInt32]) -> [Float] {
        precondition(!tokenIds.isEmpty && tokenIds.count <= cfg.block)
        B = 1
        T = tokenIds.count
        tokens.upload(tokenIds)
        gpu.begin()
        forward(withLoss: false, invCount: 1)
        gpu.end()
        let V = cfg.vocab
        let row = (T - 1) * V
        return Array(UnsafeBufferPointer(start: logits.p.advanced(by: row), count: V))
    }

    func save(_ url: URL) throws {
        var data = Data()
        data.append(contentsOf: [0x54, 0x4D, 0x41, 0x43]) // TMAC
        appendMacroU32(1, &data)
        appendMacroU32(UInt32(cfg.vocab), &data)
        appendMacroU32(UInt32(cfg.block), &data)
        appendMacroU32(UInt32(cfg.loops), &data)
        appendMacroU32(UInt32(cfg.heads), &data)
        appendMacroU32(UInt32(cfg.kvHeads), &data)
        appendMacroU32(UInt32(cfg.embd), &data)
        appendMacroU32(UInt32(cfg.intermediate), &data)
        var mix = cfg.valueMix
        withUnsafeBytes(of: &mix) { data.append(contentsOf: $0) }
        for p in params {
            data.append(Data(bytes: p.w.p, count: p.count * MemoryLayout<Float>.size))
        }
        try data.write(to: url)
    }

    static func load(gpu: GPU, url: URL, batch: Int, seed: UInt64) throws -> Macro {
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
        precondition(Array(magic) == [0x54, 0x4D, 0x41, 0x43], "not a MacroStories checkpoint")
        let version = u32()
        precondition(version == 1, "unsupported macro checkpoint \(version)")
        let vocab = Int(u32())
        let block = Int(u32())
        let nLoops = Int(u32())
        let heads = Int(u32())
        let kvHeads = Int(u32())
        let embd = Int(u32())
        let intermediate = Int(u32())
        let mixBits = take(4)
        let mix = mixBits.withUnsafeBytes { $0.load(as: Float.self) }
        let cfg = MacroConfig(vocab: vocab, block: block, batch: batch, loops: nLoops,
                              heads: heads, kvHeads: kvHeads, embd: embd, intermediate: intermediate,
                              valueMix: mix)
        let model = Macro(gpu: gpu, cfg: cfg, seed: seed)
        for p in model.params {
            let nbytes = p.count * MemoryLayout<Float>.size
            let slice = take(nbytes)
            slice.withUnsafeBytes { raw in
                p.w.p.update(from: raw.bindMemory(to: Float.self).baseAddress!, count: p.count)
            }
        }
        precondition(off == data.count, "macro checkpoint length does not match the model")
        return model
    }

    // MARK: - forward / backward

    private func bind(_ tokenIds: [UInt32], _ targetIds: [UInt32]) {
        let m = cfg.batch * cfg.block
        precondition(tokenIds.count == m && targetIds.count == m)
        B = cfg.batch
        T = cfg.block
        tokens.upload(tokenIds)
        targets.upload(targetIds)
    }

    private func forward(withLoss: Bool, invCount: Float) {
        let M = B * T
        let C = cfg.embd
        let F = cfg.intermediate
        let H = cfg.heads
        let D = cfg.headDim
        let KV = cfg.kvDim
        let V = cfg.vocab
        let mix = cfg.valueMix
        gpu.embed(bits: 32, tokens: tokens, wte: wte.w, q: nil, scale: nil, wpe: zeroPos,
                  y: xs[0], B: B, T: T, C: C, qGroup: 8)
        for p in 0..<cfg.loops {
            let L = loops[p]
            gpu.rmsFwd(x: xs[p], y: L.ln1, rstd: L.r1, weight: n1.w, rows: M, cols: C, eps: cfg.eps)
            gemm(wq, L.ln1, qPre, M: M, N: C, K: C)
            gemm(wk, L.ln1, kPre, M: M, N: KV, K: C)
            gemm(wv, L.ln1, vPre, M: M, N: KV, K: C)
            let kvN = M * KV
            if p == 0 {
                gpu.copy(vPre, vFirst, n: kvN)
                gpu.repeatHeads(src: vPre, dst: L.v, B: B, T: T, H: H, D: D)
            } else {
                gpu.mix(vPre, vFirst, vMix, n: kvN, ca: 1 - mix, cb: mix)
                gpu.repeatHeads(src: vMix, dst: L.v, B: B, T: T, H: H, D: D)
            }
            gpu.ropeFwd(x: qPre, y: L.q, cos: cosTab, sin: sinTab, B: B, T: T, H: H, D: D)
            gpu.ropeFwd(x: kPre, y: kRot, cos: cosTab, sin: sinTab, B: B, T: T, H: 1, D: D)
            gpu.repeatHeads(src: kRot, dst: L.k, B: B, T: T, H: H, D: D)
            gpu.attnFwd(q: L.q, k: L.k, v: L.v, scores: scores, probs: L.probs, av: L.av,
                        B: B, T: T, H: H, D: D)
            gemm(wo, L.av, L.attn, M: M, N: C, K: C)
            gpu.rmsFwd(x: L.attn, y: t1, rstd: L.r1b, weight: n1b.w, rows: M, cols: C, eps: cfg.eps)
            gpu.add(xs[p], t1, L.mid, n: M * C)
            gpu.rmsFwd(x: L.mid, y: L.ln2, rstd: L.r2, weight: n2.w, rows: M, cols: C, eps: cfg.eps)
            gemm(wgate, L.ln2, L.gate, M: M, N: F, K: C)
            gemm(wup, L.ln2, L.up, M: M, N: F, K: C)
            gpu.swigluFwd(gate: L.gate, up: L.up, y: L.sw, n: M * F)
            gemm(wdown, L.sw, L.mlp, M: M, N: C, K: F)
            gpu.rmsFwd(x: L.mlp, y: t1, rstd: L.r2b, weight: n2b.w, rows: M, cols: C, eps: cfg.eps)
            gpu.add(L.mid, t1, L.pre, n: M * C)
            gpu.rmsFwd(x: L.pre, y: xs[p + 1], rstd: L.rf, weight: nf.w, rows: M, cols: C, eps: cfg.eps)
        }
        gemm(wte, xs[cfg.loops], logits, M: M, N: V, K: C)
        if withLoss {
            gpu.lossMasked(logits: logits, targets: targets, nll: nll, dlogits: dlogits,
                           BT: M, V: V, ignore: MacroVocab.pad, invCount: invCount)
            gpu.reduceSum(nll, into: lossSum, n: M, slot: 0)
        }
    }

    private func backward() {
        let M = B * T
        let C = cfg.embd
        let F = cfg.intermediate
        let H = cfg.heads
        let D = cfg.headDim
        let KV = cfg.kvDim
        let V = cfg.vocab
        let mix = cfg.valueMix
        let nC = M * C
        let nKV = M * KV
        gpu.zero(dFirst, n: nKV)
        gpu.gemmDW(dy: dlogits, x: xs[cfg.loops], dw: wte.g, M: M, N: V, K: C)
        gpu.gemmDX(bits: 32, w: wte.w, q: nil, scale: nil, dy: dlogits, dx: gx,
                   M: M, N: V, K: C, qGroup: 8)

        for p in stride(from: cfg.loops - 1, through: 0, by: -1) {
            let L = loops[p]
            gpu.rmsBwd(x: L.pre, rstd: L.rf, weight: nf.w, dy: gx, dx: t1, dweight: nf.g,
                       rows: M, cols: C, eps: cfg.eps)
            gpu.rmsBwd(x: L.mlp, rstd: L.r2b, weight: n2b.w, dy: t1, dx: t2, dweight: n2b.g,
                       rows: M, cols: C, eps: cfg.eps)
            gpu.gemmDWAcc(dy: t2, x: L.sw, dw: wdown.g, M: M, N: C, K: F)
            gpu.gemmDX(bits: 32, w: wdown.w, q: nil, scale: nil, dy: t2, dx: dsw,
                       M: M, N: C, K: F, qGroup: 8)
            gpu.swigluBwd(gate: L.gate, up: L.up, dy: dsw, dgate: dg, dup: du, n: M * F)
            gpu.gemmDWAcc(dy: dg, x: L.ln2, dw: wgate.g, M: M, N: F, K: C)
            gpu.gemmDX(bits: 32, w: wgate.w, q: nil, scale: nil, dy: dg, dx: t2,
                       M: M, N: F, K: C, qGroup: 8)
            gpu.gemmDWAcc(dy: du, x: L.ln2, dw: wup.g, M: M, N: F, K: C)
            gpu.gemmDX(bits: 32, w: wup.w, q: nil, scale: nil, dy: du, dx: dln,
                       M: M, N: F, K: C, qGroup: 8)
            gpu.acc(t2, dln, n: nC)
            gpu.rmsBwd(x: L.mid, rstd: L.r2, weight: n2.w, dy: t2, dx: dln, dweight: n2.g,
                       rows: M, cols: C, eps: cfg.eps)
            gpu.acc(t1, dln, n: nC)

            gpu.rmsBwd(x: L.attn, rstd: L.r1b, weight: n1b.w, dy: t1, dx: t2, dweight: n1b.g,
                       rows: M, cols: C, eps: cfg.eps)
            gpu.gemmDWAcc(dy: t2, x: L.av, dw: wo.g, M: M, N: C, K: C)
            gpu.gemmDX(bits: 32, w: wo.w, q: nil, scale: nil, dy: t2, dx: dav,
                       M: M, N: C, K: C, qGroup: 8)
            gpu.attnBwd(q: L.q, k: L.k, v: L.v, probs: L.probs, dout: dav,
                        dq: dq, dk: dk, dv: dv, dprobs: dprobs, dscores: scores,
                        B: B, T: T, H: H, D: D)
            gpu.sumHeads(src: dk, dst: dk1, B: B, T: T, H: H, D: D)
            gpu.sumHeads(src: dv, dst: dv1, B: B, T: T, H: H, D: D)
            gpu.ropeBwd(dy: dq, dx: qPre, cos: cosTab, sin: sinTab, B: B, T: T, H: H, D: D)
            gpu.ropeBwd(dy: dk1, dx: kPre, cos: cosTab, sin: sinTab, B: B, T: T, H: 1, D: D)
            if p == 0 {
                gpu.copy(dv1, dvPre, n: nKV)
                gpu.acc(dvPre, dFirst, n: nKV)
            } else {
                gpu.mix(dv1, dv1, dvPre, n: nKV, ca: 1 - mix, cb: 0)
                gpu.accScale(dFirst, dv1, n: nKV, by: mix)
            }

            gpu.gemmDWAcc(dy: qPre, x: L.ln1, dw: wq.g, M: M, N: C, K: C)
            gpu.gemmDX(bits: 32, w: wq.w, q: nil, scale: nil, dy: qPre, dx: t2,
                       M: M, N: C, K: C, qGroup: 8)
            gpu.gemmDWAcc(dy: kPre, x: L.ln1, dw: wk.g, M: M, N: KV, K: C)
            gpu.gemmDX(bits: 32, w: wk.w, q: nil, scale: nil, dy: kPre, dx: dln,
                       M: M, N: KV, K: C, qGroup: 8)
            gpu.acc(t2, dln, n: nC)
            gpu.gemmDWAcc(dy: dvPre, x: L.ln1, dw: wv.g, M: M, N: KV, K: C)
            gpu.gemmDX(bits: 32, w: wv.w, q: nil, scale: nil, dy: dvPre, dx: dln,
                       M: M, N: KV, K: C, qGroup: 8)
            gpu.acc(t2, dln, n: nC)
            gpu.rmsBwd(x: xs[p], rstd: L.r1, weight: n1.w, dy: t2, dx: dln, dweight: n1.g,
                       rows: M, cols: C, eps: cfg.eps)
            gpu.acc(t1, dln, n: nC)
            gpu.copy(t1, gx, n: nC)
        }
        gpu.embedBwd(tokens: tokens, dx: gx, dwe: wte.g, B: B, T: T, C: C, V: V)
    }

    private func gemm(_ w: Param, _ x: Buf, _ y: Buf, M: Int, N: Int, K: Int) {
        gpu.gemmNT(bits: 32, w: w.w, q: nil, scale: nil, x: x, bias: nil, y: y,
                   M: M, N: N, K: K, qGroup: 8)
    }

    private func zeroGrads() {
        for p in params { gpu.zero(p.g, n: p.count) }
    }
}

func ropeTables(maxT: Int, dim: Int, theta: Float) -> (cos: [Float], sin: [Float]) {
    let half = dim / 2
    var inv = [Float](repeating: 0, count: half)
    for i in 0..<half {
        let exponent = Float(2 * i) / Float(dim)
        inv[i] = pow(theta, -exponent)
    }
    var cos = [Float](repeating: 0, count: maxT * dim)
    var sin = [Float](repeating: 0, count: maxT * dim)
    for t in 0..<maxT {
        for i in 0..<half {
            let freq = Float(t) * inv[i]
            let c = Darwin.cos(freq)
            let s = Darwin.sin(freq)
            cos[t * dim + i] = c
            cos[t * dim + i + half] = c
            sin[t * dim + i] = s
            sin[t * dim + i + half] = s
        }
    }
    return (cos, sin)
}

private func muonApply(_ p: Param, lr: Float, beta: Float, wd: Float) {
    let n = p.count
    let rows = p.rows
    let cols = p.cols
    var upd = [Float](repeating: 0, count: n)
    let oneMinus = 1 - beta
    for i in 0..<n {
        let g = p.g.p[i]
        let m = beta * p.m.p[i] + oneMinus * g
        p.m.p[i] = m
        upd[i] = oneMinus * g + beta * m
    }
    let orth = newtonSchulz(upd, rows: rows, cols: cols, steps: 5)
    let scale = sqrt(max(1, Float(rows) / Float(cols)))
    let decay = 1 - lr * wd
    for i in 0..<n {
        p.w.p[i] = p.w.p[i] * decay - lr * scale * orth[i]
    }
}

private func newtonSchulz(_ G: [Float], rows: Int, cols: Int, steps: Int) -> [Float] {
    let a: Float = 3.4445
    let b: Float = -4.7750
    let c: Float = 2.0315
    var transposed = false
    var X = G
    var r = rows
    var col = cols
    if rows > cols {
        X = transpose(X, rows: rows, cols: cols)
        transposed = true
        r = cols
        col = rows
    }
    var nrm: Float = 0
    for v in X { nrm += v * v }
    nrm = sqrt(nrm) + 1e-7
    for i in 0..<X.count { X[i] /= nrm }
    for _ in 0..<steps {
        var A = [Float](repeating: 0, count: r * r)
        for i in 0..<r {
            for j in 0..<r {
                var s: Float = 0
                let xi = i * col
                let xj = j * col
                for k in 0..<col { s += X[xi + k] * X[xj + k] }
                A[i * r + j] = s
            }
        }
        var A2 = [Float](repeating: 0, count: r * r)
        for i in 0..<r {
            for j in 0..<r {
                var s: Float = 0
                for k in 0..<r { s += A[i * r + k] * A[k * r + j] }
                A2[i * r + j] = s
            }
        }
        var Bmat = [Float](repeating: 0, count: r * r)
        for i in 0..<A.count { Bmat[i] = b * A[i] + c * A2[i] }
        var Y = [Float](repeating: 0, count: r * col)
        for i in 0..<r {
            for j in 0..<col {
                var s = a * X[i * col + j]
                for k in 0..<r { s += Bmat[i * r + k] * X[k * col + j] }
                Y[i * col + j] = s
            }
        }
        X = Y
    }
    if transposed { X = transpose(X, rows: r, cols: col) }
    return X
}

private func transpose(_ a: [Float], rows: Int, cols: Int) -> [Float] {
    var b = [Float](repeating: 0, count: rows * cols)
    for i in 0..<rows {
        for j in 0..<cols {
            b[j * rows + i] = a[i * cols + j]
        }
    }
    return b
}

func macroLearningRate(step: Int, total: Int, base: Float) -> Float {
    let warmup = max(Int((0.05 * Float(total)).rounded()), 1)
    if step <= warmup {
        return base * Float(step) / Float(max(warmup, 1))
    }
    let span = max(total - warmup, 1)
    let progress = min(max(Float(step - 1 - warmup) / Float(span), 0), 1)
    return base * 0.5 * (1 + Darwin.cos(Float.pi * progress))
}

func trainMacro(gpu: GPU, args: Args, words: WordList? = nil,
                dataDefault: String = "data/macro-distill.bin",
                ckptDefault: String = "build/macro.ckpt",
                title: String = "MacroStories",
                blockDefault: Int = 384) throws {
    let bits = args.int("bits", 32)
    if bits != 32 {
        fputs("\(title) trains in fp32.\n", stderr)
        exit(1)
    }
    let batch = args.int("batch", 32)
    let block = args.int("block", blockDefault)
    let loops = args.int("loops", 4)
    let steps = args.int("steps", 4000)
    let lr = args.float("lr", 0.003)
    let wd = args.float("wd", 0)
    let clip = args.float("clip", 0)
    let seed = UInt64(args.int("seed", 456))
    let dataSeed = UInt64(args.int("data-seed", 42))
    let ckptPath = args.str("ckpt", ckptDefault)
    let dataPath = args.str("data", dataDefault)
    let temp = args.float("temp", 0.5)
    let topP = args.float("top-p", 0.9)
    var cfg = MacroConfig(block: block, batch: batch, loops: loops)
    if let words {
        cfg.vocab = words.tokens.count
    }
    cfg.validate()
    let model = Macro(gpu: gpu, cfg: cfg, seed: seed)
    print("device \(gpu.device.name)")
    print("\(title)  params \(model.parameterCount)  vocab \(cfg.vocab)  embd \(cfg.embd)  ffn \(cfg.intermediate)")
    print("loops \(cfg.loops)  heads \(cfg.heads)/\(cfg.kvHeads)  block \(block)  batch \(batch)  steps \(steps)")
    print(String(format: "lr %.4f  cosine warmup 5%%  muon on decoder matrices  adamw on the rest  seed %llu", lr, seed))
    print(String(format: "uniform-guess loss %.3f", log(Float(cfg.vocab))))
    let corpus = loadMacroCorpus(dataPath)
    let longest = corpus.train.map(\.count).max() ?? 0
    print("corpus \(dataPath)  train \(corpus.train.count)  val \(corpus.val.count)  longest \(longest) tokens")
    var rng = RNG(dataSeed)
    var valRNG = RNG(dataSeed &+ 999)
    let preview = corpus.train[0]
    let previewText = words?.decode(preview) ?? MacroVocab.decode(preview)
    print("sample training story (\(preview.count) tokens):\n\(previewText)\n")
    let t0 = CFAbsoluteTimeGetCurrent()
    for step in 1...steps {
        let sched = macroLearningRate(step: step, total: steps, base: lr)
        let (x, y, valid) = macroCorpusBatch(corpus.train, batch: batch, block: block, rng: &rng)
        let loss = model.step(tokenIds: x, targetIds: y, valid: valid, lr: sched, wd: wd, clip: clip, stepIndex: step)
        if !loss.isFinite {
            fputs("loss became non-finite.\n", stderr)
            exit(1)
        }
        if step == 1 || step % 20 == 0 || step == steps {
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let toks = Double(step * batch * block) / max(dt, 1e-6)
            var line = String(format: "step %4d/%d  loss %.4f  lr %.2e  %.0f tok/s", step, steps, loss, sched, toks)
            if step % 100 == 0 || step == steps {
                let (vx, vy, vvalid) = macroCorpusBatch(corpus.val, batch: batch, block: block, rng: &valRNG)
                let vloss = model.lossOnly(tokenIds: vx, targetIds: vy, valid: vvalid)
                line += String(format: "  val %.4f", vloss)
            }
            print(line)
            fflush(stdout)
        }
        if step % 500 == 0 || step == steps {
            var sampleRNG = RNG(seed &+ UInt64(step))
            let text = generateMacro(model: model, tokens: 640, temp: temp, topP: topP, rng: &sampleRNG, words: words)
            print("---\n\(text)\n---")
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

func generateMacro(model: Macro, tokens: Int, temp: Float, topP: Float, rng: inout RNG,
                    prompt: [UInt32] = [], words: WordList? = nil) -> String {
    let bos = words?.bos ?? MacroVocab.bos
    let eos = words?.eos ?? MacroVocab.eos
    let pad = words?.pad ?? MacroVocab.pad
    let unk = words?.unk ?? MacroVocab.unk
    var ids: [UInt32] = prompt.isEmpty ? [bos] : prompt
    if ids.last == eos { return words?.decode(ids) ?? MacroVocab.decode(ids) }
    let ban: Set<Int> = [Int(pad), Int(unk), Int(bos)]
    for _ in 0..<tokens {
        let window = Array(ids.suffix(model.cfg.block))
        let logits = model.nextLogits(window)
        let next = sampleNucleus(logits, temp: temp, topP: topP, rng: &rng, ban: ban)
        ids.append(next)
        if next == eos { break }
    }
    return words?.decode(ids) ?? MacroVocab.decode(ids)
}

func sampleNucleus(_ logits: [Float], temp: Float, topP: Float, rng: inout RNG, ban: Set<Int>) -> UInt32 {
    var z = logits
    for i in ban where i >= 0 && i < z.count { z[i] = -1e30 }
    if temp <= 0 {
        var best = 0
        var bestV = -Float.infinity
        for i in 0..<z.count where z[i] > bestV {
            bestV = z[i]
            best = i
        }
        return UInt32(best)
    }
    var m = -Float.infinity
    for v in z { m = max(m, v) }
    var probs = z.map { exp(($0 - m) / max(temp, 1e-6)) }
    let sum = probs.reduce(0, +)
    let inv = sum > 0 ? 1 / sum : 0
    for i in 0..<probs.count { probs[i] *= inv }
    let order = (0..<probs.count).sorted { probs[$0] > probs[$1] }
    var keep: [Int] = []
    var cum: Float = 0
    let nucleus = min(max(topP, 0), 1)
    for i in order {
        keep.append(i)
        cum += probs[i]
        if cum >= nucleus && !keep.isEmpty { break }
    }
    if keep.isEmpty { return 0 }
    var mass: Float = 0
    for i in keep { mass += probs[i] }
    var u = rng.uniform() * mass
    for i in keep {
        u -= probs[i]
        if u <= 0 { return UInt32(i) }
    }
    return UInt32(keep[keep.count - 1])
}

struct MacroMem {
    var initial: Float
    var final: Float
}

func macroMemorize(gpu: GPU, steps: Int, lr: Float) -> MacroMem {
    let story = macroShortStory()
    let block = story.count - 1
    let batch = 2
    var x = [UInt32](repeating: 0, count: batch * block)
    var y = [UInt32](repeating: 0, count: batch * block)
    for b in 0..<batch {
        for t in 0..<block {
            x[b * block + t] = story[t]
            y[b * block + t] = story[t + 1]
        }
    }
    let cfg = MacroConfig(block: block, batch: batch, loops: 4)
    let model = Macro(gpu: gpu, cfg: cfg, seed: 456)
    var initial: Float = 0
    var final: Float = 0
    let valid = batch * block
    for step in 1...steps {
        let loss = model.step(tokenIds: x, targetIds: y, valid: valid, lr: lr, wd: 0, clip: 0, stepIndex: step)
        if step == 1 { initial = loss }
        final = loss
    }
    return MacroMem(initial: initial, final: final)
}

func macroGradCheck(gpu: GPU, _ r: inout Report) {
    let story = macroShortStory()
    let block = story.count - 1
    var x = [UInt32](repeating: 0, count: block)
    var y = [UInt32](repeating: 0, count: block)
    for t in 0..<block {
        x[t] = story[t]
        y[t] = story[t + 1]
    }
    let model = Macro(gpu: gpu, cfg: MacroConfig(block: block, batch: 1, loops: 4), seed: 7)
    r.expect(model.parameterCount == 19969, "parameter count \(model.parameterCount)")
    let probes: [(String, Param, Int)] = [
        ("wq", model.probe("wq"), 0),
        ("n1", model.probe("n1"), 0),
        ("wdown", model.probe("wdown"), 0),
        ("wte", model.probe("wte"), Int(MacroVocab.id("\u{0120}flower")) * model.cfg.embd),
    ]
    _ = model.lossAndGrad(tokenIds: x, targetIds: y, valid: block)
    var analytic = [Float](repeating: 0, count: probes.count)
    for (i, probe) in probes.enumerated() {
        analytic[i] = probe.1.g.p[probe.2]
    }
    let eps: Float = 1e-3
    for (i, probe) in probes.enumerated() {
        let p = probe.1
        let index = probe.2
        let w0 = p.w.p[index]
        p.w.p[index] = w0 + eps
        let higher = model.lossOnly(tokenIds: x, targetIds: y, valid: block)
        p.w.p[index] = w0 - eps
        let lower = model.lossOnly(tokenIds: x, targetIds: y, valid: block)
        p.w.p[index] = w0
        let numeric = (higher - lower) / (2 * eps)
        let scale = max(abs(numeric), abs(analytic[i]), 1e-3)
        let ok = abs(numeric - analytic[i]) / scale < 0.08
        r.expect(ok, String(format: "\(probe.0) grad  numeric %.5f  analytic %.5f", numeric, analytic[i]))
    }
}

private func appendMacroU32(_ v: UInt32, _ data: inout Data) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}

private extension Macro {
    func probe(_ name: String) -> Param {
        guard let p = params.first(where: { $0.name == name }) else {
            fatalError("missing parameter \(name)")
        }
        return p
    }
}
