import Darwin
import Foundation
import Metal

/// Looped flow-matching transformer. The block matches the diffusion model.
/// The path is a straight line from noise to the photograph:
/// x_t = (1 - t) * noise + t * photo. The head predicts the photograph.
/// Each Euler step walks toward that prediction,
/// x ← x + (photo − x) / (steps remaining), which is the straight-line velocity
/// (photo − x_t) / (1 − t). A velocity target would have to cancel the exact
/// noise in a 768-number patch, and the 192-wide block does not keep it.
struct FlowConfig {
    var side = Pictures.side
    var patch = Pictures.patch
    var names: Int
    var batch: Int
    var loops = 4
    var heads = 4
    var kvHeads = 1
    var embd = 64
    var intermediate = 128
    var freqDim = 64
    var eps: Float = 1e-6
    var ropeTheta: Float = 10000
    var valueMix: Float = 0.5

    var grid: Int { side / patch }
    var tokens: Int { grid * grid }
    var patchDim: Int { patch * patch * 3 }
    var headDim: Int { embd / heads }
    var kvDim: Int { kvHeads * headDim }

    func validate() {
        precondition(side > 0 && patch > 0 && side % patch == 0)
        precondition(names > 0 && batch > 0 && loops > 0)
        precondition(heads > 0 && kvHeads == 1, "this build repeats a single KV head")
        precondition(embd % heads == 0 && headDim % 2 == 0)
        precondition(freqDim >= 2 && freqDim % 2 == 0 && intermediate > 0)
    }
}

private final class FlowActs {
    let r1, r1b, r2, r2b, rf: Buf
    let ln1, q, k, v, probs, av, attn, mid, ln2, gate, up, sw, mlp, pre: Buf

    init(device: MTLDevice, batch: Int, block: Int, cfg: FlowConfig) {
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

final class Flow {
    let gpu: GPU
    let cfg: FlowConfig
    private(set) var params: [Param] = []
    private var muon: [Param] = []
    private var adam: [Param] = []
    private let wPatch, wHead, wName, wTime1, wTime2: Param
    private let wq, wk, wv, wo: Param
    private let wgate, wup, wdown: Param
    private let n1, n1b, n2, n2b, nf: Param
    private var acts: [FlowActs] = []
    private var xs: [Buf] = []
    private let cosTab, sinTab: Buf
    private let qPre, kPre, kRot, vPre, vMix, vFirst: Buf
    private let scores, dprobs: Buf
    private let xt, velocity, patchH, pred, dpred, err, lossSum, sumsq: Buf
    private let timeIn, timePre, timeHid, timeEmb, nameEmb, cond, dcond, dTimeHid, dTimePre: Buf
    private let gx, t1, t2, dln, dq, dk, dv, dk1, dv1, dvPre, dFirst, dav, dsw, dg, du: Buf
    private var nameIDs: [Int] = []
    private(set) var B = 1
    private var T = 1

    var parameterCount: Int { params.reduce(0) { $0 + $1.count } }

    init(gpu: GPU, cfg: FlowConfig, seed: UInt64) {
        cfg.validate()
        self.gpu = gpu
        self.cfg = cfg
        let device = gpu.device
        let C = cfg.embd
        let F = cfg.intermediate
        let KV = cfg.kvDim
        let P = cfg.patchDim
        let freq = cfg.freqDim
        let maxM = cfg.batch * cfg.tokens
        let maxB = cfg.batch

        var built: [Param] = []
        var muonP: [Param] = []
        var adamP: [Param] = []
        func make(_ name: String, _ rows: Int, _ cols: Int, _ kind: Int) -> Param {
            let p = Param(device: device, name: name, rows: rows, cols: cols,
                          quantized: false, decay: kind == 1, bits: 32, qGroup: 8)
            built.append(p)
            if kind == 0 { muonP.append(p) }
            if kind == 1 || kind == 2 { adamP.append(p) }
            return p
        }

        wPatch = make("wpatch", C, P, 0)
        wHead = make("whead", P, C, 0)
        wName = make("wname", cfg.names, C, 1)
        wTime1 = make("wtime1", C, freq, 0)
        wTime2 = make("wtime2", C, C, 0)
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
        params = built
        muon = muonP
        adam = adamP

        acts = (0..<cfg.loops).map { _ in FlowActs(device: device, batch: cfg.batch, block: cfg.tokens, cfg: cfg) }
        xs = (0...cfg.loops).map { _ in Buf(device, count: maxM * C) }
        let tables = ropeTables(maxT: cfg.tokens, dim: cfg.headDim, theta: cfg.ropeTheta)
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
        let attnN = cfg.batch * cfg.heads * cfg.tokens * cfg.tokens
        scores = Buf(device, count: attnN)
        dprobs = Buf(device, count: attnN)
        xt = Buf(device, count: maxM * P)
        velocity = Buf(device, count: maxM * P)
        patchH = Buf(device, count: maxM * C)
        pred = Buf(device, count: maxM * P)
        dpred = Buf(device, count: maxM * P)
        err = Buf(device, count: maxM * P)
        lossSum = Buf(device, count: 1)
        sumsq = Buf(device, count: max(muon.count + adam.count, 1))
        timeIn = Buf(device, count: maxB * freq)
        timePre = Buf(device, count: maxB * C)
        timeHid = Buf(device, count: maxB * C)
        timeEmb = Buf(device, count: maxB * C)
        nameEmb = Buf(device, count: maxB * C)
        cond = Buf(device, count: maxB * C)
        dcond = Buf(device, count: maxB * C)
        dTimeHid = Buf(device, count: maxB * C)
        dTimePre = Buf(device, count: maxB * C)
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

        var rng = RNG(seed)
        let std: Float = 0.02
        for p in [wPatch, wHead, wTime1, wTime2, wq, wk, wv, wo, wgate, wup, wdown] {
            fillNormal(p.w, std: std, rng: &rng)
        }
        fillNormal(wName.w, std: std, rng: &rng)
        for p in [n1, n1b, n2, n2b, nf] {
            for i in 0..<p.count { p.w.p[i] = 1 }
        }
        B = cfg.batch
        T = cfg.tokens
    }

    func step(xt: [Float], velocity target: [Float], times: [Float], names: [Int], lr: Float, wd: Float, clip: Float,
              stepIndex: Int, b1: Float = 0.9, b2: Float = 0.95, adamEps: Float = 1e-8,
              muonBeta: Float = 0.95) -> Float {
        let loss = lossAndGrad(xt: xt, velocity: target, times: times, names: names)
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
        for p in muon { muonApply(p, lr: lr, beta: muonBeta, wd: 0) }
        let invBC1 = 1 / (1 - pow(b1, Float(stepIndex)))
        let invBC2 = 1 / (1 - pow(b2, Float(stepIndex)))
        gpu.begin()
        for p in adam {
            gpu.adam(w: p.w, g: p.g, m: p.m, v: p.v, n: p.count, lr: lr, b1: b1, b2: b2,
                     eps: adamEps, wd: p.decay ? wd : 0, invBC1: invBC1, invBC2: invBC2)
        }
        gpu.end()
        return loss
    }

    @discardableResult
    func lossAndGrad(xt noisy: [Float], velocity target: [Float], times: [Float], names: [Int]) -> Float {
        bind(noisy, target, times, names)
        let inv = 1 / Float(B * T * cfg.patchDim)
        gpu.begin()
        zeroGrads()
        forward(withLoss: true, invCount: inv)
        backward()
        gpu.end()
        scatterName()
        return lossSum.p[0] * inv
    }

    func lossOnly(xt noisy: [Float], velocity target: [Float], times: [Float], names: [Int]) -> Float {
        bind(noisy, target, times, names)
        let inv = 1 / Float(B * T * cfg.patchDim)
        gpu.begin()
        forward(withLoss: true, invCount: inv)
        gpu.end()
        return lossSum.p[0] * inv
    }

    /// States along the path, one per Euler step, plus the four loop readouts
    /// from the first step. Each readout is the photograph that pass predicted.
    func generate(name: Int, sampleSteps: Int, rng: inout RNG) -> (frames: [[Float]], loops: [[Float]]) {
        let n = cfg.tokens * cfg.patchDim
        let steps = max(sampleSteps, 1)
        let dt = 1 / Float(steps)
        var x = (0..<n).map { _ in rng.normal() }
        var frames: [[Float]] = []
        var loopFrames: [[Float]] = []
        for i in 0..<steps {
            let t = Float(i) * dt
            let pack = predict(x, time: t, name: name, loops: i == 0)
            let hat = pack.guess.map { min(max($0, -1), 1) }
            if i == 0 {
                loopFrames = pack.loops.map { readout in
                    let clamped = readout.map { min(max($0, -1), 1) }
                    return unpatchImage(clamped, side: cfg.side, patch: cfg.patch)
                }
            }
            let remain = max(1 - t, dt)
            let jump = dt / remain
            for j in 0..<n { x[j] += (hat[j] - x[j]) * jump }
            frames.append(unpatchImage(x, side: cfg.side, patch: cfg.patch))
        }
        return (frames, loopFrames)
    }

    func save(_ url: URL) throws {
        var data = Data()
        data.append(contentsOf: [0x54, 0x46, 0x4C, 0x4F]) // TFLO
        appendFlowU32(1, &data)
        for v in [cfg.side, cfg.patch, cfg.names, cfg.loops, cfg.heads, cfg.kvHeads,
                  cfg.embd, cfg.intermediate, cfg.freqDim] {
            appendFlowU32(UInt32(v), &data)
        }
        var mix = cfg.valueMix
        withUnsafeBytes(of: &mix) { data.append(contentsOf: $0) }
        for p in params {
            data.append(Data(bytes: p.w.p, count: p.count * MemoryLayout<Float>.size))
        }
        try data.write(to: url)
    }

    static func load(gpu: GPU, url: URL, batch: Int, seed: UInt64) throws -> Flow {
        let data = try Data(contentsOf: url)
        var off = 0
        func take(_ n: Int) -> Data {
            let s = data.subdata(in: off..<(off + n))
            off += n
            return s
        }
        func u32() -> Int { Int(take(4).withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }) }
        let magic = take(4)
        precondition(Array(magic) == [0x54, 0x46, 0x4C, 0x4F], "not a flow checkpoint")
        let version = u32()
        precondition(version == 1, "unsupported flow checkpoint \(version)")
        let side = u32()
        let patch = u32()
        let names = u32()
        let loops = u32()
        let heads = u32()
        let kvHeads = u32()
        let embd = u32()
        let intermediate = u32()
        let freq = u32()
        let mix = take(4).withUnsafeBytes { $0.load(as: Float.self) }
        var cfg = FlowConfig(names: names, batch: batch, loops: loops, heads: heads, kvHeads: kvHeads,
                             embd: embd, intermediate: intermediate, freqDim: freq, valueMix: mix)
        cfg.side = side
        cfg.patch = patch
        let model = Flow(gpu: gpu, cfg: cfg, seed: seed)
        for p in model.params {
            let nbytes = p.count * MemoryLayout<Float>.size
            let slice = take(nbytes)
            slice.withUnsafeBytes { raw in
                p.w.p.update(from: raw.bindMemory(to: Float.self).baseAddress!, count: p.count)
            }
        }
        precondition(off == data.count, "flow checkpoint length does not match the model")
        return model
    }

    private func bind(_ noisy: [Float], _ target: [Float], _ times: [Float], _ names: [Int]) {
        precondition(names.count > 0 && names.count <= cfg.batch)
        precondition(times.count == names.count)
        B = names.count
        T = cfg.tokens
        let n = B * T * cfg.patchDim
        precondition(noisy.count == n && target.count == n)
        nameIDs = names
        xt.upload(noisy)
        velocity.upload(target)
        var wave: [Float] = []
        wave.reserveCapacity(B * cfg.freqDim)
        for t in times { wave.append(contentsOf: flowWave(t, dim: cfg.freqDim)) }
        timeIn.upload(wave)
        stageNameEmbed()
    }

    private func stageNameEmbed() {
        let C = cfg.embd
        for b in 0..<B {
            let src = nameIDs[b] * C
            let dst = b * C
            for c in 0..<C { nameEmb.p[dst + c] = wName.w.p[src + c] }
        }
    }

    private func scatterName() {
        let C = cfg.embd
        for b in 0..<B {
            let row = nameIDs[b] * C
            let src = b * C
            for c in 0..<C { wName.g.p[row + c] += dcond.p[src + c] }
        }
    }

    private func forward(withLoss: Bool, invCount: Float) {
        let M = B * T
        let C = cfg.embd
        let F = cfg.intermediate
        let H = cfg.heads
        let D = cfg.headDim
        let KV = cfg.kvDim
        let P = cfg.patchDim
        let mix = cfg.valueMix
        gemm(wTime1, timeIn, timePre, M: B, N: C, K: cfg.freqDim)
        gpu.geluFwd(timePre, timeHid, n: B * C)
        gemm(wTime2, timeHid, timeEmb, M: B, N: C, K: C)
        gpu.add(nameEmb, timeEmb, cond, n: B * C)
        gemm(wPatch, xt, patchH, M: M, N: C, K: P)
        gpu.addCond(x: patchH, cond: cond, y: xs[0], B: B, T: T, C: C)
        for p in 0..<cfg.loops {
            let L = acts[p]
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
                        B: B, T: T, H: H, D: D, causal: false)
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
        gemm(wHead, xs[cfg.loops], pred, M: M, N: P, K: C)
        if withLoss {
            gpu.mse(pred: pred, target: velocity, err: err, dpred: dpred, n: M * P, invCount: invCount)
            gpu.reduceSum(err, into: lossSum, n: M * P, slot: 0)
        }
    }

    private func backward() {
        let M = B * T
        let C = cfg.embd
        let F = cfg.intermediate
        let H = cfg.heads
        let D = cfg.headDim
        let KV = cfg.kvDim
        let P = cfg.patchDim
        let mix = cfg.valueMix
        let nC = M * C
        let nKV = M * KV
        gpu.zero(dFirst, n: nKV)
        gpu.gemmDWAcc(dy: dpred, x: xs[cfg.loops], dw: wHead.g, M: M, N: P, K: C)
        gpu.gemmDX(bits: 32, w: wHead.w, q: nil, scale: nil, dy: dpred, dx: gx,
                   M: M, N: P, K: C, qGroup: 8)
        for p in stride(from: cfg.loops - 1, through: 0, by: -1) {
            let L = acts[p]
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
        gpu.sumCond(dy: gx, dcond: dcond, B: B, T: T, C: C)
        gpu.gemmDWAcc(dy: gx, x: xt, dw: wPatch.g, M: M, N: C, K: P)
        gpu.gemmDWAcc(dy: dcond, x: timeHid, dw: wTime2.g, M: B, N: C, K: C)
        gpu.gemmDX(bits: 32, w: wTime2.w, q: nil, scale: nil, dy: dcond, dx: dTimeHid,
                   M: B, N: C, K: C, qGroup: 8)
        gpu.geluBwd(x: timePre, dy: dTimeHid, dx: dTimePre, n: B * C)
        gpu.gemmDWAcc(dy: dTimePre, x: timeIn, dw: wTime1.g, M: B, N: C, K: cfg.freqDim)
    }

    private func predict(_ noisy: [Float], time: Float, name: Int, loops wantLoops: Bool) -> (guess: [Float], loops: [[Float]]) {
        let n = cfg.tokens * cfg.patchDim
        precondition(noisy.count == n)
        B = 1
        T = cfg.tokens
        nameIDs = [name]
        xt.upload(noisy)
        timeIn.upload(flowWave(time, dim: cfg.freqDim))
        stageNameEmbed()
        gpu.begin()
        forward(withLoss: false, invCount: 1)
        gpu.end()
        let guess = pred.download(n)
        if !wantLoops { return (guess, []) }
        var readouts: [[Float]] = []
        for p in 1...cfg.loops {
            gpu.begin()
            gemm(wHead, xs[p], pred, M: T, N: cfg.patchDim, K: cfg.embd)
            gpu.end()
            readouts.append(pred.download(n))
        }
        return (guess, readouts)
    }

    private func gemm(_ w: Param, _ x: Buf, _ y: Buf, M: Int, N: Int, K: Int) {
        gpu.gemmNT(bits: 32, w: w.w, q: nil, scale: nil, x: x, bias: nil, y: y,
                   M: M, N: N, K: K, qGroup: 8)
    }

    private func zeroGrads() {
        for p in params { gpu.zero(p.g, n: p.count) }
    }

    fileprivate func probe(_ name: String) -> Param {
        guard let p = params.first(where: { $0.name == name }) else {
            fatalError("missing parameter \(name)")
        }
        return p
    }
}

/// Sinusoid of t * 1000, so a time in [0, 1] uses the same frequencies as a
/// thousand-step index.
private func flowWave(_ t: Float, dim: Int) -> [Float] {
    var y = [Float](repeating: 0, count: dim)
    let half = dim / 2
    let tick = min(max(t, 0), 1) * 1000
    for i in 0..<half {
        let freq = exp(-Darwin.log(Float(10000)) * Float(i) / Float(half))
        let arg = tick * freq
        y[i] = sin(arg)
        y[i + half] = cos(arg)
    }
    return y
}

private func appendFlowU32(_ v: UInt32, _ data: inout Data) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}

func trainFlow(gpu: GPU, args: Args) throws {
    let vocabPath = args.str("vocab", Pictures.vocabDefault)
    let rgbPath = args.str("rgb", Pictures.rgbDefault)
    requirePicture(vocabPath)
    requirePicture(rgbPath)
    let words = try WordList.load(vocabPath)
    let file = loadIconFile(rgbPath)
    let icons = file.icons
    let names = words.tokens.count - Int(Pictures.name0)
    if icons.count != names {
        fputs("rgb count \(icons.count) does not match \(names) names in \(vocabPath)\n", stderr)
        exit(1)
    }
    let batch = args.int("batch", 16)
    let steps = args.int("steps", 4000)
    let lr = args.float("lr", 0.008)
    let seed = UInt64(args.int("seed", 456))
    let ckptPath = args.str("ckpt", "build/images-flow.ckpt")
    var cfg = FlowConfig(names: names, batch: batch)
    cfg.side = file.side
    cfg.patch = args.int("patch", file.side == Pictures.side ? Pictures.patch : max(file.side / 8, 1))
    cfg.loops = args.int("loops", 4)
    cfg.embd = args.int("embd", 128)
    cfg.intermediate = args.int("intermediate", 512)
    cfg.heads = args.int("heads", 4)
    cfg.freqDim = cfg.embd
    cfg.validate()
    let resume = args.str("resume", "")
    let model: Flow
    if resume.isEmpty {
        model = Flow(gpu: gpu, cfg: cfg, seed: seed)
    } else {
        model = try Flow.load(gpu: gpu, url: URL(fileURLWithPath: resume), batch: batch, seed: seed)
    }
    let holdout = args.int("holdout", 1)
    let trainIcons: [IconRGB]
    let valIcons: [IconRGB]
    if holdout <= 1 {
        trainIcons = icons
        valIcons = icons
    } else {
        trainIcons = icons.enumerated().filter { $0.offset % holdout != 0 }.map(\.element)
        valIcons = icons.enumerated().filter { $0.offset % holdout == 0 }.map(\.element)
    }
    print("device \(gpu.device.name)")
    print("flow matching  params \(model.parameterCount)  names \(names)  embd \(cfg.embd)  ffn \(cfg.intermediate)")
    print("loops \(cfg.loops)  heads \(cfg.heads)/\(cfg.kvHeads)  patches \(cfg.tokens)  batch \(batch)  steps \(steps)")
    print("train \(trainIcons.count)  val \(valIcons.count)")
    if !resume.isEmpty { print("resume \(resume)") }
    print("curriculum logit-normal until step \((steps * 2) / 5), then uniform")
    var rng = RNG(UInt64(args.int("data-seed", 42)))
    var valRNG = RNG(7)
    let url = URL(fileURLWithPath: ckptPath)
    let dir = url.deletingLastPathComponent().path
    if dir != "" && dir != "." {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }
    let t0 = CFAbsoluteTimeGetCurrent()
    for step in 1...steps {
        let sched = macroLearningRate(step: step, total: steps, base: lr)
        let batchIn = flowBatch(trainIcons, batch: batch, cfg: cfg, step: step, total: steps, rng: &rng)
        let loss = model.step(xt: batchIn.xt, velocity: batchIn.velocity, times: batchIn.times, names: batchIn.names,
                              lr: sched, wd: 0, clip: 0, stepIndex: step)
        if !loss.isFinite {
            fputs("loss became non-finite.\n", stderr)
            exit(1)
        }
        if step == 1 || step % 20 == 0 || step == steps {
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let rate = Double(step * batch) / max(dt, 1e-6)
            var line = String(format: "step %4d/%d  x %.4f  lr %.2e  %.1f img/s", step, steps, loss, sched, rate)
            if step % 100 == 0 || step == steps {
                let held = flowBatch(valIcons, batch: min(batch, valIcons.count), cfg: cfg, step: step, total: steps, rng: &valRNG)
                let v = model.lossOnly(xt: held.xt, velocity: held.velocity, times: held.times, names: held.names)
                line += String(format: "  val %.4f", v)
            }
            print(line)
            fflush(stdout)
        }
        if step % 500 == 0 || step == steps {
            try model.save(url)
            print("wrote \(ckptPath)")
            fflush(stdout)
        }
    }
}

func drawImageFlow(model: Flow, words: WordList, name: String, steps: Int, rng: inout RNG) -> [String: Any] {
    guard let index = iconNameIndex(words, name) else {
        return ["error": "unknown", "unknown": [name]]
    }
    if index >= model.cfg.names {
        return ["error": "unknown", "unknown": [name]]
    }
    let drawn = model.generate(name: index, sampleSteps: steps, rng: &rng)
    let frames = drawn.frames.map(rgbBytes)
    let loops = drawn.loops.map(rgbBytes)
    return imagePayload(kind: "flow", name: words.tokens[Int(Pictures.name0) + index],
                        params: model.parameterCount, frames: frames, loops: loops, side: model.cfg.side)
}

func runImageFlowRepl(gpu: GPU, args: Args) throws {
    let words = try WordList.load(args.str("vocab", Pictures.vocabDefault))
    let model = try Flow.load(gpu: gpu, url: URL(fileURLWithPath: args.str("ckpt", "build/images-flow.ckpt")),
                              batch: 1, seed: 1)
    var rng = RNG(args.flags["seed"].flatMap(UInt64.init) ?? 1)
    let steps = args.int("steps", 20)
    while let line = readLine(strippingNewline: true) {
        let name = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { continue }
        emitImage(drawImageFlow(model: model, words: words, name: name, steps: steps, rng: &rng))
    }
}

func flowGradCheck(gpu: GPU, icon: IconRGB, _ r: inout Report) {
    let cfg = FlowConfig(names: max(icon.name + 1, 1), batch: 1)
    let model = Flow(gpu: gpu, cfg: cfg, seed: 5)
    var rng = RNG(11)
    let sample = flowBatch([icon], batch: 1, cfg: cfg, rng: &rng)
    _ = model.lossAndGrad(xt: sample.xt, velocity: sample.velocity, times: sample.times, names: sample.names)
    let probes: [(String, Int)] = [("wpatch", 0), ("whead", 0), ("n1", 0)]
    var analytic: [Float] = []
    for probe in probes {
        analytic.append(model.probe(probe.0).g.p[probe.1])
    }
    let eps: Float = 1e-3
    for (i, probe) in probes.enumerated() {
        let p = model.probe(probe.0)
        let index = probe.1
        let w0 = p.w.p[index]
        p.w.p[index] = w0 + eps
        let higher = model.lossOnly(xt: sample.xt, velocity: sample.velocity, times: sample.times, names: sample.names)
        p.w.p[index] = w0 - eps
        let lower = model.lossOnly(xt: sample.xt, velocity: sample.velocity, times: sample.times, names: sample.names)
        p.w.p[index] = w0
        let numeric = (higher - lower) / (2 * eps)
        let scale = max(abs(numeric), abs(analytic[i]), 1e-3)
        let gap = abs(numeric - analytic[i])
        let ok = gap / scale < 0.08 || gap < 5e-4
        r.expect(ok, String(format: "\(probe.0) grad  numeric %.5f  analytic %.5f", numeric, analytic[i]))
    }
}

struct FlowMem {
    var initial: Float
    var final: Float
    var mae: Float
    var gray: Float
}

func flowMemorize(gpu: GPU, icon: IconRGB) -> FlowMem {
    let cfg = FlowConfig(names: max(icon.name + 1, 1), batch: 4)
    let model = Flow(gpu: gpu, cfg: cfg, seed: 9)
    var rng = RNG(3)
    var initial: Float = 0
    var final: Float = 0
    let steps = 250
    for step in 1...steps {
        let sample = flowBatch([icon], batch: cfg.batch, cfg: cfg, rng: &rng)
        let loss = model.step(xt: sample.xt, velocity: sample.velocity, times: sample.times, names: sample.names,
                              lr: 0.01, wd: 0, clip: 0, stepIndex: step)
        if step == 1 { initial = loss }
        final = loss
        if step == 1 || step % 100 == 0 || step == steps {
            print(String(format: "  flow step %d  x %.4f", step, loss))
        }
    }
    var sampleRNG = RNG(21)
    let drawn = model.generate(name: icon.name, sampleSteps: 20, rng: &sampleRNG)
    let last = drawn.frames.last ?? []
    let mae = flowMeanAbs(last, icon.rgb)
    print(String(format: "  flow sample mae %.3f", mae))
    let gray = flowMeanAbs([Float](repeating: 0, count: icon.rgb.count), icon.rgb)
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("flow-check.ckpt")
    try? model.save(url)
    if let loaded = try? Flow.load(gpu: gpu, url: url, batch: 1, seed: 1) {
        var again = RNG(21)
        let redraw = loaded.generate(name: icon.name, sampleSteps: 20, rng: &again)
        let reload = flowMeanAbs(redraw.frames.last ?? [], icon.rgb)
        print(String(format: "  flow reload mae %.3f", reload))
    }
    try? FileManager.default.removeItem(at: url)
    return FlowMem(initial: initial, final: final, mae: mae, gray: gray)
}

func runFlowChecks(gpu: GPU) throws -> Bool {
    requirePicture(Pictures.vocabDefault)
    requirePicture(Pictures.rgbDefault)
    let words = try WordList.load(Pictures.vocabDefault)
    let icons = loadIconRGB(Pictures.rgbDefault)
    let icon = icons[0]
    let label = words.tokens[Int(Pictures.name0) + icon.name]
    var report = Report()
    print("flow gradient check")
    flowGradCheck(gpu: gpu, icon: icon, &report)
    print("flow memorization of \(label)")
    let mem = flowMemorize(gpu: gpu, icon: icon)
    report.expect(mem.final.isFinite && mem.final < mem.initial * 0.5,
                  String(format: "flow x %.3f -> %.3f", mem.initial, mem.final))
    report.expect(mem.mae < mem.gray * 0.55,
                  String(format: "flow sample mae %.3f  gray %.3f", mem.mae, mem.gray))
    if report.failed != 0 {
        fputs("images-flow-check failed\n", stderr)
        return false
    }
    print("images-flow-check passed")
    return true
}

private func flowMeanAbs(_ a: [Float], _ b: [Float]) -> Float {
    let n = min(a.count, b.count)
    if n == 0 { return 0 }
    var s: Float = 0
    for i in 0..<n { s += abs(a[i] - b[i]) }
    return s / Float(n)
}

/// Curriculum time. Sun, 2026: logit-normal for the first 40% of steps, then
/// uniform. Their clock runs from the photograph to noise, and the fastest
/// first phase is μ = 0.8, toward noise. Our clock runs from noise to the
/// photograph, so that phase is μ = -0.8.
private func flowCurriculumTime(step: Int, total: Int, rng: inout RNG) -> Float {
    let switchAt = max((total * 2) / 5, 1)
    if step > switchAt {
        return Float(rng.below(1_000_000)) / 1_000_000
    }
    let u = -0.8 + rng.normal()
    let t = 1 / (1 + exp(-u))
    return min(max(t, 0), 0.999)
}

/// Straight-line batch. The target is the photograph.
private func flowBatch(_ icons: [IconRGB], batch: Int, cfg: FlowConfig,
                       step: Int = 1, total: Int = 1,
                       rng: inout RNG) -> (xt: [Float], velocity: [Float], times: [Float], names: [Int]) {
    let n = cfg.tokens * cfg.patchDim
    var xt = [Float](repeating: 0, count: batch * n)
    var velocity = [Float](repeating: 0, count: batch * n)
    var times: [Float] = []
    var names: [Int] = []
    for b in 0..<batch {
        let icon = icons[rng.below(icons.count)]
        let t = flowCurriculumTime(step: step, total: total, rng: &rng)
        let photo = patchImage(icon.rgb, side: cfg.side, patch: cfg.patch)
        let base = b * n
        for i in 0..<n {
            let e = rng.normal()
            velocity[base + i] = photo[i]
            xt[base + i] = (1 - t) * e + t * photo[i]
        }
        times.append(t)
        names.append(icon.name)
    }
    return (xt, velocity, times, names)
}
