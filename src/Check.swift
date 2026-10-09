import Darwin
import Foundation
import Metal

struct Report {
    var failed = 0
    mutating func expect(_ ok: Bool, _ msg: String) {
        if ok {
            print("  ok   \(msg)")
        } else {
            print("  FAIL \(msg)")
            failed += 1
        }
    }
}

func maxDiff(_ a: [Float], _ b: [Float]) -> (Float, Int) {
    var m: Float = 0
    var idx = 0
    for i in 0..<min(a.count, b.count) {
        let d = abs(a[i] - b[i])
        if d > m { m = d; idx = i }
    }
    return (m, idx)
}

func runChecks(_ gpu: GPU) -> Bool {
    var r = Report()
    print("struct layouts match Metal")
    r.expect(true, "GemmP/AdamP sizes")
    checkQuant(gpu, bits: 8, &r)
    checkQuant(gpu, bits: 4, &r)
    checkGemm(gpu, bits: 32, &r)
    checkGemm(gpu, bits: 8, &r)
    checkGemm(gpu, bits: 4, &r)
    checkGelu(gpu, &r)
    checkLayerNorm(gpu, &r)
    checkAttention(gpu, &r)
    checkLoss(gpu, &r)
    checkAdam(gpu, &r)
    checkEmbed(gpu, &r)
    checkLinearLearns(gpu, bits: 32, &r)
    checkLinearLearns(gpu, bits: 8, &r)
    checkLinearLearns(gpu, bits: 4, &r)
    print("MacroStories backward")
    macroGradCheck(gpu: gpu, &r)
    let macro = macroMemorize(gpu: gpu, steps: 40, lr: 0.05)
    let macroDrop = macro.initial - macro.final
    r.expect(macro.final.isFinite && macroDrop > 1.5,
             String(format: "macro loss %.3f -> %.3f (drop %.3f)", macro.initial, macro.final, macroDrop))
    print("full backward via memorizing one batch")
    for bits in [32, 8, 4] {
        let result = memorize(gpu: gpu, bits: bits, layers: 1, heads: 2, embd: 16,
                              batch: 2, block: 8, steps: 80, lr: 8e-3, seed: 7, verbose: false)
        let drop = result.initial - result.final
        r.expect(result.final.isFinite && drop > 0.75,
                 String(format: "bits %d loss %.3f -> %.3f (drop %.3f)", bits, result.initial, result.final, drop))
    }
    if r.failed == 0 {
        print("all checks passed")
    } else {
        print("\(r.failed) check(s) failed")
    }
    return r.failed == 0
}

private func randn(_ n: Int, _ rng: inout RNG, _ std: Float) -> [Float] {
    (0..<n).map { _ in rng.normal() * std }
}

private func checkQuant(_ gpu: GPU, bits: Int, _ r: inout Report) {
    let rows = 5, cols = 16, g = 8
    var rng = RNG(UInt64(bits) * 99)
    let w = randn(rows * cols, &rng, 0.3)
    let wb = Buf(gpu.device, count: w.count)
    wb.upload(w)
    let scale = Buf(gpu.device, count: rows * cols / g)
    let qbytes = bits == 4 ? rows * cols / 2 : rows * cols
    let q = RawBuf(gpu.device, bytes: qbytes)
    gpu.begin()
    gpu.quantize(w: wb, q: q, scale: scale, rows: rows, cols: cols, bits: bits, qGroup: g)
    gpu.end()
    let cpu = cpuQuant(w, rows: rows, cols: cols, group: g, bits: bits)
    let gotS = scale.download(scale.count)
    let (ds, _) = maxDiff(gotS, cpu.scale)
    r.expect(ds == 0, "quant int\(bits) scales exact")
    if bits == 8 {
        let got = Array(UnsafeBufferPointer(start: q.raw.contents().bindMemory(to: Int8.self, capacity: qbytes), count: qbytes))
        r.expect(got == cpu.q8, "quant int8 codes exact")
    } else {
        let got = Array(UnsafeBufferPointer(start: q.raw.contents().bindMemory(to: UInt8.self, capacity: qbytes), count: qbytes))
        r.expect(got == cpu.q4, "quant int4 packing exact")
    }
}

private func checkGemm(_ gpu: GPU, bits: Int, _ r: inout Report) {
    let cases: [(Int, Int, Int)] = bits == 32 ? [(3, 5, 7), (17, 13, 16), (8, 24, 32)] : [(3, 5, 8), (15, 13, 16), (8, 24, 32)]
    var rng = RNG(1000 + UInt64(bits))
    var worst: Float = 0
    for (M, N, K) in cases {
        let x = randn(M * K, &rng, 0.5)
        let w = randn(N * K, &rng, 0.5)
        let bias = randn(N, &rng, 0.1)
        let xb = Buf(gpu.device, count: x.count); xb.upload(x)
        let wb = Buf(gpu.device, count: w.count); wb.upload(w)
        let bb = Buf(gpu.device, count: N); bb.upload(bias)
        let yb = Buf(gpu.device, count: M * N)
        let dx = Buf(gpu.device, count: M * K)
        let dw = Buf(gpu.device, count: N * K)
        let dyv = randn(M * N, &rng, 0.2)
        let dy = Buf(gpu.device, count: dyv.count); dy.upload(dyv)
        var q: RawBuf?
        var sc: Buf?
        if bits != 32 {
            sc = Buf(gpu.device, count: N * K / 8)
            q = RawBuf(gpu.device, bytes: bits == 4 ? N * K / 2 : N * K)
            gpu.begin()
            gpu.quantize(w: wb, q: q!, scale: sc!, rows: N, cols: K, bits: bits, qGroup: 8)
            gpu.end()
        }
        gpu.begin()
        gpu.gemmNT(bits: bits, w: wb, q: q, scale: sc, x: xb, bias: bb, y: yb, M: M, N: N, K: K, qGroup: 8)
        gpu.gemmDX(bits: bits, w: wb, q: q, scale: sc, dy: dy, dx: dx, M: M, N: N, K: K, qGroup: 8)
        gpu.gemmDW(dy: dy, x: xb, dw: dw, M: M, N: N, K: K)
        gpu.end()
        let refW = dequant(w, q: q, scale: sc, rows: N, cols: K, bits: bits)
        let yRef = cpuGemmNT(x, refW, bias, M, N, K)
        let dxRef = cpuGemmDX(dyv, refW, M, N, K)
        let dwRef = cpuGemmDW(dyv, x, M, N, K)
        worst = max(worst, maxDiff(yb.download(M * N), yRef).0)
        worst = max(worst, maxDiff(dx.download(M * K), dxRef).0)
        worst = max(worst, maxDiff(dw.download(N * K), dwRef).0)
    }
    r.expect(worst < 2e-3, String(format: "gemm bits %d fwd/dx/dw max abs %.3e", bits, worst))
}

private func checkGelu(_ gpu: GPU, _ r: inout Report) {
    var rng = RNG(3)
    let n = 64
    let x = randn(n, &rng, 1.2)
    let dy = randn(n, &rng, 1)
    let xb = Buf(gpu.device, count: n); xb.upload(x)
    let yb = Buf(gpu.device, count: n)
    let dyb = Buf(gpu.device, count: n); dyb.upload(dy)
    let dxb = Buf(gpu.device, count: n)
    gpu.begin()
    gpu.geluFwd(xb, yb, n: n)
    gpu.geluBwd(x: xb, dy: dyb, dx: dxb, n: n)
    gpu.end()
    let yRef = x.map(cpuGelu)
    let dxRef = zip(x, dy).map { cpuGeluPrime($0) * $1 }
    let (dyy, _) = maxDiff(yb.download(n), yRef)
    let (dxx, _) = maxDiff(dxb.download(n), dxRef)
    var fdWorst: Float = 0
    let eps: Float = 1e-3
    for i in [0, 7, 20, 40] {
        var xp = x; xp[i] += eps
        var xm = x; xm[i] -= eps
        let fd = (xp.map(cpuGelu).reduce(0, +) - xm.map(cpuGelu).reduce(0, +)) / (2 * eps)
        let analytic = cpuGeluPrime(x[i])
        fdWorst = max(fdWorst, abs(fd - analytic))
    }
    r.expect(dyy < 1e-5 && dxx < 1e-4 && fdWorst < 2e-3,
             String(format: "gelu fwd %.3e bwd %.3e fd %.3e", dyy, dxx, fdWorst))
}

private func checkLayerNorm(_ gpu: GPU, _ r: inout Report) {
    var rng = RNG(4)
    let rows = 4, cols = 8
    let x = randn(rows * cols, &rng, 1)
    let gamma = (0..<cols).map { _ in Float(0.8) + rng.uniform() }
    let beta = randn(cols, &rng, 0.2)
    let dy = randn(rows * cols, &rng, 1)
    let xb = Buf(gpu.device, count: x.count); xb.upload(x)
    let yb = Buf(gpu.device, count: x.count)
    let mean = Buf(gpu.device, count: rows)
    let rstd = Buf(gpu.device, count: rows)
    let gb = Buf(gpu.device, count: cols); gb.upload(gamma)
    let bb = Buf(gpu.device, count: cols); bb.upload(beta)
    let dyb = Buf(gpu.device, count: dy.count); dyb.upload(dy)
    let dxb = Buf(gpu.device, count: x.count)
    let dgb = Buf(gpu.device, count: cols)
    let dbb = Buf(gpu.device, count: cols)
    gpu.begin()
    gpu.lnFwd(x: xb, y: yb, mean: mean, rstd: rstd, gamma: gb, beta: bb, rows: rows, cols: cols, eps: 1e-5)
    gpu.lnBwd(x: xb, mean: mean, rstd: rstd, gamma: gb, beta: bb, dy: dyb, dx: dxb,
              dgamma: dgb, dbeta: dbb, rows: rows, cols: cols, eps: 1e-5)
    gpu.end()
    let cpu = cpuLN(x, gamma, beta, dy, rows: rows, cols: cols, eps: 1e-5)
    let ey = maxDiff(yb.download(x.count), cpu.y).0
    let ex = maxDiff(dxb.download(x.count), cpu.dx).0
    let eg = maxDiff(dgb.download(cols), cpu.dgamma).0
    let eb = maxDiff(dbb.download(cols), cpu.dbeta).0
    r.expect(ey < 1e-4 && ex < 1e-4 && eg < 1e-4 && eb < 1e-4,
             String(format: "layernorm y %.3e dx %.3e dg %.3e db %.3e", ey, ex, eg, eb))
}

private func checkAttention(_ gpu: GPU, _ r: inout Report) {
    let B = 2, T = 4, H = 2, D = 4
    let n = B * T * H * D
    let q = [Float](repeating: 0, count: n)
    let k = q
    let v = [Float](repeating: 1, count: n)
    let qb = Buf(gpu.device, count: n); qb.upload(q)
    let kb = Buf(gpu.device, count: n); kb.upload(k)
    let vb = Buf(gpu.device, count: n); vb.upload(v)
    let scores = Buf(gpu.device, count: B * H * T * T)
    let probs = Buf(gpu.device, count: B * H * T * T)
    let av = Buf(gpu.device, count: n)
    gpu.begin()
    gpu.attnFwd(q: qb, k: kb, v: vb, scores: scores, probs: probs, av: av, B: B, T: T, H: H, D: D)
    gpu.end()
    let p = probs.download(B * H * T * T)
    var maskOK = true
    for b in 0..<B {
        for h in 0..<H {
            for t in 0..<T {
                for s in 0..<T {
                    let got = p[((b * H + h) * T + t) * T + s]
                    let expect: Float = s <= t ? 1 / Float(t + 1) : 0
                    if abs(got - expect) > 1e-4 { maskOK = false }
                }
            }
        }
    }
    let avOK = av.download(n).allSatisfy { abs($0 - 1) < 1e-4 }
    r.expect(maskOK && avOK, "causal softmax is uniform over the prefix")

    var rng = RNG(11)
    let qr = randn(n, &rng, 0.4)
    let kr = randn(n, &rng, 0.4)
    let vr = randn(n, &rng, 0.4)
    let dout = randn(n, &rng, 0.3)
    qb.upload(qr); kb.upload(kr); vb.upload(vr)
    let dq = Buf(gpu.device, count: n)
    let dk = Buf(gpu.device, count: n)
    let dv = Buf(gpu.device, count: n)
    let dp = Buf(gpu.device, count: B * H * T * T)
    let ds = Buf(gpu.device, count: B * H * T * T)
    let dy = Buf(gpu.device, count: n); dy.upload(dout)
    gpu.begin()
    gpu.attnFwd(q: qb, k: kb, v: vb, scores: scores, probs: probs, av: av, B: B, T: T, H: H, D: D)
    gpu.attnBwd(q: qb, k: kb, v: vb, probs: probs, dout: dy, dq: dq, dk: dk, dv: dv,
                dprobs: dp, dscores: ds, B: B, T: T, H: H, D: D)
    gpu.end()
    let cpu = cpuAttn(qr, kr, vr, dout, B: B, T: T, H: H, D: D)
    let ea = maxDiff(av.download(n), cpu.av).0
    let eq = maxDiff(dq.download(n), cpu.dq).0
    let ek = maxDiff(dk.download(n), cpu.dk).0
    let ev = maxDiff(dv.download(n), cpu.dv).0
    r.expect(ea < 1e-4 && eq < 1e-4 && ek < 1e-4 && ev < 1e-4,
             String(format: "attention av %.3e dq %.3e dk %.3e dv %.3e", ea, eq, ek, ev))
}

private func checkLoss(_ gpu: GPU, _ r: inout Report) {
    let BT = 1, V = 2
    let logits = Buf(gpu.device, count: 2)
    logits.upload([0, 0])
    let tgt = UBuf(gpu.device, count: 1)
    tgt.upload([0])
    let nll = Buf(gpu.device, count: 1)
    let d = Buf(gpu.device, count: 2)
    let sum = Buf(gpu.device, count: 1)
    gpu.begin()
    gpu.loss(logits: logits, targets: tgt, nll: nll, dlogits: d, BT: BT, V: V)
    gpu.reduceSum(nll, into: sum, n: 1, slot: 0)
    gpu.end()
    let nllOK = abs(sum.p[0] - log(2)) < 1e-5
    let g0 = d.p[0], g1 = d.p[1]
    r.expect(nllOK && abs(g0 + 0.5) < 1e-5 && abs(g1 - 0.5) < 1e-5,
             String(format: "cross entropy log2 grad [%.4f %.4f]", g0, g1))
}

private func checkAdam(_ gpu: GPU, _ r: inout Report) {
    let w = Buf(gpu.device, count: 1); w.upload([1])
    let g = Buf(gpu.device, count: 1); g.upload([1])
    let m = Buf(gpu.device, count: 1)
    let v = Buf(gpu.device, count: 1)
    gpu.begin()
    gpu.adam(w: w, g: g, m: m, v: v, n: 1, lr: 0.1, b1: 0.9, b2: 0.95,
             eps: 1e-8, wd: 0, invBC1: 10, invBC2: 20)
    gpu.end()
    r.expect(abs(m.p[0] - 0.1) < 1e-6 && abs(v.p[0] - 0.05) < 1e-6 && abs(w.p[0] - 0.9) < 1e-5,
             String(format: "adamw w %.6f m %.6f v %.6f", w.p[0], m.p[0], v.p[0]))
}

private func checkEmbed(_ gpu: GPU, _ r: inout Report) {
    let B = 2, T = 3, C = 8, V = 4, G = 8
    var rng = RNG(5)
    let master = randn(V * C, &rng, 0.4)
    let wpe = randn(T * C, &rng, 0.2)
    let toks: [UInt32] = [1, 0, 3, 1, 2, 0]
    let wb = Buf(gpu.device, count: master.count); wb.upload(master)
    let pb = Buf(gpu.device, count: wpe.count); pb.upload(wpe)
    let y = Buf(gpu.device, count: B * T * C)
    let q = RawBuf(gpu.device, bytes: V * C)
    let sc = Buf(gpu.device, count: V * C / G)
    let tb = UBuf(gpu.device, count: toks.count); tb.upload(toks)
    gpu.begin()
    gpu.quantize(w: wb, q: q, scale: sc, rows: V, cols: C, bits: 8, qGroup: G)
    gpu.embed(bits: 8, tokens: tb, wte: wb, q: q, scale: sc, wpe: pb, y: y, B: B, T: T, C: C, qGroup: G)
    gpu.end()
    let codes = Array(UnsafeBufferPointer(start: q.raw.contents().bindMemory(to: Int8.self, capacity: V * C), count: V * C))
    let scales = sc.download(V)
    var expect = [Float](repeating: 0, count: B * T * C)
    for bt in 0..<(B * T) {
        let tok = Int(toks[bt])
        let t = bt % T
        for c in 0..<C {
            expect[bt * C + c] = Float(codes[tok * C + c]) * scales[tok] + wpe[t * C + c]
        }
    }
    let err = maxDiff(y.download(expect.count), expect).0
    r.expect(err < 1e-5, String(format: "int8 embedding lookup %.3e", err))
}

private func checkLinearLearns(_ gpu: GPU, bits: Int, _ r: inout Report) {
    let M = 16, N = 16, K = 16, G = 8
    var rng = RNG(20 + UInt64(bits))
    let x = randn(M * K, &rng, 0.5)
    let teacher = randn(N * K, &rng, 0.3)
    let target = cpuGemmNT(x, teacher, [Float](repeating: 0, count: N), M, N, K)
    let w = Buf(gpu.device, count: N * K)
    fillNormal(w, std: 0.02, rng: &rng)
    let m = Buf(gpu.device, count: N * K)
    let v = Buf(gpu.device, count: N * K)
    let dyb = Buf(gpu.device, count: M * N)
    let dwb = Buf(gpu.device, count: N * K)
    let xb = Buf(gpu.device, count: x.count); xb.upload(x)
    let yb = Buf(gpu.device, count: M * N)
    let q: RawBuf? = bits == 32 ? nil : RawBuf(gpu.device, bytes: bits == 4 ? N * K / 2 : N * K)
    let sc: Buf? = bits == 32 ? nil : Buf(gpu.device, count: N * (K / G))
    func lossOf() -> Float {
        gpu.begin()
        if bits != 32 { gpu.quantize(w: w, q: q!, scale: sc!, rows: N, cols: K, bits: bits, qGroup: G) }
        gpu.gemmNT(bits: bits, w: w, q: q, scale: sc, x: xb, bias: nil, y: yb, M: M, N: N, K: K, qGroup: G)
        gpu.end()
        let y = yb.download(M * N)
        var loss: Float = 0
        var dy = [Float](repeating: 0, count: M * N)
        let inv = 2 / Float(M * N)
        for i in 0..<y.count {
            let d = y[i] - target[i]
            loss += d * d
            dy[i] = d * inv
        }
        dyb.upload(dy)
        return loss / Float(M * N)
    }
    let initial = lossOf()
    let lr: Float = bits == 4 ? 0.05 : 0.08
    for step in 1...40 {
        let b1: Float = 0.9, b2: Float = 0.95
        gpu.begin()
        gpu.gemmDW(dy: dyb, x: xb, dw: dwb, M: M, N: N, K: K)
        gpu.adam(w: w, g: dwb, m: m, v: v, n: N * K, lr: lr, b1: b1, b2: b2, eps: 1e-8, wd: 0,
                 invBC1: 1 / (1 - pow(b1, Float(step))), invBC2: 1 / (1 - pow(b2, Float(step))))
        gpu.end()
        _ = lossOf()
    }
    let final = lossOf()
    r.expect(final.isFinite && final < initial * 0.5,
             String(format: "linear bits %d learns %.4f -> %.4f", bits, initial, final))
}

// MARK: - CPU reference

private struct QuantCPU {
    var q8: [Int8] = []
    var q4: [UInt8] = []
    var scale: [Float] = []
}

private func cpuQuant(_ w: [Float], rows: Int, cols: Int, group: Int, bits: Int) -> QuantCPU {
    let lim: Float = bits == 4 ? 7 : 127
    let groups = cols / group
    var scale = [Float](repeating: 0, count: rows * groups)
    var q8 = [Int8](repeating: 0, count: bits == 8 ? rows * cols : 0)
    var q4 = [UInt8](repeating: 0, count: bits == 4 ? rows * cols / 2 : 0)
    for row in 0..<rows {
        for g in 0..<groups {
            var maxAbs: Float = 0
            let base = row * cols + g * group
            for i in 0..<group { maxAbs = max(maxAbs, abs(w[base + i])) }
            let s = max(maxAbs / lim, 1e-8)
            scale[row * groups + g] = s
            if bits == 8 {
                for i in 0..<group { q8[base + i] = Int8(quantCode(w[base + i], s, Int(lim))) }
            } else {
                for i in stride(from: 0, to: group, by: 2) {
                    let q0 = quantCode(w[base + i], s, Int(lim))
                    let q1 = quantCode(w[base + i + 1], s, Int(lim))
                    q4[(base + i) >> 1] = UInt8((q0 & 0xF) | ((q1 & 0xF) << 4))
                }
            }
        }
    }
    return QuantCPU(q8: q8, q4: q4, scale: scale)
}

private func quantCode(_ w: Float, _ s: Float, _ lim: Int) -> Int {
    var q = Int(rintf(w / s))
    if q > lim { q = lim }
    if q < -lim { q = -lim }
    return q
}

private func dequant(_ w: [Float], q: RawBuf?, scale: Buf?, rows: Int, cols: Int, bits: Int) -> [Float] {
    if bits == 32 { return w }
    let G = 8
    let groups = cols / G
    let sc = scale!.download(rows * groups)
    var out = [Float](repeating: 0, count: rows * cols)
    if bits == 8 {
        let codes = Array(UnsafeBufferPointer(
            start: q!.raw.contents().bindMemory(to: Int8.self, capacity: rows * cols), count: rows * cols))
        for n in 0..<rows {
            for k in 0..<cols { out[n * cols + k] = Float(codes[n * cols + k]) * sc[n * groups + k / G] }
        }
    } else {
        let packed = Array(UnsafeBufferPointer(
            start: q!.raw.contents().bindMemory(to: UInt8.self, capacity: rows * cols / 2), count: rows * cols / 2))
        for n in 0..<rows {
            for k in 0..<cols {
                let b = packed[(n * cols + k) >> 1]
                let nib = (k & 1) == 0 ? Int(b & 0xF) : Int(b >> 4)
                let v = nib >= 8 ? nib - 16 : nib
                out[n * cols + k] = Float(v) * sc[n * groups + k / G]
            }
        }
    }
    return out
}

private func cpuGemmNT(_ x: [Float], _ w: [Float], _ bias: [Float], _ M: Int, _ N: Int, _ K: Int) -> [Float] {
    var y = [Float](repeating: 0, count: M * N)
    for m in 0..<M {
        for n in 0..<N {
            var acc: Float = bias.isEmpty ? 0 : bias[n]
            for k in 0..<K { acc += x[m * K + k] * w[n * K + k] }
            y[m * N + n] = acc
        }
    }
    return y
}

private func cpuGemmDX(_ dy: [Float], _ w: [Float], _ M: Int, _ N: Int, _ K: Int) -> [Float] {
    var dx = [Float](repeating: 0, count: M * K)
    for m in 0..<M {
        for k in 0..<K {
            var acc: Float = 0
            for n in 0..<N { acc += dy[m * N + n] * w[n * K + k] }
            dx[m * K + k] = acc
        }
    }
    return dx
}

private func cpuGemmDW(_ dy: [Float], _ x: [Float], _ M: Int, _ N: Int, _ K: Int) -> [Float] {
    var dw = [Float](repeating: 0, count: N * K)
    for n in 0..<N {
        for k in 0..<K {
            var acc: Float = 0
            for m in 0..<M { acc += dy[m * N + n] * x[m * K + k] }
            dw[n * K + k] = acc
        }
    }
    return dw
}

private func cpuGelu(_ x: Float) -> Float {
    let k0: Float = 0.7978845608028654
    let k1: Float = 0.044715
    let u = k0 * (x + k1 * x * x * x)
    return 0.5 * x * (1 + tanh(u))
}

private func cpuGeluPrime(_ x: Float) -> Float {
    let k0: Float = 0.7978845608028654
    let k1: Float = 0.044715
    let x2 = x * x
    let u = k0 * (x + k1 * x2 * x)
    let th = tanh(u)
    let dudx = k0 * (1 + 3 * k1 * x2)
    return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * dudx
}

private func cpuLN(_ x: [Float], _ gamma: [Float], _ beta: [Float], _ dy: [Float], rows: Int, cols: Int, eps: Float)
    -> (y: [Float], dx: [Float], dgamma: [Float], dbeta: [Float]) {
    var y = [Float](repeating: 0, count: rows * cols)
    var dx = [Float](repeating: 0, count: rows * cols)
    var dgamma = [Float](repeating: 0, count: cols)
    var dbeta = [Float](repeating: 0, count: cols)
    var mean = [Float](repeating: 0, count: rows)
    var rstd = [Float](repeating: 0, count: rows)
    for row in 0..<rows {
        var m: Float = 0
        for c in 0..<cols { m += x[row * cols + c] }
        m /= Float(cols)
        var v: Float = 0
        for c in 0..<cols {
            let d = x[row * cols + c] - m
            v += d * d
        }
        v /= Float(cols)
        let rs = 1 / sqrt(v + eps)
        mean[row] = m
        rstd[row] = rs
        for c in 0..<cols {
            y[row * cols + c] = (x[row * cols + c] - m) * rs * gamma[c] + beta[c]
        }
    }
    for row in 0..<rows {
        var sumDy: Float = 0
        var sumDyX: Float = 0
        for c in 0..<cols {
            let xhat = (x[row * cols + c] - mean[row]) * rstd[row]
            let dyhat = dy[row * cols + c] * gamma[c]
            sumDy += dyhat
            sumDyX += dyhat * xhat
        }
        sumDy /= Float(cols)
        sumDyX /= Float(cols)
        for c in 0..<cols {
            let xhat = (x[row * cols + c] - mean[row]) * rstd[row]
            let dyhat = dy[row * cols + c] * gamma[c]
            dx[row * cols + c] = rstd[row] * (dyhat - sumDy - xhat * sumDyX)
        }
    }
    for c in 0..<cols {
        var dg: Float = 0
        var db: Float = 0
        for row in 0..<rows {
            let xhat = (x[row * cols + c] - mean[row]) * rstd[row]
            dg += dy[row * cols + c] * xhat
            db += dy[row * cols + c]
        }
        dgamma[c] = dg
        dbeta[c] = db
    }
    return (y, dx, dgamma, dbeta)
}

private func cpuAttn(_ q: [Float], _ k: [Float], _ v: [Float], _ dout: [Float], B: Int, T: Int, H: Int, D: Int)
    -> (av: [Float], dq: [Float], dk: [Float], dv: [Float]) {
    let scale = 1 / sqrt(Float(D))
    var probs = [Float](repeating: 0, count: B * H * T * T)
    var av = [Float](repeating: 0, count: q.count)
    func qi(_ b: Int, _ t: Int, _ h: Int, _ d: Int) -> Int { ((b * T + t) * H + h) * D + d }
    func si(_ b: Int, _ h: Int, _ t: Int, _ s: Int) -> Int { ((b * H + h) * T + t) * T + s }
    for b in 0..<B {
        for h in 0..<H {
            for t in 0..<T {
                var row = [Float](repeating: 0, count: T)
                var maxv = -Float.infinity
                for s in 0..<T {
                    var acc: Float = 0
                    for d in 0..<D { acc += q[qi(b, t, h, d)] * k[qi(b, s, h, d)] }
                    acc *= scale
                    if s > t { acc = -1e30 }
                    row[s] = acc
                    maxv = max(maxv, acc)
                }
                var sum: Float = 0
                for s in 0..<T { sum += exp(row[s] - maxv) }
                for s in 0..<T { probs[si(b, h, t, s)] = exp(row[s] - maxv) / sum }
                for d in 0..<D {
                    var acc: Float = 0
                    for s in 0..<T { acc += probs[si(b, h, t, s)] * v[qi(b, s, h, d)] }
                    av[qi(b, t, h, d)] = acc
                }
            }
        }
    }
    var dq = [Float](repeating: 0, count: q.count)
    var dk = [Float](repeating: 0, count: q.count)
    var dv = [Float](repeating: 0, count: q.count)
    var dp = [Float](repeating: 0, count: probs.count)
    for b in 0..<B {
        for h in 0..<H {
            for s in 0..<T {
                for d in 0..<D {
                    var acc: Float = 0
                    for t in 0..<T { acc += probs[si(b, h, t, s)] * dout[qi(b, t, h, d)] }
                    dv[qi(b, s, h, d)] = acc
                }
            }
            for t in 0..<T {
                for s in 0..<T {
                    var acc: Float = 0
                    for d in 0..<D { acc += dout[qi(b, t, h, d)] * v[qi(b, s, h, d)] }
                    dp[si(b, h, t, s)] = acc
                }
            }
            for t in 0..<T {
                var dot: Float = 0
                for s in 0..<T { dot += dp[si(b, h, t, s)] * probs[si(b, h, t, s)] }
                for s in 0..<T {
                    let ds = probs[si(b, h, t, s)] * (dp[si(b, h, t, s)] - dot)
                    for d in 0..<D {
                        dq[qi(b, t, h, d)] += ds * k[qi(b, s, h, d)] * scale
                        dk[qi(b, s, h, d)] += ds * q[qi(b, t, h, d)] * scale
                    }
                }
            }
        }
    }
    return (av, dq, dk, dv)
}
