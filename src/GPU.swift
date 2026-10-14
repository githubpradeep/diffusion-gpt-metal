import Foundation
import Metal

struct RNG {
    var state: UInt64
    init(_ seed: UInt64) {
        state = seed == 0 ? 0x1234_5678_9abc_def0 : seed
    }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Float {
        let x = next() >> 40
        return (Float(x) + 1) / 16_777_216
    }
    mutating func normal() -> Float {
        let u1 = max(uniform(), 1e-7)
        let u2 = uniform()
        return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
    }
    mutating func below(_ n: Int) -> Int {
        precondition(n > 0)
        return Int(next() % UInt64(n))
    }
}

final class Buf {
    let raw: MTLBuffer
    let count: Int
    init(_ device: MTLDevice, count: Int) {
        self.count = count
        let bytes = max(count, 1) * MemoryLayout<Float>.size
        guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            fatalError("failed to allocate \(bytes) bytes")
        }
        raw = b
        memset(b.contents(), 0, bytes)
    }
    var p: UnsafeMutablePointer<Float> {
        raw.contents().bindMemory(to: Float.self, capacity: max(count, 1))
    }
    func download(_ n: Int) -> [Float] {
        precondition(n <= count)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
    func upload(_ values: [Float]) {
        precondition(values.count <= count)
        p.update(from: values, count: values.count)
    }
}

final class RawBuf {
    let raw: MTLBuffer
    let bytes: Int
    init(_ device: MTLDevice, bytes: Int) {
        self.bytes = bytes
        guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
            fatalError("failed to allocate \(bytes) raw bytes")
        }
        raw = b
        memset(b.contents(), 0, b.length)
    }
}

final class UBuf {
    let raw: MTLBuffer
    let count: Int
    init(_ device: MTLDevice, count: Int) {
        self.count = count
        let bytes = max(count, 1) * MemoryLayout<UInt32>.size
        guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            fatalError("failed to allocate token buffer")
        }
        raw = b
        memset(b.contents(), 0, bytes)
    }
    var p: UnsafeMutablePointer<UInt32> {
        raw.contents().bindMemory(to: UInt32.self, capacity: max(count, 1))
    }
    func upload(_ values: [UInt32]) {
        precondition(values.count <= count)
        p.update(from: values, count: values.count)
    }
}

final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let dummy: Buf
    private let library: MTLLibrary
    private var cache: [String: MTLComputePipelineState] = [:]
    private var cmd: MTLCommandBuffer?
    private var enc: MTLComputeCommandEncoder?
    private var firstDispatch = true
    private(set) var lastKernel = ""

    static let kernelNames = [
        "quant_i8", "quant_i4",
        "gemm_nt_f32", "gemm_nt_i8", "gemm_nt_i4",
        "gemm_dx_f32", "gemm_dx_i8", "gemm_dx_i4", "gemm_dw", "bias_grad",
        "layernorm_fwd", "layernorm_bwd_dx", "layernorm_bwd_dg",
        "gelu_fwd", "gelu_bwd", "add_f", "acc_f",
        "attn_scores", "attn_softmax", "attn_av",
        "attn_bwd_dv", "attn_bwd_dp", "attn_softmax_bwd", "attn_bwd_dq", "attn_bwd_dk",
        "embed_f32", "embed_i8", "embed_i4", "embed_bwd", "pos_bwd",
        "loss_fwd", "reduce_sum", "reduce_sumsq", "zero_f", "scale_f", "adamw",
        "rmsnorm_fwd", "rmsnorm_bwd_dx", "rmsnorm_bwd_dw",
        "rope_fwd", "rope_bwd", "swiglu_fwd", "swiglu_bwd",
        "mix_f", "copy_f", "acc_scale_f", "repeat_heads", "sum_heads",
        "gemm_dw_acc", "loss_fwd_masked",
        "mse_fwd", "add_cond", "sum_cond",
    ]

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fputs("No Metal device. This trainer needs a Mac GPU.\n", stderr)
            exit(1)
        }
        guard let queue = device.makeCommandQueue() else {
            fputs("Failed to create a Metal command queue.\n", stderr)
            exit(1)
        }
        self.device = device
        self.queue = queue
        dummy = Buf(device, count: 1)
        let url = GPU.findKernels()
        let src: String
        do {
            src = try String(contentsOf: url, encoding: .utf8)
        } catch {
            fputs("Could not read \(url.path): \(error)\n", stderr)
            exit(1)
        }
        let opts = MTLCompileOptions()
        opts.mathMode = .safe
        let t0 = CFAbsoluteTimeGetCurrent()
        fputs("compiling \(url.lastPathComponent) for \(device.name)...\n", stderr)
        do {
            library = try device.makeLibrary(source: src, options: opts)
        } catch {
            fputs("Metal compile failed:\n\(error)\n", stderr)
            exit(1)
        }
        let missing = GPU.kernelNames.filter { library.makeFunction(name: $0) == nil }
        if !missing.isEmpty {
            fputs("kernels.metal is missing: \(missing.joined(separator: ", "))\n", stderr)
            exit(1)
        }
        fputs(String(format: "kernels ready in %.2fs\n", CFAbsoluteTimeGetCurrent() - t0), stderr)
        checkLayouts()
        for name in GPU.kernelNames { _ = pipe(name) }
    }

    func begin() {
        guard let c = queue.makeCommandBuffer(), let e = c.makeComputeCommandEncoder() else {
            fatalError("failed to open a compute encoder")
        }
        c.label = "gpt"
        cmd = c
        enc = e
        firstDispatch = true
    }

    func end() {
        enc?.endEncoding()
        enc = nil
        guard let c = cmd else { return }
        cmd = nil
        c.commit()
        c.waitUntilCompleted()
        if c.status != .completed {
            let why = c.error?.localizedDescription ?? "status \(c.status.rawValue)"
            fputs("Metal error after \(lastKernel): \(why)\n", stderr)
            exit(1)
        }
    }

    func go<P>(_ name: String, _ bufs: [MTLBuffer], _ params: P, grid: MTLSize, tpg: MTLSize, asGroups: Bool = false) {
        guard let enc else { fatalError("GPU.begin() was not called before \(name)") }
        if !firstDispatch { enc.memoryBarrier(scope: .buffers) }
        firstDispatch = false
        lastKernel = name
        var p = params
        enc.setComputePipelineState(pipe(name))
        for (i, b) in bufs.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        withUnsafeBytes(of: &p) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: bufs.count)
        }
        if asGroups {
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
        } else {
            enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
        }
    }

    func zero(_ b: Buf, n: Int) {
        guard n > 0 else { return }
        go("zero_f", [b.raw], UnaryP(n: UInt32(n), pad0: 0, pad1: 0, pad2: 0),
           grid: grid1(n), tpg: tpg1)
    }

    func scale(_ b: Buf, n: Int, by s: Float) {
        guard n > 0 else { return }
        go("scale_f", [b.raw], ScaleP(n: UInt32(n), s: s, pad0: 0, pad1: 0),
           grid: grid1(n), tpg: tpg1)
    }

    func add(_ a: Buf, _ b: Buf, _ y: Buf, n: Int) {
        go("add_f", [a.raw, b.raw, y.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func acc(_ a: Buf, _ b: Buf, n: Int) {
        go("acc_f", [a.raw, b.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func quantize(w: Buf, q: RawBuf, scale: Buf, rows: Int, cols: Int, bits: Int, qGroup: Int) {
        let qmax: Float = bits == 4 ? 7 : 127
        let p = QuantP(rows: UInt32(rows), cols: UInt32(cols), qmax: qmax, qGroup: UInt32(qGroup))
        let name = bits == 4 ? "quant_i4" : "quant_i8"
        go(name, [w.raw, q.raw, scale.raw], p, grid: grid1(rows), tpg: tpg1)
    }

    func gemmNT(bits: Int, w: Buf, q: RawBuf?, scale: Buf?, x: Buf, bias: Buf?, y: Buf,
                M: Int, N: Int, K: Int, qGroup: Int) {
        if bits != 32 { precondition(K % qGroup == 0, "K=\(K) not divisible by group \(qGroup)") }
        let p = gemmP(M, N, K, bias == nil ? 0 : 1, qGroup)
        let grid = MTLSize(width: N, height: M, depth: 1)
        let biasBuf = bias?.raw ?? dummy.raw
        switch bits {
        case 32:
            go("gemm_nt_f32", [w.raw, x.raw, biasBuf, y.raw], p, grid: grid, tpg: tpg2)
        case 8:
            go("gemm_nt_i8", [q!.raw, scale!.raw, x.raw, biasBuf, y.raw], p, grid: grid, tpg: tpg2)
        case 4:
            go("gemm_nt_i4", [q!.raw, scale!.raw, x.raw, biasBuf, y.raw], p, grid: grid, tpg: tpg2)
        default:
            fatalError("unsupported bits \(bits)")
        }
    }

    func gemmDX(bits: Int, w: Buf, q: RawBuf?, scale: Buf?, dy: Buf, dx: Buf,
                M: Int, N: Int, K: Int, qGroup: Int) {
        if bits != 32 { precondition(K % qGroup == 0) }
        let p = gemmP(M, N, K, 0, qGroup)
        let grid = MTLSize(width: K, height: M, depth: 1)
        switch bits {
        case 32:
            go("gemm_dx_f32", [w.raw, dy.raw, dx.raw], p, grid: grid, tpg: tpg2)
        case 8:
            go("gemm_dx_i8", [q!.raw, scale!.raw, dy.raw, dx.raw], p, grid: grid, tpg: tpg2)
        case 4:
            go("gemm_dx_i4", [q!.raw, scale!.raw, dy.raw, dx.raw], p, grid: grid, tpg: tpg2)
        default:
            fatalError("unsupported bits \(bits)")
        }
    }

    func gemmDW(dy: Buf, x: Buf, dw: Buf, M: Int, N: Int, K: Int) {
        let p = gemmP(M, N, K, 0, defaultQGroup)
        go("gemm_dw", [dy.raw, x.raw, dw.raw], p,
           grid: MTLSize(width: K, height: N, depth: 1), tpg: tpg2)
    }

    func gemmDWAcc(dy: Buf, x: Buf, dw: Buf, M: Int, N: Int, K: Int) {
        let p = gemmP(M, N, K, 0, defaultQGroup)
        go("gemm_dw_acc", [dy.raw, x.raw, dw.raw], p,
           grid: MTLSize(width: K, height: N, depth: 1), tpg: tpg2)
    }

    func biasGrad(dy: Buf, db: Buf, M: Int, N: Int) {
        go("bias_grad", [dy.raw, db.raw], BiasP(M: UInt32(M), N: UInt32(N), pad0: 0, pad1: 0),
           grid: grid1(N), tpg: tpg1)
    }

    func lnFwd(x: Buf, y: Buf, mean: Buf, rstd: Buf, gamma: Buf, beta: Buf, rows: Int, cols: Int, eps: Float) {
        let p = LNP(rows: UInt32(rows), cols: UInt32(cols), eps: eps, pad: 0)
        go("layernorm_fwd", [x.raw, y.raw, mean.raw, rstd.raw, gamma.raw, beta.raw], p,
           grid: grid1(rows), tpg: tpg1)
    }

    func lnBwd(x: Buf, mean: Buf, rstd: Buf, gamma: Buf, beta: Buf, dy: Buf, dx: Buf,
               dgamma: Buf, dbeta: Buf, rows: Int, cols: Int, eps: Float) {
        let p = LNP(rows: UInt32(rows), cols: UInt32(cols), eps: eps, pad: 0)
        go("layernorm_bwd_dx", [x.raw, mean.raw, rstd.raw, gamma.raw, dy.raw, dx.raw], p,
           grid: grid1(rows), tpg: tpg1)
        go("layernorm_bwd_dg", [x.raw, mean.raw, rstd.raw, dy.raw, dgamma.raw, dbeta.raw], p,
           grid: grid1(cols), tpg: tpg1)
    }

    func geluFwd(_ x: Buf, _ y: Buf, n: Int) {
        go("gelu_fwd", [x.raw, y.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func geluBwd(x: Buf, dy: Buf, dx: Buf, n: Int) {
        go("gelu_bwd", [x.raw, dy.raw, dx.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func attnFwd(q: Buf, k: Buf, v: Buf, scores: Buf, probs: Buf, av: Buf, B: Int, T: Int, H: Int, D: Int,
                 causal: Bool = true) {
        let p = attnP(B, T, H, D, causal: causal)
        let tt = B * H * T * T
        let n = B * T * H * D
        go("attn_scores", [q.raw, k.raw, scores.raw], p, grid: grid1(tt), tpg: tpg1)
        go("attn_softmax", [scores.raw, probs.raw], p, grid: grid1(B * H * T), tpg: tpg1)
        go("attn_av", [probs.raw, v.raw, av.raw], p, grid: grid1(n), tpg: tpg1)
    }

    func attnBwd(q: Buf, k: Buf, v: Buf, probs: Buf, dout: Buf, dq: Buf, dk: Buf, dv: Buf,
                 dprobs: Buf, dscores: Buf, B: Int, T: Int, H: Int, D: Int) {
        let p = attnP(B, T, H, D)
        let tt = B * H * T * T
        let n = B * T * H * D
        go("attn_bwd_dv", [probs.raw, dout.raw, dv.raw], p, grid: grid1(n), tpg: tpg1)
        go("attn_bwd_dp", [dout.raw, v.raw, dprobs.raw], p, grid: grid1(tt), tpg: tpg1)
        go("attn_softmax_bwd", [probs.raw, dprobs.raw, dscores.raw], p, grid: grid1(B * H * T), tpg: tpg1)
        go("attn_bwd_dq", [dscores.raw, k.raw, dq.raw], p, grid: grid1(n), tpg: tpg1)
        go("attn_bwd_dk", [dscores.raw, q.raw, dk.raw], p, grid: grid1(n), tpg: tpg1)
    }

    func embed(bits: Int, tokens: UBuf, wte: Buf, q: RawBuf?, scale: Buf?, wpe: Buf, y: Buf,
               B: Int, T: Int, C: Int, qGroup: Int) {
        let p = EmbP(B: UInt32(B), T: UInt32(T), C: UInt32(C), qGroup: UInt32(qGroup))
        let grid = MTLSize(width: C, height: B * T, depth: 1)
        switch bits {
        case 32:
            go("embed_f32", [tokens.raw, wte.raw, wpe.raw, y.raw], p, grid: grid, tpg: tpg2)
        case 8:
            go("embed_i8", [tokens.raw, q!.raw, scale!.raw, wpe.raw, y.raw], p, grid: grid, tpg: tpg2)
        case 4:
            go("embed_i4", [tokens.raw, q!.raw, scale!.raw, wpe.raw, y.raw], p, grid: grid, tpg: tpg2)
        default:
            fatalError("unsupported bits \(bits)")
        }
    }

    func embedBwd(tokens: UBuf, dx: Buf, dwe: Buf, B: Int, T: Int, C: Int, V: Int) {
        let p = EmbBwdP(B: UInt32(B), T: UInt32(T), C: UInt32(C), V: UInt32(V))
        go("embed_bwd", [tokens.raw, dx.raw, dwe.raw], p,
           grid: MTLSize(width: C, height: V, depth: 1), tpg: tpg2)
    }

    func posBwd(dx: Buf, dwp: Buf, B: Int, T: Int, C: Int) {
        let p = PosP(B: UInt32(B), T: UInt32(T), C: UInt32(C), pad: 0)
        go("pos_bwd", [dx.raw, dwp.raw], p,
           grid: MTLSize(width: C, height: T, depth: 1), tpg: tpg2)
    }

    func loss(logits: Buf, targets: UBuf, nll: Buf, dlogits: Buf, BT: Int, V: Int) {
        let p = LossP(BT: UInt32(BT), V: UInt32(V), pad0: 0, pad1: 0)
        go("loss_fwd", [logits.raw, targets.raw, nll.raw, dlogits.raw], p, grid: grid1(BT), tpg: tpg1)
    }

    func lossMasked(logits: Buf, targets: UBuf, nll: Buf, dlogits: Buf, BT: Int, V: Int,
                    ignore: UInt32, invCount: Float) {
        let p = LossMaskP(BT: UInt32(BT), V: UInt32(V), ignore: ignore, pad: 0,
                          invCount: invCount, pad1: 0, pad2: 0, pad3: 0)
        go("loss_fwd_masked", [logits.raw, targets.raw, nll.raw, dlogits.raw], p, grid: grid1(BT), tpg: tpg1)
    }

    func rmsFwd(x: Buf, y: Buf, rstd: Buf, weight: Buf, rows: Int, cols: Int, eps: Float) {
        let p = LNP(rows: UInt32(rows), cols: UInt32(cols), eps: eps, pad: 0)
        go("rmsnorm_fwd", [x.raw, y.raw, rstd.raw, weight.raw], p, grid: grid1(rows), tpg: tpg1)
    }

    func rmsBwd(x: Buf, rstd: Buf, weight: Buf, dy: Buf, dx: Buf, dweight: Buf, rows: Int, cols: Int, eps: Float) {
        let p = LNP(rows: UInt32(rows), cols: UInt32(cols), eps: eps, pad: 0)
        go("rmsnorm_bwd_dx", [x.raw, rstd.raw, weight.raw, dy.raw, dx.raw], p, grid: grid1(rows), tpg: tpg1)
        go("rmsnorm_bwd_dw", [x.raw, rstd.raw, dy.raw, dweight.raw], p, grid: grid1(cols), tpg: tpg1)
    }

    func ropeFwd(x: Buf, y: Buf, cos: Buf, sin: Buf, B: Int, T: Int, H: Int, D: Int) {
        let p = RopeP(B: UInt32(B), T: UInt32(T), H: UInt32(H), D: UInt32(D))
        go("rope_fwd", [x.raw, y.raw, cos.raw, sin.raw], p, grid: grid1(B * T * H * D), tpg: tpg1)
    }

    func ropeBwd(dy: Buf, dx: Buf, cos: Buf, sin: Buf, B: Int, T: Int, H: Int, D: Int) {
        let p = RopeP(B: UInt32(B), T: UInt32(T), H: UInt32(H), D: UInt32(D))
        go("rope_bwd", [dy.raw, dx.raw, cos.raw, sin.raw], p, grid: grid1(B * T * H * D), tpg: tpg1)
    }

    func swigluFwd(gate: Buf, up: Buf, y: Buf, n: Int) {
        go("swiglu_fwd", [gate.raw, up.raw, y.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func swigluBwd(gate: Buf, up: Buf, dy: Buf, dgate: Buf, dup: Buf, n: Int) {
        go("swiglu_bwd", [gate.raw, up.raw, dy.raw, dgate.raw, dup.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func mix(_ a: Buf, _ b: Buf, _ y: Buf, n: Int, ca: Float, cb: Float) {
        go("mix_f", [a.raw, b.raw, y.raw], MixP(n: UInt32(n), a: ca, b: cb, pad: 0),
           grid: grid1(n), tpg: tpg1)
    }

    func copy(_ x: Buf, _ y: Buf, n: Int) {
        go("copy_f", [x.raw, y.raw], unary(n), grid: grid1(n), tpg: tpg1)
    }

    func accScale(_ a: Buf, _ b: Buf, n: Int, by s: Float) {
        go("acc_scale_f", [a.raw, b.raw], ScaleP(n: UInt32(n), s: s, pad0: 0, pad1: 0),
           grid: grid1(n), tpg: tpg1)
    }

    func repeatHeads(src: Buf, dst: Buf, B: Int, T: Int, H: Int, D: Int) {
        let p = RopeP(B: UInt32(B), T: UInt32(T), H: UInt32(H), D: UInt32(D))
        go("repeat_heads", [src.raw, dst.raw], p, grid: grid1(B * T * H * D), tpg: tpg1)
    }

    func sumHeads(src: Buf, dst: Buf, B: Int, T: Int, H: Int, D: Int) {
        let p = RopeP(B: UInt32(B), T: UInt32(T), H: UInt32(H), D: UInt32(D))
        go("sum_heads", [src.raw, dst.raw], p, grid: grid1(B * T * D), tpg: tpg1)
    }

    func reduceSum(_ x: Buf, into: Buf, n: Int, slot: Int) {
        reduce("reduce_sum", x, into, n, slot)
    }

    func reduceSumsq(_ x: Buf, into: Buf, n: Int, slot: Int) {
        reduce("reduce_sumsq", x, into, n, slot)
    }

    func adam(w: Buf, g: Buf, m: Buf, v: Buf, n: Int, lr: Float, b1: Float, b2: Float,
              eps: Float, wd: Float, invBC1: Float, invBC2: Float) {
        let p = AdamP(n: UInt32(n), lr: lr, b1: b1, b2: b2, eps: eps, wd: wd, invBC1: invBC1, invBC2: invBC2)
        go("adamw", [w.raw, g.raw, m.raw, v.raw], p, grid: grid1(n), tpg: tpg1)
    }

    private func reduce(_ name: String, _ x: Buf, _ into: Buf, _ n: Int, _ slot: Int) {
        let p = ReduceP(n: UInt32(n), slot: UInt32(slot), pad0: 0, pad1: 0)
        go(name, [x.raw, into.raw], p,
           grid: MTLSize(width: 1, height: 1, depth: 1),
           tpg: MTLSize(width: 256, height: 1, depth: 1),
           asGroups: true)
    }

    private func pipe(_ name: String) -> MTLComputePipelineState {
        if let hit = cache[name] { return hit }
        guard let fn = library.makeFunction(name: name) else { fatalError("missing kernel \(name)") }
        do {
            let p = try device.makeComputePipelineState(function: fn)
            if p.maxTotalThreadsPerThreadgroup < 256 {
                fatalError("\(name) threadgroup limit \(p.maxTotalThreadsPerThreadgroup) < 256")
            }
            cache[name] = p
            return p
        } catch {
            fatalError("pipeline \(name): \(error)")
        }
    }

    private func checkLayouts() {
        precondition(MemoryLayout<GemmP>.size == 32 && MemoryLayout<GemmP>.stride == 32)
        precondition(MemoryLayout<QuantP>.size == 16)
        precondition(MemoryLayout<LNP>.size == 16)
        precondition(MemoryLayout<UnaryP>.size == 16)
        precondition(MemoryLayout<ScaleP>.size == 16)
        precondition(MemoryLayout<BiasP>.size == 16)
        precondition(MemoryLayout<AttnP>.size == 32)
        precondition(MemoryLayout<MseP>.size == 16 && MemoryLayout<MseP>.stride == 16)
        precondition(MemoryLayout<CondP>.size == 16 && MemoryLayout<CondP>.stride == 16)
        precondition(MemoryLayout<EmbP>.size == 16)
        precondition(MemoryLayout<EmbBwdP>.size == 16)
        precondition(MemoryLayout<PosP>.size == 16)
        precondition(MemoryLayout<LossP>.size == 16)
        precondition(MemoryLayout<ReduceP>.size == 16)
        precondition(MemoryLayout<AdamP>.size == 32 && MemoryLayout<AdamP>.stride == 32)
        precondition(MemoryLayout<RopeP>.size == 16)
        precondition(MemoryLayout<MixP>.size == 16)
        precondition(MemoryLayout<LossMaskP>.size == 32 && MemoryLayout<LossMaskP>.stride == 32)
    }

    private func gemmP(_ M: Int, _ N: Int, _ K: Int, _ hasBias: Int, _ qGroup: Int) -> GemmP {
        GemmP(M: UInt32(M), N: UInt32(N), K: UInt32(K), hasBias: UInt32(hasBias),
              qGroup: UInt32(qGroup), pad0: 0, pad1: 0, pad2: 0)
    }

    private func attnP(_ B: Int, _ T: Int, _ H: Int, _ D: Int, causal: Bool = true) -> AttnP {
        AttnP(B: UInt32(B), T: UInt32(T), H: UInt32(H), D: UInt32(D),
              scale: 1 / sqrt(Float(D)), causal: causal ? 1 : 0, pad1: 0, pad2: 0)
    }

    func mse(pred: Buf, target: Buf, err: Buf, dpred: Buf, n: Int, invCount: Float) {
        let p = MseP(n: UInt32(n), pad0: 0, invCount: invCount, pad1: 0)
        go("mse_fwd", [pred.raw, target.raw, err.raw, dpred.raw], p, grid: grid1(n), tpg: tpg1)
    }

    func addCond(x: Buf, cond: Buf, y: Buf, B: Int, T: Int, C: Int) {
        let p = CondP(B: UInt32(B), T: UInt32(T), C: UInt32(C), pad: 0)
        go("add_cond", [x.raw, cond.raw, y.raw], p, grid: grid1(B * T * C), tpg: tpg1)
    }

    func sumCond(dy: Buf, dcond: Buf, B: Int, T: Int, C: Int) {
        let p = CondP(B: UInt32(B), T: UInt32(T), C: UInt32(C), pad: 0)
        go("sum_cond", [dy.raw, dcond.raw], p,
           grid: MTLSize(width: C, height: B, depth: 1), tpg: tpg2)
    }

    private func unary(_ n: Int) -> UnaryP {
        UnaryP(n: UInt32(n), pad0: 0, pad1: 0, pad2: 0)
    }

    private func grid1(_ n: Int) -> MTLSize { MTLSize(width: max(n, 1), height: 1, depth: 1) }
    private var tpg1: MTLSize { MTLSize(width: 256, height: 1, depth: 1) }
    private var tpg2: MTLSize { MTLSize(width: 16, height: 16, depth: 1) }

    private static func findKernels() -> URL {
        var cands: [URL] = []
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        cands.append(exe.appendingPathComponent("kernels.metal"))
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        cands.append(cwd.appendingPathComponent("metal/kernels.metal"))
        cands.append(cwd.appendingPathComponent("build/kernels.metal"))
        for c in cands where FileManager.default.fileExists(atPath: c.path) { return c }
        fputs("Could not find kernels.metal. Run from the repo, or place it next to the binary.\n", stderr)
        exit(1)
    }
}
