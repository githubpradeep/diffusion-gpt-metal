import Darwin
import Foundation
import Metal

/// Next-concept modeling for a photograph.
///
/// The picture is a residual palette: scale 1 is one color, and each later
/// scale is a finer grid of colors still missing. The looped block predicts
/// every code in one scale at once, from the picture the earlier scales have
/// already painted and from the name. Sampling is one forward pass per scale.
struct NCMConfig {
    var side: Int
    var finest: Int
    var names: Int
    var codes: Int
    var batch: Int
    var loops = 4
    var heads = 4
    var kvHeads = 1
    var embd = 128
    var intermediate = 512
    /// Replacement rate at the coarsest scale. It falls linearly to zero at the last scale.
    var corrupt: Float = 0.3
    var eps: Float = 1e-6
    var ropeTheta: Float = 10000
    var valueMix: Float = 0.5

    /// Repeated copies of the finest grid. The first copy is a flat color.
    /// Each later copy is a residual patch that fills the cell.
    var detail = 2
    /// Pixel size of one exact patch after the color grids. Zero skips that level.
    /// Sixteen means the last grid is the painter's 16×16 patches.
    var patch = 0
    var scales: [Int] {
        var out = Self.pyramid(side: side, finest: finest, detail: detail)
        if patch > 0 {
            precondition(side % patch == 0)
            out.append(side / patch)
        }
        return out
    }
    var levels: Int { scales.count }
    var maxTokens: Int { scales.map { $0 * $0 }.max() ?? 1 }
    var latent: Int { 3 }
    var headDim: Int { embd / heads }
    var kvDim: Int { kvHeads * headDim }

    /// Power-of-two grids that divide the side, up through `finest`.
    /// `detail` extra copies of the finest grid carry residual patches.
    static func pyramid(side: Int, finest: Int, detail: Int = 0) -> [Int] {
        precondition(side > 0 && finest > 0 && detail >= 0)
        var cap = 1
        while cap * 2 <= finest && side % (cap * 2) == 0 {
            cap *= 2
        }
        var s = 1
        var out: [Int] = []
        while s <= cap {
            out.append(s)
            s *= 2
        }
        if detail > 0, let last = out.last {
            for _ in 0..<detail { out.append(last) }
        }
        return out
    }

    func tokens(_ level: Int) -> Int {
        let s = scales[level]
        return s * s
    }

    /// Coarse scales are replaced more often. The last scale is left alone.
    func corruptRate(_ level: Int) -> Float {
        if corrupt <= 0 || levels <= 1 { return 0 }
        let u = Float(levels - 1 - level) / Float(levels - 1)
        return corrupt * u
    }

    func validate() {
        precondition(side > 0 && side % 2 == 0)
        precondition(!scales.isEmpty && scales[0] == 1 && side % scales[scales.count - 1] == 0)
        precondition(names > 0 && codes > 1 && batch > 0 && loops > 0)
        precondition(heads > 0 && kvHeads == 1, "this build repeats a single KV head")
        precondition(embd % heads == 0 && headDim % 2 == 0 && intermediate > 0)
        precondition(corrupt >= 0 && corrupt <= 1 && detail >= 0)
        precondition(patch == 0 || (patch > 0 && side % patch == 0 && detail == 0))
    }
}

enum NCMFiles {
    static let vocab = "vocab/photos.txt"
    static let rgb = "data/photos/rgb.bin"
    static let ckpt = "build/photos-ncm.ckpt"
}

/// One scale of the residual palette. The first visit to a grid is a flat color.
/// A repeated grid is a residual patch: the code fills every pixel of the cell.
final class ResidualPalette {
    let side: Int
    let scales: [Int]
    let codes: Int
    /// Exact patch size in pixels. The last scale is that grid, not a k-means book.
    let patch: Int
    var patchCount = 0
    var books: [[Float]]
    private var exactLookup: [Data: Int] = [:]

    var pixels: Int { side * side * 3 }

    /// Flat colors are 3 numbers. A repeated scale stores the whole cell.
    static func depth(side: Int, scales: [Int], level: Int) -> Int {
        let scale = scales[level]
        if scales.firstIndex(of: scale) == level { return 3 }
        let cell = side / scale
        return cell * cell * 3
    }

    func depth(_ level: Int) -> Int {
        if isExact(level) { return patch * patch * 3 }
        return Self.depth(side: side, scales: scales, level: level)
    }

    func isExact(_ level: Int) -> Bool {
        patch > 0 && level == scales.count - 1
    }

    func vocab(_ level: Int) -> Int {
        isExact(level) ? patchCount : codes
    }

    init(side: Int, scales: [Int], codes: Int, patch: Int = 0, books: [[Float]]? = nil) {
        precondition(side > 0 && codes > 1 && !scales.isEmpty)
        self.side = side
        self.scales = scales
        self.codes = codes
        self.patch = patch
        if let books {
            precondition(books.count == scales.count)
            self.books = books
            if patch > 0 {
                let dim = patch * patch * 3
                precondition(books[scales.count - 1].count % dim == 0)
                self.patchCount = books[scales.count - 1].count / dim
                rebuildLookup()
            }
            for level in 0..<scales.count where !isExact(level) {
                precondition(books[level].count == codes * Self.depth(side: side, scales: scales, level: level))
            }
        } else {
            self.books = (0..<scales.count).map { level in
                if patch > 0 && level == scales.count - 1 { return [Float]() }
                return [Float](repeating: 0, count: codes * Self.depth(side: side, scales: scales, level: level))
            }
        }
    }

    /// k-means on the residual of each scale, then freeze the books.
    @discardableResult
    func fit(_ images: [[Float]], steps: Int, seed: UInt64) -> Float {
        precondition(!images.isEmpty)
        var rng = RNG(seed)
        var residual = images
        let rounds = max(steps, 1)
        for level in 0..<scales.count where !isExact(level) {
            let dim = depth(level)
            var points: [Float] = []
            points.reserveCapacity(residual.count * scales[level] * scales[level] * dim)
            for img in residual {
                points.append(contentsOf: vectors(img, level: level))
            }
            let n = points.count / dim
            books[level] = kmeans(points, n: n, dim: dim, steps: rounds, rng: &rng)
            for i in residual.indices {
                let low = vectors(residual[i], level: level)
                let assigned = assign(low, level: level)
                var painted = [Float](repeating: 0, count: pixels)
                paint(assigned, level: level, into: &painted)
                for j in 0..<pixels { residual[i][j] -= painted[j] }
            }
        }
        if patch > 0 { installExact(images) }
        var err: Float = 0
        for img in images { err += mae(decode(encode(img)), img) }
        return err / Float(images.count)
    }

    func encode(_ image: [Float]) -> [[UInt32]] {
        var residual = image
        var out: [[UInt32]] = []
        out.reserveCapacity(scales.count)
        for level in 0..<scales.count {
            if isExact(level) {
                out.append(exactIDs(image))
                continue
            }
            let low = vectors(residual, level: level)
            let assigned = assign(low, level: level)
            out.append(assigned)
            var painted = [Float](repeating: 0, count: pixels)
            paint(assigned, level: level, into: &painted)
            for j in 0..<pixels { residual[j] -= painted[j] }
        }
        return out
    }

    func decode(_ levels: [[UInt32]]) -> [Float] {
        var image = [Float](repeating: 0, count: pixels)
        for level in 0..<min(levels.count, scales.count) {
            paint(levels[level], level: level, into: &image)
        }
        return image
    }

    /// Mean fraction of each book that the images actually use.
    func utilization(_ images: [[Float]]) -> Float {
        var sum: Float = 0
        var seen = Array(repeating: Set<UInt32>(), count: scales.count)
        for img in images {
            let coded = encode(img)
            for level in 0..<scales.count {
                for c in coded[level] { seen[level].insert(c) }
            }
        }
        for level in 0..<scales.count {
            let denom = Float(max(vocab(level), 1))
            sum += Float(seen[level].count) / denom
        }
        return sum / Float(scales.count)
    }

    func downsample(_ image: [Float], scale: Int) -> [Float] {
        let cell = side / scale
        let n = Float(cell * cell)
        var out = [Float](repeating: 0, count: scale * scale * 3)
        for y in 0..<scale {
            for x in 0..<scale {
                var r: Float = 0
                var g: Float = 0
                var b: Float = 0
                for iy in 0..<cell {
                    let row = (y * cell + iy) * side + x * cell
                    for ix in 0..<cell {
                        let s = (row + ix) * 3
                        r += image[s]
                        g += image[s + 1]
                        b += image[s + 2]
                    }
                }
                let d = (y * scale + x) * 3
                out[d] = r / n
                out[d + 1] = g / n
                out[d + 2] = b / n
            }
        }
        return out
    }

    /// Mean color, or the raw cell, whichever this level quantizes.
    func vectors(_ image: [Float], level: Int) -> [Float] {
        if depth(level) == 3 { return downsample(image, scale: scales[level]) }
        let scale = scales[level]
        let cell = side / scale
        let dim = cell * cell * 3
        var out = [Float](repeating: 0, count: scale * scale * dim)
        var w = 0
        for y in 0..<scale {
            for x in 0..<scale {
                for iy in 0..<cell {
                    let row = (y * cell + iy) * side + x * cell
                    for ix in 0..<cell {
                        let s = (row + ix) * 3
                        out[w] = image[s]
                        out[w + 1] = image[s + 1]
                        out[w + 2] = image[s + 2]
                        w += 3
                    }
                }
            }
        }
        return out
    }

    /// The last level copies the painter's patches. Same byte rounding, so a
    /// correct code is the sharp photograph, not a leftover added on top.
    func installExact(_ images: [[Float]]) {
        precondition(patch > 0 && isExact(scales.count - 1))
        var lookup: [Data: Int] = [:]
        var book: [Float] = []
        let grid = scales[scales.count - 1]
        for image in images {
            for y in 0..<grid {
                for x in 0..<grid {
                    let (key, floats) = patchKey(image, y: y, x: x)
                    if lookup[key] == nil {
                        lookup[key] = book.count / floats.count
                        book.append(contentsOf: floats)
                    }
                }
            }
        }
        books[scales.count - 1] = book
        exactLookup = lookup
        patchCount = lookup.count
        precondition(patchCount > 1)
    }

    private func quantByte(_ v: Float) -> UInt8 {
        var u = (v + 1) * 127.5
        if u < 0 { u = 0 }
        if u > 255 { u = 255 }
        return UInt8(u.rounded())
    }

    private func patchKey(_ image: [Float], y: Int, x: Int) -> (Data, [Float]) {
        let n = patch * patch * 3
        var bytes = [UInt8](repeating: 0, count: n)
        var floats = [Float](repeating: 0, count: n)
        var i = 0
        for iy in 0..<patch {
            for ix in 0..<patch {
                let s = ((y * patch + iy) * side + (x * patch + ix)) * 3
                for c in 0..<3 {
                    let byte = quantByte(image[s + c])
                    bytes[i] = byte
                    floats[i] = Float(byte) / 127.5 - 1
                    i += 1
                }
            }
        }
        return (Data(bytes), floats)
    }

    private func exactIDs(_ image: [Float]) -> [UInt32] {
        let grid = scales[scales.count - 1]
        var out = [UInt32](repeating: 0, count: grid * grid)
        var i = 0
        for y in 0..<grid {
            for x in 0..<grid {
                let (key, _) = patchKey(image, y: y, x: x)
                guard let id = exactLookup[key] else {
                    preconditionFailure("patch is missing from the book")
                }
                out[i] = UInt32(id)
                i += 1
            }
        }
        return out
    }

    private func rebuildLookup() {
        exactLookup = [:]
        guard patch > 0, isExact(scales.count - 1) else { return }
        let dim = patch * patch * 3
        let book = books[scales.count - 1]
        let n = book.count / dim
        patchCount = n
        for v in 0..<n {
            var bytes = [UInt8](repeating: 0, count: dim)
            for i in 0..<dim { bytes[i] = quantByte(book[v * dim + i]) }
            exactLookup[Data(bytes)] = v
        }
    }

    func paint(_ codesIn: [UInt32], level: Int, into image: inout [Float]) {
        let scale = scales[level]
        let cell = side / scale
        let book = books[level]
        let dim = depth(level)
        if isExact(level) {
            for i in 0..<codesIn.count {
                let y = i / scale
                let x = i % scale
                var e = Int(codesIn[i]) * dim
                for iy in 0..<cell {
                    let row = (y * cell + iy) * side + x * cell
                    for ix in 0..<cell {
                        let d = (row + ix) * 3
                        image[d] = book[e]
                        image[d + 1] = book[e + 1]
                        image[d + 2] = book[e + 2]
                        e += 3
                    }
                }
            }
            return
        }
        if dim == 3 {
            for i in 0..<codesIn.count {
                let y = i / scale
                let x = i % scale
                let e = Int(codesIn[i]) * 3
                let r = book[e]
                let g = book[e + 1]
                let b = book[e + 2]
                for iy in 0..<cell {
                    let row = (y * cell + iy) * side + x * cell
                    for ix in 0..<cell {
                        let d = (row + ix) * 3
                        image[d] += r
                        image[d + 1] += g
                        image[d + 2] += b
                    }
                }
            }
            return
        }
        for i in 0..<codesIn.count {
            let y = i / scale
            let x = i % scale
            var e = Int(codesIn[i]) * dim
            for iy in 0..<cell {
                let row = (y * cell + iy) * side + x * cell
                for ix in 0..<cell {
                    let d = (row + ix) * 3
                    image[d] += book[e]
                    image[d + 1] += book[e + 1]
                    image[d + 2] += book[e + 2]
                    e += 3
                }
            }
        }
    }

    private func assign(_ low: [Float], level: Int) -> [UInt32] {
        let book = books[level]
        let dim = depth(level)
        let n = low.count / dim
        var out = [UInt32](repeating: 0, count: n)
        for i in 0..<n {
            let base = i * dim
            var best = 0
            var bestD = Float.greatestFiniteMagnitude
            for v in 0..<codes {
                var d: Float = 0
                let e = v * dim
                for k in 0..<dim {
                    let z = low[base + k] - book[e + k]
                    d += z * z
                }
                if d < bestD {
                    bestD = d
                    best = v
                }
            }
            out[i] = UInt32(best)
        }
        return out
    }

    private func kmeans(_ points: [Float], n: Int, dim: Int, steps: Int, rng: inout RNG) -> [Float] {
        precondition(n > 0 && dim > 0)
        var book = [Float](repeating: 0, count: codes * dim)
        for v in 0..<codes {
            let i = rng.below(n) * dim
            for k in 0..<dim { book[v * dim + k] = points[i + k] }
        }
        for _ in 0..<steps {
            var sum = [Float](repeating: 0, count: codes * dim)
            var count = [Int](repeating: 0, count: codes)
            for i in 0..<n {
                let base = i * dim
                var best = 0
                var bestD = Float.greatestFiniteMagnitude
                for v in 0..<codes {
                    var d: Float = 0
                    let e = v * dim
                    for k in 0..<dim {
                        let z = points[base + k] - book[e + k]
                        d += z * z
                    }
                    if d < bestD {
                        bestD = d
                        best = v
                    }
                }
                count[best] += 1
                let e = best * dim
                for k in 0..<dim { sum[e + k] += points[base + k] }
            }
            for v in 0..<codes {
                if count[v] > 0 {
                    let inv = 1 / Float(count[v])
                    let e = v * dim
                    for k in 0..<dim { book[e + k] = sum[e + k] * inv }
                } else {
                    let i = rng.below(n) * dim
                    for k in 0..<dim { book[v * dim + k] = points[i + k] }
                }
            }
        }
        return book
    }
}

struct NCMExample {
    var name: Int
    var rgb: [Float]
    var codes: [[UInt32]]
}

struct NCMLevelBatch {
    var level: Int
    var names: [Int]
    var latent: [Float]
    var targets: [UInt32]
}

private final class NCMActs {
    let r1, r1b, r2, r2b, rf: Buf
    let ln1, q, k, v, probs, av, attn, mid, ln2, gate, up, sw, mlp, pre: Buf

    init(device: MTLDevice, batch: Int, block: Int, cfg: NCMConfig) {
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

final class NCM {
    let gpu: GPU
    let cfg: NCMConfig
    let palette: ResidualPalette
    private(set) var params: [Param] = []
    private var muon: [Param] = []
    private var adam: [Param] = []
    private let wProj, wHead, wName, wLevel, wRow, wCol: Param
    private let wExact: Param?
    private let wq, wk, wv, wo: Param
    private let wgate, wup, wdown: Param
    private let n1, n1b, n2, n2b, nf: Param
    private var acts: [NCMActs] = []
    private var xs: [Buf] = []
    private let cosTab, sinTab: Buf
    private let qPre, kPre, kRot, vPre, vMix, vFirst: Buf
    private let scores, dprobs: Buf
    private let latent, proj, logits, dlogits, nll, lossSum, sumsq: Buf
    private let cond, pos: Buf
    private let nameTok, levelTok, rowTok, colTok: UBuf
    private let targets: UBuf
    private let gx, t1, t2, dln, dq, dk, dv, dk1, dv1, dvPre, dFirst, dav, dsw, dg, du: Buf
    private var nameIDs: [Int] = []
    private(set) var B = 1
    private var T = 1
    private var level = 0

    var parameterCount: Int { params.reduce(0) { $0 + $1.count } }

    init(gpu: GPU, cfg: NCMConfig, palette: ResidualPalette, seed: UInt64) {
        cfg.validate()
        precondition(palette.side == cfg.side && palette.scales == cfg.scales && palette.codes == cfg.codes)
        if cfg.patch > 0 { precondition(palette.patchCount > 1 && palette.patch == cfg.patch) }
        self.gpu = gpu
        self.cfg = cfg
        self.palette = palette
        let device = gpu.device
        let C = cfg.embd
        let F = cfg.intermediate
        let KV = cfg.kvDim
        let V = max(cfg.codes, palette.patchCount)
        let maxM = cfg.batch * cfg.maxTokens
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

        let grid = cfg.scales.max() ?? 1
        wProj = make("wproj", C, cfg.latent, 0)
        wHead = make("whead", cfg.codes, C, 0)
        if palette.patchCount > 1 {
            wExact = make("wexact", palette.patchCount, C, 0)
        } else {
            wExact = nil
        }
        wName = make("wname", cfg.names, C, 1)
        wLevel = make("wlevel", cfg.levels, C, 1)
        // A flat coarse color is the same vector in every cell. Row and column
        // embeddings give each cell an address RoPE can sharpen but not invent.
        wRow = make("wrow", grid, C, 1)
        wCol = make("wcol", grid, C, 1)
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

        acts = (0..<cfg.loops).map { _ in NCMActs(device: device, batch: cfg.batch, block: cfg.maxTokens, cfg: cfg) }
        xs = (0...cfg.loops).map { _ in Buf(device, count: maxM * C) }
        let tables = ropeTables(maxT: cfg.maxTokens, dim: cfg.headDim, theta: cfg.ropeTheta)
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
        let attnN = cfg.batch * cfg.heads * cfg.maxTokens * cfg.maxTokens
        scores = Buf(device, count: attnN)
        dprobs = Buf(device, count: attnN)
        latent = Buf(device, count: maxM * cfg.latent)
        proj = Buf(device, count: maxM * C)
        logits = Buf(device, count: maxM * V)
        dlogits = Buf(device, count: maxM * V)
        nll = Buf(device, count: maxM)
        lossSum = Buf(device, count: 1)
        sumsq = Buf(device, count: max(muon.count + adam.count, 1))
        cond = Buf(device, count: maxB * C)
        pos = Buf(device, count: maxM * C)
        nameTok = UBuf(device, count: maxM)
        levelTok = UBuf(device, count: maxM)
        rowTok = UBuf(device, count: maxM)
        colTok = UBuf(device, count: maxM)
        targets = UBuf(device, count: maxM)
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
        var wide = [wHead, wq, wk, wv, wo, wgate, wup, wdown]
        if let wExact { wide.append(wExact) }
        for p in wide {
            fillNormal(p.w, std: std, rng: &rng)
        }
        // RGB lives near [-1, 1]. A wider projection keeps that signal next to the name row.
        fillNormal(wProj.w, std: 0.15, rng: &rng)
        fillNormal(wName.w, std: std, rng: &rng)
        fillNormal(wLevel.w, std: std, rng: &rng)
        fillNormal(wRow.w, std: std, rng: &rng)
        fillNormal(wCol.w, std: std, rng: &rng)
        for p in [n1, n1b, n2, n2b, nf] {
            for i in 0..<p.count { p.w.p[i] = 1 }
        }
    }

    /// Color cells count once. The exact patch grid counts as much as every color cell together.
    private func levelMass(_ level: Int) -> Float {
        if palette.isExact(level) {
            let color = (0..<level).reduce(0) { $0 + cfg.tokens($1) }
            return Float(max(color, cfg.tokens(level)))
        }
        return Float(cfg.tokens(level))
    }

    private func massSum() -> Float {
        (0..<cfg.levels).reduce(Float(0)) { $0 + levelMass($1) }
    }

    func step(_ batches: [NCMLevelBatch], lr: Float, stepIndex: Int, clip: Float = 0,
              b1: Float = 0.9, b2: Float = 0.95, adamEps: Float = 1e-8,
              muonBeta: Float = 0.95) -> Float {
        gpu.begin()
        zeroGrads()
        gpu.end()
        var total: Float = 0
        var mass: Float = 0
        for batch in batches {
            bind(batch)
            gpu.begin()
            forward(withLoss: true)
            backward()
            gpu.end()
            let m = levelMass(level)
            total += lossSum.p[0] / Float(B * T) * m
            mass += m
        }
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
                     eps: adamEps, wd: p.decay ? 0 : 0, invBC1: invBC1, invBC2: invBC2)
        }
        gpu.end()
        return total / max(mass, 1)
    }

    @discardableResult
    func lossAndGrad(_ batches: [NCMLevelBatch]) -> Float {
        gpu.begin()
        zeroGrads()
        gpu.end()
        var total: Float = 0
        var mass: Float = 0
        for batch in batches {
            bind(batch)
            gpu.begin()
            forward(withLoss: true)
            backward()
            gpu.end()
            let m = levelMass(level)
            total += lossSum.p[0] / Float(B * T) * m
            mass += m
        }
        return total / max(mass, 1)
    }

    func lossOnly(_ batches: [NCMLevelBatch]) -> Float {
        var total: Float = 0
        var mass: Float = 0
        for batch in batches {
            bind(batch)
            gpu.begin()
            forward(withLoss: true)
            gpu.end()
            let m = levelMass(level)
            total += lossSum.p[0] / Float(B * T) * m
            mass += m
        }
        return total / max(mass, 1)
    }

    /// One scale at a time, from a blank image. `frames` is the picture after each scale.
    /// `loops` is the four passes of the last scale, each painted on top of the coarser picture.
    func generate(name: Int, temp: Float, rng: inout RNG) -> (frames: [[Float]], loops: [[Float]]) {
        let pixels = palette.pixels
        var recon = [Float](repeating: 0, count: pixels)
        var frames: [[Float]] = []
        var loopFrames: [[Float]] = []
        frames.reserveCapacity(cfg.levels)
        for k in 0..<cfg.levels {
            let low = palette.downsample(recon, scale: cfg.scales[k])
            let last = k == cfg.levels - 1
            let pack = predict(low, name: name, level: k, loops: last)
            let n = cfg.tokens(k)
            var chosen = [UInt32](repeating: 0, count: n)
            for i in 0..<n {
                chosen[i] = UInt32(pickCode(pack.logits, row: i, V: palette.vocab(k), temp: temp, rng: &rng))
            }
            if last {
                let prefix = recon
                for pass in pack.passes {
                    var image = prefix
                    var codes = [UInt32](repeating: 0, count: n)
                    for i in 0..<n {
                        codes[i] = UInt32(pickCode(pass, row: i, V: palette.vocab(k), temp: 0, rng: &rng))
                    }
                    palette.paint(codes, level: k, into: &image)
                    loopFrames.append(image)
                }
            }
            palette.paint(chosen, level: k, into: &recon)
            frames.append(recon)
        }
        return (frames, loopFrames)
    }

    /// Teacher-forced fraction of codes that match, conditioning on the true coarser scales.
    /// The second value is that fraction per scale, coarse first.
    func codeAccuracy(_ examples: [NCMExample]) -> (all: Float, levels: [Float]) {
        var hit = [Int](repeating: 0, count: cfg.levels)
        var total = [Int](repeating: 0, count: cfg.levels)
        var scratch = RNG(1)
        for ex in examples {
            var recon = [Float](repeating: 0, count: palette.pixels)
            for k in 0..<cfg.levels {
                let low = palette.downsample(recon, scale: cfg.scales[k])
                let pack = predict(low, name: ex.name, level: k, loops: false)
                let n = cfg.tokens(k)
                for i in 0..<n {
                    let guess = pickCode(pack.logits, row: i, V: palette.vocab(k), temp: 0, rng: &scratch)
                    if UInt32(guess) == ex.codes[k][i] { hit[k] += 1 }
                    total[k] += 1
                }
                palette.paint(ex.codes[k], level: k, into: &recon)
            }
        }
        let allHit = hit.reduce(0, +)
        let allN = total.reduce(0, +)
        let levels = zip(hit, total).map { n, d in d == 0 ? Float(0) : Float(n) / Float(d) }
        return (allN == 0 ? 0 : Float(allHit) / Float(allN), levels)
    }

    func save(_ url: URL) throws {
        var data = Data()
        data.append(contentsOf: [0x54, 0x4E, 0x43, 0x4D]) // TNCM
        ncmAppendU32(cfg.patch > 0 ? 2 : 1, &data)
        for v in [cfg.side, cfg.finest, cfg.names, cfg.codes, cfg.loops, cfg.heads, cfg.kvHeads,
                  cfg.embd, cfg.intermediate, cfg.levels] {
            ncmAppendU32(UInt32(v), &data)
        }
        if cfg.patch > 0 {
            ncmAppendU32(UInt32(cfg.patch), &data)
            ncmAppendU32(UInt32(palette.patchCount), &data)
        }
        var mix = cfg.valueMix
        withUnsafeBytes(of: &mix) { data.append(contentsOf: $0) }
        var corrupt = cfg.corrupt
        withUnsafeBytes(of: &corrupt) { data.append(contentsOf: $0) }
        for s in cfg.scales { ncmAppendU32(UInt32(s), &data) }
        for book in palette.books {
            book.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        for p in params {
            data.append(Data(bytes: p.w.p, count: p.count * MemoryLayout<Float>.size))
        }
        try data.write(to: url)
    }

    static func load(gpu: GPU, url: URL, batch: Int, seed: UInt64) throws -> NCM {
        let data = try Data(contentsOf: url)
        var off = 0
        func take(_ n: Int) -> Data {
            let s = data.subdata(in: off..<(off + n))
            off += n
            return s
        }
        func u32() -> Int { Int(take(4).withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }) }
        func f32() -> Float { take(4).withUnsafeBytes { $0.load(as: Float.self) } }
        let magic = take(4)
        precondition(Array(magic) == [0x54, 0x4E, 0x43, 0x4D], "not a next-concept checkpoint")
        let version = u32()
        precondition(version == 1 || version == 2, "unsupported next-concept checkpoint \(version)")
        let side = u32()
        let finest = u32()
        let names = u32()
        let codes = u32()
        let loops = u32()
        let heads = u32()
        let kvHeads = u32()
        let embd = u32()
        let intermediate = u32()
        let levels = u32()
        var patch = 0
        var patchCount = 0
        if version == 2 {
            patch = u32()
            patchCount = u32()
        }
        let mix = f32()
        let corrupt = f32()
        var scales: [Int] = []
        for _ in 0..<levels { scales.append(u32()) }
        var colorScales = scales
        if patch > 0 { colorScales.removeLast() }
        let cap = colorScales.max() ?? 1
        let detail = max(0, colorScales.filter { $0 == cap }.count - 1)
        var books: [[Float]] = []
        for level in 0..<levels {
            let n: Int
            if patch > 0 && level == levels - 1 {
                n = patchCount * patch * patch * 3
            } else {
                n = codes * ResidualPalette.depth(side: side, scales: scales, level: level)
            }
            let slice = take(n * MemoryLayout<Float>.size)
            var book = [Float](repeating: 0, count: n)
            _ = book.withUnsafeMutableBytes { raw in slice.copyBytes(to: raw) }
            books.append(book)
        }
        var cfg = NCMConfig(side: side, finest: finest, names: names, codes: codes, batch: batch,
                            loops: loops, heads: heads, kvHeads: kvHeads, embd: embd,
                            intermediate: intermediate, corrupt: corrupt, valueMix: mix)
        cfg.detail = detail
        cfg.patch = patch
        precondition(cfg.scales == scales, "checkpoint scales do not match the pyramid")
        let palette = ResidualPalette(side: side, scales: scales, codes: codes, patch: patch, books: books)
        if patch > 0 { precondition(palette.patchCount == patchCount) }
        let model = NCM(gpu: gpu, cfg: cfg, palette: palette, seed: seed)
        for p in model.params {
            let nbytes = p.count * MemoryLayout<Float>.size
            let slice = take(nbytes)
            slice.withUnsafeBytes { raw in
                p.w.p.update(from: raw.bindMemory(to: Float.self).baseAddress!, count: p.count)
            }
        }
        precondition(off == data.count, "next-concept checkpoint length does not match the model")
        return model
    }

    // MARK: - forward / backward

    private func bind(_ batch: NCMLevelBatch) {
        let tokens = cfg.tokens(batch.level)
        precondition(batch.names.count > 0 && batch.names.count <= cfg.batch)
        precondition(batch.level >= 0 && batch.level < cfg.levels)
        B = batch.names.count
        T = tokens
        level = batch.level
        let m = B * T
        precondition(batch.latent.count == m * cfg.latent)
        precondition(batch.targets.count == m)
        nameIDs = batch.names
        latent.upload(batch.latent)
        targets.upload(batch.targets)
        var nameIds = [UInt32](repeating: 0, count: m)
        let levelIds = [UInt32](repeating: UInt32(level), count: m)
        var rowIds = [UInt32](repeating: 0, count: m)
        var colIds = [UInt32](repeating: 0, count: m)
        let scale = cfg.scales[level]
        let C = cfg.embd
        for b in 0..<B {
            let id = nameIDs[b]
            precondition(id >= 0 && id < cfg.names)
            let nameRow = id * C
            let levelRow = level * C
            for c in 0..<C {
                cond.p[b * C + c] = wName.w.p[nameRow + c] + wLevel.w.p[levelRow + c]
            }
            for t in 0..<T {
                let y = t / scale
                let x = t % scale
                let at = b * T + t
                nameIds[at] = UInt32(id)
                rowIds[at] = UInt32(y)
                colIds[at] = UInt32(x)
                let dst = at * C
                let row = y * C
                let col = x * C
                for c in 0..<C {
                    pos.p[dst + c] = wRow.w.p[row + c] + wCol.w.p[col + c]
                }
            }
        }
        nameTok.upload(nameIds)
        levelTok.upload(levelIds)
        rowTok.upload(rowIds)
        colTok.upload(colIds)
    }

    private func predict(_ low: [Float], name: Int, level k: Int, loops wantLoops: Bool) -> (logits: [Float], passes: [[Float]]) {
        let tokens = cfg.tokens(k)
        precondition(low.count == tokens * cfg.latent)
        let batch = NCMLevelBatch(level: k, names: [name], latent: low,
                                  targets: [UInt32](repeating: 0, count: tokens))
        bind(batch)
        gpu.begin()
        forward(withLoss: false)
        gpu.end()
        let main = logits.download(tokens * palette.vocab(k))
        if !wantLoops { return (main, []) }
        var passes: [[Float]] = []
        for p in 1...cfg.loops {
            gpu.begin()
            let head = palette.isExact(k) ? wExact! : wHead
            gemm(head, xs[p], logits, M: T, N: palette.vocab(k), K: cfg.embd)
            gpu.end()
            passes.append(logits.download(tokens * palette.vocab(k)))
        }
        return (main, passes)
    }

    private func forward(withLoss: Bool) {
        let M = B * T
        let C = cfg.embd
        let F = cfg.intermediate
        let H = cfg.heads
        let D = cfg.headDim
        let KV = cfg.kvDim
        let V = palette.vocab(level)
        let mix = cfg.valueMix
        let head = palette.isExact(level) ? wExact! : wHead
        gemm(wProj, latent, proj, M: M, N: C, K: cfg.latent)
        gpu.addCond(x: proj, cond: cond, y: xs[0], B: B, T: T, C: C)
        gpu.acc(xs[0], pos, n: M * C)
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
        gemm(head, xs[cfg.loops], logits, M: M, N: V, K: C)
        if withLoss {
            gpu.loss(logits: logits, targets: targets, nll: nll, dlogits: dlogits, BT: M, V: V)
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
        let V = palette.vocab(level)
        let mix = cfg.valueMix
        let head = palette.isExact(level) ? wExact! : wHead
        let nC = M * C
        let nKV = M * KV
        gpu.zero(dFirst, n: nKV)
        // Color cells share the loss by count. The exact patch grid weighs as
        // much as all of those color cells together.
        gpu.scale(dlogits, n: M * V, by: levelMass(level) / massSum())
        gpu.gemmDWAcc(dy: dlogits, x: xs[cfg.loops], dw: head.g, M: M, N: V, K: C)
        gpu.gemmDX(bits: 32, w: head.w, q: nil, scale: nil, dy: dlogits, dx: gx,
                   M: M, N: V, K: C, qGroup: 8)
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
        let grid = cfg.scales[cfg.scales.count - 1]
        gpu.embedBwd(tokens: nameTok, dx: gx, dwe: wName.g, B: B, T: T, C: C, V: cfg.names)
        gpu.embedBwd(tokens: levelTok, dx: gx, dwe: wLevel.g, B: B, T: T, C: C, V: cfg.levels)
        gpu.embedBwd(tokens: rowTok, dx: gx, dwe: wRow.g, B: B, T: T, C: C, V: grid)
        gpu.embedBwd(tokens: colTok, dx: gx, dwe: wCol.g, B: B, T: T, C: C, V: grid)
        gpu.gemmDWAcc(dy: gx, x: latent, dw: wProj.g, M: M, N: C, K: cfg.latent)
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

private func ncmAppendU32(_ v: UInt32, _ data: inout Data) {
    var le = v.littleEndian
    withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
}

private func ncmMAE(_ a: [Float], _ b: [Float]) -> Float {
    let n = min(a.count, b.count)
    if n == 0 { return 0 }
    var s: Float = 0
    for i in 0..<n { s += abs(a[i] - b[i]) }
    return s / Float(n)
}

private func mae(_ a: [Float], _ b: [Float]) -> Float { ncmMAE(a, b) }

func pickCode(_ logits: [Float], row: Int, V: Int, temp: Float, rng: inout RNG) -> Int {
    let base = row * V
    if temp <= 0 {
        var best = 0
        var m = -Float.greatestFiniteMagnitude
        for v in 0..<V {
            let z = logits[base + v]
            if z > m {
                m = z
                best = v
            }
        }
        return best
    }
    var maxv = -Float.greatestFiniteMagnitude
    for v in 0..<V { maxv = max(maxv, logits[base + v]) }
    var sum: Float = 0
    var weights = [Float](repeating: 0, count: V)
    let invT = 1 / temp
    for v in 0..<V {
        let w = exp((logits[base + v] - maxv) * invT)
        weights[v] = w
        sum += w
    }
    var u = rng.uniform() * sum
    for v in 0..<V {
        u -= weights[v]
        if u <= 0 { return v }
    }
    return V - 1
}

func corruptCodes(_ codes: [UInt32], rate: Float, vocab: Int, rng: inout RNG) -> [UInt32] {
    if rate <= 0 || vocab <= 1 { return codes }
    var out = codes
    for i in 0..<out.count {
        if rng.uniform() >= rate { continue }
        var j = rng.below(vocab - 1)
        if j >= Int(out[i]) { j += 1 }
        out[i] = UInt32(j)
    }
    return out
}

/// One training draw. Corrupted coarser codes build the conditioning picture.
/// The labels stay clean. Levels do not attend to each other: each is its own sequence.
func ncmBatches(_ examples: [NCMExample], picks: [Int], cfg: NCMConfig, palette: ResidualPalette,
                rng: inout RNG) -> [NCMLevelBatch] {
    let pixels = palette.pixels
    var latents = Array(repeating: [Float](), count: cfg.levels)
    var targets = Array(repeating: [UInt32](), count: cfg.levels)
    var names: [Int] = []
    for pick in picks {
        let ex = examples[pick]
        names.append(ex.name)
        var recon = [Float](repeating: 0, count: pixels)
        for k in 0..<cfg.levels {
            latents[k].append(contentsOf: palette.downsample(recon, scale: cfg.scales[k]))
            targets[k].append(contentsOf: ex.codes[k])
            let used = corruptCodes(ex.codes[k], rate: cfg.corruptRate(k), vocab: palette.vocab(k), rng: &rng)
            palette.paint(used, level: k, into: &recon)
        }
    }
    return (0..<cfg.levels).map { k in
        NCMLevelBatch(level: k, names: names, latent: latents[k], targets: targets[k])
    }
}

func trainNCM(gpu: GPU, args: Args) throws {
    let vocabPath = args.str("vocab", NCMFiles.vocab)
    let rgbPath = args.str("rgb", NCMFiles.rgb)
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
    let batch = args.int("batch", 4)
    let steps = args.int("steps", 800)
    let lr = args.float("lr", 0.004)
    let seed = UInt64(args.int("seed", 456))
    let ckptPath = args.str("ckpt", NCMFiles.ckpt)
    let finest = args.int("finest", file.side >= 64 ? 16 : 8)
    let codes = args.int("codes", 64)
    var cfg = NCMConfig(side: file.side, finest: finest, names: names, codes: codes, batch: batch)
    cfg.loops = args.int("loops", 4)
    cfg.embd = args.int("embd", 128)
    cfg.intermediate = args.int("intermediate", 512)
    cfg.heads = args.int("heads", 4)
    cfg.corrupt = args.float("corrupt", 0.3)
    cfg.detail = args.int("detail", 0)
    cfg.patch = args.int("patch", file.side >= 64 ? 16 : 0)
    cfg.validate()
    let holdout = args.int("holdout", 1)
    let trainIcons: [IconRGB]
    if holdout <= 1 {
        trainIcons = icons
    } else {
        trainIcons = icons.enumerated().filter { $0.offset % holdout != 0 }.map(\.element)
    }
    let palette = ResidualPalette(side: cfg.side, scales: cfg.scales, codes: cfg.codes, patch: cfg.patch)
    let tokSteps = args.int("tok-steps", 8)
    let fitMAE = palette.fit(trainIcons.map(\.rgb), steps: tokSteps, seed: seed &+ 17)
    var examples: [NCMExample] = []
    for icon in trainIcons {
        examples.append(NCMExample(name: icon.name, rgb: icon.rgb, codes: palette.encode(icon.rgb)))
    }
    let used = palette.utilization(trainIcons.map(\.rgb))
    let model = NCM(gpu: gpu, cfg: cfg, palette: palette, seed: seed)
    print("device \(gpu.device.name)")
    print("next-concept  params \(model.parameterCount)  names \(names)  embd \(cfg.embd)  ffn \(cfg.intermediate)")
    print("scales \(cfg.scales)  codes \(cfg.codes)  patches \(palette.patchCount)  loops \(cfg.loops)  heads \(cfg.heads)/\(cfg.kvHeads)")
    print(String(format: "palette mae %.4f  utilization %.0f%%  corrupt %.2f  batch %d  steps %d",
                 fitMAE, used * 100, cfg.corrupt, batch, steps))
    print("train \(examples.count)")
    var rng = RNG(UInt64(args.int("data-seed", 42)))
    var valRNG = RNG(7)
    let t0 = CFAbsoluteTimeGetCurrent()
    for step in 1...steps {
        let sched = macroLearningRate(step: step, total: steps, base: lr)
        var picks: [Int] = []
        for _ in 0..<batch { picks.append(rng.below(examples.count)) }
        let drawn = ncmBatches(examples, picks: picks, cfg: cfg, palette: palette, rng: &rng)
        let loss = model.step(drawn, lr: sched, stepIndex: step)
        if !loss.isFinite {
            fputs("loss became non-finite.\n", stderr)
            exit(1)
        }
        if step == 1 || step % 20 == 0 || step == steps {
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let rate = Double(step * batch) / max(dt, 1e-6)
            var line = String(format: "step %4d/%d  ce %.4f  lr %.2e  %.2f img/s", step, steps, loss, sched, rate)
            if step % 100 == 0 || step == steps {
                var held: [Int] = []
                let n = min(batch, examples.count)
                for i in 0..<n { held.append(i) }
                var quiet = cfg
                quiet.corrupt = 0
                let cleanBatch = ncmBatches(examples, picks: held, cfg: quiet, palette: palette, rng: &valRNG)
                let v = model.lossOnly(cleanBatch)
                line += String(format: "  val %.4f", v)
            }
            print(line)
            fflush(stdout)
        }
    }
    let scored = model.codeAccuracy(examples)
    let acc = scored.all
    let perLevel = scored.levels.map { String(format: "%.0f%%", $0 * 100) }.joined(separator: " ")
    var sampleRNG = RNG(21)
    var photoErr: Float = 0
    var codeErr: Float = 0
    for ex in examples {
        let drawn = model.generate(name: ex.name, temp: 0, rng: &sampleRNG)
        let last = drawn.frames.last ?? []
        photoErr += ncmMAE(last, ex.rgb)
        codeErr += ncmMAE(last, palette.decode(ex.codes))
    }
    let n = Float(examples.count)
    print(String(format: "teacher-forced codes %.1f%%  [\(perLevel)]  sample vs palette %.4f  sample vs photo %.4f",
                 acc * 100, codeErr / n, photoErr / n))
    let url = URL(fileURLWithPath: ckptPath)
    let dir = url.deletingLastPathComponent().path
    if dir != "" && dir != "." {
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }
    try model.save(url)
    print("wrote \(ckptPath)")
}

func drawImageNCM(model: NCM, words: WordList, name: String, temp: Float, rng: inout RNG) -> [String: Any] {
    guard let index = iconNameIndex(words, name) else {
        return ["error": "unknown", "unknown": [name]]
    }
    if index >= model.cfg.names {
        return ["error": "unknown", "unknown": [name]]
    }
    let drawn = model.generate(name: index, temp: temp, rng: &rng)
    return imagePayload(kind: "ncm", name: words.tokens[Int(Pictures.name0) + index],
                        params: model.parameterCount,
                        frames: drawn.frames.map(rgbBytes),
                        loops: drawn.loops.map(rgbBytes),
                        side: model.cfg.side)
}

func runImageNCMRepl(gpu: GPU, args: Args) throws {
    let words = try WordList.load(args.str("vocab", NCMFiles.vocab))
    let model = try NCM.load(gpu: gpu, url: URL(fileURLWithPath: args.str("ckpt", NCMFiles.ckpt)),
                             batch: 1, seed: 1)
    var rng = RNG(args.flags["seed"].flatMap(UInt64.init) ?? 1)
    let temp = args.float("temp", 0)
    while let line = readLine(strippingNewline: true) {
        let name = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { continue }
        emitImage(drawImageNCM(model: model, words: words, name: name, temp: temp, rng: &rng))
    }
}

func runNCMChecks(gpu: GPU) throws -> Bool {
    requirePicture(NCMFiles.vocab)
    requirePicture(NCMFiles.rgb)
    let words = try WordList.load(NCMFiles.vocab)
    let file = loadIconFile(NCMFiles.rgb)
    var report = Report()
    let photoScales = NCMConfig.pyramid(side: 128, finest: 32)
    report.expect(photoScales == [1, 2, 4, 8, 16, 32], "128 photo pyramid \(photoScales)")
    let patchScales = NCMConfig.pyramid(side: 128, finest: 32, detail: 2)
    report.expect(patchScales == [1, 2, 4, 8, 16, 32, 32, 32], "patch rounds \(patchScales)")
    var exactScales = NCMConfig(side: 128, finest: 16, names: 1, codes: 64, batch: 1)
    exactScales.detail = 0
    exactScales.patch = 16
    report.expect(exactScales.scales == [1, 2, 4, 8, 16, 8], "exact patches \(exactScales.scales)")
    let iconScales = NCMConfig.pyramid(side: 24, finest: 8)
    report.expect(iconScales == [1, 2, 4, 8], "24 icon pyramid \(iconScales)")

    let icon = file.icons[0]
    let label = words.tokens[Int(Pictures.name0) + icon.name]
    let cfg = NCMConfig(side: file.side, finest: 8, names: max(icon.name + 1, 1), codes: 32, batch: 2,
                        loops: 4, heads: 4, embd: 64, intermediate: 128, corrupt: 0)
    cfg.validate()
    let palette = ResidualPalette(side: cfg.side, scales: cfg.scales, codes: cfg.codes)
    let fitMAE = palette.fit([icon.rgb], steps: 6, seed: 3)
    let coded = palette.encode(icon.rgb)
    let decoded = palette.decode(coded)
    let round = ncmMAE(decoded, icon.rgb)
    report.expect(abs(round - fitMAE) < 1e-5, String(format: "palette round-trip %.4f", round))
    let gray = ncmMAE([Float](repeating: 0, count: icon.rgb.count), icon.rgb)
    report.expect(round < gray * 0.75, String(format: "palette mae %.4f under gray %.4f", round, gray))
    var prefix = [Float](repeating: 0, count: palette.pixels)
    var firstErr: Float = 0
    var lastErr: Float = 0
    for k in 0..<cfg.scales.count {
        palette.paint(coded[k], level: k, into: &prefix)
        let err = ncmMAE(prefix, icon.rgb)
        if k == 0 { firstErr = err }
        lastErr = err
    }
    report.expect(lastErr < firstErr && lastErr < gray,
                  String(format: "finer scale %.4f beats the first color %.4f and gray %.4f", lastErr, firstErr, gray))

    let model = NCM(gpu: gpu, cfg: cfg, palette: palette, seed: 5)
    let example = NCMExample(name: icon.name, rgb: icon.rgb, codes: coded)
    var gradRNG = RNG(1)
    let batches = ncmBatches([example, example], picks: [0, 0], cfg: cfg, palette: palette, rng: &gradRNG)
    _ = model.lossAndGrad(batches)
    let probes = ["wproj", "whead", "n1", "wrow"]
    var analytic: [Float] = []
    for name in probes { analytic.append(model.probe(name).g.p[0]) }
    let eps: Float = 1e-3
    for (i, name) in probes.enumerated() {
        let p = model.probe(name)
        let w0 = p.w.p[0]
        p.w.p[0] = w0 + eps
        let higher = model.lossOnly(batches)
        p.w.p[0] = w0 - eps
        let lower = model.lossOnly(batches)
        p.w.p[0] = w0
        let numeric = (higher - lower) / (eps + eps)
        let gap = abs(numeric - analytic[i])
        let scale = max(abs(numeric), abs(analytic[i]), 1e-3)
        // A gradient near 1e-3 moves the loss by a few fp32 ulps across this step.
        let ok = gap / scale < 0.10 || gap < 5e-4
        report.expect(ok, String(format: "\(name) grad  numeric %.5f  analytic %.5f", numeric, analytic[i]))
    }

    print("next-concept memorization of \(label)")
    var trainCfg = cfg
    trainCfg.batch = 4
    let trainer = NCM(gpu: gpu, cfg: trainCfg, palette: palette, seed: 9)
    var rng = RNG(3)
    var initial: Float = 0
    var final: Float = 0
    let steps = 120
    for step in 1...steps {
        let loss = trainer.step(ncmBatches([example], picks: [0, 0, 0, 0], cfg: trainCfg, palette: palette, rng: &rng),
                                lr: 0.01, stepIndex: step)
        if step == 1 { initial = loss }
        final = loss
        if step == 1 || step % 40 == 0 || step == steps {
            print(String(format: "  ncm step %d  ce %.4f", step, loss))
        }
    }
    report.expect(final < initial - 0.5, String(format: "loss fell  %.3f → %.3f", initial, final))
    let scored = trainer.codeAccuracy([example])
    let acc = scored.all
    let perLevel = scored.levels.map { String(format: "%.0f%%", $0 * 100) }.joined(separator: " ")
    var sampleRNG = RNG(21)
    let drawn = trainer.generate(name: icon.name, temp: 0, rng: &sampleRNG)
    let last = drawn.frames.last ?? []
    let vsPalette = ncmMAE(last, decoded)
    let vsPhoto = ncmMAE(last, icon.rgb)
    print(String(format: "  codes %.1f%%  [\(perLevel)]  vs palette %.4f  vs photo %.4f  frames %d",
                 acc * 100, vsPalette, vsPhoto, drawn.frames.count))
    report.expect(drawn.frames.count == cfg.levels && drawn.loops.count == cfg.loops,
                  "one frame per scale and one loop per pass")
    report.expect(acc > 0.9 && vsPalette < 0.05,
                  String(format: "sample follows the palette  acc %.1f%%  mae %.4f", acc * 100, vsPalette))
    report.expect(vsPhoto < gray, String(format: "sample %.4f closer than gray %.4f", vsPhoto, gray))
    let ckpt = URL(fileURLWithPath: "build/ncm-check.ckpt")
    try FileManager.default.createDirectory(atPath: "build", withIntermediateDirectories: true)
    try trainer.save(ckpt)
    let loaded = try NCM.load(gpu: gpu, url: ckpt, batch: 1, seed: 1)
    let loadedAcc = loaded.codeAccuracy([example]).all
    report.expect(loadedAcc == acc, String(format: "checkpoint reloads the codes  %.1f%%", loadedAcc * 100))
    try FileManager.default.removeItem(at: ckpt)

    var exact = NCMConfig(side: file.side, finest: 8, names: max(icon.name + 1, 1), codes: 32, batch: 2,
                          loops: 4, heads: 4, embd: 64, intermediate: 128, corrupt: 0)
    exact.detail = 0
    exact.patch = 16
    exact.validate()
    let exactPalette = ResidualPalette(side: exact.side, scales: exact.scales, codes: exact.codes, patch: exact.patch)
    let exactMAE = exactPalette.fit([icon.rgb], steps: 4, seed: 3)
    report.expect(exactMAE < 0.01 && exactPalette.patchCount > 1 && exactPalette.patchCount <= 64,
                  String(format: "exact patches %.4f  book %d", exactMAE, exactPalette.patchCount))
    let exactExample = NCMExample(name: icon.name, rgb: icon.rgb, codes: exactPalette.encode(icon.rgb))
    print("exact-patch memorization of \(label)")
    let exactModel = NCM(gpu: gpu, cfg: exact, palette: exactPalette, seed: 9)
    var exactRNG = RNG(3)
    var exactLoss: Float = 0
    for step in 1...80 {
        exactLoss = exactModel.step(ncmBatches([exactExample], picks: [0, 0], cfg: exact, palette: exactPalette, rng: &exactRNG),
                                     lr: 0.01, stepIndex: step)
    }
    let exactAcc = exactModel.codeAccuracy([exactExample]).all
    var exactSample = RNG(4)
    let exactDrawn = exactModel.generate(name: icon.name, temp: 0, rng: &exactSample)
    let exactPhoto = ncmMAE(exactDrawn.frames.last ?? [], icon.rgb)
    print(String(format: "  exact ce %.4f  codes %.1f%%  vs photo %.4f", exactLoss, exactAcc * 100, exactPhoto))
    report.expect(exactAcc > 0.9 && exactPhoto < 0.02,
                  String(format: "exact sample follows the photo  acc %.1f%%  mae %.4f", exactAcc * 100, exactPhoto))
    try exactModel.save(ckpt)
    let exactLoaded = try NCM.load(gpu: gpu, url: ckpt, batch: 1, seed: 1)
    report.expect(exactLoaded.codeAccuracy([exactExample]).all == exactAcc, "exact checkpoint reloads")
    try FileManager.default.removeItem(at: ckpt)

    if report.failed > 0 {
        print("ncm checks failed: \(report.failed)")
        return false
    }
    print("ncm checks passed")
    return true
}
