// Parameter blocks shared with metal/kernels.metal.
// Field order is the Metal layout. Every struct is a multiple of 4 bytes
// with 4-byte alignment so Swift and MSL agree.

let defaultQGroup = 8

struct GemmP {
    var M: UInt32
    var N: UInt32
    var K: UInt32
    var hasBias: UInt32
    var qGroup: UInt32
    var pad0: UInt32
    var pad1: UInt32
    var pad2: UInt32
}

struct QuantP {
    var rows: UInt32
    var cols: UInt32
    var qmax: Float
    var qGroup: UInt32
}

struct LNP {
    var rows: UInt32
    var cols: UInt32
    var eps: Float
    var pad: UInt32
}

struct UnaryP {
    var n: UInt32
    var pad0: UInt32
    var pad1: UInt32
    var pad2: UInt32
}

struct ScaleP {
    var n: UInt32
    var s: Float
    var pad0: UInt32
    var pad1: UInt32
}

struct BiasP {
    var M: UInt32
    var N: UInt32
    var pad0: UInt32
    var pad1: UInt32
}

struct AttnP {
    var B: UInt32
    var T: UInt32
    var H: UInt32
    var D: UInt32
    var scale: Float
    var causal: UInt32
    var pad1: UInt32
    var pad2: UInt32
}

struct MseP {
    var n: UInt32
    var pad0: UInt32
    var invCount: Float
    var pad1: UInt32
}

struct CondP {
    var B: UInt32
    var T: UInt32
    var C: UInt32
    var pad: UInt32
}

struct EmbP {
    var B: UInt32
    var T: UInt32
    var C: UInt32
    var qGroup: UInt32
}

struct EmbBwdP {
    var B: UInt32
    var T: UInt32
    var C: UInt32
    var V: UInt32
}

struct PosP {
    var B: UInt32
    var T: UInt32
    var C: UInt32
    var pad: UInt32
}

struct LossP {
    var BT: UInt32
    var V: UInt32
    var pad0: UInt32
    var pad1: UInt32
}

struct ReduceP {
    var n: UInt32
    var slot: UInt32
    var pad0: UInt32
    var pad1: UInt32
}

struct RopeP {
    var B: UInt32
    var T: UInt32
    var H: UInt32
    var D: UInt32
}

struct MixP {
    var n: UInt32
    var a: Float
    var b: Float
    var pad: UInt32
}

struct LossMaskP {
    var BT: UInt32
    var V: UInt32
    var ignore: UInt32
    var pad: UInt32
    var invCount: Float
    var pad1: UInt32
    var pad2: UInt32
    var pad3: UInt32
}

struct AdamP {
    var n: UInt32
    var lr: Float
    var b1: Float
    var b2: Float
    var eps: Float
    var wd: Float
    var invBC1: Float
    var invBC2: Float
}

struct GPTConfig {
    var vocab: Int
    var block: Int
    var batch: Int
    var layers: Int
    var heads: Int
    var embd: Int
    var bits: Int
    var qGroup: Int = defaultQGroup
    var eps: Float = 1e-5

    var headDim: Int { embd / heads }
    var mlp: Int { embd * 4 }

    func validate() {
        precondition(batch > 0 && block > 0 && layers > 0 && heads > 0 && embd > 0)
        precondition(vocab > 1, "vocab must be at least 2")
        precondition(embd % heads == 0, "embd \(embd) must be divisible by heads \(heads)")
        precondition(bits == 4 || bits == 8 || bits == 32, "bits must be 4, 8, or 32")
        if bits != 32 {
            precondition(qGroup > 0 && embd % qGroup == 0, "embd must be a multiple of the quant group")
            precondition(mlp % qGroup == 0, "mlp width must be a multiple of the quant group")
            precondition(qGroup % 2 == 0, "quant group must be even for int4 packing")
        }
    }
}
