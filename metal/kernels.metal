// Metal kernels for a decoder-only GPT.
//
// Linear layers store a fp32 master copy. The forward GEMM consumes a
// group-quantized copy:
//   int8  symmetric, codes -127..127, one byte each
//   int4  symmetric, codes -7..7, two codes per byte (low nibble = even k)
// Scales are per output row and per group of qGroup inputs (8).
// Backward is the straight-through estimator: dW is the full-precision
// outer product, applied to the fp32 master. The scale is treated as a
// stop-gradient.
//
// GEMM shapes, W stored row-major [N, K]:
//   Y[m, n]   = sum_k X[m, k] * dequant(W[n, k]) + bias[n]
//   dX[m, k]  = sum_n dY[m, n] * dequant(W[n, k])
//   dW[n, k]  = sum_m dY[m, n] * X[m, k]
//
// Struct layouts must match src/Params.swift.

#include <metal_stdlib>
using namespace metal;

struct GemmP {
    uint M, N, K, hasBias, qGroup, pad0, pad1, pad2;
};
struct QuantP {
    uint rows, cols;
    float qmax;
    uint qGroup;
};
struct LNP {
    uint rows, cols;
    float eps;
    uint pad;
};
struct UnaryP { uint n, pad0, pad1, pad2; };
struct ScaleP { uint n; float s; uint pad0, pad1; };
struct BiasP { uint M, N, pad0, pad1; };
struct AttnP {
    uint B, T, H, D;
    float scale;
    uint causal, pad1, pad2;
};
struct MseP { uint n, pad0; float invCount; uint pad1; };
struct CondP { uint B, T, C, pad; };
struct EmbP { uint B, T, C, qGroup; };
struct EmbBwdP { uint B, T, C, V; };
struct PosP { uint B, T, C, pad; };
struct LossP { uint BT, V, pad0, pad1; };
struct RopeP { uint B, T, H, D; };
struct MixP { uint n; float a, b; uint pad; };
struct LossMaskP {
    uint BT, V, ignore, pad;
    float invCount;
    uint pad1, pad2, pad3;
};
struct ReduceP { uint n, slot, pad0, pad1; };
struct AdamP {
    uint n;
    float lr, b1, b2, eps, wd, invBC1, invBC2;
};

inline int quant_code(float w, float s, int lim) {
    int q = int(rint(w / s));
    if (q > lim) q = lim;
    if (q < -lim) q = -lim;
    return q;
}

inline float load_i4(device const uchar* W, uint row, uint col, uint cols) {
    uchar b = W[(row * cols + col) >> 1];
    uint nib = (col & 1u) ? (uint)(b >> 4) : (uint)(b & 0xFu);
    int v = int(nib);
    if (v >= 8) v -= 16;
    return float(v);
}

// ---- quantize ----------------------------------------------------------------

kernel void quant_i8(
    device const float* W [[buffer(0)]],
    device char* Q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant QuantP& p [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint G = p.qGroup;
    uint groups = p.cols / G;
    int lim = int(p.qmax);
    for (uint g = 0; g < groups; ++g) {
        uint base = row * p.cols + g * G;
        float maxAbs = 0.0f;
        for (uint i = 0; i < G; ++i) maxAbs = max(maxAbs, fabs(W[base + i]));
        float s = max(maxAbs / p.qmax, 1e-8f);
        scale[row * groups + g] = s;
        for (uint i = 0; i < G; ++i) Q[base + i] = char(quant_code(W[base + i], s, lim));
    }
}

kernel void quant_i4(
    device const float* W [[buffer(0)]],
    device uchar* Q [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant QuantP& p [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint G = p.qGroup;
    uint groups = p.cols / G;
    int lim = int(p.qmax);
    for (uint g = 0; g < groups; ++g) {
        uint base = row * p.cols + g * G;
        float maxAbs = 0.0f;
        for (uint i = 0; i < G; ++i) maxAbs = max(maxAbs, fabs(W[base + i]));
        float s = max(maxAbs / p.qmax, 1e-8f);
        scale[row * groups + g] = s;
        for (uint i = 0; i < G; i += 2u) {
            int q0 = quant_code(W[base + i], s, lim);
            int q1 = quant_code(W[base + i + 1u], s, lim);
            uchar packed = uchar((q0 & 0xF) | ((q1 & 0xF) << 4));
            Q[(base + i) >> 1] = packed;
        }
    }
}

// ---- GEMM: one thread per output ---------------------------------------------

kernel void gemm_nt_f32(
    device const float* W [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant GemmP& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.x;
    uint m = gid.y;
    if (n >= p.N || m >= p.M) return;
    const device float* wrow = W + n * p.K;
    const device float* xrow = X + m * p.K;
    float acc = 0.0f;
    uint k = 0;
    for (; k + 4u <= p.K; k += 4u) {
        acc += wrow[k] * xrow[k] + wrow[k + 1] * xrow[k + 1]
             + wrow[k + 2] * xrow[k + 2] + wrow[k + 3] * xrow[k + 3];
    }
    for (; k < p.K; ++k) acc += wrow[k] * xrow[k];
    if (p.hasBias) acc += bias[n];
    Y[m * p.N + n] = acc;
}

kernel void gemm_nt_i8(
    device const char* W [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GemmP& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.x;
    uint m = gid.y;
    if (n >= p.N || m >= p.M) return;
    uint G = p.qGroup;
    uint groups = p.K / G;
    const device char* wrow = W + n * p.K;
    const device float* xrow = X + m * p.K;
    const device float* srow = scale + n * groups;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float part = 0.0f;
        uint base = g * G;
        for (uint i = 0; i < G; ++i) part += float(wrow[base + i]) * xrow[base + i];
        acc += part * srow[g];
    }
    if (p.hasBias) acc += bias[n];
    Y[m * p.N + n] = acc;
}

kernel void gemm_nt_i4(
    device const uchar* W [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant GemmP& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.x;
    uint m = gid.y;
    if (n >= p.N || m >= p.M) return;
    uint G = p.qGroup;
    uint groups = p.K / G;
    const device float* xrow = X + m * p.K;
    const device float* srow = scale + n * groups;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float part = 0.0f;
        uint base = g * G;
        for (uint i = 0; i < G; i += 2u) {
            float w0 = load_i4(W, n, base + i, p.K);
            float w1 = load_i4(W, n, base + i + 1u, p.K);
            part += w0 * xrow[base + i] + w1 * xrow[base + i + 1u];
        }
        acc += part * srow[g];
    }
    if (p.hasBias) acc += bias[n];
    Y[m * p.N + n] = acc;
}

kernel void gemm_dx_f32(
    device const float* W [[buffer(0)]],
    device const float* dY [[buffer(1)]],
    device float* dX [[buffer(2)]],
    constant GemmP& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint k = gid.x;
    uint m = gid.y;
    if (k >= p.K || m >= p.M) return;
    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) acc += dY[m * p.N + n] * W[n * p.K + k];
    dX[m * p.K + k] = acc;
}

kernel void gemm_dx_i8(
    device const char* W [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* dY [[buffer(2)]],
    device float* dX [[buffer(3)]],
    constant GemmP& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint k = gid.x;
    uint m = gid.y;
    if (k >= p.K || m >= p.M) return;
    uint G = p.qGroup;
    uint g = k / G;
    uint groups = p.K / G;
    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        float w = float(W[n * p.K + k]) * scale[n * groups + g];
        acc += dY[m * p.N + n] * w;
    }
    dX[m * p.K + k] = acc;
}

kernel void gemm_dx_i4(
    device const uchar* W [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* dY [[buffer(2)]],
    device float* dX [[buffer(3)]],
    constant GemmP& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint k = gid.x;
    uint m = gid.y;
    if (k >= p.K || m >= p.M) return;
    uint G = p.qGroup;
    uint g = k / G;
    uint groups = p.K / G;
    float acc = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        float w = load_i4(W, n, k, p.K) * scale[n * groups + g];
        acc += dY[m * p.N + n] * w;
    }
    dX[m * p.K + k] = acc;
}

kernel void gemm_dw(
    device const float* dY [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* dW [[buffer(2)]],
    constant GemmP& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint k = gid.x;
    uint n = gid.y;
    if (k >= p.K || n >= p.N) return;
    float acc = 0.0f;
    for (uint m = 0; m < p.M; ++m) acc += dY[m * p.N + n] * X[m * p.K + k];
    dW[n * p.K + k] = acc;
}

kernel void bias_grad(
    device const float* dY [[buffer(0)]],
    device float* db [[buffer(1)]],
    constant BiasP& p [[buffer(2)]],
    uint n [[thread_position_in_grid]]
) {
    if (n >= p.N) return;
    float acc = 0.0f;
    for (uint m = 0; m < p.M; ++m) acc += dY[m * p.N + n];
    db[n] = acc;
}

// ---- layernorm, gelu, residual ------------------------------------------------

kernel void layernorm_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device float* mean [[buffer(2)]],
    device float* rstd [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    constant LNP& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint C = p.cols;
    const device float* xr = x + row * C;
    float m = 0.0f;
    for (uint c = 0; c < C; ++c) m += xr[c];
    m /= float(C);
    float v = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float d = xr[c] - m;
        v += d * d;
    }
    v /= float(C);
    float rs = rsqrt(v + p.eps);
    mean[row] = m;
    rstd[row] = rs;
    device float* yr = y + row * C;
    for (uint c = 0; c < C; ++c) yr[c] = (xr[c] - m) * rs * gamma[c] + beta[c];
}

kernel void layernorm_bwd_dx(
    device const float* x [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* rstd [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* dx [[buffer(5)]],
    constant LNP& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint C = p.cols;
    float m = mean[row];
    float rs = rstd[row];
    float sumDy = 0.0f;
    float sumDyX = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float xhat = (x[row * C + c] - m) * rs;
        float dyhat = dy[row * C + c] * gamma[c];
        sumDy += dyhat;
        sumDyX += dyhat * xhat;
    }
    sumDy /= float(C);
    sumDyX /= float(C);
    for (uint c = 0; c < C; ++c) {
        float xhat = (x[row * C + c] - m) * rs;
        float dyhat = dy[row * C + c] * gamma[c];
        dx[row * C + c] = rs * (dyhat - sumDy - xhat * sumDyX);
    }
}

kernel void layernorm_bwd_dg(
    device const float* x [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* rstd [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* dgamma [[buffer(4)]],
    device float* dbeta [[buffer(5)]],
    constant LNP& p [[buffer(6)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= p.cols) return;
    uint C = p.cols;
    float dg = 0.0f;
    float db = 0.0f;
    for (uint row = 0; row < p.rows; ++row) {
        float xhat = (x[row * C + c] - mean[row]) * rstd[row];
        dg += dy[row * C + c] * xhat;
        db += dy[row * C + c];
    }
    dgamma[c] = dg;
    dbeta[c] = db;
}

inline float gelu_f(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float u = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanh(u));
}

inline float gelu_prime(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x2 = x * x;
    float u = k0 * (x + k1 * x2 * x);
    float th = tanh(u);
    float sech2 = 1.0f - th * th;
    float dudx = k0 * (1.0f + 3.0f * k1 * x2);
    return 0.5f * (1.0f + th) + 0.5f * x * sech2 * dudx;
}

kernel void gelu_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant UnaryP& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) y[i] = gelu_f(x[i]);
}

kernel void gelu_bwd(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant UnaryP& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) dx[i] = gelu_prime(x[i]) * dy[i];
}

kernel void add_f(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant UnaryP& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) y[i] = a[i] + b[i];
}

kernel void acc_f(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant UnaryP& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) a[i] += b[i];
}

// ---- attention ----------------------------------------------------------------
// q, k, v, av: [B, T, H, D] packed as ((b*T+t)*H+h)*D+d
// scores, probs: [B, H, T, T] packed as ((b*H+h)*T+t)*T+s
// causal != 0: position t may attend to s <= t. Diffusion sets causal to 0.

kernel void attn_scores(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint TT = p.T * p.T;
    uint rows = p.B * p.H * TT;
    if (idx >= rows) return;
    uint s = idx % p.T;
    uint t = (idx / p.T) % p.T;
    uint h = (idx / TT) % p.H;
    uint b = idx / (TT * p.H);
    float acc = 0.0f;
    for (uint d = 0; d < p.D; ++d) {
        float qv = q[((b * p.T + t) * p.H + h) * p.D + d];
        float kv = k[((b * p.T + s) * p.H + h) * p.D + d];
        acc += qv * kv;
    }
    acc *= p.scale;
    if (p.causal && s > t) acc = -1.0e30f;
    scores[idx] = acc;
}

kernel void attn_softmax(
    device const float* scores [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant AttnP& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    uint rows = p.B * p.H * p.T;
    if (row >= rows) return;
    const device float* s = scores + row * p.T;
    device float* o = probs + row * p.T;
    float maxv = s[0];
    for (uint i = 1; i < p.T; ++i) maxv = max(maxv, s[i]);
    float sum = 0.0f;
    for (uint i = 0; i < p.T; ++i) sum += exp(s[i] - maxv);
    float inv = 1.0f / sum;
    for (uint i = 0; i < p.T; ++i) o[i] = exp(s[i] - maxv) * inv;
}

kernel void attn_av(
    device const float* probs [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* av [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = p.B * p.T * p.H * p.D;
    if (idx >= total) return;
    uint d = idx % p.D;
    uint h = (idx / p.D) % p.H;
    uint t = (idx / (p.D * p.H)) % p.T;
    uint b = idx / (p.D * p.H * p.T);
    float acc = 0.0f;
    for (uint s = 0; s < p.T; ++s) {
        float pr = probs[((b * p.H + h) * p.T + t) * p.T + s];
        float vv = v[((b * p.T + s) * p.H + h) * p.D + d];
        acc += pr * vv;
    }
    av[idx] = acc;
}

kernel void attn_bwd_dv(
    device const float* probs [[buffer(0)]],
    device const float* dout [[buffer(1)]],
    device float* dv [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = p.B * p.T * p.H * p.D;
    if (idx >= total) return;
    uint d = idx % p.D;
    uint h = (idx / p.D) % p.H;
    uint s = (idx / (p.D * p.H)) % p.T;
    uint b = idx / (p.D * p.H * p.T);
    float acc = 0.0f;
    for (uint t = 0; t < p.T; ++t) {
        float pr = probs[((b * p.H + h) * p.T + t) * p.T + s];
        float dy = dout[((b * p.T + t) * p.H + h) * p.D + d];
        acc += pr * dy;
    }
    dv[idx] = acc;
}

kernel void attn_bwd_dp(
    device const float* dout [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* dp [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint TT = p.T * p.T;
    uint rows = p.B * p.H * TT;
    if (idx >= rows) return;
    uint s = idx % p.T;
    uint t = (idx / p.T) % p.T;
    uint h = (idx / TT) % p.H;
    uint b = idx / (TT * p.H);
    float acc = 0.0f;
    for (uint d = 0; d < p.D; ++d) {
        float dy = dout[((b * p.T + t) * p.H + h) * p.D + d];
        float vv = v[((b * p.T + s) * p.H + h) * p.D + d];
        acc += dy * vv;
    }
    dp[idx] = acc;
}

kernel void attn_softmax_bwd(
    device const float* probs [[buffer(0)]],
    device const float* dp [[buffer(1)]],
    device float* ds [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    uint rows = p.B * p.H * p.T;
    if (row >= rows) return;
    const device float* pr = probs + row * p.T;
    const device float* g = dp + row * p.T;
    device float* o = ds + row * p.T;
    float dot = 0.0f;
    for (uint i = 0; i < p.T; ++i) dot += g[i] * pr[i];
    for (uint i = 0; i < p.T; ++i) o[i] = pr[i] * (g[i] - dot);
}

kernel void attn_bwd_dq(
    device const float* ds [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* dq [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = p.B * p.T * p.H * p.D;
    if (idx >= total) return;
    uint d = idx % p.D;
    uint h = (idx / p.D) % p.H;
    uint t = (idx / (p.D * p.H)) % p.T;
    uint b = idx / (p.D * p.H * p.T);
    float acc = 0.0f;
    for (uint s = 0; s < p.T; ++s) {
        float g = ds[((b * p.H + h) * p.T + t) * p.T + s];
        float kv = k[((b * p.T + s) * p.H + h) * p.D + d];
        acc += g * kv;
    }
    dq[idx] = acc * p.scale;
}

kernel void attn_bwd_dk(
    device const float* ds [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* dk [[buffer(2)]],
    constant AttnP& p [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = p.B * p.T * p.H * p.D;
    if (idx >= total) return;
    uint d = idx % p.D;
    uint h = (idx / p.D) % p.H;
    uint s = (idx / (p.D * p.H)) % p.T;
    uint b = idx / (p.D * p.H * p.T);
    float acc = 0.0f;
    for (uint t = 0; t < p.T; ++t) {
        float g = ds[((b * p.H + h) * p.T + t) * p.T + s];
        float qv = q[((b * p.T + t) * p.H + h) * p.D + d];
        acc += g * qv;
    }
    dk[idx] = acc * p.scale;
}

// ---- embeddings and loss ------------------------------------------------------

kernel void embed_f32(
    device const uint* tokens [[buffer(0)]],
    device const float* wte [[buffer(1)]],
    device const float* wpe [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant EmbP& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint bt = gid.y;
    if (c >= p.C || bt >= p.B * p.T) return;
    uint t = bt % p.T;
    uint tok = tokens[bt];
    y[bt * p.C + c] = wte[tok * p.C + c] + wpe[t * p.C + c];
}

kernel void embed_i8(
    device const uint* tokens [[buffer(0)]],
    device const char* wte [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* wpe [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant EmbP& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint bt = gid.y;
    if (c >= p.C || bt >= p.B * p.T) return;
    uint t = bt % p.T;
    uint tok = tokens[bt];
    uint groups = p.C / p.qGroup;
    float s = scale[tok * groups + (c / p.qGroup)];
    y[bt * p.C + c] = float(wte[tok * p.C + c]) * s + wpe[t * p.C + c];
}

kernel void embed_i4(
    device const uint* tokens [[buffer(0)]],
    device const uchar* wte [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* wpe [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant EmbP& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint bt = gid.y;
    if (c >= p.C || bt >= p.B * p.T) return;
    uint t = bt % p.T;
    uint tok = tokens[bt];
    uint groups = p.C / p.qGroup;
    float s = scale[tok * groups + (c / p.qGroup)];
    y[bt * p.C + c] = load_i4(wte, tok, c, p.C) * s + wpe[t * p.C + c];
}

// Adds token-embedding gradients on top of an existing buffer (the tied lm head).
kernel void embed_bwd(
    device const uint* tokens [[buffer(0)]],
    device const float* dx [[buffer(1)]],
    device float* dwe [[buffer(2)]],
    constant EmbBwdP& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint v = gid.y;
    if (c >= p.C || v >= p.V) return;
    float sum = 0.0f;
    uint BT = p.B * p.T;
    for (uint bt = 0; bt < BT; ++bt) {
        if (tokens[bt] == v) sum += dx[bt * p.C + c];
    }
    dwe[v * p.C + c] += sum;
}

kernel void pos_bwd(
    device const float* dx [[buffer(0)]],
    device float* dwp [[buffer(1)]],
    constant PosP& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint t = gid.y;
    if (c >= p.C || t >= p.T) return;
    float sum = 0.0f;
    for (uint b = 0; b < p.B; ++b) sum += dx[(b * p.T + t) * p.C + c];
    dwp[t * p.C + c] = sum;
}

kernel void loss_fwd(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* nll [[buffer(2)]],
    device float* dlogits [[buffer(3)]],
    constant LossP& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.BT) return;
    uint V = p.V;
    const device float* L = logits + row * V;
    uint tgt = targets[row];
    float maxv = L[0];
    for (uint v = 1; v < V; ++v) maxv = max(maxv, L[v]);
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(L[v] - maxv);
    float inv = 1.0f / sum;
    nll[row] = log(sum) + maxv - L[tgt];
    float invBT = 1.0f / float(p.BT);
    for (uint v = 0; v < V; ++v) {
        float prob = exp(L[v] - maxv) * inv;
        float g = prob;
        if (v == tgt) g -= 1.0f;
        dlogits[row * V + v] = g * invBT;
    }
}

// One threadgroup. Every thread must reach the barriers.
kernel void reduce_sum(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ReduceP& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float sh[256];
    float v = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) v += x[i];
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[p.slot] = sh[0];
}

kernel void reduce_sumsq(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ReduceP& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float sh[256];
    float v = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float a = x[i];
        v += a * a;
    }
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[p.slot] = sh[0];
}

kernel void zero_f(
    device float* x [[buffer(0)]],
    constant UnaryP& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) x[i] = 0.0f;
}

kernel void scale_f(
    device float* x [[buffer(0)]],
    constant ScaleP& p [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) x[i] *= p.s;
}

kernel void adamw(
    device float* w [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant AdamP& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    float gi = g[i];
    float mi = p.b1 * m[i] + (1.0f - p.b1) * gi;
    float vi = p.b2 * v[i] + (1.0f - p.b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float upd = (mi * p.invBC1) / (sqrt(vi * p.invBC2) + p.eps);
    w[i] -= p.lr * (upd + p.wd * w[i]);
}

// ---- MacroStories: RMSNorm, RoPE, SwiGLU, GQA helpers -------------------------
// RMSNorm matches the Ouro / T5 definition: y = x * rsqrt(mean(x^2) + eps) * weight.

kernel void rmsnorm_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device float* rstd [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    constant LNP& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint C = p.cols;
    const device float* xr = x + row * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) ss += xr[c] * xr[c];
    ss /= float(C);
    float rs = rsqrt(ss + p.eps);
    rstd[row] = rs;
    device float* yr = y + row * C;
    for (uint c = 0; c < C; ++c) yr[c] = xr[c] * rs * weight[c];
}

kernel void rmsnorm_bwd_dx(
    device const float* x [[buffer(0)]],
    device const float* rstd [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* dx [[buffer(4)]],
    constant LNP& p [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.rows) return;
    uint C = p.cols;
    float rs = rstd[row];
    float meanDyX = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float dyhat = dy[row * C + c] * weight[c];
        meanDyX += dyhat * x[row * C + c];
    }
    meanDyX /= float(C);
    float rs3 = rs * rs * rs;
    for (uint c = 0; c < C; ++c) {
        float dyhat = dy[row * C + c] * weight[c];
        dx[row * C + c] = rs * dyhat - x[row * C + c] * rs3 * meanDyX;
    }
}

// Adds into dweight so the same RMSNorm can be applied on every recurrent pass.
kernel void rmsnorm_bwd_dw(
    device const float* x [[buffer(0)]],
    device const float* rstd [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dweight [[buffer(3)]],
    constant LNP& p [[buffer(4)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= p.cols) return;
    uint C = p.cols;
    float dg = 0.0f;
    for (uint row = 0; row < p.rows; ++row) {
        dg += dy[row * C + c] * x[row * C + c] * rstd[row];
    }
    dweight[c] += dg;
}

// q/k layout [B, T, H, D]. cos/sin layout [T, D], duplicated across the two halves.
kernel void rope_fwd(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const float* cosv [[buffer(2)]],
    device const float* sinv [[buffer(3)]],
    constant RopeP& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    uint D = p.D;
    uint total = p.B * p.T * p.H * D;
    if (i >= total) return;
    uint d = i % D;
    uint mid = D >> 1;
    uint t = (i / (p.H * D)) % p.T;
    float rot = (d < mid) ? -x[i + mid] : x[i - mid];
    float c = cosv[t * D + d];
    float s = sinv[t * D + d];
    y[i] = x[i] * c + rot * s;
}

kernel void rope_bwd(
    device const float* dy [[buffer(0)]],
    device float* dx [[buffer(1)]],
    device const float* cosv [[buffer(2)]],
    device const float* sinv [[buffer(3)]],
    constant RopeP& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    uint D = p.D;
    uint total = p.B * p.T * p.H * D;
    if (i >= total) return;
    uint d = i % D;
    uint mid = D >> 1;
    uint t = (i / (p.H * D)) % p.T;
    float c = cosv[t * D + d];
    float s = sinv[t * D + d];
    if (d < mid) dx[i] = dy[i] * c + dy[i + mid] * s;
    else dx[i] = dy[i] * c - dy[i - mid] * s;
}

inline float silu_sigmoid(float z) {
    if (z >= 0.0f) {
        float e = exp(-z);
        return 1.0f / (1.0f + e);
    }
    float e = exp(z);
    return e / (1.0f + e);
}

kernel void swiglu_fwd(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant UnaryP& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    float z = gate[i];
    y[i] = (z * silu_sigmoid(z)) * up[i];
}

kernel void swiglu_bwd(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dgate [[buffer(3)]],
    device float* dup [[buffer(4)]],
    constant UnaryP& p [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    float z = gate[i];
    float s = silu_sigmoid(z);
    float silu = z * s;
    float siluP = s * (1.0f + z * (1.0f - s));
    float g = dy[i];
    dgate[i] = siluP * up[i] * g;
    dup[i] = silu * g;
}

kernel void mix_f(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MixP& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) y[i] = p.a * a[i] + p.b * b[i];
}

kernel void copy_f(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant UnaryP& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) y[i] = x[i];
}

kernel void acc_scale_f(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant ScaleP& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p.n) a[i] += p.s * b[i];
}

// src is one KV head [B, T, D]. dst is [B, T, H, D], every head a copy.
kernel void repeat_heads(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant RopeP& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    uint D = p.D;
    uint total = p.B * p.T * p.H * D;
    if (i >= total) return;
    uint d = i % D;
    uint t = (i / (p.H * D)) % p.T;
    uint b = i / (p.T * p.H * D);
    dst[i] = src[(b * p.T + t) * D + d];
}

kernel void sum_heads(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant RopeP& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    uint D = p.D;
    uint total = p.B * p.T * D;
    if (i >= total) return;
    uint d = i % D;
    uint t = (i / D) % p.T;
    uint b = i / (p.T * D);
    float acc = 0.0f;
    for (uint h = 0; h < p.H; ++h) {
        acc += src[((b * p.T + t) * p.H + h) * D + d];
    }
    dst[i] = acc;
}

kernel void gemm_dw_acc(
    device const float* dY [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* dW [[buffer(2)]],
    constant GemmP& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint k = gid.x;
    uint n = gid.y;
    if (k >= p.K || n >= p.N) return;
    float acc = 0.0f;
    for (uint m = 0; m < p.M; ++m) acc += dY[m * p.N + n] * X[m * p.K + k];
    dW[n * p.K + k] += acc;
}

// Same as loss_fwd, but rows whose target equals ignore contribute neither loss nor gradient.
kernel void loss_fwd_masked(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* nll [[buffer(2)]],
    device float* dlogits [[buffer(3)]],
    constant LossMaskP& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= p.BT) return;
    uint V = p.V;
    device float* drow = dlogits + row * V;
    uint tgt = targets[row];
    if (tgt == p.ignore) {
        nll[row] = 0.0f;
        for (uint v = 0; v < V; ++v) drow[v] = 0.0f;
        return;
    }
    const device float* L = logits + row * V;
    float maxv = L[0];
    for (uint v = 1; v < V; ++v) maxv = max(maxv, L[v]);
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(L[v] - maxv);
    float inv = 1.0f / sum;
    nll[row] = log(sum) + maxv - L[tgt];
    for (uint v = 0; v < V; ++v) {
        float prob = exp(L[v] - maxv) * inv;
        float g = prob;
        if (v == tgt) g -= 1.0f;
        drow[v] = g * p.invCount;
    }
}

// Mean squared error. err stores the squared residual; dpred is the gradient of the mean.
kernel void mse_fwd(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* err [[buffer(2)]],
    device float* dpred [[buffer(3)]],
    constant MseP& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    float d = pred[i] - target[i];
    err[i] = d * d;
    dpred[i] = 2.0f * d * p.invCount;
}

// y[b, t, c] = x[b, t, c] + cond[b, c]
kernel void add_cond(
    device const float* x [[buffer(0)]],
    device const float* cond [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant CondP& p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    uint n = p.B * p.T * p.C;
    if (i >= n) return;
    uint c = i % p.C;
    uint b = i / (p.C * p.T);
    y[i] = x[i] + cond[b * p.C + c];
}

// dcond[b, c] = sum over t of dy[b, t, c]
kernel void sum_cond(
    device const float* dy [[buffer(0)]],
    device float* dcond [[buffer(1)]],
    constant CondP& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint b = gid.y;
    if (c >= p.C || b >= p.B) return;
    float s = 0.0f;
    for (uint t = 0; t < p.T; ++t) s += dy[(b * p.T + t) * p.C + c];
    dcond[b * p.C + c] = s;
}
