#!/usr/bin/env python3
"""Turn the packed photographs into a short autoregressive sequence.

Each 128×128 photo is 64 patches of 16×16, the same grid the diffusion model
uses. Identical patches share one codebook entry. The sequence is the name,
then those 64 patch ids, then end. That is 66 tokens, not 16,384 pixels.
"""

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "data" / "photos" / "rgb.bin"
TOKENS = ROOT / "data" / "photos" / "tokens.bin"
BOOK = ROOT / "data" / "photos" / "codebook.bin"
NAMES = ROOT / "vocab" / "photos.txt"
VOCAB = ROOT / "vocab" / "photos-ar.txt"
PATCH = 16
EOS = 2


def load_rgb():
    raw = RGB.read_bytes()
    off = 0

    def u32():
        nonlocal off
        value = struct.unpack_from("<I", raw, off)[0]
        off += 4
        return value

    if raw[off:off + 4] != b"IRGB":
        raise SystemExit(f"{RGB} is not an icon rgb file")
    off += 4
    version, count, side, channels = u32(), u32(), u32(), u32()
    if version != 1 or channels != 3 or side % PATCH != 0:
        raise SystemExit(f"unexpected rgb header {version} {side} {channels}")
    span = side * side * channels
    images = []
    for _ in range(count):
        u32()
        vals = struct.unpack_from("<" + "f" * span, raw, off)
        off += span * 4
        pixels = []
        for v in vals:
            pixels.append(max(0, min(255, int(round((v + 1) * 127.5)))))
        images.append(pixels)
    return side, images


def patches_of(pixels, side):
    grid = side // PATCH
    out = []
    for py in range(grid):
        for px in range(grid):
            blob = bytearray()
            for iy in range(PATCH):
                for ix in range(PATCH):
                    src = ((py * PATCH + iy) * side + (px * PATCH + ix)) * 3
                    blob += bytes(pixels[src:src + 3])
            out.append(bytes(blob))
    return out


def main() -> None:
    if not RGB.exists() or not NAMES.exists():
        raise SystemExit("missing photographs. Run python3 scripts/make_photos.py")
    names = [line.strip() for line in NAMES.read_text(encoding="utf-8").splitlines() if line.strip()][36:]
    side, images = load_rgb()
    if len(images) != len(names):
        raise SystemExit(f"{len(images)} photos and {len(names)} names")
    codebook = []
    index = {}
    rows = []
    for pixels in images:
        ids = []
        for blob in patches_of(pixels, side):
            if blob not in index:
                index[blob] = len(codebook)
                codebook.append(blob)
            ids.append(4 + index[blob])
        rows.append(ids)
    name0 = 4 + len(codebook)
    vocab = ["<|pad|>", "<|start|>", "<|end|>", "<|unk|>"]
    vocab += [f"p{i}" for i in range(len(codebook))]
    vocab += names
    VOCAB.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokens = bytearray(b"MDST" + struct.pack("<II", 1, len(rows)))
    for n, ids in enumerate(rows):
        row = [name0 + n] + ids + [EOS]
        tokens += struct.pack("<I", len(row))
        tokens += struct.pack("<" + "I" * len(row), *row)
    TOKENS.write_bytes(tokens)

    book = bytearray(b"PCBK" + struct.pack("<IIIII", 1, side, PATCH, len(codebook), name0))
    for blob in codebook:
        book += blob
    BOOK.write_bytes(book)
    print(f"photos {len(rows)}  patches {side // PATCH}x{side // PATCH}  codebook {len(codebook)}")
    print(f"sequence {1 + (side // PATCH) ** 2 + 1}  vocab {len(vocab)}  name0 {name0}")
    print(f"wrote {TOKENS}")
    print(f"wrote {BOOK}")
    print(f"wrote {VOCAB}")


if __name__ == "__main__":
    main()
