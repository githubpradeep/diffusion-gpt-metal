#!/usr/bin/env python3
"""Download a closed set of Twemoji icons and pack them for the two image runs.

Twemoji is CC-BY 4.0, Copyright 2020 Twitter, Inc and other contributors.
https://github.com/jdecked/twemoji

Each icon is drawn on white, resized to 24×24, and quantized to one shared
32-color palette. The diffusion run reads the continuous RGB. The
autoregressive run reads palette indices in raster order.
"""

import io
import json
import struct
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "images"
VOCAB = ROOT / "vocab" / "images.txt"
BASE = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/"
SIDE = 24
COLORS = 32

# name, twemoji codepoint. About 128 concrete icons that still read at 24px.
ICONS = [
    ("cat", "1f431"), ("dog", "1f436"), ("mouse", "1f42d"), ("hamster", "1f439"),
    ("rabbit", "1f430"), ("fox", "1f98a"), ("bear", "1f43b"), ("panda", "1f43c"),
    ("koala", "1f428"), ("tiger", "1f42f"), ("lion", "1f981"), ("cow", "1f42e"),
    ("pig", "1f437"), ("frog", "1f438"), ("monkey", "1f412"), ("chicken", "1f414"),
    ("penguin", "1f427"), ("bird", "1f426"), ("chick", "1f424"), ("duck", "1f986"),
    ("eagle", "1f985"), ("owl", "1f989"), ("bat", "1f987"), ("wolf", "1f43a"),
    ("horse", "1f434"), ("unicorn", "1f984"), ("bee", "1f41d"), ("butterfly", "1f98b"),
    ("snail", "1f40c"), ("turtle", "1f422"), ("snake", "1f40d"), ("octopus", "1f419"),
    ("fish", "1f41f"), ("dolphin", "1f42c"), ("whale", "1f433"), ("crab", "1f980"),
    ("shrimp", "1f990"), ("squid", "1f991"), ("blowfish", "1f421"), ("elephant", "1f418"),
    ("cherry_blossom", "1f338"), ("blossom", "1f33c"), ("sunflower", "1f33b"), ("tulip", "1f337"),
    ("rose", "1f339"), ("herb", "1f33f"), ("clover", "1f340"), ("maple", "1f341"),
    ("cactus", "1f335"), ("palm", "1f334"), ("tree", "1f333"), ("seedling", "1f331"),
    ("sun", "2600"), ("moon", "1f319"), ("star", "2b50"), ("cloud", "2601"),
    ("rainbow", "1f308"), ("snowflake", "2744"), ("fire", "1f525"), ("droplet", "1f4a7"),
    ("mushroom", "1f344"), ("chestnut", "1f330"), ("leaves", "1f343"), ("globe", "1f30d"),
    ("apple", "1f34e"), ("pear", "1f350"), ("tangerine", "1f34a"), ("lemon", "1f34b"),
    ("banana", "1f34c"), ("watermelon", "1f349"), ("grapes", "1f347"), ("strawberry", "1f353"),
    ("cherries", "1f352"), ("peach", "1f351"), ("pineapple", "1f34d"), ("tomato", "1f345"),
    ("eggplant", "1f346"), ("avocado", "1f951"), ("carrot", "1f955"), ("corn", "1f33d"),
    ("bread", "1f35e"), ("cheese", "1f9c0"), ("pizza", "1f355"), ("hamburger", "1f354"),
    ("taco", "1f32e"), ("icecream", "1f366"), ("doughnut", "1f369"), ("cake", "1f370"),
    ("kite", "1fa81"), ("balloon", "1f388"), ("gift", "1f381"), ("house", "1f3e0"),
    ("car", "1f697"), ("bicycle", "1f6b2"), ("airplane", "1f6e9"), ("rocket", "1f680"),
    ("sailboat", "26f5"), ("anchor", "2693"), ("hourglass", "231b"), ("alarm", "23f0"),
    ("watch", "231a"), ("camera", "1f4f7"), ("book", "1f4d5"), ("pencil", "270f"),
    ("scissors", "2702"), ("key", "1f511"), ("bulb", "1f4a1"), ("candle", "1f56f"),
    ("gem", "1f48e"), ("crown", "1f451"), ("umbrella", "2602"), ("bell", "1f514"),
    ("soccer", "26bd"), ("basketball", "1f3c0"), ("guitar", "1f3b8"), ("drum", "1f941"),
    ("heart", "2764"), ("yellow_heart", "1f49b"), ("green_heart", "1f49a"), ("blue_heart", "1f499"),
    ("purple_heart", "1f49c"), ("smile", "1f604"), ("grin", "1f601"), ("joy", "1f602"),
    ("wink", "1f609"), ("heart_eyes", "1f60d"), ("thumbsup", "1f44d"), ("ghost", "1f47b"),
]

PAD, BOS, EOS, UNK = 0, 1, 2, 3
COLOR0 = 4
NAME0 = COLOR0 + COLORS


def fetch(code: str) -> bytes:
    for name in (code, f"{code}-fe0f"):
        url = BASE + name + ".png"
        req = urllib.request.Request(url, headers={"User-Agent": "transformer-metal"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            continue
    raise SystemExit(f"could not download twemoji {code}")


def raster(png: bytes) -> Image.Image:
    src = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
    bg.alpha_composite(src)
    return bg.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BOX)


def quantize_method():
    return getattr(getattr(Image, "Quantize", Image), "MEDIANCUT")


def main() -> None:
    names = [name for name, _ in ICONS]
    if len(names) != len(set(names)):
        raise SystemExit("icon names are not unique")
    if len(ICONS) < 64:
        raise SystemExit("need a closed set large enough to hold icons out")

    images = []
    for i, (name, code) in enumerate(ICONS):
        images.append(raster(fetch(code)))
        if (i + 1) % 32 == 0 or i + 1 == len(ICONS):
            print(f"downloaded {i + 1}/{len(ICONS)}")

    mosaic = Image.new("RGB", (SIDE, SIDE * len(images)))
    for i, im in enumerate(images):
        mosaic.paste(im, (0, i * SIDE))
    palette_img = mosaic.quantize(colors=COLORS, method=quantize_method(), dither=Image.Dither.NONE)
    raw_pal = palette_img.getpalette()[: COLORS * 3]
    palette = [raw_pal[i : i + 3] for i in range(0, COLORS * 3, 3)]

    indices = []
    flat_rgb = []
    for im in images:
        q = im.quantize(palette=palette_img, dither=Image.Dither.NONE)
        idxs = list(q.getdata())
        if len(idxs) != SIDE * SIDE:
            raise SystemExit("quantized icon has the wrong size")
        if max(idxs) >= COLORS:
            raise SystemExit("palette index out of range")
        indices.append(idxs)
        for r, g, b in im.getdata():
            flat_rgb.append((r / 255) * 2 - 1)
            flat_rgb.append((g / 255) * 2 - 1)
            flat_rgb.append((b / 255) * 2 - 1)

    OUT.mkdir(parents=True, exist_ok=True)
    VOCAB.parent.mkdir(parents=True, exist_ok=True)

    vocab = ["<|pad|>", "<|start|>", "<|end|>", "<|unk|>"]
    vocab += [f"c{i:02d}" for i in range(COLORS)]
    vocab += names
    VOCAB.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokens = bytearray(b"MDST" + struct.pack("<II", 1, len(images)))
    seq = 1 + SIDE * SIDE + 1
    for n, idxs in enumerate(indices):
        row = [NAME0 + n] + [COLOR0 + i for i in idxs] + [EOS]
        if len(row) != seq:
            raise SystemExit("token row length")
        tokens += struct.pack("<I", len(row))
        tokens += struct.pack("<" + "I" * len(row), *row)
    (OUT / "tokens.bin").write_bytes(tokens)

    rgb = bytearray(b"IRGB" + struct.pack("<IIII", 1, len(images), SIDE, 3))
    span = SIDE * SIDE * 3
    for n in range(len(images)):
        rgb += struct.pack("<I", n)
        chunk = flat_rgb[n * span : (n + 1) * span]
        rgb += struct.pack("<" + "f" * span, *chunk)
    (OUT / "rgb.bin").write_bytes(rgb)

    (OUT / "palette.json").write_text(json.dumps({
        "side": SIDE,
        "colors": palette,
        "names": names,
        "credit": "Twemoji, CC-BY 4.0, Copyright 2020 Twitter, Inc and other contributors",
    }), encoding="utf-8")

    held = sum(1 for i in range(len(images)) if i % 8 == 0)
    print(f"icons {len(images)}  held out every 8th ({held})")
    print(f"vocab {len(vocab)}  sequence {seq}  palette {COLORS}")
    print(f"wrote {OUT / 'tokens.bin'}")
    print(f"wrote {OUT / 'rgb.bin'}")
    print(f"wrote {OUT / 'palette.json'}")
    print(f"wrote {VOCAB}")


if __name__ == "__main__":
    main()
