#!/usr/bin/env python3
"""Pack a closed set of real photographs for the diffusion run.

Each picture is a Wikimedia Commons photograph, center-cropped to a square
and resized to 128×128. The file format matches data/images/rgb.bin, so the
diffusion trainer can read it. Names sit after the same four special tokens
and 32 unused color ids the icon vocab uses.
"""

import io
import json
import struct
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "photos"
VOCAB = ROOT / "vocab" / "photos.txt"
SIDE = 128
COLORS = 32
NAME0 = 4 + COLORS

# name, commons filename. Photographs, not icons.
PHOTOS = [
    ("cat", "Cat_November_2010-1a.jpg"),
    ("dog", "YellowLabradorLooking_new.jpg"),
    ("horse", "Nokota_Horses_cropped.jpg"),
    ("bird", "Erithacus_rubecula_with_cocked_head.jpg"),
    ("lion", "Lion_waiting_in_Namibia.jpg"),
    ("elephant", "African_Bush_Elephant.jpg"),
    ("pizza", "Pizza_with_tomatoes.jpg"),
    ("burger", "Hamburger.jpg"),
    ("car", "2018_Porsche_911_Carrera_S.jpg"),
    ("house", "Semi-detached houses, Hall Green.jpg"),
    ("flower", "Sunflower sky backdrop.jpg"),
    ("mountain", "Matterhorn Riffelsee 2005-06-11.jpg"),
    ("beach", "Navagio beach Zakynthos.jpg"),
    ("tree", "The Solitary Oak (The Old Oak).jpeg"),
]


def fetch(filename: str) -> bytes:
    quoted = urllib.request.quote(filename)
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width=640"
    req = urllib.request.Request(url, headers={"User-Agent": "transformer-metal/1.0 (photo pack)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def square(png: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    crop = im.crop((left, top, left + side, top + side))
    return crop.resize((SIDE, SIDE), Image.Resampling.LANCZOS)


def main() -> None:
    names = [name for name, _ in PHOTOS]
    if len(names) != len(set(names)):
        raise SystemExit("photo names are not unique")
    images = []
    for i, (name, filename) in enumerate(PHOTOS):
        try:
            images.append(square(fetch(filename)))
        except Exception as exc:
            raise SystemExit(f"could not download {name} ({filename}): {exc}") from exc
        print(f"downloaded {i + 1}/{len(PHOTOS)}  {name}")

    OUT.mkdir(parents=True, exist_ok=True)
    VOCAB.parent.mkdir(parents=True, exist_ok=True)
    vocab = ["<|pad|>", "<|start|>", "<|end|>", "<|unk|>"]
    vocab += [f"c{i:02d}" for i in range(COLORS)]
    vocab += names
    VOCAB.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    rgb = bytearray(b"IRGB" + struct.pack("<IIII", 1, len(images), SIDE, 3))
    flat = []
    for n, im in enumerate(images):
        rgb += struct.pack("<I", n)
        for r, g, b in im.getdata():
            flat.append((r / 255) * 2 - 1)
            flat.append((g / 255) * 2 - 1)
            flat.append((b / 255) * 2 - 1)
        chunk = flat[-SIDE * SIDE * 3 :]
        rgb += struct.pack("<" + "f" * len(chunk), *chunk)
    (OUT / "rgb.bin").write_bytes(rgb)
    (OUT / "credit.json").write_text(json.dumps({
        "side": SIDE,
        "names": names,
        "files": [filename for _, filename in PHOTOS],
        "credit": "Photographs from Wikimedia Commons. See each file page for its license.",
    }, indent=2), encoding="utf-8")
    print(f"photos {len(images)}  side {SIDE}  vocab {len(vocab)}  name0 {NAME0}")
    print(f"wrote {OUT / 'rgb.bin'}")
    print(f"wrote {VOCAB}")


if __name__ == "__main__":
    main()
