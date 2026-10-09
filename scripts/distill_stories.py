"""Sample stories from raincandy-u/MacroStories and write data/macro-distill.bin.

The published Gemma set was not released. These samples are the teacher's
own distribution: temperature 0.5, nucleus 0.9, the same 378-word ids.
"""

import argparse
import struct
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAGIC = b"MDST"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--raincandy-u--MacroStories/snapshots/dc1ef998bf5331f7f70a77c815ed3e4f643a0ba7"


def trim(row):
    ids = [int(t) for t in row]
    if 2 in ids:
        ids = ids[: ids.index(2) + 1]
    else:
        while ids and ids[-1] == 0:
            ids.pop()
    if not ids or ids[0] != 1:
        ids = [1] + [t for t in ids if t != 1]
    return ids


def load_existing(path: Path):
    if not path.exists() or path.stat().st_size < 12:
        return []
    blob = path.read_bytes()
    if blob[:4] != MAGIC:
        return []
    count = struct.unpack_from("<I", blob, 8)[0]
    offset = 12
    stories = []
    for _ in range(count):
        if offset + 4 > len(blob):
            break
        length = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        end = offset + length * 4
        if end > len(blob):
            break
        stories.append(list(struct.unpack_from(f"<{length}I", blob, offset)))
        offset = end
    return stories


def write_corpus(path: Path, stories):
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [MAGIC, struct.pack("<I", 1), struct.pack("<I", len(stories))]
    for story in stories:
        parts.append(struct.pack("<I", len(story)))
        parts.append(struct.pack(f"<{len(story)}I", *story))
    path.write_bytes(b"".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(SNAPSHOT))
    parser.add_argument("--out", default="data/macro-distill.bin")
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=768)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32)
    model.to(device).eval()

    out = Path(args.out)
    stories = load_existing(out)
    seen = {tuple(story) for story in stories}
    print(f"device {device}  have {len(stories)}  target {args.count}", flush=True)

    attempts = 0
    t0 = time.time()
    while len(stories) < args.count:
        batch = min(args.batch, args.count - len(stories) + args.batch)
        prompt = torch.full((batch, 1), 1, dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=True,
                temperature=args.temp,
                top_p=args.top_p,
                max_new_tokens=args.max_new,
                eos_token_id=2,
                pad_token_id=0,
                use_cache=True,
            )
        attempts += batch
        fresh = 0
        for row in output:
            story = trim(row)
            if len(story) < 16 or 2 not in story:
                continue
            key = tuple(story)
            if key in seen:
                continue
            seen.add(key)
            stories.append(story)
            fresh += 1
            if len(stories) >= args.count:
                break
        if len(stories) % 256 < batch or fresh == 0:
            write_corpus(out, stories)
            elapsed = time.time() - t0
            rate = attempts / max(elapsed, 1e-6)
            lens = [len(s) for s in stories]
            print(
                f"unique {len(stories)}/{args.count}  tried {attempts}  "
                f"{rate:.1f} samples/s  len {min(lens)}-{max(lens)}",
                flush=True,
            )
        if attempts > args.count * 6 and len(stories) < args.count:
            print(f"stopping early at {len(stories)} unique stories", flush=True)
            break

    write_corpus(out, stories)
    lens = sorted(len(s) for s in stories)
    def pct(p):
        return lens[min(len(lens) - 1, int(p * (len(lens) - 1)))]
    print(
        f"wrote {out}  n={len(stories)}  "
        f"len p50={pct(0.50)} p90={pct(0.90)} p99={pct(0.99)} max={lens[-1]}",
        flush=True,
    )
    sample = stories[0][1:-1]
    text = tokenizer.decode(sample, skip_special_tokens=False)
    print(text[:500], flush=True)


if __name__ == "__main__":
    main()
