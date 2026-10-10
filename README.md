# transformer-metal

A decoder-only GPT trained from scratch on a Mac GPU. The math lives in Metal compute kernels, in the same spirit as nanoGPT and llm.c. One run learns Tiny Shakespeare. The other is a 19,969-parameter story model.

Forward GEMMs use **int8** (default) or **int4** weights. Master weights, activations, gradients, and Adam moments stay in fp32. After each optimizer step the master copy is quantized again. Backward uses the straight-through estimator: the rounding and the per-group scale are a stop-gradient, and `dW` is the ordinary outer product. That is what actually converges. A pure 4-bit optimizer does not.

## Build

Xcode or the Command Line Tools are enough. Kernels in `metal/kernels.metal` are compiled on first launch with Metal's runtime compiler.

```bash
make check      # kernel tests, then a tiny model must memorize one batch
make overfit    # larger one-batch memorization, int8
make train      # Tiny Shakespeare, int8, 800 steps
./build/gpt sample --prompt "ROMEO:"
```

`make train` downloads Karpathy's Tiny Shakespeare into `data/input.txt` on first run.

## Useful flags

```bash
./build/gpt train --bits 4 --steps 800
./build/gpt train --bits 32 --layers 6 --heads 6 --embd 384 --batch 32 --block 128 --steps 2000
./build/gpt overfit --bits 4
./build/gpt sample --ckpt build/model.ckpt --tokens 300 --temp 0.8
```

`--bits` is 4, 8, or 32. 32 keeps every GEMM in fp32 so you can compare the quantized run against a full-precision one. Quantized weights use a scale for every 8 inputs along each output row. int4 packs two codes per byte, range -7..7. int8 uses -127..127.

## Model

Character-level, pre-norm, learned positions, causal attention, tanh-approximate GELU, 4× MLP, tied token embedding and output head, AdamW, mean cross-entropy. No dropout.

A fresh model should start near `ln(vocab)`, about 4.17 for Shakespeare. On an M1 Pro the default int8 net (810k parameters, batch 16, block 64) trains at about 13k tokens/s. After 1000 steps the loss is about 2.1 and the samples start to pick up English words. This is a small model, so a wider net or a few thousand more steps keeps improving it. `./build/gpt train --bits 32` is the full-precision comparison.

## MacroStories

`make stories` trains the same shape as [raincandy-u/MacroStories](https://huggingface.co/raincandy-u/MacroStories): 19,969 parameters, hidden size 32, SwiGLU width 48, one decoder block applied 4 times with shared weights, 2 query heads and 1 key/value head, RoPE, sandwich RMSNorm, and a tied 378-word vocabulary. Muon updates the decoder matrices. AdamW updates the embeddings and norms. The published run is 4,000 steps, batch 32, learning rate 0.003, cosine decay with 5% warmup, seed 456.

The original stories were written by a local Gemma model and were not released. `scripts/distill_stories.py` samples the published Hugging Face model at temperature 0.5 and top-p 0.9, and `make stories` trains on those token ids with the same 4,000-step recipe. Names render as Alex, Robin, and Casey.

```bash
make stories-check
make stories
./build/gpt stories-sample --ckpt build/macro.ckpt
./build/gpt stories-sample --ckpt build/macro.ckpt --prompt '"I want to fix this kite," Alex said.'
```

Each run picks a new seed. Pass `--seed N` to repeat that sample. `--prompt` continues that opening; the words have to be in the 378-word list. Sampling defaults match the published model: temperature 0.5 and top-p 0.9.

The block diagram, the parameter table, the corpus format, and sample outputs are in [docs/architecture.md](docs/architecture.md).

## Hindi stories

The same Metal block trains on Hindi. The 120 stories in `data/hindi-by-hand/` were written directly, then packed into `vocab/hindi.txt` and `data/hindi-stories.bin`. The checkpoint is `build/hindi.ckpt`, separate from the English run.

```bash
make stories-hi
./build/gpt stories-hi-sample --ckpt build/hindi.ckpt --seed 7
./build/gpt stories-hi-sample --ckpt build/hindi.ckpt --prompt "एक दिन मीना" --seed 17
```

## Layout

- `metal/kernels.metal` — quantize, grouped int4/int8 GEMM, LayerNorm, RMSNorm, RoPE, GELU, SwiGLU, causal attention, embeddings, loss, AdamW
- `src/GPT.swift` — the Shakespeare block, the backward pass, checkpoints
- `src/Macro.swift` — the 19,969-parameter story model, Muon, training, sampling
- `src/MacroData.swift` — the teacher-story corpus used by `make stories`
- `src/MacroVocab.swift` — the 378-word list and prompt encoding
- `scripts/distill_stories.py` — sample `raincandy-u/MacroStories` into `data/macro-distill.bin`
- `data/hindi-by-hand/` — the 120 handwritten Hindi stories
- `vocab/hindi.txt` — the Hindi word list
- `src/WordList.swift` — loads that list at train and sample time
- `src/Train.swift` — Tiny Shakespeare, the training loop, sampling
- `src/Check.swift` — CPU references and finite-difference checks
