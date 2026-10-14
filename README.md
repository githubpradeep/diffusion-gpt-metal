# transformer-metal

A decoder-only GPT trained from scratch on a Mac GPU. The math lives in Metal compute kernels, in the same spirit as nanoGPT and llm.c. One run learns Tiny Shakespeare. Another is a 19,969-parameter story model. A third is a 1,000,193-parameter looped controller for ten offline actions.

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

## Tool calls

`make tools` trains the same looped block at a wider size: hidden size 256, SwiGLU width 758, 4 query heads, 1 key/value head, 4 passes, 987 words, **1,000,193** parameters. The checkpoint is `build/tools.ckpt`. It is one call per sentence. The arguments are copied out of the sentence. If the sentence is not one of the ten actions, the model emits `<refuse>`.

The ten actions are timer, alarm, reminder, message, calendar event, weather, music, directions, note, and opening an app. `scripts/distill_tools.py` fixes each label before a teacher writes the sentence, and drops any sentence that does not contain the argument values. The packed corpus is `data/tools/calls.bin` (gitignored) with the word list in `vocab/tools.txt`. After 4,000 steps the training loss was 0.577 and the held-out loss was 1.268. Greedy decode matches the gold call on **200 of 207** held-out requests. Short fragments sometimes keep writing the sentence instead of starting the call.

```bash
make tools
./build/gpt tools-eval --ckpt build/tools.ckpt
./build/gpt tools-sample --ckpt build/tools.ckpt --prompt "Could you set a timer for 10 minutes for pasta" --temp 0
make demo
```

`make demo` serves http://127.0.0.1:8787 and keeps the checkpoint loaded. `demo/controller.mp4` is a 24-second recording of that page. Words outside `vocab/tools.txt` are rejected before the model runs. A colon is its own token, so a time comes back as `6: 45`.

## Logit lens

The controller is a looped block, so the same weights run four times before anything is emitted. `http://127.0.0.1:8787/lens` reads the tied output head off the residual stream after every pass and shows the whole ladder: which word the model expected at each depth, and how confident it was.

The interesting column is the last one. After the final token of a request, confidence on the correct next word climbs across the passes. For `Could you set a timer for 10 minutes for pasta`, the readout after `pasta` goes `<|end|>` at 0.50, then `?` at 0.89, 0.92, and 1.00. The answer is not there after one pass. It is there by three.

The `embedding` column is the readout before any block has run. Because the head is the embedding matrix, it hands the input straight back — a flat identity that every later column has to beat.

```bash
make demo
./build/gpt lens-repl --ckpt build/tools.ckpt --vocab vocab/tools.txt --k 5
```

`lens-repl` speaks the same one-prompt-per-line protocol as `tools-repl`, so both models stay resident in one server.

## Drawings

The painter and the diffusion model are both the looped block from the story run. One set of weights runs four times. RMSNorm, RoPE, the value residual, and SwiGLU are the same pieces. The four small frames under each canvas are those four passes. Each run has its own checkpoint.

You prompt with a single name. The page at http://127.0.0.1:8787/draw shows fourteen photographs. `demo/draw.mp4` records that page for 26 seconds: house, then cat, then pizza.

### Photographs

`make photos` downloads 14 Wikimedia Commons pictures and center-crops them to 128×128. The names are cat, dog, horse, bird, lion, elephant, pizza, burger, car, house, flower, mountain, beach, and tree. Credits are in `data/photos/credit.json`. The originals are in `demo/photos/`.

The diffusion copy sees the whole photograph at once: 64 patches of 16×16. Attention is open, so every patch sees every other patch. A learned name row and a sinusoidal timestep are added onto every token. The head predicts the clean patch, and the loss is that prediction's mean squared error. Sampling is 20 deterministic DDIM steps. The block is wider than the painter (embedding 192, SwiGLU 768). The checkpoint is `build/photos-dit.ckpt`, **906,816** parameters. Training uses 100 linear noise levels and sets the last alpha-bar to `0.0001`, and half of each batch is drawn from the noisiest fifth of those levels, so a sample starts from noise. With fourteen pictures the weights memorize each name.

The painter for the same photographs predicts patch ids. `make photo-tokens` gives every distinct 16×16 patch its own id, so the sequence is the name, 64 ids, and an end token. Attention stays causal. The checkpoint is `build/photos-ar.ckpt`, **93,697** parameters. A finished sample matches the photograph, because each id is a stored patch.

```bash
make photos
make photo-tokens
./build/gpt photos-ar
./build/gpt images-dit --vocab vocab/photos.txt --rgb data/photos/rgb.bin \
  --ckpt build/photos-dit.ckpt --patch 16 --embd 192 --intermediate 768 \
  --batch 8 --steps 2500 --lr 0.006 --last-alpha 0.0001
make demo
```

```bash
./build/gpt photos-ar-sample --prompt cat
./build/gpt images-dit-sample --ckpt build/photos-dit.ckpt --vocab vocab/photos.txt --prompt cat
```

Both sample commands print one JSON document: the frames, and the four loop snapshots.

### Twemoji icons

The same two runs also train on 128 Twemoji icons at 24×24. You prompt with a name such as `kite` or `heart`.

The autoregressive run reads the icon in raster order. Each pixel is one of 32 shared colors, so a picture is 576 tokens after the name. Attention stays causal. The checkpoint is `build/images-ar.ckpt`, **45,697** parameters.

The diffusion run uses 36 patches of 4×4, with the causal mask off, and predicts the clean patch. It is wider than that painter (embedding 128, SwiGLU 512). The checkpoint is `build/images-dit.ckpt`, **299,648** parameters. Training uses 100 linear noise levels, from beta `1e-4` up to about `0.089`, which leaves the last alpha-bar near `0.01`. Both icon runs train on all 128 names.

```bash
make pictures
make images-check
make images-ar
make images-dit
```

`make pictures` downloads the icons. The artwork is [Twemoji](https://github.com/jdecked/twemoji), CC-BY 4.0, copyright 2020 Twitter, Inc and other contributors. `images-check` memorizes one icon with each model.

## Layout

- `metal/kernels.metal` — quantize, grouped int4/int8 GEMM, LayerNorm, RMSNorm, RoPE, GELU, SwiGLU, causal attention, embeddings, loss, AdamW
- `src/GPT.swift` — the Shakespeare block, the backward pass, checkpoints
- `src/Macro.swift` — the looped block, Muon, training, sampling, tool-call eval
- `scripts/distill_tools.py` — fixed tool labels, then a teacher writes the sentence
- `vocab/tools.txt` — the 987-word tool list
- `demo/` — the controller page, the logit lens, the drawing page, `controller.mp4`, `draw.mp4`, and the server that keeps the checkpoints loaded
- `scripts/make_pictures.py` — download 128 Twemoji icons and pack the two training files
- `scripts/make_photos.py` — download 14 Wikimedia photographs and pack the diffusion file
- `scripts/make_photo_tokens.py` — turn those photographs into 64 patch ids
- `vocab/photos.txt`, `vocab/photos-ar.txt`, `vocab/images.txt` — picture names for the two drawing runs
- `src/DiT.swift` — the diffusion transformer: the looped block with full attention, clean-image loss, and a 20-step sampler
- `src/MacroData.swift` — the teacher-story corpus used by `make stories`
- `src/MacroVocab.swift` — the 378-word list and prompt encoding
- `scripts/distill_stories.py` — sample `raincandy-u/MacroStories` into `data/macro-distill.bin`
- `data/hindi-by-hand/` — the 120 handwritten Hindi stories
- `vocab/hindi.txt` — the Hindi word list
- `src/WordList.swift` — loads that list at train and sample time
- `src/Train.swift` — Tiny Shakespeare, the training loop, sampling
- `src/Check.swift` — CPU references and finite-difference checks
