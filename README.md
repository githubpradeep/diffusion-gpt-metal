# diffusion-gpt-metal

A looped transformer that draws, trained from scratch on a Mac GPU. The arithmetic is Metal compute kernels.

Two runs share that block. The autoregressive painter fills a photograph one 16×16 patch at a time. The diffusion transformer starts from noise and predicts the clean photograph. One set of weights runs four times in each model. RMSNorm, RoPE, the value residual, and SwiGLU are the same pieces. Each run has its own checkpoint. The four small frames under each canvas are those four passes.

You prompt with a single name. The page at http://127.0.0.1:8787/draw shows fourteen photographs. `demo/draw.mp4` records that page for 26 seconds: house, then cat, then pizza.

The block diagram and the noise schedule are in [docs/architecture.md](docs/architecture.md).

## Photographs

`make photos` downloads 14 Wikimedia Commons pictures and center-crops them to 128×128. The names are cat, dog, horse, bird, lion, elephant, pizza, burger, car, house, flower, mountain, beach, and tree. Credits are in `data/photos/credit.json`. The originals are in `demo/photos/`.

The diffusion model sees the whole photograph at once: 64 patches of 16×16. Attention is open, so every patch sees every other patch. A learned name row and a sinusoidal timestep are added onto every token. The head predicts the clean patch, and the loss is that prediction's mean squared error. Sampling is 20 deterministic DDIM steps. The block is wider than the painter (embedding 192, SwiGLU 768). The checkpoint is `build/photos-dit.ckpt`, **906,816** parameters. Training uses 100 linear noise levels and sets the last alpha-bar to `0.0001`, and half of each batch is drawn from the noisiest fifth of those levels, so a sample starts from noise. With fourteen pictures the weights memorize each name.

The painter predicts patch ids. `make photo-tokens` gives every distinct 16×16 patch its own id, so the sequence is the name, 64 ids, and an end token. Attention stays causal. The checkpoint is `build/photos-ar.ckpt`, **93,697** parameters. A finished sample matches the photograph, because each id is a stored patch.

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

Both sample commands print one JSON document: the frames, and the four loop snapshots. Xcode or the Command Line Tools are enough. Kernels in `metal/kernels.metal` are compiled on first launch.

## Twemoji icons

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

## The block underneath

The image runs sit on a looped decoder this repo already trained for text. `make stories` is the [MacroStories](https://huggingface.co/raincandy-u/MacroStories) shape: 19,969 parameters, hidden size 32, one block applied 4 times. `make stories-hi` is that block on 120 handwritten Hindi stories. `make tools` is the same block at 1,000,193 parameters, one call per sentence for ten offline actions, with a controller page and a logit lens at http://127.0.0.1:8787/lens. `make train` is the quantized GPT the kernels were proved on, character-level Tiny Shakespeare, int8 or int4 GEMMs with fp32 master weights.

```bash
make check
make stories
make tools
make demo
```

`demo/controller.mp4` is a 24-second recording of the controller page.

## Layout

- `src/DiT.swift` — the diffusion transformer: the looped block with full attention, clean-image loss, and a 20-step sampler
- `src/Pictures.swift` — icon and photograph loading, the patch painter, sampling
- `src/Macro.swift` — the looped block, Muon, training, sampling
- `metal/kernels.metal` — quantize, grouped int4/int8 GEMM, LayerNorm, RMSNorm, RoPE, GELU, SwiGLU, attention, embeddings, loss, AdamW
- `scripts/make_photos.py` — download 14 Wikimedia photographs and pack the diffusion file
- `scripts/make_photo_tokens.py` — turn those photographs into 64 patch ids
- `scripts/make_pictures.py` — download 128 Twemoji icons and pack the two training files
- `vocab/photos.txt`, `vocab/photos-ar.txt`, `vocab/images.txt` — picture names
- `demo/` — the drawing page, `draw.mp4`, the controller, the logit lens, and the server that keeps the checkpoints loaded
- `src/GPT.swift` — the Shakespeare block the kernels were proved on
- `src/Train.swift` — commands, the training loop, sampling
- `src/Check.swift` — CPU references and finite-difference checks
