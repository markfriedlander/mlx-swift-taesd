# MLXTAESD

A tiny, low-memory Stable Diffusion decoder for [MLX Swift](https://github.com/ml-explore/mlx-swift).

**TAESD** (Tiny AutoEncoder for Stable Diffusion, by [Ollin Boer Bohan](https://github.com/madebyollin/taesd))
turns an SD latent into an image at a fraction of the full VAE's memory: about 1.2 M parameters versus about
49.5 M, a 4.9 MB weights file, and no GroupNorm and no attention, so its feature maps stay a constant 64
channels wide. On an iPhone, the full VAE decode of a 64x64 latent to 512x512 can peak around 4.5 GB in
intermediate activations; TAESD decodes the same latent while adding almost nothing to the memory peak.

The trade is honest: TAESD is measurably **softer** than the full VAE and drops fine high-frequency detail.
Use it as a low-memory or fast option, not as a silent replacement for a full VAE where fidelity matters.

As far as we could find, there was no MLX Swift TAESD, so this is offered back. It is a transliteration of
the reference [`taesd.py`](https://github.com/madebyollin/taesd), and it has run on-device in
[Thomas](https://github.com/markfriedlander/Thomas), an iPhone camera app, where it fixed an out-of-memory
crash in the full VAE decode.

## Verified against the original

The package's tests decode the same latent with this port and with the reference PyTorch TAESD, using the
official `taesd_decoder.safetensors`. The largest per-pixel difference is about **5e-6** on a 0 to 1 scale,
which is floating-point rounding. You can rerun it yourself (see Tests below).

## Install

Swift Package Manager:

```swift
.package(url: "https://github.com/markfriedlander/mlx-swift-taesd", from: "1.0.0"),
```

Then add `"MLXTAESD"` to your target's dependencies. Requires iOS 17 or macOS 14 (the same as MLX Swift).

## Use

```swift
import MLX
import MLXTAESD

// Weights: taesd_decoder.safetensors from https://huggingface.co/madebyollin/taesd
let decoder = try loadTAESDDecoder(url: weightsURL)

// `latent` is a RAW Stable Diffusion latent [B, H, W, 4]. Do NOT pre-scale it.
let image = decoder.decodePixels(latent)   // [B, H*8, W*8, 3], clamped to [0, 1]
```

The loader checks that all 67 parameters are present with the right shapes, so a wrong or truncated
weights file fails with an error instead of decoding noise.

## Two scaling facts that matter

Get either wrong and the image comes out as washed-out garbage. Both come from the reference source:

1. **Input:** feed the **raw** SD latent straight in. TAESD does *not* use the `0.18215` VAE scaling (its
   `config.scaling_factor` is `1.0`); the first layer's `tanh(x/3)*3` clamp absorbs the range. Hand it the
   latent directly, **not** `latent / scalingFactor`.
2. **Output:** the decoder already emits pixels in `[0, 1]` (the reference is
   `taesd.decoder(x).clamp(0, 1)`). Do **not** apply the `x/2 + 0.5` that a full VAE (which emits
   `[-1, 1]`) needs. `decodePixels` clamps to `[0, 1]` for you.

## Weights

Download `taesd_decoder.safetensors` (4,895,612 bytes, 67 tensors) from
[madebyollin/taesd](https://huggingface.co/madebyollin/taesd). Its keys map one-to-one onto this module's 67
parameters; `loadTAESDDecoder` handles the renaming and the PyTorch-to-MLX convolution weight transpose.
Weights are not bundled in this package.

## Tests

`xcodebuild test -scheme MLXTAESD -destination 'platform=macOS'` runs a parameter-count check and a shape
and range check. The PyTorch comparison runs when you provide the weights and a reference decode:

```sh
# needs torch, safetensors, numpy, and taesd.py from https://github.com/madebyollin/taesd
python Scripts/make_reference.py path/to/taesd_decoder.safetensors /tmp/taesd-ref

TEST_RUNNER_TAESD_WEIGHTS=path/to/taesd_decoder.safetensors \
TEST_RUNNER_TAESD_REFERENCE_DIR=/tmp/taesd-ref \
xcodebuild test -scheme MLXTAESD -destination 'platform=macOS'
```

## License

MIT. The original TAESD architecture and weights are © 2023 Ollin Boer Bohan
([madebyollin/taesd](https://github.com/madebyollin/taesd)); the Swift/MLX port is © 2026 Mark Friedlander.
The small `upsampleNearest` helper matches the StableDiffusion example in
[ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (MIT). See [LICENSE](LICENSE).
