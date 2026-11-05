//
//  TAESD.swift
//  MLXTAESD: a tiny, low-memory Stable Diffusion decoder for MLX Swift.
//
//  TAESD (Tiny AutoEncoder for Stable Diffusion) is a distilled decoder that turns an SD latent
//  into an image at a fraction of the full VAE's memory: 1.2 M parameters vs ~49.5 M, a ~4.9 MB
//  weights file, and (crucially) no GroupNorm and no attention, so its feature maps stay a
//  constant 64 channels wide. Those three absences are exactly why it is light. The trade is
//  honest: TAESD is measurably softer than the full VAE, dropping fine high-frequency detail.
//
//  This is a Swift/MLX port. The architecture is transliterated from madebyollin/taesd `taesd.py`
//  (MIT, © 2023 Ollin Boer Bohan). There is no other MLX-Swift TAESD, so it is offered back to the
//  ecosystem. Two scaling facts are load-bearing (get either wrong and the image is washed-out
//  garbage) and were read from the reference source, not remembered:
//
//    1. INPUT: feed the RAW SD latent straight in. TAESD does NOT apply the 0.18215 VAE scaling
//       (its diffusers `config.scaling_factor` is 1.0). The first layer's `tanh(x/3)*3` clamp
//       absorbs the range. So hand it the latent directly, NOT `latent / scalingFactor`.
//    2. OUTPUT: the decoder emits pixels already in [0, 1] (the reference demo is literally
//       `taesd.decoder(x).clamp(0, 1)`). Do NOT apply the `x/2 + 0.5` a full VAE (which emits
//       [-1, 1]) uses.
//
//  Weights: madebyollin/taesd: `taesd_decoder.safetensors` (4,895,612 bytes). Its 67 tensors map
//  key-for-key onto this module's 67 parameters. Load with `loadTAESDDecoder(url:)`.
//
//  License: MIT. See LICENSE (© 2023 Ollin Boer Bohan for the original TAESD; Swift port
//  © 2026 Mark Friedlander).
//

import Foundation
import MLX
import MLXNN

/// Nearest-neighbour 2x upsample for channels-last `[B, H, W, C]`. Same six lines as the
/// StableDiffusion example in ml-explore/mlx-swift-examples (MIT); MLXNN's own copy is private.
func upsampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
    precondition(x.ndim == 4)
    let (B, H, W, C) = x.shape4
    let x = broadcast(
        x[0..., 0..., .newAxis, 0..., .newAxis, 0...], to: [B, H, scale, W, scale, C])
    return x.reshaped(B, H * scale, W * scale, C)
}

/// One TAESD residual block: `conv → relu → conv → relu → conv`, added to a skip, fused by a relu.
///
/// In the SD decoder every block is 64→64, so the reference's skip (`Conv2d(n_in,n_out,1)` only
/// when `n_in != n_out`) is always `Identity` here; no skip parameters exist in the weights.
///
class TAESDBlock: Module, UnaryLayer {

    @ModuleInfo(key: "conv0") var conv0: Conv2d
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    init(_ channels: Int = 64) {
        self._conv0.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(conv0(x))
        h = relu(conv1(h))
        h = conv2(h)
        return relu(h + x)   // skip is Identity (n_in == n_out), so `+ x` is the whole skip
    }
}

/// The TAESD decoder: latent `[B, H, W, 4]` → image `[B, H*8, W*8, 3]` in [0, 1].
///
/// Layer-for-layer the reference `Decoder(latent_channels=4, use_midblock_gn=False)`:
///   Clamp · conv(4,64)·relu · Block×3 · up·conv(64,64,nobias) · Block×3 · up·conv · Block×3 ·
///   up·conv · Block · conv(64,3).
/// The three post-upsample convs carry no bias (reference `conv(64,64,bias=False)`); every other
/// conv does. Upsampling is nearest ×2 (reference `nn.Upsample(scale_factor=2)`).
public class TAESDDecoder: Module, UnaryLayer {

    @ModuleInfo(key: "convIn") var convIn: Conv2d
    @ModuleInfo(key: "stage0") var stage0: [TAESDBlock]
    @ModuleInfo(key: "up0") var up0: Conv2d
    @ModuleInfo(key: "stage1") var stage1: [TAESDBlock]
    @ModuleInfo(key: "up1") var up1: Conv2d
    @ModuleInfo(key: "stage2") var stage2: [TAESDBlock]
    @ModuleInfo(key: "up2") var up2: Conv2d
    @ModuleInfo(key: "tail") var tail: TAESDBlock
    @ModuleInfo(key: "convOut") var convOut: Conv2d

    override public init() {
        self._convIn.wrappedValue = Conv2d(
            inputChannels: 4, outputChannels: 64, kernelSize: 3, padding: 1)
        self._stage0.wrappedValue = [TAESDBlock(), TAESDBlock(), TAESDBlock()]
        self._up0.wrappedValue = Conv2d(
            inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1, bias: false)
        self._stage1.wrappedValue = [TAESDBlock(), TAESDBlock(), TAESDBlock()]
        self._up1.wrappedValue = Conv2d(
            inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1, bias: false)
        self._stage2.wrappedValue = [TAESDBlock(), TAESDBlock(), TAESDBlock()]
        self._up2.wrappedValue = Conv2d(
            inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1, bias: false)
        self._tail.wrappedValue = TAESDBlock()
        self._convOut.wrappedValue = Conv2d(
            inputChannels: 64, outputChannels: 3, kernelSize: 3, padding: 1)
    }

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var x = tanh(z / 3) * 3          // Clamp(): the input-range guard; also the reason no
                                         // 0.18215 pre-scaling is needed.
        x = relu(convIn(x))

        for b in stage0 { x = b(x) }
        x = up0(upsampleNearest(x))      // 64 → 128 px

        for b in stage1 { x = b(x) }
        x = up1(upsampleNearest(x))      // 128 → 256 px

        for b in stage2 { x = b(x) }
        x = up2(upsampleNearest(x))      // 256 → 512 px

        x = tail(x)
        x = convOut(x)
        return x                         // already ~[0, 1]; caller clamps
    }

    /// Decode a final SD latent to pixels: `[B, H*8, W*8, 3]` clamped to [0, 1]. **Feed the RAW
    /// latent**, no `/ scalingFactor`.
    public func decodePixels(_ z: MLXArray) -> MLXArray {
        clip(callAsFunction(z), min: 0, max: 1)
    }
}

/// Rewrite a `taesd_decoder.safetensors` key onto this module's key paths, and transpose conv
/// weights from PyTorch layout `[O, I, kH, kW]` to MLX's channels-last `[O, kH, kW, I]`. Returns
/// `nil` for keys this decoder does not own (for example, stray encoder tensors), which the loader drops.
///
/// The reference decoder is an `nn.Sequential`, so its keys are position-numbered: `1.weight` is
/// `conv(4,64)`, `3.conv.0.weight` is the first block's first conv, `7.weight` is the first
/// post-upsample bias-less conv, and so on. This table maps those leading indices onto the named
/// stages above.
let taesdIndexMap: [String: String] = [
    "1": "convIn",
    "3": "stage0.0", "4": "stage0.1", "5": "stage0.2",
    "7": "up0",
    "8": "stage1.0", "9": "stage1.1", "10": "stage1.2",
    "12": "up1",
    "13": "stage2.0", "14": "stage2.1", "15": "stage2.2",
    "17": "up2",
    "18": "tail",
    "19": "convOut",
]

func taesdRemap(key: String, value: MLXArray) -> [(String, MLXArray)]? {
    // Keys look like "3.conv.0.weight", "1.weight", or "7.weight". Split off the leading index.
    guard let dot = key.firstIndex(of: ".") else { return nil }
    let head = String(key[key.startIndex ..< dot])
    guard let mapped = taesdIndexMap[head] else { return nil }   // not ours: drop it

    // Inside a block, `self.conv` is Sequential(conv, relu, conv, relu, conv) → indices 0/2/4.
    let rest = String(key[key.index(after: dot)...])
        .replacingOccurrences(of: "conv.0.", with: "conv0.")
        .replacingOccurrences(of: "conv.2.", with: "conv1.")
        .replacingOccurrences(of: "conv.4.", with: "conv2.")

    var v = value
    if v.ndim == 4 {
        v = v.transposed(0, 2, 3, 1)
        v = v.reshaped(-1).reshaped(v.shape)   // force contiguity
    }
    return [(mapped + "." + rest, v)]
}

/// Build a TAESD decoder and load its weights from a `taesd_decoder.safetensors` on disk.
/// Weights are float32 by default (the file is tiny; no reason to quantize the one cheap part of
/// the pipeline).
public func loadTAESDDecoder(url: URL, dType: DType = .float32) throws -> TAESDDecoder {
    let model = TAESDDecoder()
    let weights = try loadArrays(url: url)
        .compactMap { taesdRemap(key: $0.key, value: $0.value.asType(dType)) }
        .flatMap { $0 }
    // `.all`: every one of the decoder's 67 parameters must be filled, with matching shapes, so a
    // wrong or truncated weights file fails loudly here instead of decoding noise.
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    eval(model)
    return model
}
