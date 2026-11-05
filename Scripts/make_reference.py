"""Make a PyTorch reference decode for MLXTAESDTests.testMatchesPyTorchReference.

Needs: torch, safetensors, numpy, and taesd.py from https://github.com/madebyollin/taesd
Usage: python make_reference.py <taesd_decoder.safetensors> <out_dir>
Writes latent.npy (channels-last [1, 16, 16, 4]) and image.npy (channels-last, clamped to [0, 1]).
"""
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from taesd import Decoder

weights, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

decoder = Decoder()
decoder.load_state_dict(load_file(weights))
decoder.eval()

torch.manual_seed(0)
latent = torch.randn(1, 4, 16, 16)  # a raw SD latent, NCHW
with torch.no_grad():
    image = decoder(latent).clamp(0, 1)

np.save(os.path.join(out, "latent.npy"), latent.permute(0, 2, 3, 1).numpy().astype(np.float32))
np.save(os.path.join(out, "image.npy"), image.permute(0, 2, 3, 1).numpy().astype(np.float32))
print("wrote", out, tuple(image.shape))
