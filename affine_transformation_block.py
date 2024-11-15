#!/usr/bin/env python3
"""uint8 tensor affine transformation demo."""

from dataclasses import dataclass
import sys

try:
  import torch
except ImportError:
  print("Run: pip install -r requirements.txt", file=sys.stderr)
  sys.exit(1)


@dataclass
class block_qx_1:
  """A block of weights, similar to block_q8_1 at
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-common.h
  """
  scale: torch.Tensor # float16
  zero_point: torch.Tensor # float16
  packed: torch.Tensor # 32x uint8 for Q8_1 or as appropriate.


@dataclass
class quantized:
  """A whole quantized tensor as a list of blocks."""
  blocks: list[block_qx_1]


def quantize_to_uint(t: torch.Tensor, bits: int) -> quantized:
  """Quantize (compress) the tensor."""
  # TODO: this codes assumes a tensor of length that is a multiple of
  # batch_size. Production code would have to store the actual length.
  batch_size = 32
  assert 2 <= bits <= 8
  bitsmax = (1<<bits)-1
  l = t.size(0)
  out = quantized([])
  for i in range((l+batch_size-1)//batch_size):
    start = i * batch_size
    end = min(start + batch_size, l)
    batch = t[start:end]
    min_val = batch.min()
    max_val = batch.max()

    # Determine the scale and the zero point: the value in the tensor to map to 0.
    scale = (max_val - min_val) / bitsmax
    zero_point = min_val

    # Bias then scale to change the range to [0, bitsmax] then reduce to uint8
    # to reduce storage by 4x.
    # TODO: For bits < 8, we'd want to patch the bits more efficiently.
    packed = ((batch - zero_point) / scale).clamp(0, bitsmax).to(torch.uint8)
    scale = scale.to(torch.float16)
    zero_point = zero_point.to(torch.float16)
    out.blocks.append(block_qx_1(scale, zero_point, packed))
  return out


def dequantize_from_uint(q: quantized) -> torch.Tensor:
  """Dequantize (decompress) the quantized tensor."""
  if not q.blocks:
    return torch.empty(0)
  return torch.cat(
      tuple(b.packed.to(torch.float32) * b.scale + b.zero_point
            for b in q.blocks))


def main():
  g = torch.Generator(device="cpu")
  g.manual_seed(1)
  t = torch.rand(256, generator=g)*10. - 5
  print(f"Original tensor of 256 random values between -5 and 5:")
  print(f"- [{', '.join('{:.4f}'.format(x) for x in t[:8].tolist())}, ...]")
  print(f"- storage: {len(t)*4} bytes ({len(t)}*4)")

  for bits in [8, 4]:
    print(f"\nQuantizing as {bits} bits:")
    q = quantize_to_uint(t, bits)
    print(f"  Quantized tensor:")
    b0 = q.blocks[0]
    print(f"  - scale:      {b0.scale:g}")
    print(f"  - zero_point: {b0.zero_point:g}")
    print(f"  - packed:     [{', '.join(str(x) for x in b0.packed[:8].tolist())}, ...]")
    print(f"  - storage:    {len(q.blocks)*int(2+2+len(b0.packed)*bits/8)} bytes ({len(q.blocks)}*(2+2+{len(b0.packed)}*{bits}/8))")

    d = dequantize_from_uint(q)
    print(f"  Dequantized tensor:")
    print(f"  - [{', '.join('{:.4f}'.format(x) for x in d[:8].tolist())}, ...]")
    # https://en.wikipedia.org/wiki/Mean_squared_error
    mse = (t - d) ** 2
    print(f"  - Mean squared error: {sum(mse)/len(mse):g}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

