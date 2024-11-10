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
class quantized:
  """A vector long quantized based on an affine transformation.
  """
  scale: torch.Tensor # float16
  zero_point: torch.Tensor # float16
  packed: torch.Tensor # 32x uint8 for Q8_1 or as appropriate.


def quantize_to_uint(t: torch.Tensor, bits: int) -> quantized:
  """Quantize (compress) the tensor."""
  assert 2 <= bits <= 8
  bitsmax = (1<<bits)-1
  l = t.size(0)
  min_val = t.min()
  max_val = t.max()

  # Determine the scale and the zero point: the value in the tensor to map to 0.
  scale = (max_val - min_val) / bitsmax
  zero_point = min_val

  # Bias then scale to change the range to [0, bitsmax] then reduce to uint8
  # to reduce storage by 4x.
  # TODO: For bits < 8, we'd want to patch the bits more efficiently.
  q = ((t - zero_point) / scale).clamp(0, bitsmax).to(torch.uint8)
  scale = scale.to(torch.float16)
  zero_point = zero_point.to(torch.float16)
  return quantized(scale, zero_point, q)


def dequantize_from_uint(q: quantized) -> torch.Tensor:
  """Dequantize (decompress) the quantized tensor."""
  return q.packed.to(torch.float32) * q.scale + q.zero_point


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
    print(f"  - scale:      {q.scale:g}")
    print(f"  - zero_point: {q.zero_point:g}")
    print(f"  - values:     [{', '.join(str(x) for x in q.packed[:8].tolist())}, ...]")
    print(f"  - storage:    {int(2+2+len(q.packed)*bits/8)} bytes ((2+2+{len(q.packed)}*{bits}/8))")

    d = dequantize_from_uint(q)
    print(f"  Dequantized tensor:")
    print(f"  - [{', '.join('{:.4f}'.format(x) for x in d[:8].tolist())}, ...]")
    # https://en.wikipedia.org/wiki/Mean_squared_error
    mse = (t - d) ** 2
    print(f"  - Mean squared error: {sum(mse)/len(mse):g}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

