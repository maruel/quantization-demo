#!/usr/bin/env python3
"""uint8 tensor affine transformation demo."""

import sys

try:
  import torch
except ImportError:
  print("Run: pip install -r requirements.txt", file=sys.stderr)
  sys.exit(1)


def quantize_to_uint8(t: torch.Tensor):
  min_val = t.min()
  max_val = t.max()

  # Determine the scale and the zero point: the value in the tensor to map to 0.
  scale = (max_val - min_val) / 255.
  zero_point = min_val

  # Bias then scale to change the range to [0, 255] then reduce to uint8 to
  # reduce storage by 4x.
  q = (t - zero_point) / scale
  q = torch.clamp(q, min=0, max=255)
  q = q.type(torch.uint8)
  return q, scale, zero_point


def dequantize_from_uint8(t: torch.Tensor, scale: float, zero_point: float):
  return t.to(torch.float32) * scale + zero_point


def main():
  t = torch.rand(8)*10.
  print(f"Original tensor of 8 random values between 0 and 10:")
  print(f"- {t}")
  print(f"- storage: {len(t)*4} bytes ({len(t)}*4)\n")

  q, scale, zero_point = quantize_to_uint8(t)
  print(f"Quantized tensor:")
  print(f"- {q}")
  print(f"- scale:      {scale}")
  print(f"- zero_point: {zero_point}")
  print(f"- storage: {len(q)+4+4} bytes ({len(q)}+4+4)\n")

  d = dequantize_from_uint8(q, scale, zero_point)
  print(f"Dequantized tensor:")
  print(f"- {d}\n")

  err = t - d
  print(f"Numerical error induced by the quantization:")
  print(f"- absolute: {err}")
  print(f"- relative: {err/t}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

