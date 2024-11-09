#!/usr/bin/env python3
"""uint8 tensor k-quantization demo.

This code is not optimized.
"""

from dataclasses import dataclass
import math
import sys

try:
  import torch
except ImportError:
  print("Run: pip install -r requirements.txt", file=sys.stderr)
  sys.exit(1)


@dataclass
class superblock:
  """Based on block_q4_K at
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-common.h

  Ultimately, subscales would be packed as 12 bytes.
  """
  scale: float = None # torch.Tensor # float16
  zero_point: float = None # torch.Tensor # float16
  subscales: torch.Tensor = None  # 8x 6-bits encoded values encoded as uint8
  suboffsets: torch.Tensor = None  # 8x 6-bits encoded values encoded as uint8
  values: torch.Tensor = torch.zeros(256, dtype=torch.uint8) # 256x values stored as uint8


@dataclass
class quantized:
  """A whole quantized tensor as a list of blocks."""
  blocks: list[superblock]


def nearest_int(f):
  return int(round(f))


def make_qp_quants(nmax: int, x: torch.Tensor, quant_weights: torch.Tensor):
  #print(f"make_qp_quants({nmax}, {x}, {quant_weights}")
  n = len(x)
  # float make_qp_quants(int n, int nmax, const float * x, uint8_t * L, const float * quant_weights):
  fmax = x.max()
  if not fmax.item():
    # All zeros.
    return torch.zeros(1), torch.zeros(len(x)).to(torch.uint8)
  iscale = nmax / fmax
  L = (iscale * x).to(torch.uint8)
  # Search for the best Mean squared error. First calculate for iis 0.
  scale = 1./iscale
  diff = x - (scale*L)
  best_mse = (quant_weights*(diff**2)).sum()
  for iis in range(-4, 5):
    if not iis:
      continue
    iscale_is = (0.1*iis + nmax)/fmax
    scale_is = 1./iscale_is
    # Do not update L here.
    l = torch.minimum((iscale_is*x).to(torch.uint8).min(), nmax)
    diff = x - (scale_is*l)
    mse = (quant_weights*(diff**2)).sum()
    if mse.item() < best_mse.item():
      best_mse = mse
      iscale = iscale_is

  L = torch.minimum((iscale_is*x).to(torch.uint8), nmax)
  sumlx = (quant_weights*x*L).sum()
  suml2 = (quant_weights*(L**2)).sum()
  for itry in range(5):
    #print("  itry", itry)
    n_changed = 0
    slx = sumlx - quant_weights*x*L
    sl2 = suml2 - quant_weights*(L**2)
    new_l = torch.minimum((x * sl2 / slx).to(torch.uint8), nmax)
    for i in range(n):
      if slx[i] > 0 and sl2[i] > 0:
        if new_l[i] != L[i]:
          slx[i] += quant_weights[i]*x[i]*new_l[i]
          sl2[i] += quant_weights[i]*(new_l[i]**2)
          if ((slx[i]**2)*suml2.item()) > ((sumlx.item()**2)*sl2[i]):
            L[i] = new_l[i]
            sumlx = slx[i]
            suml2 = sl2[i]
            ++n_changed
    if not n_changed:
      break
  #print(f"make_qp_quants() -> {sumlx/suml2}, {L}")
  return sumlx/suml2, L



def make_qkx3_quants(nmax: torch.Tensor, x: torch.Tensor, weights: torch.Tensor):
  """Simplified version from make_qkx3_quant() in
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c

  In the original code, use_mad is always false and weights is always specified.

  Note: we ignore the quantized values generated! We only want to figure out the
  scale and fmin.
  """
  # These are always the same values in the original code:
  rmin = -0.9
  rdelta = 0.05
  nstep = 36

  #print(f"make_qkx3_quants()")
  n = len(x)
  fmin = torch.minimum(x.min(), torch.zeros(1))
  fmax = x.max()
  if fmax <= fmin:
    # All negative values.
    return 0., -fmin
  sum_w = weights.sum()
  sum_x = (weights*(x**2)).sum()
  iscale = nmax/(fmax - fmin)
  scale = 1./iscale
  l = (iscale*(x - fmin)).to(torch.uint8)
  L = torch.maximum(torch.minimum(l, nmax), torch.zeros(1).to(torch.uint8))
  diff = scale * L + fmin - x
  diff = diff**2
  best_mad = (weights * diff).sum()
  Laux = torch.zeros(32).to(torch.uint8)
  for iis in range(nstep):
    iscale = (rmin + rdelta*iis + nmax)/(fmax - fmin)
    l = (iscale*(x-fmin)).to(torch.uint8)
    Laux = torch.maximum(torch.minimum(l, nmax), torch.zeros(1).to(torch.uint8))
    sum_l = (weights*l).sum()
    sum_l2 = (weights*(l**2)).sum()
    sum_xl = (weights*l*x).sum()
    D = sum_w * sum_l2 - sum_l * sum_l
    if D.item() > 0:
      this_scale = (sum_w * sum_xl - sum_x * sum_l)/D
      this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D
      if this_min.item() > 0:
        this_min = torch.zeros(1)
        this_scale = sum_xl / sum_l2
      diff = this_scale * Laux + this_min - x
      diff = diff**2
      mad = (weights*diff).sum()
      if mad.item() < best_mad.item():
        L = Laux
        best_mad = mad
        scale = this_scale
        fmin = this_min
        print(f"  fmin = {fmin}")
  #print(f"make_qkx3_quants({nmax}, {x}, {weights}) -> {scale}, {-fmin}, {L}")
  #print(f"make_qkx3_quants() -> {scale}, {-fmin}")
  return scale, -fmin


def quantize_to_Q4_K_block(x: torch.Tensor, quant_weights = None) -> superblock:
  bits = 4
  # Batch size, aka QK_K
  bs = len(x)
  bitsmax = torch.tensor((1<<bits)-1).to(torch.uint8)
  sw = torch.zeros(bs//32)
  mins = torch.zeros(bs//32)
  scales = torch.zeros(bs//32)

  #print(f"  quantize block {i}: {x}")
  sum_x2 = (x * x).sum()
  sigma2 = 2.*sum_x2/bs
  av_x = sigma2.sqrt()
  # For each subblock:
  for j in range(bs//32):
    subx = x[32*j:32*(j+1)]
    if quant_weights:
      weights = quant_weights[bs*i+32*j:bs*i+32*(j+1)] * (sigma2 + (subx**2)).sqrt()
    else:
      weights = av_x + subx.abs()
    sw[j] = weights.sum().item()
    scales[j], mins[j] = make_qkx3_quants(bitsmax, subx, weights)
  scale, subscales = make_qp_quants(bitsmax, scales, sw)
  zero_point, suboffsets = make_qp_quants(bitsmax, mins, sw)
  values = []
  for j in range(bs//32):
    d = scale * subscales[j]
    if not d.item():
      values.append(torch.zeros(32).to(torch.uint8))
    else:
      dm = zero_point * suboffsets[j]
      # Note: Add the zero point instead of subtracting.
      # While it is stored as one value per uint8 here for simplicity, in
      # production there should be two values packed per uint8.
      values.append(((x[32*j:32*(j+1)] + dm)/d).to(torch.uint8).clamp(0, bitsmax))
  return superblock(
      scale=scale.to(torch.float16),
      zero_point=zero_point.to(torch.float16),
      subscales=subscales,
      suboffsets=suboffsets,
      values=torch.cat(values))


def quantize_to_Q4_K(t: torch.Tensor, quant_weights = None) -> quantized:
  """Similar implementation to quantize_row_q4_K_impl() at
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c

  quant_weights is the optional imatrix to improve the quantization. It must be
  the same length as t.
  """
  # Batch size, aka QK_K
  bs = 256
  nb = (len(t)+bs-1)//bs
  return quantized([quantize_to_Q4_K_block(t[bs*i:bs*(i+1)], quant_weights) for i in range(nb)])


def dequantize_from_Qx_K(quant: quantized) -> torch.Tensor:
  """Simplified implementation of dequantize_row_q4_K() at
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c
  """
  out = []
  for sb in quant.blocks:
    subblock_size = len(sb.values) // len(sb.subscales)
    for j in range(len(sb.subscales)):
      q = sb.values[j*subblock_size:(j+1)*subblock_size]
      d = sb.scale * sb.subscales[j]
      m = sb.zero_point * sb.suboffsets[j]
      # In contrast with the affine transformation, the minimum value is
      # subtracted. As I (Marc-Antoine Rul) understand, this is because the
      # value is stored as a uint.
      out.append(q * d - m)
  if not out:
    return torch.zeros(0)
  return torch.cat(out)


def main():
  # We need to have negative values otherwise zero_offset and suboffsets are
  # always 0 and it's not really useful from a demonstration perspective.
  t = torch.rand(256)*10. - 5
  print(f"Original tensor of 256 random values between -5 and 5:")
  print(f"- [{', '.join('{:.4f}'.format(x) for x in t[:8].tolist())}, ...]")
  print(f"- storage: {len(t)*4} bytes ({len(t)}*4)\n")

  q = quantize_to_Q4_K(t)
  b0 = q.blocks[0]
  print(f"Quantized tensor:")
  print(f"- scale:      {b0.scale.item():g}")
  print(f"- zero_point: {b0.zero_point.item():g}")
  print(f"- subscales:  [{', '.join(f'{x}' for x in b0.subscales.tolist())}]")
  print(f"- suboffsets: [{', '.join(f'{x}' for x in b0.suboffsets.tolist())}]")
  print(f"- values:     [{', '.join(f'{x}' for x in b0.values[:16].tolist())}, ...]")
  # TODO:
  # - subscales and sufoffsets should be packed as 12 bytes (instead of 16)
  # - values should be packed two values per uint8
  v = int(2 + 2 + len(b0.subscales)*6/8 + len(b0.suboffsets)*6/8 + len(b0.values)*4/8)
  print(f"- storage:    {v} bytes (2+2+{int(len(b0.subscales)*6/8)}+{int(len(b0.suboffsets)*6/8)}+{int(len(b0.values)*4/8)})\n")

  d = dequantize_from_Qx_K(q)
  print(f"Dequantized tensor:")
  print(f"- [{', '.join('{:.4f}'.format(x) for x in d[:8].tolist())}, ...]")
  # https://en.wikipedia.org/wiki/Mean_squared_error
  mse = (t[:8] - d[:8]) ** 2
  print(f"- Mean squared error: {sum(mse)/len(mse):g}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

