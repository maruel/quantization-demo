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


def make_qp_quants(nmax: int, x: torch.Tensor, quant_weights: torch.Tensor):
  """Calculates quantized values of tensor x.

  It tries to reduce the Mean Squared Error (MSE) by fudging the values a bit.
  https://en.wikipedia.org/wiki/Mean_squared_error

  :param nmax: highest value as a unsigned bit that is representable.
  :param x: tensor to quantize, where each values returned are within [0, nmax].
  :param quant_weights: Relative importance of each weight. The higher the
  value, the more important the precision of the corresponding weight is.

  :return: scale and quantized values.

  Simplified version of make_qp_quants() at
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c
  """
  n = len(x)
  fmax = x.max()
  if not fmax.item():
    # All zeros.
    return torch.zeros(1), torch.zeros(len(x)).to(torch.uint8)
  iscale = nmax / fmax
  # Start with a naive quantization.
  L = (iscale * x).to(torch.uint8)
  # Search for the best Mean squared error. Offset values between [-0.4, 0.4] to
  # find the best MSE..
  diff = x - ((1./iscale) * L)
  # This is not strictly speaking the MSE, this is a weighted MSE based on the
  # importance of each weight.
  best_mse = (quant_weights*(diff**2)).sum()
  for iis in range(-4, 5):
    if not iis:
      continue
    iscale_is = (0.1*iis + nmax)/fmax
    l = torch.minimum((iscale_is*x).to(torch.uint8).min(), nmax)
    diff = x - ((1./iscale_is) * l)
    mse = (quant_weights*(diff**2)).sum()
    if mse.item() < best_mse.item():
      # Shifting values improved MSE, use this.
      best_mse = mse
      iscale = iscale_is

  # Recalculate the quantized values with the best bias.
  # TODO: Keep all the `l` to save unnecessary calculation.
  L = torch.minimum((iscale*x).to(torch.uint8), nmax)
  sumlx = (quant_weights*x*L).sum()
  suml2 = (quant_weights*(L**2)).sum()
  for itry in range(5):
    # I have no idea what this does.
    changed = False
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
            changed = True
    if not changed:
      break
  return sumlx/suml2, L



def make_qkx3_quants(nmax: torch.Tensor, x: torch.Tensor, weights: torch.Tensor):
  """Quantize tensor x into values [0, nmax] according to importance weights.

  It tries to reduce the "MAD" by fudging the values a bit. It is similar to MSE
  (Mean Squared Error) except that each error is weighted by `weights`.

  :param nmax: highest value as a unsigned bit that is representable.
  :param x: tensor to quantize, where each values are to be quantized between [0, nmax].
  :param weights: Relative importance of each weight. The higher the value, the
  more important the precision of the corresponding weight is.

  Simplified version from make_qkx3_quant() in
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c

  In the original code, use_mad is always false and weights is always specified.

  Note: we ignore the quantized values generated! We only want to figure out the
  scale and fmin.
  """
  # These are always the same values in the original code:
  rmin = -0.9
  rdelta = 0.05
  nstep = 36

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
  l = (iscale * (x-fmin)).to(torch.uint8)
  # Calculate the quantized values without offset.
  L = torch.maximum(torch.minimum(l, nmax), torch.zeros(1).to(torch.uint8))
  # This is the Mean Squared Error. Each value is reconstructed from the
  # quantized value and then subtracted from the original value.
  mse = ((scale*L + fmin) - x)**2
  # This is the "MAD" which is the MSE but with each error multiplied by the
  # relative importance for each weight.
  best_mad = (weights*mse).sum()
  for iis in range(nstep):
    # Try a offset starting at rmin and iterating nstep times, increasing the
    # value by rdelta. Compare with iscale above.
    iscale = (rmin + rdelta*iis + nmax)/(fmax - fmin)
    l = (iscale * (x-fmin)).to(torch.uint8)
    # Calculate the quantized values with the offset.
    Laux = torch.maximum(torch.minimum(l, nmax), torch.zeros(1).to(torch.uint8))
    sum_l = (weights*l).sum()
    sum_l2 = (weights * (l**2)).sum()
    sum_xl = (weights*l*x).sum()
    # Please send a PR updating this comment to explain what D is.
    D = sum_w*sum_l2 - sum_l*sum_l
    if D.item() > 0:
      # This is a probable candidate!
      this_scale = (sum_w*sum_xl - sum_x*sum_l)/D
      this_min   = (sum_l2*sum_x - sum_l*sum_xl)/D
      if this_min.item() > 0:
        # Realign at zero.
        this_min = torch.zeros(1)
        this_scale = sum_xl / sum_l2
      # Recalculate MSE.
      mse = ((this_scale*Laux + this_min) - x)**2
      # Recalculate MAD.
      mad = (weights*mse).sum()
      if mad.item() < best_mad.item():
        # Bingo.
        best_mad = mad
        scale = this_scale
        fmin = this_min
  return scale, -fmin


def quantize_to_Q4_K_block(x: torch.Tensor, quant_weights = None) -> superblock:
  """Quantize (compress) a block using K-Quantization algorithm.

  :param t: tensor to quantize.
  :param quant_weights: optional imatrix to improve the quantization. It must be
  the same length as x. Not tested.

  Similar implementation to quantize_row_q4_K_impl() at
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c
  """
  bits = 4
  # Batch size, aka QK_K, hardcoded to 256.
  bs = len(x)
  bitsmax = torch.tensor((1<<bits)-1).to(torch.uint8)
  # Assign a weight important either based on quant_weights if provided or based
  # on the absolute value.
  weight_importance = torch.zeros(bs//32)

  # Values that will be used to calculate the scale and zero point.
  scales = torch.zeros(bs//32)
  mins = torch.zeros(bs//32)

  # Properties of the number distribution.
  sum_x2 = (x**2).sum()
  # This was changed to 2 in https://github.com/ggerganov/llama.cpp/pull/5361
  # but unclear as to why.
  sigma2 = 2.*sum_x2/bs
  av_x = sigma2.sqrt()
  # For each subblock, calculate the weight importance, scale and min.
  for j in range(bs//32):
    subx = x[32*j:32*(j+1)]
    if quant_weights:
      weights = quant_weights[32*j:32*(j+1)] * (sigma2 + (subx**2)).sqrt()
    else:
      weights = av_x + subx.abs()
    # The more the values are large, the larger the importance.
    weight_importance[j] = weights.sum().item()
    scales[j], mins[j] = make_qkx3_quants(bitsmax, subx, weights)
  # Requantize to get the final scaling and offset values.
  scale, subscales = make_qp_quants(bitsmax, scales, weight_importance)
  zero_point, suboffsets = make_qp_quants(bitsmax, mins, weight_importance)
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


def quantize_to_Q4_K(t: torch.Tensor) -> quantized:
  """Quantize (compress) a tensor using K-Quantization algorithm."""
  # Batch size, aka QK_K
  bs = 256
  nb = (len(t)+bs-1)//bs
  # If quant_weights were to be provided, it should be the same length as t and
  # a view passed to the function, i.e. quant_weights[bs*i:bs*(i+1)]
  return quantized([quantize_to_Q4_K_block(t[bs*i:bs*(i+1)], None) for i in range(nb)])


def dequantize_from_Qx_K(quant: quantized) -> torch.Tensor:
  """Dequantize (decompress) the quantized tensor.

  Simplified implementation of dequantize_row_q4_K() at
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
      # subtracted. As I (Marc-Antoine Ruel) understand, this is because the
      # value is stored as a uint.
      out.append(q * d - m)
  if not out:
    return torch.zeros(0)
  # TODO: This could be made faster by reducing memory allocations.
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
  v = len(q.blocks)*int(2 + 2 + len(b0.subscales)*6/8 + len(b0.suboffsets)*6/8 + len(b0.values)*4/8)
  print(f"- storage:    {v} bytes ({len(q.blocks)}*(2+2+{int(len(b0.subscales)*6/8)}+{int(len(b0.suboffsets)*6/8)}+{int(len(b0.values)*4/8)}))\n")

  d = dequantize_from_Qx_K(q)
  print(f"Dequantized tensor:")
  print(f"- [{', '.join('{:.4f}'.format(x) for x in d[:8].tolist())}, ...]")
  # https://en.wikipedia.org/wiki/Mean_squared_error
  mse = (t[:8] - d[:8]) ** 2
  print(f"- Mean squared error: {sum(mse)/len(mse):g}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

