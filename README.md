# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)


## Affine transformation

This encodes the numbers with `y = ax + b`.

By saving one `a` (scale) and `b` (zero point/bias) per block, we can reduce the precision of each `x`
(weight) significantly.

Source: [affine_transformation.py](affine_transformation.py)

```
$ python3 affine_transformation.py
Original tensor of 256 random values between -5 and 5:
- [4.0744, -1.8795, 1.4729, -2.1683, -4.5729, 3.8113, -0.5243, 2.2338, ...]
- storage: 1024 bytes (256*4)

Quantizing as 8 bits:
  Quantized tensor:
  - scale:      0.0388489
  - zero_point: -4.90625
  - values:     [231, 77, 164, 70, 8, 224, 112, 183, ...]
  - storage:    288 bytes (8*(2+2+32*8/8))
  Dequantized tensor:
  - [4.0678, -1.9149, 1.4650, -2.1868, -4.5955, 3.7959, -0.5552, 2.2031, ...]
  - Mean squared error: 0.000402794

Quantizing as 4 bits:
  Quantized tensor:
  - scale:      0.660645
  - zero_point: -4.90625
  - values:     [13, 4, 9, 4, 0, 13, 6, 10, ...]
  - storage:    160 bytes (8*(2+2+32*4/8))
  Dequantized tensor:
  - [3.6821, -2.2637, 1.0396, -2.2637, -4.9062, 3.6821, -0.9424, 1.7002, ...]
  - Mean squared error: 0.11608
```

We see reasonable [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) with
great size reduction; from 1024 to 288 or 160 bytes.


## K-Quantization

This encodes the numbers with two sets of scales and offsets. One global and one per subblock. This leads to
`y = c(ax + b) + d` where `c` (scale) and `d` (zero point) have the same precision as for the affine
transformation, but the subblock `a` (scale) and `b` (zero point) are saved with only 6 bits of precision.

Source: [k_quantization.py](k_quantization.py)

```
Original tensor of 256 random values between -5 and 5:
- [4.4640, 2.3434, 1.0826, -2.5315, 4.4279, 4.3040, -2.1526, -2.6482, ...]
- storage: 1024 bytes (256*4)

Quantized tensor:
- scale:      0.0450439
- zero_point: 0.326416
- subscales:  [13, 14, 14, 14, 14, 15, 14, 14]
- suboffsets: [14, 14, 14, 15, 13, 15, 14, 15]
- values:     [15, 11, 9, 3, 15, 15, 4, 3, 14, 2, 12, 3, 2, 8, 9, 0, ...]
- storage:    144 bytes (2+2+6+6+128)

Dequantized tensor:
- [4.2109, 1.8711, 0.6992, -2.8145, 4.2109, 4.2109, -2.2285, -2.8145, ...]
- Mean squared error: 0.0754133
```

We see reasonable [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) with
great size reduction; from 1024 to 144 bytes.
