# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)


## Affine transformation

This encodes the numbers with `y = ax + b`.

By saving one `a` (scale) and `b` (zero point/bias) per block, we can reduce the precision of each `x`
(weight) significantly.

```
$ python3 affine_transformation.py
Original tensor of 8 random values between 0 and 10:
- [4.9217, 8.5881, 4.2807, 3.0488, 8.6625, 0.1261, 9.7784, 6.1589]
- storage: 32 bytes (8*4)

Quantizing as 8 bits:
  Quantized tensor:
  - scale:      0.0378418
  - zero_point: 0.126099
  - values:     [126, 223, 109, 77, 225, 0, 255, 159]
  - storage:    12 bytes (8*8/8+2+2)
  Dequantized tensor:
  - [4.8942, 8.5648, 4.2509, 3.0399, 8.6405, 0.1261, 9.7758, 6.1429]
  - Mean squared error: 0.000377167

Quantizing as 4 bits:
  Quantized tensor:
  - scale:      0.643555
  - zero_point: 0.126099
  - values:     [7, 13, 6, 4, 13, 0, 15, 9]
  - storage:    8 bytes (8*4/8+2+2)
  Dequantized tensor:
  - [4.6310, 8.4923, 3.9874, 2.7003, 8.4923, 0.1261, 9.7794, 5.9181]
  - Mean squared error: 0.0485167
```


## K-Quantization

This encodes the numbers with two sets of scales and offsets. One global and one per subblock. This leads to
`y = c(ax + b) + d` where `c` (scale) and `d` (zero point) have the same precision as for the affine
transformation, but the subblock `a` (scale) and `b` (zero point) are saved with only 6 bits of precision.

```
Original tensor of 256 random values between -5 and 5:
- [-4.5823, 0.1796, -4.9258, 2.4787, -3.1214, 4.8178, -0.8871, -0.4964, ...]
- storage: 1024 bytes (256*4)

Quantized tensor:
- scale:      0.0436401
- zero_point: 0.328613
- subscales:  [15, 14, 15, 14, 15, 15, 14, 14]
- suboffsets: [15, 13, 15, 14, 15, 15, 13, 13]
- values:     [0, 7, 0, 11, 2, 14, 6, 6, 7, 13, 2, 5, 12, 0, 8, 12, ...]
- storage:    144 bytes (2+2+6+6+128)

Dequantized tensor:
- [-4.9297, -0.3477, -4.9297, 2.2734, -3.6211, 4.2344, -1.0000, -1.0000, ...]
- Mean squared error: 0.162163
```
