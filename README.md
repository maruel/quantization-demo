# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)


## Affine transformation

This is encoding the numbers with `y = ax + b`. By saving one `a` (scale) and
`b` (zero point/bias) per block, we can reduce the precision of each `x`
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
