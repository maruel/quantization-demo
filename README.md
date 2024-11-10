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
- [2.5763, -2.2069, -0.9693, 2.3468, -4.7072, 2.9986, -1.0286, 2.5437, ...]
- storage: 1024 bytes (256*4)

Quantizing as 8 bits:
  Quantized tensor:
  - scale:      0.0387268
  - zero_point: -4.88672
  - values:     [192, 69, 101, 186, 4, 203, 99, 191, ...]
  - storage:    260 bytes ((2+2+256*8/8))
  Dequantized tensor:
  - [2.5488, -2.2146, -0.9753, 2.3165, -4.7318, 2.9748, -1.0528, 2.5101, ...]
  - Mean squared error: 0.000525236

Quantizing as 4 bits:
  Quantized tensor:
  - scale:      0.658691
  - zero_point: -4.88672
  - values:     [11, 4, 5, 10, 0, 11, 5, 11, ...]
  - storage:    132 bytes ((2+2+256*4/8))
  Dequantized tensor:
  - [2.3589, -2.2520, -1.5933, 1.7002, -4.8867, 2.3589, -1.5933, 2.3589, ...]
  - Mean squared error: 0.153035
```

We see reasonable [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) with
great size reduction; from 1024 (float32) to 260 or 132 bytes.


## Affine transformation in blocks

This is an affine transformation where the tensor is sliced up in blocks. This
improves precision a little bit at the cost of a few more bytes.

Source: [affine_transformation.py](affine_transformation.py)

```
$ python3 affine_transformation_block.py
Original tensor of 256 random values between -5 and 5:
- [2.5763, -2.2069, -0.9693, 2.3468, -4.7072, 2.9986, -1.0286, 2.5437, ...]
- storage: 1024 bytes (256*4)

Quantizing as 8 bits:
  Quantized tensor:
  - scale:      0.0377808
  - zero_point: -4.88672
  - values:     [197, 70, 103, 191, 4, 208, 102, 196, ...]
  - storage:    288 bytes (8*(2+2+32*8/8))
  Dequantized tensor:
  - [2.5561, -2.2421, -0.9953, 2.3294, -4.7356, 2.9717, -1.0331, 2.5183, ...]
  - Mean squared error: 0.0004177

Quantizing as 4 bits:
  Quantized tensor:
  - scale:      0.64209
  - zero_point: -4.88672
  - values:     [11, 4, 6, 11, 0, 12, 6, 11, ...]
  - storage:    160 bytes (8*(2+2+32*4/8))
  Dequantized tensor:
  - [2.1763, -2.3184, -1.0342, 2.1763, -4.8867, 2.8184, -1.0342, 2.1763, ...]
  - Mean squared error: 0.122715
```

We see improved [MSE](https://en.wikipedia.org/wiki/Mean_squared_error)
compared to whole tensor encoding. 8 bits improved from 0.000525236 to
0.0004177 and 4 bits improved from 0.153035 to 0.122715. The cost is a
corresponding increase (260->288 bytes) and (132->160 bytes) in storage size.


## K-Quantization

This encodes the numbers with two sets of scales and offsets. One global and one per subblock. This leads to
`y = c(ax + b) + d` where `c` (scale) and `d` (zero point) have the same precision as for the affine
transformation, but the subblock `a` (scale) and `b` (zero point) are saved with only 6 bits of precision.

Source: [k_quantization.py](k_quantization.py)

```
$ python3 k_quantization.py
Original tensor of 256 random values between -5 and 5:
- [2.5763, -2.2069, -0.9693, 2.3468, -4.7072, 2.9986, -1.0286, 2.5437, ...]
- storage: 1024 bytes (256*4)

Quantized tensor:
- scale:      0.0456238
- zero_point: 0.338623
- subscales:  [14, 14, 12, 14, 14, 14, 14, 14]
- suboffsets: [14, 14, 14, 14, 14, 13, 13, 14]
- values:     [11, 3, 5, 11, 0, 12, 5, 11, 8, 6, 9, 7, 10, 4, 6, 6, ...]
- storage:    144 bytes (1*(2+2+6+6+128))

Dequantized tensor:
- [2.2812, -2.8262, -1.5488, 2.2812, -4.7422, 2.9219, -1.5488, 2.2812, ...]
- Mean squared error: 0.144666
```

Compared to affine transformation in block, the
[MSE](https://en.wikipedia.org/wiki/Mean_squared_error) increases a little
(0.122715->0.144666) for a reduction in storage (160->144 bytes).
