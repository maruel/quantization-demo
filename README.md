# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)

```
$ python3 affine_transformation.py
Original tensor of 8 random values between 0 and 10:
- tensor([9.9698, 1.2025, 6.2956, 1.3270, 6.1061, 2.6999, 5.1529, 3.4568])
- storage: 32 bytes (8*4)

Quantized tensor:
- tensor([255,   0, 148,   3, 142,  43, 114,  65], dtype=torch.uint8)
- scale:      0.034393310546875
- zero_point: 1.2021484375
- storage: 12 bytes (8+2+2)

Dequantized tensor:
- tensor([9.9724, 1.2021, 6.2924, 1.3053, 6.0860, 2.6811, 5.1230, 3.4377])

Numerical error induced by the quantization:
- absolute: tensor([-0.0026,  0.0004,  0.0032,  0.0217,  0.0201,  0.0188,  0.0300,  0.0191])
- relative: tensor([-0.0003,  0.0003,  0.0005,  0.0164,  0.0033,  0.0070,  0.0058,  0.0055])
```
