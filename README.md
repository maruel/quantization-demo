# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)

```
$ python3 affine_transformation.py
Original tensor of 8 random values between 0 and 10:
- tensor([8.9891, 0.1838, 2.2898, 0.0931, 1.9074, 0.9179, 7.1204, 9.9374])
- storage: 32 bytes (8*4)

Quantized tensor:
- tensor([230,   2,  56,   0,  46,  21, 182, 255], dtype=torch.uint8)
- scale:      0.038604993373155594
- zero_point: 0.09310603141784668
- storage: 16 bytes (8+4+4)

Dequantized tensor:
- tensor([8.9723, 0.1703, 2.2550, 0.0931, 1.8689, 0.9038, 7.1192, 9.9374])

Numerical error induced by the quantization:
- absolute: tensor([0.0168, 0.0135, 0.0348, 0.0000, 0.0385, 0.0141, 0.0012, 0.0000])
- relative: tensor([0.0019, 0.0735, 0.0152, 0.0000, 0.0202, 0.0154, 0.0002, 0.0000])
```
