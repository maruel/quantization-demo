# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)

```
$ python3 affine_transformation.py
Original tensor of 8 random values between 0 and 10:
- [1.3517, 0.0395, 6.6755, 4.6396, 0.4514, 8.1571, 8.1088, 9.1571]
- storage: 32 bytes (8*4)

Quantizing as 8 bits:
  Quantized tensor:
  - [36, 0, 185, 128, 11, 227, 225, 255]
  - scale:      0.0357666
  - zero_point: 0.0394897
  - storage: 12 bytes (8*8/8+2+2)
  Dequantized tensor:
  - [1.3271, 0.0395, 6.6563, 4.6176, 0.4329, 8.1585, 8.0870, 9.1600]
  - Mean squared error: 0.000285141

Quantizing as 4 bits:
  Quantized tensor:
  - [2, 0, 10, 7, 0, 13, 13, 14]
  - scale:      0.60791
  - zero_point: 0.0394897
  - storage: 8 bytes (8*4/8+2+2)
  Dequantized tensor:
  - [1.2553, 0.0395, 6.1186, 4.2949, 0.0395, 7.9423, 7.9423, 8.5502]
  - Mean squared error: 0.131248
```
