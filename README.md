# Quantization demo

Python code to demonstrate quantization techniques as described in my blog post
[maruel.ca/post/weights-part-1/](https://maruel.ca/post/weights-part-1/)

```
$ python3 affine_transformation.py
Original tensor of 8 random values between 0 and 10:
- tensor([1.9590, 7.1657, 6.0784, 6.3131, 2.4895, 7.7106, 6.6957, 9.2660])
- storage: 32 bytes (8*4)

Quantized tensor:
- tensor([  0, 181, 143, 151,  18, 200, 165, 255], dtype=torch.uint8)
- scale:      0.028654834255576134
- zero_point: 1.9590240716934204
- storage: 16 bytes (8+4+4)

Dequantized tensor:
- tensor([1.9590, 7.1455, 6.0567, 6.2859, 2.4748, 7.6900, 6.6871, 9.2660])

Numerical error induced by the quantization:
- tensor([0.0000, 0.0201, 0.0218, 0.0272, 0.0147, 0.0206, 0.0086, 0.0000])
```
