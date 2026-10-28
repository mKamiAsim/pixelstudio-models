# PixelStudio models

Model files that the PixelStudio photo editor downloads, when no original source publishes the
file in the form the app needs. Each file is a conversion of published weights. It is not
retrained or changed in any other way.

The app checks every download against the SHA-256 in its catalog. A file that differs is deleted.

## Files

| Release | File | From | License |
|---|---|---|---|
| `birefnet-lite-512-v1` | `birefnet-lite-512.onnx` | BiRefNet lite by Peng Zheng and others, weights from [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) at commit `aa62cd8` | MIT, [licenses/birefnet-mit.txt](licenses/birefnet-mit.txt) |
| `realesrgan-general-x4v3-v1` | `realesrgan-general-x4v3.onnx` | Real-ESRGAN by Xintao Wang and others, weights `realesr-general-x4v3.pth` from [the v0.2.5.0 release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) | BSD-3-Clause, [licenses/realesrgan-bsd-3-clause.txt](licenses/realesrgan-bsd-3-clause.txt) |

### birefnet-lite-512.onnx

- What it does: finds the main subject of a photo, with fine edges such as hair.
- Why a conversion: the published ONNX file is fixed at 1024 x 1024 pixels, where it needs about 6 GB of memory. At 512 x 512 it needs about 1.2 GB, which a phone can give.
- How: exported with PyTorch by `tools/convert_birefnet.py` in the PixelStudio repository, at side 512, ONNX opset 19. The decoder's deformable convolutions become ONNX `DeformConv`.
- Input `input_image`: 1 x 3 x 512 x 512, RGB divided by 255, then normalized with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
- Output `output_image`: 1 x 1 x 512 x 512 logits. The mask is their sigmoid, 1 / (1 + e^-x).
- Size: 180,826,801 bytes. SHA-256: `82a8e05569ce210d174e7dabe6765de80be42203f2fbd071cd6926a62fee7d91`.

The BiRefNet paper: Zheng, Gao, Fan, Liu, Laaksonen, Ouyang and Sebe, "Bilateral Reference for High-Resolution Dichotomous Image Segmentation", CAAI Artificial Intelligence Research, 2024.

### realesrgan-general-x4v3.onnx

- What it does: enlarges a photo 4 times, drawing clean edges, and smooths noise and JPEG blocks while it enlarges.
- Why a conversion: the authors publish PyTorch weights only.
- How: the network (SRVGGNetCompact: 34 3x3 convolutions with PReLU, PixelShuffle x4, plus the input enlarged by nearest pixel) written out in `tools/convert_sr.py` in the PixelStudio repository; the released weights loaded strictly; exported with PyTorch, ONNX opset 17, free height and width.
- Input `image`: 1 x 3 x H x W, RGB from 0 to 1. Output `output`: 1 x 3 x 4H x 4W, RGB from 0 to 1.
- Size: 4,866,422 bytes. SHA-256: `db8b6c03ff50c0f315bb710b93d39a3df45481f0ef26d70aa1b0b1dc3e3cc0f2`.

The Real-ESRGAN paper: Wang, Xie, Dong and Shan, "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCV Workshops, 2021.
