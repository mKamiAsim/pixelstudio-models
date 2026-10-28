# PixelStudio models

Model files that the PixelStudio photo editor downloads, when no original source publishes the
file in the form the app needs. Each file is a conversion of published weights. It is not
retrained or changed in any other way.

The app checks every download against the SHA-256 in its catalog. A file that differs is deleted.

## Files

| Release | File | From | License |
|---|---|---|---|
| `birefnet-lite-512-v1` | `birefnet-lite-512.onnx` | BiRefNet lite by Peng Zheng and others, weights from [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) at commit `aa62cd8` | MIT, [licenses/birefnet-mit.txt](licenses/birefnet-mit.txt) |

### birefnet-lite-512.onnx

- What it does: finds the main subject of a photo, with fine edges such as hair.
- Why a conversion: the published ONNX file is fixed at 1024 x 1024 pixels, where it needs about 6 GB of memory. At 512 x 512 it needs about 1.2 GB, which a phone can give.
- How: exported with PyTorch by `tools/convert_birefnet.py` in the PixelStudio repository, at side 512, ONNX opset 19. The decoder's deformable convolutions become ONNX `DeformConv`.
- Input `input_image`: 1 x 3 x 512 x 512, RGB divided by 255, then normalized with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
- Output `output_image`: 1 x 1 x 512 x 512 logits. The mask is their sigmoid, 1 / (1 + e^-x).
- Size: 180,826,801 bytes. SHA-256: `82a8e05569ce210d174e7dabe6765de80be42203f2fbd071cd6926a62fee7d91`.

The BiRefNet paper: Zheng, Gao, Fan, Liu, Laaksonen, Ouyang and Sebe, "Bilateral Reference for High-Resolution Dichotomous Image Segmentation", CAAI Artificial Intelligence Research, 2024.
