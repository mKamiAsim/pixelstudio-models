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
| `mediapipe-face-landmarker-v1` | `mediapipe-face-detector.onnx`, `mediapipe-face-landmarks.onnx` | MediaPipe Face Landmarker by Google, [face_landmarker.task version 1](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | Apache-2.0, [licenses/apache-2.0.txt](licenses/apache-2.0.txt) |
| `ddcolor-tiny-512-v1` | `ddcolor-tiny-512.onnx` | DDColor tiny by Xiaoyang Kang and others, weights from [piddnad/ddcolor_paper_tiny](https://huggingface.co/piddnad/ddcolor_paper_tiny) at revision `cf9fd99` | Apache-2.0, [licenses/apache-2.0.txt](licenses/apache-2.0.txt) |

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

### mediapipe-face-detector.onnx and mediapipe-face-landmarks.onnx

- What they do: the detector finds faces in a photo; the landmarks model then gives 478 points on a face (eyes, brows, lips, the outline), which retouching uses to tell skin from eyes and lips.
- Why a conversion: Google publishes them as TensorFlow Lite inside `face_landmarker.task`; the app runs ONNX. The model cards of both (BlazeFace short range and Face Mesh V2) say Apache License 2.0.
- How: `tools/convert_face.py` in the PixelStudio repository unpacks `face_landmarker.task` version 1 (SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`) and converts both models with tf2onnx 1.16.1 (TensorFlow 2.15.1), ONNX opset 17. The half-precision weights come out as float. The expressions model in the same file is not included.
- Checked: on 4 portraits the ONNX outputs differ from TensorFlow Lite's by at most 0.0003 (detector) and 0.0005 pixels (landmarks).
- `mediapipe-face-detector.onnx`: input `input` 1 x 128 x 128 x 3, RGB scaled to -1..1; outputs `regressors` 1 x 896 x 16 (box and 6 key points per anchor) and `classificators` 1 x 896 x 1 (a score per anchor, before a sigmoid). Size: 418,502 bytes. SHA-256: `a4bdbe66dbedf468e90205aa9d3bd0f310b1d634733be93495e4fdc8206519ed`.
- `mediapipe-face-landmarks.onnx`: input `input_12` 1 x 256 x 256 x 3, RGB from 0 to 1, a crop around one face; outputs `Identity` 1 x 1 x 1 x 1434 (478 points x, y, z in crop pixels), `Identity_1` and `Identity_2`, two single values that PixelStudio does not use. Size: 4,921,043 bytes. SHA-256: `0d5c53356ece4c02ec7dc7d4e6722c79dd9c19f69b8b2b0c9ddb6570f8b7579a`.

The model cards: [BlazeFace short range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf) and [Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf).

### ddcolor-tiny-512.onnx

- What it does: gives a black and white photo likely colours. It predicts the a and b channels of Lab; the app joins them to the photo's own lightness, so no detail is lost.
- Why a conversion: the authors publish PyTorch weights only.
- How: `tools/convert_ddcolor.py` in the PixelStudio repository loads the authors' network (github.com/piddnad/DDColor, commit `2adb63f`) with the weights of `piddnad/ddcolor_paper_tiny` (`pytorch_model.bin`, SHA-256 `8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce`) strictly, exports it with PyTorch, ONNX opset 17, at 512 x 512, then stores the weights in half precision with float input and output (onnxconverter-common). The script gives the same file byte for byte each run.
- Checked: the float export matches PyTorch within 0.002 on a and b; half precision moves them by under 1 unit.
- Input `image`: 1 x 3 x 512 x 512, the photo's lightness alone as RGB from 0 to 1 (resize, take Lab L, set a and b to 0, back to RGB). Output `ab`: 1 x 2 x 512 x 512, Lab a and b.
- Size: 135,444,379 bytes. SHA-256: `fe13c421dad616223591ed388504a6f3e3628028c55b5961832cf28b3d51acf4`.

The DDColor paper (its README's citation): Kang, Yang, Ouyang, Ren, Li and Xie, "DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders", ICCV 2023, pages 328 to 338.
