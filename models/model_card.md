# Model Card: YOLO26s Construction Safety PPE

## Overview

- Model name: `yolo26s-construction-safety`
- Architecture: YOLO26s
- Format: ONNX FP32 baseline, ONNX FP16 GPU candidate
- Dataset: Roboflow construction-safety-gsnvb v1
- Number of classes: 5
- Intended use: construction-site PPE detection for helmets and safety vests

## Training

- Training platform: Kaggle Kernel
- Kernel: `dungsunf/construction-safety-yolo26-fine-tune`
- Training script: `models/kaggle/train_yolo26_ppe.py`
- Base weights: `yolo26s.pt`
- Input size: 640
- Epochs: 30
- Batch size: 16
- Frozen layers: 10
- Training time: 0.160 hours
- FP32 export: `/kaggle/working/yolo26_ppe.onnx`
- FP16 export: `/kaggle/working/yolo26_ppe_fp16.onnx`

## Metrics

Validation metrics from the completed Kaggle run:

```text
all        precision 0.816  recall 0.796  mAP50 0.836  mAP50-95 0.478
helmet    precision 0.889  recall 0.901  mAP50 0.929  mAP50-95 0.527
no_helmet precision 0.717  recall 0.591  mAP50 0.587  mAP50-95 0.275
no_vest   precision 0.737  recall 0.727  mAP50 0.798  mAP50-95 0.387
person    precision 0.916  recall 0.927  mAP50 0.968  mAP50-95 0.657
vest      precision 0.822  recall 0.835  mAP50 0.897  mAP50-95 0.546
```

`no_helmet` is the weakest class and should be tested carefully with real demo footage.

## Class Names

```text
0: helmet
1: no_helmet
2: no_vest
3: person
4: vest
```

## Limitations

- `no_helmet` has limited validation support and may need more examples.
- Accuracy may degrade under low light, blur, occlusion, dust, rain, or unusual camera angles.
- Small helmets and vests may be missed at low image resolution.
- FP16 is intended for GPU serving; CPU inference should use FP32.

## Deployment

- Runtime: ONNX Runtime
- Default FP32 model path: `models/yolo26_ppe.onnx`
- Optional FP16 model path: `models/yolo26_ppe_fp16.onnx`
- FastAPI health check: `/health`
- FastAPI metrics endpoint: `/metrics`
