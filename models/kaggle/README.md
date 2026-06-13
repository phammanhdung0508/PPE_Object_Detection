# Kaggle YOLO26 PPE Training

This folder is the Kaggle Kernel workspace for fine-tuning the PPE detector.

The current training script is dataset-agnostic and can use the new 5-class Roboflow construction-safety dataset in `data/`:

```text
helmet
no_helmet
no_vest
person
vest
```

The API server is still monolithic and uses the exported ONNX model at:

```text
models/yolo26_ppe.onnx
```

## Files

```text
models/kaggle/
├── kernel-metadata.json
└── train_yolo26_ppe.py
```

## Push To Kaggle

From this folder:

```bash
kaggle kernels push -p .
```

Or from the project root:

```bash
kaggle kernels push -p models/kaggle
```

## Dataset

For the new construction-safety dataset, upload the local `data/` folder as a Kaggle dataset first, then attach that dataset to the kernel.

The script expects a YOLO dataset with `data.yaml`, for example:

```text
data.yaml
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

On Kaggle, set `DATASET_ROOT` if the attached dataset path differs from the default:

```text
DATASET_ROOT=/kaggle/input/<your-dataset-slug>
```

## Classes

The training script reads class names from `data.yaml` by default. For the new dataset it should discover 5 classes:

```text
helmet,no_helmet,no_vest,person,vest
```

This is what reduces the model head from COCO's default 80 classes to the PPE dataset classes during training.

It discovers class names in this order:

1. `PPE_CLASS_NAMES` environment variable
2. Dataset files like `classes.txt`, `obj.names`, `data.names`, or `labels.txt`
3. Dataset YAML files with a `names` field
4. Pascal VOC XML object names
5. Fallback names only when `NUM_CLASSES` is explicitly set

If class discovery is wrong, set names explicitly:

```bash
PPE_CLASS_NAMES="helmet,no_helmet,no_vest,person,vest"
```

## Output

The trained ONNX model is exported to:

```text
/kaggle/working/yolo26_ppe.onnx
```

When `EXPORT_FP16=true`, the script also attempts a GPU-oriented FP16 export:

```text
/kaggle/working/yolo26_ppe_fp16.onnx
```

Use FP32 as the accuracy baseline first. Use FP16 for GPU serving after validation.

Download that file and place it in the API project as:

```text
models/yolo26_ppe.onnx
```

## Tunable Environment Variables

```text
MODEL_ARCH=yolo26s.pt
NUM_CLASSES=5
PPE_CLASS_NAMES=helmet,no_helmet,no_vest,person,vest
EPOCHS=30
IMG_SIZE=640
BATCH_SIZE=16
CONF_THRESHOLD=0.25
DATASET_ROOT=/kaggle/input/construction-safety-ppe
PREPARED_DIR=/kaggle/temp/ppe_yolo_dataset
DEVICE=0
EXPORT_DYNAMIC=true
EXPORT_FP16=true
LR0=0.001
FREEZE=10
PATIENCE=15
RUN_NAME=construction_safety_yolo26_finetune
USE_EXISTING_DATA_YAML=true
```

Recommended first run for the 5-class dataset:

```text
FREEZE=10
EPOCHS=30
LR0=0.001
```

`FREEZE=10` freezes early YOLO layers and trains the later/head layers, which is faster and usually safer for a small class-count fine-tune. If validation metrics are weak or the model underfits, run a second pass with `FREEZE=0` to fine-tune the full model.

Recommended export flow for GPU serving:

```text
1. Export and validate yolo26_ppe.onnx as FP32 baseline.
2. Export yolo26_ppe_fp16.onnx for GPU serving.
3. Promote FP16 only if detection quality is acceptable on real demo images/videos.
```

This run starts from base `yolo26s.pt` weights by default. It does not use a previous checkpoint unless you explicitly set `MODEL_ARCH` to a checkpoint path.

Important: after exporting a new ONNX model, update the API metadata/class names before serving it. The class order in `models/model_metadata.json` and `CLASS_NAMES` must match the exported model.

Fine-tune kernel slug:

```text
dungsunf/construction-safety-yolo26-fine-tune
```

## GPU Note

The first full run reached training but failed on Kaggle P100 because Kaggle's installed PyTorch build supported CUDA architectures `sm_70+`, while P100 is `sm_60`.

Recommended fix:

```text
Use T4 or another newer Kaggle GPU runtime instead of P100.
```

The script uses `/kaggle/temp/ppe_yolo_dataset` for prepared data so failed runs do not publish thousands of converted dataset files as Kaggle output.

ONNX export uses `EXPORT_DYNAMIC=true` by default so the API can attempt true batch inference in `/predict-batch`.
