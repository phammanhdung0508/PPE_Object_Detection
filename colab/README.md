# Colab Demo Setup

Use this folder to run the PPE FastAPI demo on Google Colab.

## Drive Folder

Create this folder in Google Drive:

```text
MyDrive/PPE_Object_Detection_Demo/
```

Copy these project files/folders into it:

```text
app/
models/yolo26_ppe.onnx
models/model_metadata.json
requirements-gpu.txt
scripts/test_predict.py
```

Recommended final structure:

```text
MyDrive/PPE_Object_Detection_Demo/
  app/
  models/
    yolo26_ppe.onnx
    model_metadata.json
  requirements-gpu.txt
  scripts/
    test_predict.py
```

## Run Notebook

Open `ppe_fastapi_demo.ipynb` in Google Colab.

Before running cells:

1. Set runtime to GPU: `Runtime > Change runtime type > T4 GPU`.
2. Run cells from top to bottom.
3. When the Cloudflare tunnel URL appears, open:

```text
https://<generated-url>/docs
```

Then use `/predict` to upload a construction/PPE image.

## Notes

- The tunnel URL is temporary and changes each runtime.
- The current production model is `models/yolo26_ppe.onnx`.
- If GPU ONNX Runtime is unavailable in Colab, the API may still run on CPU but inference will be slower.
