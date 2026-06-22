## 2026-06-22 - Vectorized YOLO Post-processing
**Learning:** Python loops iterating over thousands of YOLO detection candidates (typically 8400 for 640x640 input) in post-processing are a massive performance bottleneck. NumPy vectorized operations provide a ~120x speedup by pushing these computations into C.
**Action:** Always prefer NumPy vectorization over explicit loops for large array processing in computer vision pipelines.
