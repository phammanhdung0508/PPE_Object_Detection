## 2025-05-15 - Vectorized YOLO Post-processing
**Learning:** Python loops over thousands of detection candidates (e.g., 8400 for YOLOv8) are a major bottleneck. Vectorizing filtering and coordinate transformations with NumPy provides a ~100x speedup.
**Action:** Always check for row-by-row processing in model output handling and replace with vectorized NumPy operations where possible.
