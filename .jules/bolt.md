## 2025-05-15 - Vectorizing YOLO Post-processing
**Learning:** Python loops are extremely slow for processing thousands of detection candidates (e.g., 8400 in YOLO). NumPy vectorization provides massive speedups (60x to 120x) by offloading operations to C.
**Action:** Always check if a loop over model predictions or results can be replaced with NumPy vectorized operations (clipping, mask filtering, coordinate transforms).
