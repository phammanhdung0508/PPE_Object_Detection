## 2025-05-14 - Vectorized YOLO Post-processing
**Learning:** Python loops are a major bottleneck for ML post-processing (NMS prep) when candidate counts are high. Vectorizing coordinate scaling, format conversion, and filtering with NumPy provides ~10x-100x speedup.
**Action:** Always prefer NumPy vectorization over row-by-row iteration for processing model outputs before passing to NMS or returning responses.
