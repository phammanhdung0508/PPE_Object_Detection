## 2025-05-15 - Vectorized YOLO post-processing
**Learning:** Python loops iterating over thousands of YOLO detection candidates in post-processing are a primary bottleneck. Replacing these loops with NumPy vectorized operations (filtering, coordinate scaling) yielded a ~3x speedup in this codebase (76ms to 25ms for 8400 candidates).
**Action:** Always check for row-by-row processing of large NumPy arrays or tensors and replace with vectorized operations where possible.
