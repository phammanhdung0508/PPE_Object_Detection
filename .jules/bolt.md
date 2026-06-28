## 2025-05-14 - Vectorized YOLO Post-processing
**Learning:** Iterative Python loops for YOLO post-processing (e.g., 8400 candidates) are a massive bottleneck, taking ~30-70ms per frame. Vectorizing these operations using NumPy reduces this to < 10ms, enabling higher frame rates on CPU.
**Action:** Always check for row-by-row loops in computer vision pipelines and replace with NumPy vectorized operations for coordinate scaling, filtering, and conversion.
