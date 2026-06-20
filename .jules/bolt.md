## 2025-05-15 - Vectorized YOLO Post-processing
**Learning:** Python loops iterating over thousands of raw YOLO detection rows (e.g., 8400 boxes) are extremely slow (~50ms). Vectorizing these operations with NumPy (filtering, scaling, clipping) reduces the time to ~1ms, a 50x speedup.
**Action:** Always check for Python loops in data-intensive areas like preprocessing or post-processing and replace them with NumPy vectorized operations where possible.
