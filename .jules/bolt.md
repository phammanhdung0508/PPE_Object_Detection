## 2025-05-15 - Vectorized YOLO Post-processing
**Learning:** Replacing Python loops with vectorized NumPy operations in YOLO post-processing eliminates a major O(N) bottleneck where N is the number of candidate detections (often 8400+).
**Action:** Always vectorize coordinate transformations and confidence filtering using NumPy before passing to NMS. Use `cv2.dnn.NMSBoxes` with NumPy arrays directly for optimal speed.
