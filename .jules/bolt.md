## 2025-05-22 - Image Preprocessing Reordering
**Learning:** Moving the resize operation to the beginning of the preprocessing pipeline significantly reduces the workload for subsequent pixel-wise operations like color conversion and brightness calculation. For a 1080p -> 640px resize, the number of pixels to process drops by ~80%, leading to measurable latency reduction.
**Action:** Always check if spatial reduction (resize/crop) can be performed before intensive pixel-wise transformations or statistical calculations.
