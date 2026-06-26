## 2025-05-22 - Accessible Custom File Inputs
**Learning:** Using `display: none` on file inputs removes them from the tab order, making them inaccessible to keyboard users even if an associated `<label>` is visible.
**Action:** Use a `visually-hidden` CSS pattern (absolute positioning, 1px size, clip) on the input and use the `:focus-within` or adjacent sibling selector (`input:focus-visible + label`) to provide visual focus feedback on the custom-styled label.
