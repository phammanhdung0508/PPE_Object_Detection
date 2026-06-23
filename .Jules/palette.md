## 2025-05-22 - Focus Styling for Visually Hidden Inputs
**Learning:** Hiding interactive inputs with `display: none` or `visibility: hidden` removes them from the tab order, making them inaccessible to keyboard users. Using a 'visually-hidden' CSS pattern preserves focusability.
**Action:** Place the 'visually-hidden' input immediately BEFORE its associated label in the DOM. This allows styling the label's focused state using the adjacent sibling selector (e.g., `input:focus-visible + label`).
