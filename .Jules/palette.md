## 2025-05-15 - Accessible File Uploads and Dynamic Toggle States
**Learning:** Using `display: none` on interactive elements like file inputs completely removes them from the tab order, breaking keyboard accessibility. A `visually-hidden` pattern preserves accessibility while maintaining the desired visual design.
**Action:** Always use the `visually-hidden` pattern for hidden interactive elements and ensure associated labels have `:focus-within` or sibling `:focus-visible` styles to provide visual feedback.

**Learning:** Interactive toggles (like Maximize/Restore) that don't update their ARIA state (`aria-pressed`) or text content can be confusing for screen reader users and sighted users alike.
**Action:** Ensure toggle buttons dynamically update their text, `aria-label`, and `aria-pressed` state to reflect the current UI condition.
