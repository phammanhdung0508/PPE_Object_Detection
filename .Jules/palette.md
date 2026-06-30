## 2025-05-15 - Accessible File Inputs and Stateful Toggles
**Learning:** Using `display: none` on file inputs breaks keyboard accessibility. The `visually-hidden` pattern combined with adjacent sibling selectors allows for custom-styled, fully accessible file controls. Additionally, dynamic state changes (like Maximize/Restore) must be reflected both in text and ARIA attributes to maintain a predictable UX for all users.
**Action:** Always prefer `visually-hidden` for inputs and ensure toggle states update their ARIA labels dynamically.
