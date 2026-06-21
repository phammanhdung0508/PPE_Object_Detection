## 2026-06-21 - Accessible File Inputs in Static HTML
**Learning:** Using `display: none` on `<input type="file">` completely removes it from the tab order, making it inaccessible to keyboard users, even if an associated `<label>` is styled as a button. In a pure HTML/CSS environment (like this dashboard's single-file demo), focus states must be explicitly passed to the label.
**Action:** Use a "visually-hidden" CSS pattern for inputs (clip, 1x1 size, absolute position) and the sibling selector `input:focus-visible + label` to provide visual focus indicators for the user-facing label.
