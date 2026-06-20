## 2025-06-20 - Dashboard Accessibility Patterns
**Learning:** Labels acting as buttons for file uploads are common but often inaccessible to keyboard users unless explicitly given a role, tab index, and keyboard event listeners. Sliders and icon buttons also frequently lack descriptive ARIA labels.
**Action:** Always check labels for `htmlFor` that target hidden inputs and ensure they are keyboard-navigable. Consistently add `aria-label` to range inputs and decorative `aria-hidden="true"` to SVGs.
