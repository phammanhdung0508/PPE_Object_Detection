## 2025-05-14 - [Accessible File Input Pattern]
**Learning:** Using 'display: none' on file inputs breaks keyboard accessibility. Using a 'visually-hidden' utility allows the input to remain focusable while being hidden visually.
**Action:** Always use 'visually-hidden' for hidden interactive elements and ensure their associated labels show focus states using adjacent sibling selectors (e.g., 'input:focus-visible + label').

## 2025-05-14 - [Dynamic ARIA Updates]
**Learning:** Screen readers need state updates for toggles like "Maximize/Restore". Just changing visual text is insufficient.
**Action:** Dynamically update 'aria-pressed' and 'aria-label' in JavaScript when toggling UI states to ensure assistive technology remains in sync with the visual state.
