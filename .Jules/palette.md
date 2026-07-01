## 2025-07-01 - Accessible Visual Feedback Patterns

**Learning:** In dark-themed dashboards with complex grid layouts, standard focus rings can be hard to see. Combining `outline` with a `box-shadow` glow creates a high-contrast focus indicator that remains visible across various background colors and element states. Additionally, separating decorative status indicators (like pulsing dots) from their text labels using `aria-hidden` prevents redundant screen reader announcements while maintaining visual appeal.

**Action:** Use `outline: 2px solid var(--text-primary); box-shadow: 0 0 0 2px var(--color-brand); outline-offset: 1px;` for focus-visible states. Always wrap decorative icons and status dots in `span aria-hidden="true"` when they are adjacent to descriptive text.
