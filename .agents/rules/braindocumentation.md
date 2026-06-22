---
trigger: always_on
---

# Antigravity Developer Rules — Memory & Change Log Enforcement

## 1. Catching Up on Context
- At the start of any new coding session or conversation, you MUST inspect the change log file `docs/brains.md` to understand what was previously implemented, what architectural decisions were made, and what state the database/system is in.

## 2. Dynamic Documentation & Logging
- After completing any structural change (modifying scripts, adding new features, altering database formats, or re-indexing files), you MUST update the `docs/brains.md` file.
- Every entry in `docs/brains.md` must follow the established structure:
  - **Timestamp:** The exact local date and time of the change.
  - **Decision ID:** A sequential identifier (e.g., `DEC-004`).
  - **Context & Motivation:** Why the change was made and what problem it resolves.
  - **Before vs. After:** A clear comparison showing what the system was like before and what it is like now.
  - **Impacted Files:** Clickable markdown file links (using `file://` scheme) to all modified or created files.
  - **Validation:** How you verified that the change works (e.g., ran a script, got a zip, query results).
- The language of `docs/brains.md` must always be English.
