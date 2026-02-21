# Agent Notes

- Do not add `from __future__ import annotations` in project code.
- Do not use import fallback `try/except` blocks for local package imports.
- Use direct imports that match the repository layout.
- Add a descriptive message string to every `assert` statement.
- Prefer `assert` with descriptive messages over explicit exception raising for internal invariants.
