# Domain Docs

This repository uses a single-context domain-documentation layout.

## Before exploring

Read these when they exist:

- `CONTEXT.md` at the repository root.
- Relevant architectural decisions under `docs/adr/`.

If they do not exist, proceed silently. Domain-modeling work creates them when terminology or architectural decisions are resolved.

## Layout

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── src/
```

Use terminology defined in `CONTEXT.md` in issue titles, specifications, tests, and implementation. Explicitly flag proposals that conflict with an existing ADR.
