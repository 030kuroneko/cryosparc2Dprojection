# Retire the Tk/ttk desktop launcher

## Status

Accepted.

## Decision

Retire the native desktop launcher and its `cryosparc2d-gui` command. Keep the
web launcher and both scientific workflow CLIs. This supersedes ADR 0014 and
ADR 0015's decision to retain the desktop launcher.

Move shared CLI defaults and argument validation into `workflow_config.py`,
and web field labels and grouping into `workflow_fields.py`. Web requests,
job dispatch and workers no longer import desktop modules.

Remove desktop-only settings persistence, subprocess runner, worker entry point,
Tk widgets, their tests and the unreferenced HTML design prototype. Preserve
shared validation tests and existing Web, CLI and scientific regression tests.

## Consequences

No desktop display or Tk installation is needed. Existing desktop settings files
are no longer consumed. Users launch the web UI or workflow CLI commands and
reinstall the package to refresh command entry points. Scientific workflow
behavior and web job execution are unchanged by this removal.
