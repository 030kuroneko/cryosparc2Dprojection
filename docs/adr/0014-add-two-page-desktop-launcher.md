# Add a two-page desktop launcher

## Status

Accepted for the GUI feature branch.

## Decision

Use a native Tk/ttk desktop launcher with two main pages: Class Orientation
and Axis Search. The connection URL, Project UID and Workspace UID are shared.
Each page exposes the existing CLI's parameters, keeping uncommon parameters
behind an expandable section. Parameter defaults and choices come from the
CLI parsers; GUI validation also instantiates existing domain configurations
before connecting or creating a job.

Execute the selected existing CLI in a separate Python process with the same
interpreter and saved CryoSPARC Tools token. Stream stdout/stderr through a
queue; never call Tk from the reader thread. Allow one active job and keep the
window open until completion. Do not offer cancellation without a supported
External Job cancellation contract. A failed run is explicit and retry is a
user action. Results remain in the CryoSPARC workspace.

Save both pages as opt-in, versioned JSON containing only known GUI fields.
Do not collect credentials or accept credential-bearing URLs. No web server,
new Python dependency, private CryoSPARC frontend modification, or scientific
algorithm change is introduced. This implements the separately researched
launcher anticipated in ADR 0001.

## Consequences

Tk and a desktop display (or X forwarding) must be available on the machine
running the existing workflows. CPU, memory, CryoSPARC connectivity and file
access requirements are unchanged. This is a local launcher, not a Slurm
submission service. For unattended work the existing CLI remains available.
