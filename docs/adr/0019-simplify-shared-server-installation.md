# Simplify shared-server installation

Accepted and implemented as two installation modes.

Prioritize first installation and connection setup for lab members unfamiliar
with Python. A single-command basic installation provides the scientific CLIs
and their dependencies without requiring Web dependencies or a persistent
service. A subsequent single command adds all Web GUI dependencies, interactive
setup, and the persistent service to the installation. GUI means the existing
browser interface, preserving ADR 0016's retirement of the desktop launcher.
Install the optional Web mode once on a shared Linux server for browser access,
preserving ADR 0015's private multi-user architecture. Default computation to
the installation host and retain Slurm as advanced configuration, so cluster
setup does not become a prerequisite for every installation.

Distribute the basic CLI as a Python package on PyPI and a Conda package in a
maintainer-owned Anaconda channel, following the clarified user requirement for
`uv tool install` and `conda install`. GitHub Actions builds and tests packages
before publication. The checkout installer remains an optional deployment path.
The managed Web runtime continues to use uv and a source checkout.
For Web mode, manage the service through systemd,
with a `cryosparc2d-service` command exposing `start`, `stop`, `restart`,
`status`, `enable`, and `disable`: stopping the current service does not disable
startup after a reboot. Adding Web mode should start the service and enable
boot startup. Basic CLI installation does not require sudo; adding Web mode may
use sudo to create a dedicated service account and register the service.
Routine service control should be narrowly authorized for the deployer.

Keep the shared Web runtime root-managed and separate from the deployer's
personal CLI environment. This costs an additional environment but avoids
depending on a private home directory and prevents personal CLI updates from
changing the running shared service's dependencies.

Use an interactive setup flow to collect required settings, including the
CryoSPARC URL, save them, and check connectivity. Support subsequent changes to
required settings through the same guided flow. Stopping or restarting the
service stops web access and new job dispatch while allowing already-started
computations to finish.

Expose subsequent setup through `cryosparc2d-service configure`, showing current
settings and retaining each value on Enter. Validate changes before applying
them. Reject changes to the CryoSPARC connection, data directory, or computation
environment while jobs are queued or running, and explain that those jobs must
finish before the change can be applied.

Use a host IP and port on the lab network or VPN as the default browser entry
point, retaining configuration for an existing HTTPS reverse proxy. Automated
domain and certificate provisioning is outside this initial simplification.
