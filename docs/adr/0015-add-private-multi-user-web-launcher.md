# Add a private multi-user web launcher

The accepted Abyss visual design now serves a multi-user Linux lab/VPN web
launcher. This supersedes ADR 0014's desktop-only, no-web-server and no-Slurm
constraints for this new entry point. The desktop launcher was initially retained;
[ADR 0016](0016-retire-desktop-launcher.md) subsequently retires it.
Use Flask and Waitress, individual CryoSPARC authentication, server-owned job
records and a single durable dispatch queue. This separates browser lifetimes
from computation while reusing existing CLI validation and scientific workflows.

Slurm execution uses a dedicated Linux service account and administrator-defined
resource profiles. CryoSPARC accounts and web job histories remain isolated per
user. Existing External Jobs cannot be queued through CryoSPARC's supported SDK,
so the web service submits the worker and the worker publishes results through
the existing External Job adapter. Local sbatch integration is implemented first;
actual cluster topology and resource settings require deployment confirmation.

Credentials are server-side, with private per-job token files available to the
worker on shared storage until completion. The service account and administrators
are trusted; users do not supply shell commands, executable paths, server URLs
or Slurm directives. Uncertain submissions are not automatically retried because
creating duplicate External Jobs is more harmful than requiring reconciliation.
