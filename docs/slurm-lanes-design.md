# Multiple administrator-managed Slurm Lanes

Status: design interview in progress; implementation has not been approved.

## Confirmed requirements

- Provide a web administration interface for multiple named Slurm Lanes.
- Target one Slurm cluster; lanes represent different resource configurations, such as CPU, GPU or high memory.
- Administrators define fixed resource allocations. Users select a lane without overriding its resource settings.
- Make the administrator-provided lanes available to all launcher users.
- Permit concurrent execution across lanes. Each lane defaults to one submitted, unfinished Slurm job; administrators may adjust this limit.
- Lane edits affect new submissions only. Disabling a lane prevents new submissions while existing jobs complete with their original settings. Support re-enabling rather than permanent deletion.
- Retain CryoSPARC sign-in plus the launcher administrator key for administration. Provide creation, duplication, editing and disabling in the web interface.
- Preserve CPU fallback for recognized GPU unavailability, reporting the reason and actual execution device.
- Combine the web administration interface with external Slurm Submission Templates supporting variable substitution, in the style of CryoSPARC or CCP-EM. Templates are authored by administrators.
- Use `{{ variable }}` placeholders. Provide built-in resource/job variables and administrator-defined variables, such as GPU model or module name; ordinary users continue to select a lane only.
- Templates may contain complete batch scripts, including `#SBATCH` directives and environment initialization. The application retains responsibility for `sbatch` submission and `squeue`/`sacct` state tracking.
- Administrators explicitly reload and apply external template changes through the web interface, with a preview before activation. Existing jobs preserve their original template version.

## Decisions still open

- Resource authority when a template hard-codes values that conflict with the lane's web settings.
- Configuration ownership and compatibility with existing execution profiles: the user has not identified any existing Slurm configuration file. An import must not be assumed necessary. Proposed normal flow: an empty lane list populated through web administration. Compatibility handling would apply only if an existing launcher Slurm profile is actually present; it would not discover lanes from Slurm or CryoSPARC.
- Dispatch edge cases: uncertain submissions and lowering concurrency limits. The accepted rule that existing jobs finish with their original settings also covers jobs already queued in a subsequently disabled lane.
- Local execution alongside concurrent Slurm Lanes.

## Existing behavior informing the design

The current launcher permits multiple configuration-file profiles but its web form manages only one Slurm profile. All profiles share a single active execution slot. Submitted jobs retain a resource snapshot; administrator edits do not change that snapshot. Administrator access currently requires CryoSPARC sign-in and a separate launcher administrator key.

Sources: `src/web_jobs.py`, `src/web_execution.py`, `src/web.py`, and ADR 0015. These are existing behaviors, not decisions to preserve them in the new design.

The JSON passed through `--config` is optional. Service installation may generate a service configuration containing a local execution profile (`src/service_setup.py`); that does not imply a Slurm profile exists. Slurm settings entered in the current web form are stored in `jobs.sqlite3`, not a separate Slurm configuration file.

## Reference behavior

CryoSPARC uses a parameterized `cluster_script.sh` plus cluster command configuration. Registering the configuration stores it in the database; job generation reads that registered content rather than live-reloading the original files. This is a reference model, not yet a decision to reproduce its entire API. See the [official cluster integration examples](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/cryosparc-cluster-integration-script-examples) and [installation guide](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc).

CCP-EM Pipeliner also supports an external submission template with placeholders and queue-related job options; Doppio exposes configurable extra queue variables. These sources establish the template/UI pattern, not a named-lane management contract: [Pipeliner queue submission](https://ccpem-pipeliner.readthedocs.io/en/latest/source/getting_started.html#submitting-jobs-to-a-queue), [Doppio custom queue variables](https://www.ccpem.ac.uk/docs/doppio/user_guide.html#custom-queue-submission-variables).
