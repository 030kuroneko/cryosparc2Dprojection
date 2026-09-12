# Multiple administrator-managed Slurm Lanes

Status: implemented; local automated and browser verification complete.

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
- Lane resource values entered in the web interface are authoritative when a template contains conflicting hard-coded resource directives. Templates should reference the corresponding variables.
- Local execution and Slurm Lanes may run concurrently. Administrators can configure the Local concurrent-job limit in the web interface, defaulting to one; Local and each Slurm Lane account for their own slots.
- An uncertain submission is not automatically resent and continues to occupy its lane's slot. Other lanes may proceed within their own limits.
- Lowering a concurrent-job limit does not terminate existing jobs. It delays further dispatch until the active count falls below the new limit.
- Existing queued jobs in a disabled lane continue under their original submission settings. Disabling prevents new submissions, rather than pausing previously accepted work.

## Setup and compatibility

The normal flow starts with an empty Slurm Lane list, populated through web administration and linked to administrator-authored external templates. No pre-existing Slurm configuration file or manual import step is required. Existing configuration-file profiles remain configuration-driven until saved through web administration, which then takes precedence for that ID. Preserve existing launcher profiles and job references if present; this compatibility handling does not discover lanes from Slurm or CryoSPARC.

## Acceptance scenarios

- An administrator creates, duplicates, edits, disables and re-enables multiple lanes through the web interface; ordinary users select available lanes without changing resources, templates or limits.
- Rendering a template fills built-in and administrator-defined variables, supports complete environment setup, and preserves the lane's authoritative resource allocation.
- External file edits take effect only after explicit reload and activation. Previously submitted jobs retain their original template and resource snapshot across launcher restarts.
- Local and different Slurm Lanes run concurrently, with administrator-configurable limits enforced independently. Pending Slurm jobs and uncertain submissions continue to consume their lane's slots.
- Lowered limits and disabled lanes preserve already accepted work; new submissions to a disabled lane are rejected.
- Existing GPU fallback reporting, per-user job isolation, administrator authentication and credential cleanup behavior remain intact.
- A fresh installation needs no legacy Slurm settings; an upgraded installation preserves existing lane references and job history.

## Existing behavior informing the design

The current launcher permits multiple configuration-file profiles but its web form manages only one Slurm profile. All profiles share a single active execution slot. Submitted jobs retain a resource snapshot; administrator edits do not change that snapshot. Administrator access currently requires CryoSPARC sign-in and a separate launcher administrator key.

Sources: `src/web_jobs.py`, `src/web_execution.py`, `src/web.py`, and ADR 0015. These are existing behaviors, not decisions to preserve them in the new design.

The JSON passed through `--config` is optional. Service installation may generate a service configuration containing a local execution profile (`src/service_setup.py`); that does not imply a Slurm profile exists. Slurm settings entered in the current web form are stored in `jobs.sqlite3`, not a separate Slurm configuration file.

## Reference behavior

CryoSPARC uses a parameterized `cluster_script.sh` plus cluster command configuration. Registering the configuration stores it in the database; job generation reads that registered content rather than live-reloading the original files. This is a reference model, not yet a decision to reproduce its entire API. See the [official cluster integration examples](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/cryosparc-cluster-integration-script-examples) and [installation guide](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/downloading-and-installing-cryosparc).

CCP-EM Pipeliner also supports an external submission template with placeholders and queue-related job options; Doppio exposes configurable extra queue variables. These sources establish the template/UI pattern, not a named-lane management contract: [Pipeliner queue submission](https://ccpem-pipeliner.readthedocs.io/en/latest/source/getting_started.html#submitting-jobs-to-a-queue), [Doppio custom queue variables](https://www.ccpem.ac.uk/docs/doppio/user_guide.html#custom-queue-submission-variables).

## Validation boundaries

The accepted workflows are verified through the authenticated Web API, the public dispatcher/store lifecycle and generated Slurm commands/scripts at the OS boundary. Browser checks cover lane publication, cloning, template preview/activation and Local capacity. Tests use isolated temporary storage and fake scheduler executables; they do not submit real cluster or CryoSPARC jobs.

On 2026-09-12, `uv run pytest -q` completed with 555 passed and one hardware-dependent CUDA test skipped. The existing image-camera test emitted a gimbal-lock warning. Browser verification covered Local capacity changes, Slurm lane creation and duplication, template preview/activation, lane selection and preview invalidation after edits. Real Slurm/CryoSPARC execution and CUDA hardware verification remain pending.
