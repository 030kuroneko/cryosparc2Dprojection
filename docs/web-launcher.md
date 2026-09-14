# Multi-user web launcher

The Abyss (deep-ocean) web UI supports Class Orientation and Axis Search,
individual CryoSPARC sign-in, private job histories, versioned settings files,
and a durable queue with configurable per-lane concurrency. Closing a browser or signing out does not stop
a submitted job. The workflow CLI commands remain available; the Tk/ttk desktop launcher has been retired.

The top-right Light/Dark control is available on the sign-in screen and every
workflow page. It changes the entire launcher palette, including Slurm settings
and activity logs, without changing scientific rendering options. The preference
is saved in this browser for this site only; no credentials are stored with it.
If browser storage is blocked, switching still works for the current page.

## Stop and delete jobs

Select a job in your history to use **Stop job** or **Delete job** above the
activity log. Stop requests immediate termination of Local worker processes or
Slurm execution and keeps the record. The UI shows **stopping** until termination
is confirmed, then **stopped**; a completed result keeps its completed outcome.
Unconfirmed stops retain their record and can be retried.

Delete asks for confirmation, stops active computation first, then removes the
record from this interface. Local result files and CryoSPARC jobs/results remain;
the result-viewing and class-selection entry points disappear from this interface.
Credential cleanup continues even after the record is removed.

The launcher attempts to mark the associated CryoSPARC External Job as failed
with a user-stop explanation, while preserving jobs already completed. A sync
failure remains visible, including after deletion. Legacy runs without a saved
process identity may require administrator reconciliation after a service restart.

Slurm termination combines an immediate KILL signal for running steps with
allocation cancellation, then checks scheduler evidence. See the
[scancel documentation](https://slurm.schedmd.com/scancel.html#SECTION_NOTES).

## Execution and Slurm

CryoSPARC Tools' `ExternalJobController.queue()` explicitly rejects queuing
External Jobs. They cannot be sent to an existing CryoSPARC lane. This launcher
submits its own worker with `sbatch`, and that worker creates and publishes the
existing External Job in CryoSPARC. The web UI reports the Slurm Job ID; scientific
outputs and previews remain in the chosen CryoSPARC workspace.

Slurm jobs run under a dedicated Linux service account approved by the cluster
administrator. This is distinct from each user's CryoSPARC identity. Slurm
account/partition/QoS and CPU, memory and time budgets are administrator-defined
profiles, not user-supplied commands. SSH and slurmrestd are not implemented.

References: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html),
[Slurm sacct](https://slurm.schedmd.com/sacct.html),
[CryoSPARC cluster integration](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/cryosparc-cluster-integration-script-examples).

## Install and configure

### Managed installation (recommended)

Install the CLI through uv or Conda as described in the
[package installation guide](../README.md#install). Managed service setup also
requires uv in PATH and a checkout containing `pyproject.toml`, `uv.lock`, and
`src`. On a Linux server running systemd:

```bash
cryosparc2d-service install --source /absolute/path/to/checkout
```

The command uses sudo for setup. It creates a dedicated `cryosparc2d` service
account, installs Python and the locked Web dependencies into a root-managed
runtime, asks for required settings, validates HTTP reachability and the local
computation environment as the service account, and enables/starts the service.
HTTP reachability does not authenticate a CryoSPARC user or submit a test job;
users sign in with their own accounts in the browser. The host must have systemd,
`useradd`, `runuser`, and `visudo`; package downloads require network access.

The defaults are direct lab/VPN HTTP at `http://SERVER-IP:40000`, private state
and work directories under `/var/lib/cryosparc2d`, and local sequential execution.
The setup prompts also support an existing HTTPS reverse proxy and custom
directories. Slurm remains an administrator setting in the Web UI.

```bash
cryosparc2d-service configure
cryosparc2d-service status
cryosparc2d-service start
cryosparc2d-service stop
cryosparc2d-service restart
cryosparc2d-service enable
cryosparc2d-service disable
```

`configure` displays current values; Enter keeps a value and `-` clears an
optional public URL. It validates a candidate before replacing the saved file.
An active service is briefly stopped and restarted; a stopped service stays
stopped. Failed activation restores the previous settings. Concurrent setup
sessions are refused. Queued/running jobs block changes to the CryoSPARC URL,
state/work directories and computation environment; pending credential cleanup
also blocks these changes. The queue is checked again after dispatch stops.

A state directory is permanently associated with its CryoSPARC instance and
work directory. To change either association after work finishes, choose a new
empty state directory; existing history and results stay in the old location.
This command does not migrate or delete history. Existing custom directories
must already be private (0700), owned by `cryosparc2d`, and accessible to that
account; setup does not recursively change ownership of existing data.

`stop` and `restart` leave already-started workers running; the restarted
dispatcher observes their completion records. `enable`/`disable` only change
boot startup and do not start/stop the current service. The sudo deployer receives
a validated sudoers rule for only these six systemctl operations on this service;
installation and configuration still require administrative sudo authorization.

Managed files:

| Location | Purpose |
| --- | --- |
| `/opt/cryosparc2d` | Root-managed Python, application and installation state |
| `/etc/cryosparc2d/web.json` | Current settings, root-owned and service-group readable |
| `/etc/cryosparc2d/web.previous.json` | Settings before the most recent change |
| `/etc/systemd/system/cryosparc2d.service` | Persistent service |
| `/etc/sudoers.d/cryosparc2d` | Narrow deployer service-control authorization |
| `/usr/local/bin/cryosparc2d-service` | System-wide service management command |

If setup fails, correct the reported problem and rerun `cryosparc2d-service install`;
an incomplete installation can resume. An already completed installation is left
unchanged. Inspect startup failures with `cryosparc2d-service status` and
`sudo journalctl -u cryosparc2d.service`. Installation does not provision a domain,
certificate, firewall rule or Slurm cluster.

The service intentionally uses `KillMode=process` with its dedicated account so
workers can finish after Web shutdown; see the
[systemd kill documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.kill.html).
Dependency installation uses uv's locked, explicit-extra synchronization; see
[uv syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

### Manual foreground session

Use Python 3.10–3.12. From the repository directory, activate your existing
Conda environment or create one, then install the web extra:

```bash
conda create -n cryosparc2d python=3.12 pip
conda activate cryosparc2d
python -m pip install -e '.[web]'
cryosparc2d --url https://your-cryosparc-server
```

Open `http://127.0.0.1:40000` and sign in with your CryoSPARC account. Defaults:
loopback binding, port **40000**, local sequential computation using the active
environment's Python, private state in `~/.local/state/cryosparc2d`. No Slurm
installation, Slurm flags or JSON configuration is required for local use.
`http://localhost:40000` is also accepted in loopback HTTP mode. Stay on the
same hostname while signed in: browser cookies are host-specific. Write requests
must match that hostname and the configured port; HTTPS deployments continue to
accept only their configured public origin.
For a server reached by SSH, you may forward your own local port 40000 to the
server's loopback port 40000; for shared HTTPS deployment see below.

Optional flags: `--port`, `--host`, `--public-url`, `--data-dir`, `--config`.
Run `cryosparc2d --help` for details. Flags override optional JSON settings.
`--public-url` is optional. Without it, direct HTTP accepts IP literals,
`localhost`, the operating system's hostname and an explicitly configured bind
hostname, on the configured port. Custom DNS aliases require `--public-url`.
To use an existing HTTPS reverse proxy:

```bash
cryosparc2d --url https://your-cryosparc-server \
  --public-url https://projection.lab.example
```

This does not provision DNS, TLS or a reverse proxy. For advanced existing
deployments, [web-config.example.json](web-config.example.json) remains supported;
replace example URLs and paths. Only the configured CryoSPARC server receives
credentials; the browser cannot change this destination.

### Direct lab-network HTTP (explicit opt-in)

For a trusted lab network/VPN, bind all IPv4 interfaces:

```bash
cryosparc2d --url http://your-cryosparc-server:39000 \
  --host 0.0.0.0
```

Users open `http://192.168.1.20:40000` (replace with the server's actual IP),
**not** `http://0.0.0.0:40000`. Use `--port` to change 40000.
Optionally add `--public-url http://projection.lab.example:40000` to restrict
access to a fixed address. The same settings can be supplied through JSON `host`
and optional `public_url`; an existing JSON public URL remains in effect even
when the flag is omitted. The default remains loopback-only.

This mode sends passwords, session cookies and administrator keys over unencrypted
HTTP. It is authorized only for trusted lab/VPN use, not Internet exposure.
Restrict port 40000 with your existing firewall/network policy; the launcher does
not open firewall rules. CSRF checks, authentication and per-user job isolation
remain enabled. Writes must match the request's exact scheme, hostname and port;
arbitrary DNS hosts and forwarded headers are not trusted. An explicit public URL
retains fixed-host/origin validation. Stay on one hostname while signed in because
cookies are host-specific.

Request IDs are generated server-side so job submission does not require the
browser's secure-context-only
[`crypto.randomUUID()`](https://developer.mozilla.org/en-US/docs/Web/API/Crypto/randomUUID).
Clipboard copying may be unavailable on HTTP; use Save settings instead.

### Execution lanes and external Slurm templates

1. Run the launcher on a Linux submission host under its dedicated service
   account, with `sbatch`, `squeue` and `sacct` on its PATH.
2. Sign in and open **Administrator · Execution lanes** below the workflow form.
   Enter the private key from the startup terminal's `admin-token` file and click
   **Unlock settings**. Administrator unlock is separate from CryoSPARC sign-in
   and ends with the browser session or logout.
3. Choose **New Slurm lane**. Give it a stable ID (letters, numbers, hyphens or
   underscores) and a display name such as **GPU** or **Large memory**. Existing
   IDs cannot be renamed; **Duplicate lane** creates a new ID with copied settings
   and the currently registered template, even if its original file is gone.
4. Set a dedicated shared working directory, outside CryoSPARC-managed project
   folders. The directory must exist, be owned by the service account with mode
   `700`, and be accessible at the same absolute path on the submission and
   compute nodes. Keep the launcher's SQLite data directory on local disk.
5. Set the absolute **Compute-node Python** executable and fixed CPU, GPU,
   memory and time allocations. Partition, account and QoS may be blank for
   cluster defaults. GPU count is 0 or 1; GPU search needs compatible CuPy in this
   Python environment. Users select lanes without overriding these resources.
6. Optionally enter an **External template path** on the launcher host and
   **Custom variables (JSON)**. Select **Reload template & preview** to read that
   file. Leaving the path blank uses the built-in template. A preview does not
   execute the script or submit a job.
7. Review the generated script and click **Apply preview**. New settings become
   available to all users. Changing any field invalidates the preview. An edit by
   another administrator requires reloading before applying stale settings.

No configuration file or import operation is required to create lanes. Existing
configuration-file profiles and saved single-Slurm settings remain available;
web-saved settings take precedence for the same ID. Users with an already-open
page can reload to see newly published lanes. Local remains usable without Slurm.

#### Template format

Start from [slurm-template.example.sh](slurm-template.example.sh), or the built-in
example shown in the administration panel. Templates contain complete shell
scripts, including module loading and environment setup before `{{ run_cmd }}`.
Administrators must select trusted templates; the launcher does not validate
file ownership or write permissions. Only authenticated, unlocked administrators
can preview and activate them. The launcher continues to own submission and
status tracking; it does not execute custom scheduler adapters.

Supported placeholders are `{{ name }}` and `{{ name | quote }}`. The `quote`
filter protects a single shell argument; use it for paths or custom values in
shell commands. `run_cmd` is already a shell-quoted complete worker command and
must appear in executable script content without another quoting filter. This
is variable substitution, not the full Jinja language: expressions, includes,
loops and template conditionals are not supported. Shell logic is allowed in the
script body. Unknown variables or a missing worker command prevent activation.

Built-in variables:

| Purpose | Variables |
| --- | --- |
| Resources | `cpus`, `gpus`, `memory_mb`, `time_minutes`, `partition`, `account`, `qos` |
| Execution | `python`, `run_cmd`, `job_dir`, `log_path`, `job_id`, `job_name`, `lane_id` |
| Workflow | `project_uid`, `workspace_uid`, `workflow` |

Custom variables are up to 32 named, single-line scalar values. Built-in names
cannot be overridden. For example, define
`{"module_name": "cuda/12", "gpu_model": "a100"}` and add:

```bash
#SBATCH --constraint={{ gpu_model }}
module load {{ module_name | quote }}
exec {{ run_cmd }}
```

The GPU model example requires matching cluster node features. Use the lane's
GPU count for allocation. Resource directives and aliases such as `--gres`,
`--gpus-per-node`, `--mem-per-cpu` and `-c` are replaced by the lane's authoritative
settings. The launcher also controls the one-node/one-task worker geometry,
logging paths, environment export and retry behavior. Other directives retain
their order. Put all `#SBATCH` lines before shell commands: Slurm stops reading
directives at the first command and does not expand shell variables in them.
See [SchedMD's sbatch documentation](https://slurm.schedmd.com/sbatch.html).

Template files are limited to 32 KiB of UTF-8 text. File edits alone do not take
effect: reload, preview and apply explicitly. Jobs snapshot the activated template,
variables, resources, Python and working directory when submitted, preserving
queued work across edits and restarts. The active template hash is shown in the
interface; the generated `submit.sh` is retained in each job directory.

#### Capacity and lifecycle

Each Slurm lane and **Local** have an independent **Concurrent jobs** setting
(default 1, configurable from 1 to 32). Local and different lanes can execute in
parallel. Pending Slurm allocations and uncertain submissions consume their own
lane's capacity; they do not block unrelated lanes. Unknown submissions are never
automatically resent. Lowering a limit leaves existing jobs running and pauses
further dispatch until capacity becomes available.

Clear **Enabled for new submissions**, preview and apply to disable a Slurm lane.
It disappears from users' choices, but previously accepted jobs, including queued
ones, finish with their saved settings. Re-enable the lane to accept new work;
there is no permanent-delete operation.

The launcher still accepts at most 32 nonterminal jobs overall and eight per
user. It uses one dispatcher and eight HTTP threads; run only one service per
state directory, not multiple WSGI workers or replicas. Job records survive
restart; browser sessions do not. Compute nodes need access to the configured
CryoSPARC API and scientific data paths. Slurm does not copy job input files, and
`sacct` accounting must be available. Preview validation checks local paths and
commands, not remote-node accessibility, GPU readiness or allocation permissions.

## HTTPS in the lab / VPN

Terminate HTTPS at the lab reverse proxy and keep the application bound to
`127.0.0.1:40000`. Set `public_url` to the exact user-facing HTTPS origin. Pass the
original Host header. The service checks the Origin and a session-bound CSRF
token on every write. Cookies are HttpOnly, SameSite=Strict and Secure on HTTPS.
Do not expose the upstream port to bypass TLS.

Example nginx location, inside an existing TLS server block:

```nginx
location / {
    proxy_pass http://127.0.0.1:40000;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $remote_addr;
    client_max_body_size 64k;
}
```

The application does not trust forwarded client-IP headers. Its conservative
sign-in throttle therefore applies to the proxy as a whole (10 attempts per
five minutes); the proxy may enforce additional per-client limits. The app is
intended for trusted lab/VPN access, not unrestricted public Internet service.

For a loopback-only development session, omit `public_url` and use the default
`host` of `127.0.0.1`. Direct LAN HTTP uses `--host 0.0.0.0` as shown above.
Existing explicit loopback HTTP configurations with `allow_http: true` remain
supported.

Example service unit after creating the dedicated account and private directories:

```ini
[Unit]
Description=2D Projection web launcher
After=network-online.target

[Service]
User=projection
Group=projection
UMask=0077
Environment=PATH=/opt/slurm/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/projection/.venv/bin/cryosparc2d --config /etc/cryosparc-projection/web.json
Restart=on-failure
KillMode=process
TimeoutStopSec=75
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

`KillMode=process` allows an existing local worker to finish when the web process
restarts. Use it only with this dedicated service account. This is a manual
example; the managed installer generates its own unit. Neither setup modifies
Slurm, opens a firewall or provisions certificates.

## Credentials and lifecycle

The login adapter accepts SDK model objects and plain JSON token/user responses,
including CryoSPARC's `_id` user field. An SDK unregistered-schema warning may
still appear, but a plain JSON response no longer fails attribute access. Missing
or invalid tokens and user identities are rejected rather than assigned a fallback
identity. Warnings are not globally suppressed.

Users sign in with their own CryoSPARC email/password through HTTPS. The SDK
exchanges credentials for a token, then obtains the authoritative user ID for
record ownership. Passwords are never saved. Tokens are kept in server memory
for the eight-hour browser session and in private `0600` per-job SDK auth files
while a job is queued or running. These files are not encrypted at rest: the
Linux service account and administrators are trusted. Do not include them in
general-purpose shared backups. Active job credentials are needed if the web
process restarts or a Slurm allocation starts later. They are removed on observed
completion/failure, with persistent retries if removal fails. The SDK auth file has a 24-hour local lifetime; upstream token
expiry may be earlier. Expired credentials fail the run and require a fresh login
and explicit resubmission.

Workers use an isolated SDK configuration directory. The service account's saved
login and environment credentials are not used. Slurm scripts and arguments
contain paths and resource settings, not passwords or tokens. Workflow logs
redact the known token and are capped at 2 MiB; the UI displays the latest 64 KiB.
Each user can list, inspect and read logs only for their own web jobs, even when
they share a CryoSPARC project. Settings import/export never includes credentials.

The worker writes an atomic completion marker. A successful `sbatch` is only
submission, not completion. Missing accounting or uncertain submission outcomes
remain `unknown` and block further dispatch, preventing duplicate work. The web
service never automatically retries a failed or ambiguous scientific job.

Web Job Credential Cleanup is tracked separately from the execution outcome.
Cleanup failure preserves that outcome and does not block the next queued job.
The dispatcher retries after 5, 10, 20, 40, 80 and 160 seconds, then every 300
seconds until removal succeeds (on its next polling cycle). Retry progress and
the next attempt time survive restarts. The selected job's Activity log section
shows a cleanup-pending warning, which disappears after successful cleanup;
no separate notification is sent. Cleanup exceptions do not expose file paths
or exception details in that warning. Existing terminal records are checked for
leftover credentials once when upgrading to cleanup tracking.

The job lifecycle module owns queue claims, execution reconciliation and cleanup
state. Local and Slurm execution observations are evaluated against the worker's
completion record before applying a terminal outcome. A worker still records
its execution result if its own credential removal fails.

If a job remains unknown after a crash, an administrator must inspect its
directory, Slurm job name (`projection-<web-job-id>`), `squeue`, `sacct` and
CryoSPARC External Job state before reconciling it. Do not delete its credentials
or start another copy while the original might still run. There is no UI cancel
button: Slurm cancellation and CryoSPARC External Job status must be handled
together by an administrator. Scheduler failure before the worker's cleanup can
leave a CryoSPARC External Job needing manual status reconciliation.

Terminal job records and logs are retained until an administrator archives them;
there is no automatic retention policy in this version. Monitor data-directory
disk space. A changed CryoSPARC instance should use a new data directory.

## Verification

```bash
uv run --extra web pytest tests/test_web.py tests/test_web_execution.py tests/test_web_job_lifecycle.py tests/test_web_job_details.py -q
```

Tests exercise authentication, CSRF, user isolation, idempotency, concurrent
submission, parameter validation, persisted history, a serial local process queue,
worker failure cleanup, persistent cleanup backoff and recovery, detail warnings,
and Slurm command/status handling using simulated external
commands. A real cluster acceptance run is still required after configuring its
shared paths, API access, profile resources and TLS. It should verify both
workflows with real data and two separate users.
