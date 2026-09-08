# Multi-user web launcher

The Abyss (deep-ocean) web UI supports Class Orientation and Axis Search,
individual CryoSPARC sign-in, private job histories, versioned settings files,
and a durable sequential queue. Closing a browser or signing out does not stop
a submitted job. The existing desktop launcher and CLI remain available.

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
`--host 0.0.0.0` requires an explicit `--public-url`, so the service knows which
hostname/IP and origin to accept.
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

For a trusted lab network/VPN, bind all IPv4 interfaces and specify the actual
address users will open. Replace the example IP with this web server's address:

```bash
cryosparc2d --url http://your-cryosparc-server:39000 \
  --host 0.0.0.0 \
  --public-url http://192.168.1.20:40000
```

Users open `http://192.168.1.20:40000`, **not** `http://0.0.0.0:40000`.
Use `--port` and the matching public URL port if changing 40000. The same settings
can be supplied through JSON `host` and `public_url`. The default without these
options remains loopback-only; HTTPS deployments keep their existing policy.

This mode sends passwords, session cookies and administrator keys over unencrypted
HTTP. It is authorized only for trusted lab/VPN use, not Internet exposure.
Restrict port 40000 with your existing firewall/network policy; the launcher does
not open firewall rules. CSRF checks, exact public-host/origin validation,
authentication and per-user job isolation remain enabled. Other aliases are not
automatically trusted in LAN mode; use the configured address consistently.

Request IDs are generated server-side so job submission does not require the
browser's secure-context-only
[`crypto.randomUUID()`](https://developer.mozilla.org/en-US/docs/Web/API/Crypto/randomUUID).
Clipboard copying may be unavailable on HTTP; use Save settings instead.

### Optional Slurm setup — entirely in the browser

1. Run the web service on a Linux host allowed to submit under the dedicated
   service account, with `sbatch`, `squeue` and `sacct` on its PATH.
2. Sign in, choose **Slurm · configure in browser**, and open **Administrator ·
   Slurm setup**. The startup terminal shows the location of a private
   `admin-token` file. The service administrator reads that file and pastes its
   contents into **Administrator key**, then clicks **Unlock settings**.
   This is separate from CryoSPARC authentication. Do not distribute the key;
   it grants permission to select executable paths for the service account.
3. Choose an existing dedicated shared working directory such as
   `/shared/cryosparc2d/jobs`. Do **not** place it in a CryoSPARC-managed project
   folder. It holds scripts, private job credentials and logs; scientific results
   are still published to CryoSPARC. The example path is not created automatically
   and is not guaranteed to be a shared mount on your cluster.
4. Confirm the directory is owned by the service account, has permissions `700`,
   and is readable/writable at the **same absolute path** on submission and
   compute hosts. Keep SQLite state on the web host's local disk.
5. **Compute-node Python** defaults to the active environment's interpreter.
   Its environment, this package and dependencies must also exist at that exact
   path on compute nodes. The service does not run `conda activate` inside jobs.
6. Set CPUs, memory in MiB and time in minutes. Partition, account and QoS are
   optional when cluster defaults are appropriate. Confirm the shared-path
   checkbox and click **Save Slurm settings**. Saving checks local paths and
   command availability, but cannot prove remote-node accessibility or allocation
   permissions and does not submit a test job.

Other users simply select **Slurm** and run their workflow. Users who already had
the page open before initial setup should reload it to see the saved profile.
Settings persist in local SQLite state; jobs snapshot their execution resources,
Python and directory when submitted, so later edits do not alter queued jobs.
Administrator unlock expires with the browser session and is cleared by logout.
Local jobs do not require valid Slurm settings. No additional Slurm launch command
or `--slurm` flag is used. Regular users cannot edit resource limits or paths.

The service account must own both directories with mode `0700`. `data_dir`
contains SQLite records and should reside on local disk; optional `work_dir`
contains the default private per-job directories. Without `work_dir`, local
jobs reside inside `data_dir`; browser-configured Slurm jobs use their separate
shared directory. Job directories must be
visible at the same absolute path on the submission and compute hosts; Slurm
does not copy job input files. The configured filesystem must support reliable
SQLite/POSIX locking. Run only one web service per data directory. Do not run
multiple WSGI worker processes, multiple replicas or `flask run`.

Compute nodes need network access to the configured CryoSPARC API and access to
the same scientific data paths as the CLI. The Slurm profile's Python environment
must include this package and its dependencies. No shell activation is performed;
use an absolute environment Python path. Nonstandard Slurm commands must be on
the service's PATH. Slurm accounting (`sacct`) must be configured.

```bash
cryosparc2d --config /etc/cryosparc-projection/web.json
```

This starts Waitress, with eight HTTP threads and one independent dispatcher.
Only one computation is active across local and Slurm profiles, including a
Slurm job waiting for allocation. Up to 32 nonterminal jobs can be accepted,
with at most eight per user. Job records survive restart; browser sessions do not.

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

For a loopback-only development session, set `public_url` to
`http://127.0.0.1:40000`, `host` to `127.0.0.1` and `allow_http` to `true`.
Loopback HTTP mode accepts the documented local aliases; direct LAN HTTP requires
the explicit all-interface bind and actual public URL shown above.

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
restarts. Use it only with this dedicated service account. This document does not
install services, modify Slurm, open a firewall or provision certificates.

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
completion/failure. The SDK auth file has a 24-hour local lifetime; upstream token
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
uv run --extra web pytest tests/test_web.py tests/test_web_execution.py -q
```

Tests exercise authentication, CSRF, user isolation, idempotency, concurrent
submission, parameter validation, persisted history, a serial local process queue,
worker failure cleanup, and Slurm command/status handling using simulated external
commands. A real cluster acceptance run is still required after configuring its
shared paths, API access, profile resources and TLS. It should verify both
workflows with real data and two separate users.
