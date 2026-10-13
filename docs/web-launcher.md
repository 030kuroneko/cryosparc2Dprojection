# Web deployment

The browser interface runs Class Orientation and Axis Search, with individual
CryoSPARC sign-in, private histories, and Local or Slurm execution.

## Manual foreground session

[Install the Web extra](../README.md#install), then start the interface:

```bash
uv run cryosparc2d --url https://your-cryosparc-server --port 40000
```

With Conda, omit `uv run`. Open [localhost:40000](http://127.0.0.1:40000).
Default state directory: `~/.local/state/cryosparc2d`. Local execution needs no Slurm setup.

Use `--config` for [JSON settings](web-config.example.json); flags override them.
Replace example URLs and paths. `--public-url` sets the user-facing address.

## Managed installation

On Linux with systemd, install a persistent service from a checkout:

```bash
cryosparc2d-service install --source /absolute/path/to/checkout
```

Requires uv in PATH and sudo. Setup creates the `cryosparc2d` account, installs
Web dependencies under `/opt/cryosparc2d`, and enables startup at boot.
Settings live in `/etc/cryosparc2d/web.json`; default data is under `/var/lib/cryosparc2d`.

Manage the service:

```bash
cryosparc2d-service configure  # Change settings
cryosparc2d-service status     # Check status
cryosparc2d-service start      # Start now
cryosparc2d-service stop       # Stop Web; workers may finish
cryosparc2d-service restart    # Restart Web
cryosparc2d-service enable     # Enable boot startup
cryosparc2d-service disable    # Disable boot startup
```

`configure` preserves values on Enter. Changing the CryoSPARC instance or work-directory
association requires a new empty state directory. Setup does not migrate history,
configure Slurm, open firewall ports, or provision HTTPS.

## Network access

For a trusted lab network or VPN, listen on all IPv4 interfaces:

```bash
cryosparc2d --url https://your-cryosparc-server --host 0.0.0.0 --port 40000
```

Open `http://SERVER-IP:40000`. HTTP sends credentials unencrypted; restrict access
to trusted clients. Custom DNS aliases require `--public-url`. Stay on the same
hostname while signed in.

For HTTPS, use an existing reverse proxy and keep the app on `127.0.0.1:40000`.
Set `--public-url https://projection.lab.example` and preserve the Host header.
Example inside an existing nginx TLS server:

```nginx
location / {
    proxy_pass http://127.0.0.1:40000;
    proxy_set_header Host $http_host;
    client_max_body_size 64k;
}
```

Do not expose the upstream port. Sign-in throttling applies to the proxy as a whole.

## Execution lanes and Slurm

1. Run on a submission host with `sbatch`, `squeue`, and `sacct` available to the service account.
2. Open **Administrator · Execution lanes** and unlock using the private `admin-token` file.
3. Add a Slurm lane with its Python executable and CPU, GPU, memory, and time limits.
4. Choose a private shared work directory owned by the service account (mode `700`),
   outside CryoSPARC project folders and available at the same path on compute nodes.
5. Preview and apply. Users can then select the lane.

Keep the SQLite state directory on local disk. Compute nodes need access to CryoSPARC,
input data, and the application's Python environment. GPU lanes need CuPy and **GPUs = 1**.
The launcher submits its own Slurm workers, not jobs to CryoSPARC lanes.

Each lane has independent concurrency (default 1, maximum 32). The overall queue
allows 32 active/queued jobs and eight per user. Run one service per state directory.
Disabling a lane prevents new submissions; accepted jobs retain their saved settings.

### External templates

Use the [example template](slurm-template.example.sh) or the built-in template.
Only use trusted scripts. Reload, preview, and apply after editing a template.

- `{{ run_cmd }}` is required and already shell-quoted.
- `{{ name | quote }}` safely quotes a single shell argument.
- Resource variables include `cpus`, `gpus`, `memory_mb`, and `time_minutes`.
- Execution variables include `python`, `job_dir`, `log_path`, and `lane_id`.
- Custom variables are single-line scalar values; expressions and loops are unsupported.

Lane settings control resource directives. Put `#SBATCH` lines before shell commands.
See [Slurm sbatch](https://slurm.schedmd.com/sbatch.html).

## Jobs and credentials

- Closing the browser or signing out does not stop submitted work.
- **Stop job** terminates execution and retains its record.
- **Delete job** stops active work and removes the Web record; local files and CryoSPARC results remain.
- Service restarts preserve job records, but require users to sign in again.
- Unknown submissions are not retried automatically. An administrator must reconcile
  worker, Slurm, and CryoSPARC state before creating another copy.
- Passwords are not saved. Tokens remain in memory and private per-job auth files
  while work is pending; cleanup is retried after completion. These files are not
  encrypted at rest and must not enter shared backups.
- Expired credentials require a fresh login and explicit resubmission. Monitor disk
  usage; there is no automatic retention policy.
