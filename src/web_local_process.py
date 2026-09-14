"""Persist Local worker identity so a restarted dispatcher can stop its process group."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'command='],
                            capture_output=True, text=True, timeout=5)
    if result.returncode not in (0, 1):
        raise RuntimeError('Process identity unavailable')
    return result.stdout.strip()


def record_process(directory, process):
    signature = identity(process.pid)
    if signature:
        path = Path(directory) / 'local-process.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'pid': process.pid, 'identity': signature}))
        temporary.replace(path)


def group_alive(pgid):
    result = subprocess.run(['ps', '-axo', 'pgid=,stat='], capture_output=True,
                            text=True, timeout=5, check=True)
    return any(parts[0] == str(pgid) and not parts[1].startswith('Z')
               for line in result.stdout.splitlines() if len(parts := line.split()) == 2)


def kill_local(directory, process=None):
    if process is not None:
        pid = process.pid
        # The unreaped child handle is authoritative even if its leader has exited.
        alive = process.poll() is None
        if not alive and not group_alive(pid):
            return
    else:
        saved = json.loads((Path(directory) / 'local-process.json').read_text())
        pid = saved['pid']
        current = identity(pid)
        if not current and not group_alive(pid):
            return
        if not current or current != saved['identity'] or os.getpgid(pid) != pid:
            raise RuntimeError('Cannot confirm process identity')
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process is not None:
        process.wait(timeout=10)
    deadline = time.monotonic() + 5
    while group_alive(pid):
        if time.monotonic() >= deadline:
            raise RuntimeError('Process group termination not yet confirmed')
        time.sleep(0.05)
