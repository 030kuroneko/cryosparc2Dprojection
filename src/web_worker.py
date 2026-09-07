"""Single-use worker entry point. All credentials are scoped to one private job."""
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys


class JobLog:
    """Bounded line buffering prevents token fragments from leaking between writes."""
    def __init__(self, source, tokens):
        self.source, self.tokens = source, tokens
        self.pending = ''
        self.written = 0

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            self._emit(line + '\n')
        if len(self.pending) > 65536:
            # Do not split an untrusted arbitrarily long line across redaction boundaries.
            self.pending = ''
            self._emit('[Oversized log line omitted]\n')
        return len(text)

    def _emit(self, text):
        for token in self.tokens:
            text = text.replace(token, '[REDACTED]')
        if self.written < 2 * 1024 * 1024:
            self.source.write(text)
            self.written += len(text)
            self.source.flush()

    def flush(self):
        self.source.flush()

    def finish(self):
        self._emit(self.pending)
        self.pending = ''


def main(argv=None):
    directory = Path((argv or sys.argv[1:])[0]).resolve()
    os.umask(0o077)
    # Configure before importing the SDK, which caches its configuration path.
    for key in list(os.environ):
        if key.startswith('CRYOSPARC_'):
            del os.environ[key]
    os.environ.update(XDG_CONFIG_HOME=str(directory / 'config'), MPLBACKEND='Agg')
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = os.environ['SLURM_CPUS_PER_TASK']
    auth_path = directory / 'config' / 'cryosparc-tools' / 'auth.json'
    code, detail = 1, 'Worker setup failed; check the activity log.'
    with (directory / 'output.log').open('w', encoding='utf-8') as out:
        log = JobLog(out, [])
        with redirect_stdout(log), redirect_stderr(log):
            try:
                auth = json.loads(auth_path.read_text())
                log.tokens = [session['token']['access_token'] for users in auth.values()
                              for session in users.values()]
                payload = json.loads((directory / 'request.json').read_text())
                os.environ['CRYOSPARC_EMAIL'] = payload['email']
                from cryosparc_2d_projection.gui_model import WORKFLOWS
                print('Starting ' + payload['workflow'] + ' workflow.')
                code = WORKFLOWS[payload['workflow']].main(payload['argv']) or 0
                detail = '' if code == 0 else 'Workflow returned an error. Check the activity log.'
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 1
                detail = 'Workflow command failed. Check the activity log.'
            except Exception as error:
                # Upstream exception strings may contain credentials or HTTP headers.
                print('Workflow failed (' + type(error).__name__ + '). Check inputs, CryoSPARC access and the External Job event log.')
                detail = 'Workflow failed. Check the activity log and CryoSPARC External Job.'
            finally:
                auth_path.unlink(missing_ok=True)
                print('Workflow completed.' if code == 0 else 'Workflow failed.')
                log.finish()
    temporary = directory / 'result.tmp'
    temporary.write_text(json.dumps({'exit_code': code, 'detail': detail}), encoding='utf-8')
    temporary.replace(directory / 'result.json')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
