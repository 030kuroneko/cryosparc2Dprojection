"""Synchronize an explicitly stopped Web Job without deleting scientific results."""
import json
from pathlib import Path


SYNC_WARNING = 'CryoSPARC stop status could not be synchronized. Check the corresponding External Job in CryoSPARC.'


def sync_external_stop(directory, *, api=None):
    directory = Path(directory)
    marker = directory / 'external-job.json'
    if not marker.exists():
        return ''
    try:
        target = json.loads(marker.read_text())
        if target.get('pending'):
            return SYNC_WARNING
        if api is None:
            from cryosparc import __version__
            from cryosparc.api import APIClient
            from cryosparc.constants import API_SUFFIX
            auth = json.loads((directory / 'config/cryosparc-tools/auth.json').read_text())
            url, users = next(iter(auth.items()))
            session = next(iter(users.values()))
            api = APIClient(url + API_SUFFIX, auth=session['token']['access_token'], timeout=10,
                            headers={'User-Agent': f'cryosparc-tools/{__version__}'})
        project, job = target['project_uid'], target['job_uid']
        current = api.jobs.find_one(project, job)
        status = current['status'] if isinstance(current, dict) else current.status
        if status not in ('completed', 'failed', 'killed'):
            # The SDK ExternalJob.stop(error=...) uses mark_failed; ExternalJob.kill is unsupported.
            api.jobs.mark_failed(project, job, error='Stopped by user from the web launcher.')
        return ''
    except Exception:
        return SYNC_WARNING
