import json
from threading import Event, Thread

import pytest

from cryosparc_2d_projection.job_progress import JobProgress


def test_progress_estimates_only_measured_stage_and_resets_for_upload(tmp_path):
    now = [0.0]
    messages = []
    estimate_sent, upload_sent = Event(), Event()
    def emit(message):
        messages.append(message)
        if '2/4 classes completed' in message:
            estimate_sent.set()
        if message.startswith('Uploading results'):
            upload_sent.set()
    path = tmp_path / 'progress.json'
    with JobProgress(emit, path=path, clock=lambda: now[0], heartbeat_seconds=0) as progress:
        progress.start('Finding class orientations', total=4, unit='classes')
        assert json.loads(path.read_text())['remaining_seconds'] is None
        now[0] = 20
        progress.advance(1)
        now[0] = 40
        progress.advance(2)
        state = json.loads(path.read_text())
        assert state['completed'] == 2
        assert state['remaining_scope'] == 'stage'
        assert state['remaining_seconds'][0] <= 40 <= state['remaining_seconds'][1]
        assert estimate_sent.wait(2)
        assert 'Estimated stage time remaining:' in messages[-1]
        progress.start('Uploading results')
        assert json.loads(path.read_text())['remaining_seconds'] is None
        assert upload_sent.wait(2)
        assert 'Estimating' in messages[-1]
    assert json.loads(path.read_text())['state'] == 'completed'


def test_heartbeat_reports_elapsed_without_inventing_work_and_stops_on_failure(tmp_path):
    now = [0.0]
    heartbeat = Event()
    messages = []
    def emit(message):
        messages.append(message)
        if 'Last progress update: 2 min ago' in message:
            heartbeat.set()
    path = tmp_path / 'progress.json'
    with pytest.raises(RuntimeError):
        with JobProgress(emit, path=path, clock=lambda: now[0], heartbeat_seconds=.02) as progress:
            progress.start('Reading input data')
            now[0] = 120
            assert heartbeat.wait(2)
            state = json.loads(path.read_text())
            assert state['completed'] == 0
            assert state['remaining_seconds'] is None
            raise RuntimeError('upstream failure')
    assert json.loads(path.read_text())['state'] == 'failed'
    assert messages[-1].startswith('Failed during Reading input data')
    assert not any(message.startswith('Completed') for message in messages)


def test_new_search_events_refresh_last_progress_even_before_class_completes(tmp_path):
    now = [0.0]
    path = tmp_path / 'progress.json'
    with JobProgress(lambda message: None, path=path, clock=lambda: now[0], heartbeat_seconds=0) as progress:
        progress.start('Comparing symmetry axes', total=4)
        progress.advance(0, detail='Class 1')
        now[0] = 20
        progress.advance(0, detail='Class 1')
        assert json.loads(path.read_text())['last_progress_seconds'] == 0


@pytest.mark.parametrize('fail', [False, True])
def test_slow_heartbeat_delivery_does_not_block_work_or_completion(tmp_path, fail):
    path = tmp_path / 'progress.json'
    blocked, release, work_finished, delivery_finished = Event(), Event(), Event(), Event()
    messages = []
    now = [0.0]
    final_prefix = 'Failed during' if fail else 'Completed'

    def emit(message):
        if 'Last progress update: 2 min ago' in message:
            blocked.set()
            release.wait(5)
        messages.append(message)
        if message.startswith(final_prefix):
            delivery_finished.set()

    def run():
        try:
            with JobProgress(emit, path=path, clock=lambda: now[0], heartbeat_seconds=.02) as progress:
                progress.start('Reading input data')
                now[0] = 120
                assert blocked.wait(2)
                progress.start('Finding class orientations', total=200)
                for completed in range(1, 201):
                    progress.advance(completed)
                progress.start('Uploading results')
                if fail:
                    raise RuntimeError('Publication failed')
        except RuntimeError:
            if not fail:
                raise
        work_finished.set()

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert blocked.wait(2)
        assert work_finished.wait(2), 'Slow logging blocked the workflow'
        assert json.loads(path.read_text())['state'] == ('failed' if fail else 'completed')
    finally:
        release.set()
        worker.join(3)
    assert delivery_finished.wait(2)
    assert messages[-1].startswith(final_prefix)
    assert not any(message.startswith('Finding class orientations') for message in messages)
