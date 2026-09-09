"""Human-readable progress shared by both workflows and the web launcher."""

import json
import math
import os
from collections import deque
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from time import monotonic, time


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    if seconds < 60:
        return f'{seconds} sec'
    if seconds < 3600:
        return f'{math.ceil(seconds / 60)} min'
    return f'{seconds // 3600} hr {math.ceil(seconds % 3600 / 60)} min'


class JobProgress:
    """Report measured stage work; unknown future stages never inherit its ETA."""

    def __init__(self, emit, *, path=None, clock=monotonic, heartbeat_seconds=30):
        self.emit = emit
        self.clock = clock
        self.path = Path(path) if path else None
        self.web_path = os.environ.get('CRYOSPARC2D_PROGRESS_PATH')
        self.heartbeat_seconds = heartbeat_seconds
        self.lock = RLock()
        self.stop = Event()
        self.thread = None
        self.delivery_condition = Condition()
        self.pending_messages = deque(maxlen=64)
        self.delivery_closing = False
        self.delivery_thread = None
        self.started = self.stage_started = self.last_work = clock()
        self.stage = 'Starting workflow'
        self.state = 'running'
        self.completed = 0
        self.total = None
        self.unit = 'items'
        self.detail = ''
        self.samples = []
        self.sample_at = self.started
        self.last_emit = float('-inf')

    def __enter__(self):
        self.delivery_thread = Thread(target=self._deliver, daemon=True)
        self.delivery_thread.start()
        if self.heartbeat_seconds > 0:
            self.thread = Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        with self.lock:
            self.state = 'failed' if exc_type else 'completed'
            if exc_type:
                self.detail = f'{exc_type.__name__}. Check warnings and the job log.'
            else:
                self.stage = 'Completed'
                self.detail = 'All required results uploaded.'
            self._report()
        with self.delivery_condition:
            self.delivery_closing = True
            self.delivery_condition.notify()
        if self.delivery_thread:
            # Fast sinks drain normally. A stalled remote logger must not delay
            # publication completion or keep the worker alive indefinitely.
            self.delivery_thread.join(timeout=1)
            if self.delivery_thread.is_alive():
                with self.delivery_condition:
                    if self.pending_messages:
                        latest = self.pending_messages[-1]
                        self.pending_messages.clear()
                        self.pending_messages.append(latest)

    def start(self, stage, *, total=None, unit='items'):
        with self.lock:
            self.stage = stage
            self.total, self.unit = total, unit
            self.completed = 0
            self.detail = ''
            self.samples = []
            self.stage_started = self.last_work = self.sample_at = self.clock()
            self._report()

    def advance(self, completed, *, detail=''):
        with self.lock:
            now = self.clock()
            if completed > self.completed:
                self.samples.append((now - self.sample_at) / (completed - self.completed))
                self.sample_at = now
            changed = completed != self.completed or detail != self.detail
            self.completed, self.detail = completed, detail
            self.last_work = now
            # Observe fine-grained work in memory without writing two files per
            # search angle. Heartbeats persist it even within one long class.
            if changed or now - self.last_emit >= self.heartbeat_seconds:
                self._report()

    def _heartbeat(self):
        while not self.stop.wait(self.heartbeat_seconds):
            with self.lock:
                if self.state == 'running':
                    self._report()

    def _deliver(self):
        while True:
            with self.delivery_condition:
                self.delivery_condition.wait_for(
                    lambda: self.pending_messages or self.delivery_closing
                )
                if not self.pending_messages:
                    return
                text = self.pending_messages.popleft()
            # Neither the state lock nor the queue lock is held across callbacks.
            # A single sender preserves ordering and bounds concurrent requests.
            try:
                self.emit(text)
            except Exception:
                pass

    def _report(self, *, emit=True):
        now = self.clock()
        remaining = None
        if self.state == 'running' and self.total and self.completed == self.total:
            remaining = [0, 0]
        if self.state == 'running' and self.total and 0 < self.completed < self.total and len(self.samples) >= 2:
            recent = self.samples[-8:]
            outstanding = self.total - self.completed
            # A heuristic range, not a statistical confidence interval. If the
            # current unit exceeds our observations, withdraw the stale estimate.
            if now - self.sample_at <= max(recent) * 2:
                remaining = [max(0, min(recent) * .75 * outstanding - (now - self.sample_at)),
                             max(recent) * 1.5 * outstanding]
        snapshot = dict(stage=self.stage, state=self.state, completed=self.completed,
                        total=self.total, unit=self.unit, detail=self.detail,
                        elapsed_seconds=max(0, now - self.started),
                        stage_elapsed_seconds=max(0, now - self.stage_started),
                        last_progress_seconds=max(0, now - self.last_work),
                        updated_at=time(), remaining_seconds=remaining,
                        remaining_scope='stage' if remaining else None)
        for path in {str(p) for p in (self.path, self.web_path) if p}:
            try:
                destination = Path(path)
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(json.dumps(snapshot), encoding='utf-8')
                temporary.replace(destination)
            except OSError:
                pass  # Reporting must not fail scientific work.
        if not emit:
            return
        count = f' · {self.completed}/{self.total} {self.unit} completed' if self.total is not None else ''
        text = f'{self.stage}{count} · Elapsed: {duration(snapshot["elapsed_seconds"])}'
        if self.state == 'running':
            text += f' · Time in stage: {duration(snapshot["stage_elapsed_seconds"])}'
            text += (' · Estimated stage time remaining: ' + duration(remaining[0]) + '–' + duration(remaining[1])
                     if remaining else ' · Estimated time remaining: Estimating…')
            text += f' · Last progress update: {duration(snapshot["last_progress_seconds"])} ago'
        elif self.state == 'failed':
            text = 'Failed during ' + text
        if self.detail:
            text += ' · ' + self.detail
        with self.delivery_condition:
            self.pending_messages.append(text)
            self.delivery_condition.notify()
        self.last_emit = now
