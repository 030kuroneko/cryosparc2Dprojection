"""Durable lifecycle for publishing Class Selection Outputs.

The HTTP layer owns manifest validation and saved selections.  This module
owns the export snapshot, durable state transitions, retry rules, recovery and
the bounded background workers that call the CryoSPARC publisher.
"""

import json
import re
import threading
import uuid


ACTIVE_STATES = ('preparing', 'creating', 'publishing')
CAPACITY = 4
UNKNOWN_CREATION_DETAIL = (
    'CryoSPARC creation outcome is unknown. Administrator reconciliation is '
    'required before another export; do not create a replacement job.'
)
RESTARTED_CREATION_DETAIL = (
    'Service restarted; retry reconciles the recorded job. Without a job ID, '
    'administrator reconciliation is required.'
)
RESTARTED_PREPARING_DETAIL = 'Service restarted before remote creation; retry this export.'
RETRY_DETAIL = 'Export could not finish. Sign in again if needed, then retry this export.'


class ExportLifecycleError(Exception):
    """A lifecycle rejection that the HTTP layer can map to a status code."""

    status_code = 409


class ExportRequestConflict(ExportLifecycleError):
    """The request ID already identifies a different selection."""


class ExportCapacityReached(ExportLifecycleError):
    """The durable export capacity is full."""

    status_code = 429


class ExportUnknownCreation(ExportLifecycleError):
    """A remote job may have been created but its UID was not recorded."""


class ExportNotFound(ExportLifecycleError):
    """The source or export is outside the caller's ownership scope."""

    status_code = 404


class ClassSelectionExportLifecycle:
    """Public lifecycle interface for owner-scoped Class Selection exports.

    ``identity`` is the authenticated identity mapping used by the project
    factory.  It must contain ``owner``; all public operations check that the
    source Web Job belongs to that owner before reading or changing exports.
    The HTTP boundary validates the manifest and selected class numbers before
    calling ``start``; this module persists those values as one immutable
    snapshot and applies lifecycle rules.
    """

    def __init__(self, store, *, project_factory, launch=None):
        if not callable(project_factory):
            raise TypeError('project_factory must be callable')
        self.store = store
        self.project_factory = project_factory
        self._launch = launch if launch is not None else self._launch_thread
        self._initialize()

    @staticmethod
    def _launch_thread(target, *args):
        threading.Thread(target=target, args=args, daemon=True).start()

    def _initialize(self):
        with self.store.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS class_exports (
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL, request_id TEXT NOT NULL,
                selected TEXT NOT NULL, manifest TEXT NOT NULL, state TEXT NOT NULL,
                job_uid TEXT, detail TEXT NOT NULL DEFAULT '', UNIQUE(job_id,request_id))''')
            db.execute(
                "UPDATE class_exports SET state='unknown', detail=? "
                "WHERE state IN ('creating','publishing')",
                (RESTARTED_CREATION_DETAIL,))
            db.execute(
                "UPDATE class_exports SET state='failed', detail=? "
                "WHERE state='preparing'",
                (RESTARTED_PREPARING_DETAIL,))

    @staticmethod
    def _owner(identity):
        owner = identity.get('owner') if isinstance(identity, dict) else None
        if not isinstance(owner, str) or not owner:
            raise ExportNotFound('Not Found')
        return owner

    def _require_source(self, identity, job_id):
        owner = self._owner(identity)
        if self.store.get(owner, job_id) is None:
            raise ExportNotFound('Not Found')
        return owner

    @staticmethod
    def _public(row):
        return dict(
            id=row['id'],
            state=row['state'],
            job_uid=row['job_uid'],
            detail=row['detail'],
            selected_class_numbers=json.loads(row['selected']),
        )

    def list_exports(self, identity, job_id):
        """Return exports for an owned source Web Job in newest-first order."""
        self._require_source(identity, job_id)
        with self.store.connect() as db:
            rows = db.execute(
                'SELECT * FROM class_exports WHERE job_id=? ORDER BY rowid DESC',
                (job_id,)).fetchall()
        return [self._public(row) for row in rows]

    def _active_count(self, db):
        return db.execute(
            "SELECT COUNT(*) FROM class_exports WHERE state IN (?,?,?)",
            ACTIVE_STATES).fetchone()[0]

    def _submit(self, export_id, identity):
        # Copy the mapping because Flask's request context and session object
        # are unavailable in the worker thread.
        self._launch(self._execute, export_id, dict(identity))

    def start(self, identity, job_id, request_id, selected_class_numbers, manifest):
        """Persist an immutable export snapshot and schedule it once.

        The returned value is the public export record at acceptance time.
        Duplicate request IDs return their original record and snapshot.
        """
        self._require_source(identity, job_id)
        try:
            request_id = str(uuid.UUID(request_id))
        except (ValueError, TypeError, AttributeError):
            raise ValueError('Invalid request ID') from None
        selected = sorted(selected_class_numbers)
        if not selected:
            raise ValueError('Select at least one class')
        selected_json = json.dumps(selected)
        manifest_json = json.dumps(manifest)
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute(
                'SELECT * FROM class_exports WHERE job_id=? AND request_id=?',
                (job_id, request_id)).fetchone()
            if prior:
                if json.loads(prior['selected']) != selected:
                    raise ExportRequestConflict('Request ID already used with another selection')
                return self._public(prior)
            if db.execute(
                    "SELECT 1 FROM class_exports "
                    "WHERE job_id=? AND state='unknown' AND job_uid IS NULL",
                    (job_id,)).fetchone():
                raise ExportUnknownCreation(
                    'Resolve the unknown export with an administrator before creating another job.')
            if self._active_count(db) >= CAPACITY:
                raise ExportCapacityReached('Export capacity reached; try again later')
            export_id = uuid.uuid4().hex
            db.execute(
                'INSERT INTO class_exports '
                '(id,job_id,request_id,selected,manifest,state) VALUES (?,?,?,?,?,?)',
                (export_id, job_id, request_id, selected_json, manifest_json, 'preparing'))
            row = db.execute('SELECT * FROM class_exports WHERE id=?', (export_id,)).fetchone()
        self._submit(export_id, identity)
        return self._public(row)

    def retry(self, identity, job_id, export_id):
        """Claim a failed export for retry, preserving its snapshot and UID."""
        self._require_source(identity, job_id)
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(
                'SELECT * FROM class_exports WHERE id=? AND job_id=?',
                (export_id, job_id)).fetchone()
            if row is None:
                raise ExportNotFound('Export not found')
            if row['state'] in (*ACTIVE_STATES, 'completed'):
                return self._public(row)
            if row['state'] == 'unknown' and not row['job_uid']:
                raise ExportUnknownCreation(
                    'Administrator must reconcile the unknown CryoSPARC job before retry.')
            if self._active_count(db) >= CAPACITY:
                raise ExportCapacityReached('Export capacity reached; try again later.')
            db.execute(
                "UPDATE class_exports SET state='preparing',detail='' WHERE id=?",
                (export_id,))
            row = db.execute('SELECT * FROM class_exports WHERE id=?', (export_id,)).fetchone()
        self._submit(export_id, identity)
        return self._public(row)

    def _update_state(self, export_id, state, *, job_uid=None, detail=None):
        fields, values = ['state=?'], [state]
        if job_uid is not None:
            fields.append('job_uid=?')
            values.append(job_uid)
        if detail is not None:
            fields.append('detail=?')
            values.append(detail)
        values.append(export_id)
        with self.store.connect() as db:
            db.execute(f"UPDATE class_exports SET {','.join(fields)} WHERE id=?", values)

    def _execute(self, export_id, identity):
        try:
            from cryosparc_2d_projection.class_selection_export import export_class_selection

            with self.store.connect() as db:
                row = db.execute('SELECT * FROM class_exports WHERE id=?', (export_id,)).fetchone()
            if row is None:
                return
            manifest = json.loads(row['manifest'])
            project = self.project_factory(identity, manifest['project_uid'])

            def created(uid):
                if not isinstance(uid, str) or not re.fullmatch(r'J\d+', uid):
                    raise ValueError('Invalid remote job identity')
                self._update_state(export_id, 'publishing', job_uid=uid)

            def creating():
                self._update_state(export_id, 'creating')

            result = export_class_selection(
                project,
                manifest['workspace_uid'],
                manifest,
                json.loads(row['selected']),
                job_uid=row['job_uid'],
                on_created=created,
                on_creating=creating,
            )
            self._update_state(export_id, 'completed', job_uid=result['job_uid'], detail='')
        except Exception:
            with self.store.connect() as db:
                row = db.execute(
                    'SELECT job_uid,state FROM class_exports WHERE id=?',
                    (export_id,)).fetchone()
                if row is None:
                    return
                uncertain = row['state'] == 'creating' and not row['job_uid']
                detail = UNKNOWN_CREATION_DETAIL if uncertain else RETRY_DETAIL
                db.execute(
                    'UPDATE class_exports SET state=?,detail=? WHERE id=?',
                    ('unknown' if uncertain else 'failed', detail, export_id))


__all__ = [
    'ACTIVE_STATES',
    'CAPACITY',
    'ClassSelectionExportLifecycle',
    'ExportCapacityReached',
    'ExportLifecycleError',
    'ExportNotFound',
    'ExportRequestConflict',
    'ExportUnknownCreation',
]
