"""Progress presentation at the browser boundary."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_progress_labels_scope_and_never_advances_work_with_time():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for browser contracts')
    script = Path(__file__).parents[1] / 'src/web_assets/progress.js'
    result = subprocess.run([node, '-e', r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const describe = context.window.JobProgressView.describe;
const p = {stage:'Finding class orientations',state:'running',completed:2,total:4,unit:'classes',
  elapsed_seconds:40,stage_elapsed_seconds:40,last_progress_seconds:0,updated_at:100,
  remaining_seconds:[30,60],remaining_scope:'stage'};
const first = describe(p, 'running', 100);
assert.ok(first.text.includes('Estimated stage time remaining:'));
assert.ok(first.text.includes('2/4 classes'));
const later = describe(p, 'running', 120);
assert.equal(later.value, first.value);
assert.ok(later.text.includes('1 min'));
assert.ok(describe({...p,remaining_seconds:null},'running',100).text.includes('Estimating'));
assert.ok(!describe(p,'failed',120).text.includes('remaining:'));
assert.ok(describe(p,'unknown',120).text.includes('Status unavailable'));
assert.ok(describe(null,'queued',120).text.includes('Queue wait is not included'));
assert.ok(describe(p,'completed',120).text.startsWith('Completed'));
assert.ok(!describe(p,'completed',120).text.includes('2/4'));
''', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
