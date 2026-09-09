"""Help disclosure behavior at the browser event/DOM boundary."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_help_supports_hover_focus_click_and_escape():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for browser contracts')
    script = Path(__file__).parents[1] / 'src/web_assets/help.js'
    result = subprocess.run([node, '-e', r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const listeners = {};
const button = {attrs: {}, setAttribute(k,v) {this.attrs[k]=v;}, focus() {this.focused=true;}};
const panel = {hidden: true};
const wrapper = {dataset: {}, contains(target) {return target === button || target === panel;},
  querySelector(selector) {return selector === 'button' ? button : panel;}};
button.closest = panel.closest = selector => selector === '.field-help' ? wrapper : null;
const document = {addEventListener(event, fn) {listeners[event]=fn;},
  querySelectorAll() {return [wrapper];}};
const timers = new Map(); let nextTimer = 0;
function flushTimers() {
  const pending = [...timers.values()]; timers.clear();
  for (const callback of pending) callback();
}
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {document,
  setTimeout(fn) {timers.set(++nextTimer, fn); return nextTimer;},
  clearTimeout(id) {timers.delete(id);}});
const event = {target: button, relatedTarget: null};
listeners.pointerover(event);
assert.equal(panel.hidden, false);
assert.equal(button.attrs['aria-expanded'], 'true');
listeners.pointerout({...event, relatedTarget: panel});
assert.equal(panel.hidden, false);
listeners.pointerout(event);
assert.equal(panel.hidden, false); // cross the gap without losing the panel
listeners.pointerover({target: panel, relatedTarget: null});
flushTimers();
assert.equal(panel.hidden, false); // entering content cancels delayed close
listeners.pointerout({target: panel, relatedTarget: null});
flushTimers();
assert.equal(panel.hidden, true);
listeners.focusin(event);
assert.equal(panel.hidden, false);
listeners.focusout(event);
flushTimers();
assert.equal(panel.hidden, true);
listeners.click(event);
assert.equal(panel.hidden, false);
listeners.pointerout(event);
assert.equal(panel.hidden, false); // clicked help remains open
listeners.keydown({...event, key:'Escape'});
assert.equal(panel.hidden, true);
assert.equal(button.focused, true);
flushTimers();
assert.equal(panel.hidden, true);
listeners.click(event);
listeners.click({target: {closest() {return null;}}});
assert.equal(panel.hidden, true);
''', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
