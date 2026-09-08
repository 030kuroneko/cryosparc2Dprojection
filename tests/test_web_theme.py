"""Theme control at the browser DOM/storage boundary."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_theme_toggle_restores_preference_and_works_without_storage():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser theme contract')
    script = Path(__file__).parents[1] / 'src/web_assets/theme.js'
    result = subprocess.run([node, '-e', r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
for (const saved of [null, 'light', 'dark', 'invalid', 'unavailable']) {
  let stored = saved;
  const root = {dataset: {}};
  const button = {attrs: {}, setAttribute(k,v) {this.attrs[k]=v;},
    addEventListener(event, handler) {this[event]=handler;}};
  const context = {document: {documentElement:root,
    getElementById() {return button;}, addEventListener(event, handler) {handler();}},
    localStorage: {getItem() {if(saved==='unavailable') throw Error('blocked'); return saved;},
      setItem(key,value) {if(saved==='unavailable') throw Error('blocked'); stored=value;}}};
  vm.runInNewContext(source, context);
  const initial = saved === 'light' ? 'light' : 'dark';
  assert.equal(root.dataset.theme, initial);
  button.click();
  assert.equal(root.dataset.theme, initial === 'dark' ? 'light' : 'dark');
  assert.equal(button.attrs['aria-pressed'], String(root.dataset.theme === 'light'));
  assert.ok(button.attrs['aria-label'].includes(root.dataset.theme === 'light' ? 'dark' : 'light'));
  if (saved !== 'unavailable') assert.equal(stored, root.dataset.theme);
  button.click();
  assert.equal(root.dataset.theme, initial);
}
''', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
