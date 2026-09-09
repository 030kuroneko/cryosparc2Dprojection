"""Job-detail warnings through browser HTTP responses and DOM events."""
from pathlib import Path
import shutil
import subprocess


def test_job_cleanup_warning_tracks_selected_job_and_disappears_after_cleanup():
    assets = Path(__file__).parents[1] / 'src/web_assets'
    node = shutil.which('node')
    assert node, 'Node is required for the job-detail browser contract'
    result = subprocess.run([node, '-e', r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const html = fs.readFileSync(process.argv[1] + '/index.html', 'utf8');
const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(([,id]) =>
  [id, {hidden:true, textContent:'', innerHTML:'', value:'local', listeners:{},
    addEventListener(event, fn) {this.listeners[event]=fn;},
    insertAdjacentHTML() {}, replaceChildren() {this.innerHTML='';}}]));
assert.ok(elements['job-cleanup-warning'], 'Job details must contain a cleanup warning');
const job = {id:'a'.repeat(32), workflow:'orientation', profile:'local', state:'completed',
  created:'2026-09-09T00:00:00Z', values:{project:'P1', workspace:'W2'}, cleanup_pending:true};
const other = {...job, id:'b'.repeat(32), cleanup_pending:false};
let jobs = [job, other], poll;
const context = {window:{}, document:{getElementById(id) {return elements[id];}, querySelectorAll() {return [];}},
  setInterval(fn) {poll=fn;}, fetch:async path => ({ok:true, json:async () => {
    if(path==='/api/session') return {csrf:'test', email:'alice', cryosparc_url:'https://cryo.example'};
    if(path==='/api/schema') return {workflows:{orientation:{title:'Class Orientation',fields:[]}},
      profiles:[{id:'local',label:'Local',backend:'local'}]};
    if(path==='/api/jobs') return {jobs};
    if(path.endsWith('/log')) return {log:'Workflow completed.'};
    throw Error(path);
  }})};
vm.runInNewContext(fs.readFileSync(process.argv[1] + '/progress.js', 'utf8'), context);
vm.runInNewContext(fs.readFileSync(process.argv[1] + '/app.js', 'utf8'), context);
(async () => {
  await new Promise(setImmediate);
  assert.equal(elements['login-error'].textContent, '');
  const warning = elements['job-cleanup-warning'];
  assert.equal(warning.hidden, false);
  assert.match(warning.textContent, /Computation succeeded.*credential cleanup pending retry/i);
  assert.match(elements['job-progress'].textContent, /^Completed/);
  assert.ok(elements['job-list'].innerHTML.includes('completed'));
  assert.ok(!elements['job-list'].innerHTML.includes('cleanup pending'));
  await elements['job-list'].listeners.click({target:{closest(){return {dataset:{job:other.id}};}}});
  assert.equal(warning.hidden, true);
  await elements['job-list'].listeners.click({target:{closest(){return {dataset:{job:job.id}};}}});
  assert.equal(warning.hidden, false);
  jobs = [{...job, cleanup_pending:false}, other];
  await poll();
  assert.equal(warning.hidden, true);
  assert.equal(warning.textContent, '');
  jobs = [{...job, state:'failed'}, other];
  await poll();
  assert.equal(warning.hidden, false);
  assert.match(warning.textContent, /Computation failed/i);
  assert.match(elements['job-progress'].textContent, /^Failed/);
  assert.ok(!warning.textContent.includes('succeeded'));
})().catch(error => {console.error(error); process.exitCode=1;});
''', str(assets)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
