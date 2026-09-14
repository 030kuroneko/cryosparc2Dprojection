"""Selection behavior exercised through the browser view's public actions."""
from pathlib import Path
import shutil
import subprocess

import pytest


def run_browser(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for browser contracts')
    asset = Path(__file__).parents[1] / 'src/web_assets/selection.js'
    result = subprocess.run([node, '-e', script, str(asset)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_result_page_exposes_selection_controls_and_script(tmp_path):
    from cryosparc_2d_projection.web import create_app
    app = create_app(dict(data_dir=str(tmp_path),cryosparc_url='https://cryo.example',
                         public_url='http://localhost',allow_http=True))
    client = app.test_client()
    page = client.get('/').text
    for control in ('class-selection','selection-all','selection-clear','selection-invert',
                    'selection-sort','selection-export','selection-cards','selection-status'):
        assert f'id="{control}"' in page
    assert '/assets/selection.js' in page
    assert client.get('/assets/selection.js').status_code == 200


def test_selection_controls_preserve_class_identity_and_saved_choices():
    run_browser(r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const context = {window:{}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'), context);
(async()=>{
  let view, saved = [], revision = 0;
  const classes = [{class_number:9,particle_count:30,score:.4},
                   {class_number:2,particle_count:10,score:.9}];
  const api = async (path, method, body) => {
    if (method === 'PUT') {saved = body.selected_class_numbers; revision++;}
    return {available:true,classes,selected_class_numbers:saved,revision,exports:[]};
  };
  const ui = context.window.ClassSelectionView.create({api,render:v=>view=v});
  await ui.setJob({id:'a',workflow:'orientation',state:'completed'});
  assert.deepEqual(Array.from(view.classes, c=>c.class_number), [2,9]);
  assert.equal(view.canExport,false);
  await ui.toggle(9);
  ui.sort('score');
  assert.deepEqual(Array.from(view.selected_class_numbers), [9]);
  assert.deepEqual(Array.from(view.classes, c=>c.class_number), [2,9]);
  await ui.invert();
  assert.deepEqual(Array.from(saved), [2]);
  await ui.selectAll();
  assert.deepEqual(Array.from(saved), [2,9]);
  await ui.clear();
  assert.equal(view.canExport,false);
  await ui.toggle(9);
  const reopened = context.window.ClassSelectionView.create({api,render:v=>view=v});
  await reopened.setJob({id:'a',workflow:'orientation',state:'completed'});
  assert.deepEqual(Array.from(view.selected_class_numbers), [9]);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_export_snapshots_selection_and_retries_uncertain_request_without_new_id():
    run_browser(r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const context = {window:{}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'), context);
(async()=>{
  let view, attempts = [], fail = true, release, requested;
  const began = new Promise(r=>requested=r);
  const gate = new Promise(r=>release=r);
  const api = async (path,method,body) => {
    if(path==='/api/request-id') return {request_id:'one-request'};
    if(path.endsWith('/exports')) {
      attempts.push(body); requested(); await gate;
      if(fail) {fail=false; throw Error('Connection lost');}
      return {id:'export1',state:'completed',job_uid:'J99'};
    }
    return {available:true,classes:[{class_number:2},{class_number:9}],
      selected_class_numbers:[2],revision:1,exports:[]};
  };
  const ui = context.window.ClassSelectionView.create({api,render:v=>view=v});
  await ui.setJob({id:'a',workflow:'orientation',state:'completed'});
  const exporting = ui.exportSelection();
  await began;
  await ui.toggle(9);
  release(); await exporting;
  assert.deepEqual(Array.from(attempts[0].selected_class_numbers), [2]);
  assert.deepEqual(Array.from(view.selected_class_numbers), [2,9]);
  assert.equal(view.retryRequest,true);
  await ui.exportSelection();
  assert.equal(attempts[1].request_id, attempts[0].request_id);
  assert.deepEqual(Array.from(attempts[1].selected_class_numbers), [2]);
  assert.equal(view.exports[0].job_uid,'J99');
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_failed_save_does_not_silently_replace_local_choices_on_poll():
    run_browser(r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert/strict');
const context={window:{}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
  let view;
  const api=async(path,method)=>{
    if(method==='PUT') throw Error('Selection changed in another tab');
    return {available:true,classes:[{class_number:1}],selected_class_numbers:[],revision:0,exports:[]};
  };
  const ui=context.window.ClassSelectionView.create({api,render:v=>view=v});
  const job={id:'a',workflow:'orientation',state:'completed'};
  await ui.setJob(job); await ui.toggle(1); await ui.setJob(job);
  assert.deepEqual(Array.from(view.selected_class_numbers),[1]);
  assert.ok(view.error.includes('another tab')); assert.equal(view.canExport,false);
  await ui.reload(); assert.deepEqual(Array.from(view.selected_class_numbers),[]);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_switching_jobs_retains_inflight_selection_and_revision():
    run_browser(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const context={window:{}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
  let view, release, began, first=true;
  const gate=new Promise(r=>release=r), started=new Promise(r=>began=r);
  const saved={a:[],b:[]}, revisions={a:0,b:0};
  const api=async(path,method,body)=>{
    const id=path.split('/')[3];
    if(method==='PUT') {
      if(first) {first=false;began();await gate;}
      if(body.revision!==revisions[id]) throw Error('Revision conflict');
      saved[id]=body.selected_class_numbers;revisions[id]++;
    }
    return {available:true,classes:[{class_number:1},{class_number:2}],
      selected_class_numbers:saved[id],revision:revisions[id],exports:[]};
  };
  const ui=context.window.ClassSelectionView.create({api,render:v=>view=v});
  const a={id:'a',workflow:'orientation',state:'completed'},b={...a,id:'b'};
  await ui.setJob(a);const saving=ui.toggle(1);await started;
  await ui.setJob(b);await ui.setJob(a);
  assert.deepEqual(Array.from(view.selected_class_numbers),[1]);
  const second=ui.toggle(2);release();await saving;await second;
  assert.deepEqual(Array.from(saved.a),[1,2]);
  assert.deepEqual(Array.from(view.selected_class_numbers),[1,2]);
  assert.equal(view.error,'');
  await ui.setJob(null);await ui.setJob(a);
  assert.deepEqual(Array.from(view.selected_class_numbers),[1,2]);
})().catch(e=>{console.error(e);process.exitCode=1});
''')
