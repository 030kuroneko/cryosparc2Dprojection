"""Job-control behavior through browser actions."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_stop_is_immediate_delete_is_confirmed_and_failures_are_visible():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const context={window:{}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
 let view, allow=false, confirmations=[],calls=[], refreshed=0;
 const ui=context.window.JobControls.create({
  api:async(path,method)=>{calls.push([path,method]);return {state:'stopping'};},
  confirm:text=>{confirmations.push(text);return allow;},
  render:v=>view=v, refresh:async()=>refreshed++
 });
 ui.setJob({id:'abc',state:'running'});
 await ui.stop();
 assert.deepEqual(calls,[['/api/jobs/abc/stop','POST']]);
 assert.equal(confirmations.length,0);
 await ui.remove();assert.equal(calls.length,1);
 assert.ok(confirmations[0].includes('immediately'));
 assert.ok(confirmations[0].includes('CryoSPARC'));
 allow=true;await ui.remove();
 assert.deepEqual(calls[1],['/api/jobs/abc','DELETE']);
 ui.setJob({id:'abc',state:'completed'});
 assert.equal(view.canStop,false);
 ui.setJob(null);assert.equal(view.visible,false);
 const bad=context.window.JobControls.create({api:async()=>{throw Error('Stop unavailable')},
  confirm:()=>true,render:v=>view=v,refresh:async()=>{}});
 bad.setJob({id:'abc',state:'running'});await bad.stop();
 assert.equal(view.error,'Stop unavailable');assert.equal(view.busy,false);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    asset = Path(__file__).parents[1] / 'src/web_assets/job-controls.js'
    result = subprocess.run([node, '-e', script, str(asset)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_page_exposes_stop_delete_and_accessible_feedback(tmp_path):
    from cryosparc_2d_projection.web import create_app
    page = create_app(dict(data_dir=str(tmp_path), cryosparc_url='https://cryo.example',
                           public_url='http://localhost', allow_http=True)).test_client().get('/').text
    assert '/assets/job-controls.js' in page
    assert 'id="stop-job"' in page and 'id="delete-job"' in page
    assert 'id="job-control-status"' in page
