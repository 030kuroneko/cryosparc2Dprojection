// NODE_PATH=<playwright packages> node tests/browser_job_controls.cjs
const {chromium} = require('playwright');
const fs = require('fs'), path = require('path'), assert = require('assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors = [], calls = [];
    page.on('pageerror', error => errors.push(error.message));
    let jobs = [{id:'one',workflow:'orientation',state:'running',created:'2026-09-14',
      profile:'local',values:{project:'P1',workspace:'W1'}}];
    const assets = path.join(__dirname,'../src/web_assets');
    await page.route('http://localhost:4311/**', async route => {
      const p = new URL(route.request().url()).pathname, method = route.request().method();
      const json = value => route.fulfill({json:value});
      if (p === '/') return route.fulfill({path:path.join(assets,'index.html'),contentType:'text/html'});
      if (p.startsWith('/assets/')) return route.fulfill({path:path.join(assets,path.basename(p)),contentType:p.endsWith('.js')?'text/javascript':'text/css'});
      if (p === '/api/session') return json({csrf:'test',email:'test@example.org',cryosparc_url:'https://cryo.example'});
      if (p === '/api/schema') return json({profiles:[{id:'local',label:'Local',backend:'local'}],workflows:{orientation:{title:'Class Orientation',fields:[]},axis:{title:'Axis Search',fields:[]}}});
      if (p === '/api/jobs') return json({jobs});
      if (p.endsWith('/stop')) {calls.push('stop');jobs[0].state='interrupted';return json(jobs[0]);}
      if (p === '/api/jobs/one' && method === 'DELETE') {calls.push('delete');jobs=[];return json({state:'deleted'});}
      if (p.endsWith('/log')) return json({log:'Test job',details:'',progress:null});
      if (p.endsWith('/selection')) return json({available:false});
      return json({});
    });
    await page.goto('http://localhost:4311');
    await page.locator('#job-controls').waitFor({state:'visible'});
    assert.equal(await page.locator('#stop-job').isEnabled(), true);
    await page.locator('#stop-job').click();
    await page.waitForFunction(() => document.querySelector('#stop-job').disabled);
    assert.deepEqual(calls,['stop']);
    page.once('dialog', async dialog => {assert.ok(dialog.message().includes('CryoSPARC')); await dialog.dismiss();});
    await page.locator('#delete-job').click();
    assert.deepEqual(calls,['stop']);
    await page.locator('#job-controls').screenshot({path:'/tmp/job-controls-ui.png'});
    page.once('dialog', dialog => dialog.accept());
    await page.locator('#delete-job').click();
    await page.locator('#job-controls').waitFor({state:'hidden'});
    assert.deepEqual(calls,['stop','delete']);
    assert.equal(await page.locator('#class-selection').isVisible(),false);
    assert.equal(await page.locator('#job-log').textContent(),'Select a run to inspect its activity.');
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),true);
    assert.deepEqual(errors,[]);
    console.log('Browser job controls checks passed');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
