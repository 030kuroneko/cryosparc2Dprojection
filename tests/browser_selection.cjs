// Optional browser integration check: NODE_PATH=<playwright packages> node tests/browser_selection.cjs
const {chromium} = require('playwright');
const fs = require('fs'), path = require('path'), assert = require('assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors=[]; page.on('pageerror', e=>errors.push(e.message));
    let selected=[],revision=0,exports=[];
    const classes=[{class_number:2,particle_count:1200,score:.85,orientation_method:'particle_pose_local_search',confidence:'high',image:'class_2.png'},
      {class_number:9,particle_count:3500,score:.5,orientation_method:'image_global_search',confidence:'low',image:'class_9.png'}];
    const selection=()=>({available:true,classes,selected_class_numbers:selected,revision,exports});
    const assetRoot=path.join(__dirname,'../src/web_assets');
    await page.route('http://localhost:4311/**', async route=>{
      const url=new URL(route.request().url()), p=url.pathname;
      const json=value=>route.fulfill({json:value});
      if(p==='/') return route.fulfill({path:path.join(assetRoot,'index.html'),contentType:'text/html'});
      if(p.startsWith('/assets/')) return route.fulfill({path:path.join(assetRoot,path.basename(p)),contentType:p.endsWith('.js')?'text/javascript':'text/css'});
      if(p==='/api/session') return json({csrf:'test',email:'test@example.org',cryosparc_url:'https://cryo.example'});
      if(p==='/api/schema') return json({profiles:[{id:'local',label:'Local',backend:'local'}],workflows:{orientation:{title:'Class Orientation',description:'',fields:[]},axis:{title:'Axis Search',fields:[]}}});
      if(p==='/api/jobs') return json({jobs:[{id:'one',workflow:'orientation',state:'completed',created:'2026-09-14',profile:'local',values:{project:'P1',workspace:'W1'}}]});
      if(p.endsWith('/log')) return json({log:'Workflow completed.',details:'',progress:null});
      if(p.endsWith('/selection')) {
        if(route.request().method()==='PUT') {selected=route.request().postDataJSON().selected_class_numbers;revision++;}
        return json(selection());
      }
      if(p.includes('/images/')) return route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><rect width="900" height="300" fill="#182637"/><g fill="#b9d4d9" font-family="sans-serif" font-size="22"><text x="40" y="150">Class average</text><text x="320" y="150">Matched projection</text><text x="650" y="150">Camera view</text></g></svg>'});
      if(p==='/api/request-id') return json({request_id:'dd33caa9-ecb7-4399-b317-657f9b2901f8'});
      if(p.endsWith('/exports')) {exports=[{id:'export1',state:'completed',job_uid:'J99',selected_class_numbers:route.request().postDataJSON().selected_class_numbers}];return json(exports[0]);}
      return json({});
    });
    await page.goto('http://localhost:4311');
    await page.locator('#class-selection').waitFor({state:'visible'});
    await page.locator('[data-class="9"]').check();
    await page.waitForFunction(()=>document.querySelector('#selection-status').textContent.startsWith('Selections saved'));
    assert.deepEqual(selected,[9]);
    await page.locator('#selection-sort').selectOption('particle_count');
    assert.equal(await page.locator('.selection-card input').first().getAttribute('data-class'),'9');
    assert.equal(await page.locator('[data-class="9"]').isChecked(),true);
    await page.locator('#selection-export').click();
    await page.getByText('J99',{exact:true}).waitFor();
    assert.deepEqual(exports[0].selected_class_numbers,[9]);
    await page.reload();
    await page.locator('[data-class="9"]:checked').waitFor();
    await page.locator('#selection-clear').click();
    await page.waitForFunction(()=>document.querySelector('#selection-export').disabled);
    await page.locator('#selection-all').click();
    await page.waitForFunction(()=>document.querySelector('#selection-count').textContent==='2 / 2 selected');
    assert.equal(await page.locator('.selection-card').count(),2);
    await page.locator('#class-selection').screenshot({path:process.env.SELECTION_SCREENSHOT || '/tmp/class-selection-ui.png'});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
    assert.deepEqual(errors,[]);
    console.log('Browser selection checks passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
