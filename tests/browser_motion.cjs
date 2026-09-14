// NODE_PATH=<Playwright packages> node tests/browser_motion.cjs
const {chromium} = require('playwright');
const path = require('path'), assert = require('assert/strict');
(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors=[]; page.on('pageerror', e=>errors.push(e.message));
    const field = (key, group) => ({key,group,label:key,type:'string',choices:[],default:'10',hint:'Test field'});
    const fields=[field('sample','basic'),field('search_limit','search')];
    await page.route('http://localhost:4312/**', async route=>{
      const p=new URL(route.request().url()).pathname;
      const json=value=>route.fulfill({json:value});
      if(p==='/') return route.fulfill({path:path.join(__dirname,'../src/web_assets/index.html'),contentType:'text/html'});
      if(p.startsWith('/assets/')) return route.fulfill({path:path.join(__dirname,'../src/web_assets',path.basename(p)),contentType:p.endsWith('.js')?'text/javascript':'text/css'});
      if(p==='/api/session') return json({csrf:'test',email:'test@example.org',cryosparc_url:'https://cryo.example'});
      if(p==='/api/schema') return json({profiles:[{id:'local',label:'Local',backend:'local'}],workflows:{orientation:{title:'Class Orientation',description:'',fields},axis:{title:'Axis Search',description:'',fields}}});
      if(p==='/api/jobs') return json({jobs:[]});
      return json({});
    });
    await page.goto('http://localhost:4312');
    const heading=page.getByText('Search settings',{exact:true});
    await heading.waitFor();
    await heading.click();
    const moving=await page.locator('#advanced-fields').evaluate(el=>el.getAnimations({subtree:true}).length);
    assert.ok(moving>0,'Expanding settings should animate');
    await page.getByLabel('search_limit',{exact:true}).fill('42');
    await heading.focus(); await page.keyboard.press('Enter');
    await page.waitForTimeout(300);
    assert.equal(await page.getByLabel('search_limit',{exact:true}).isVisible(),false);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(300);
    assert.equal(await page.getByLabel('search_limit',{exact:true}).inputValue(),'42');
    await page.locator('[data-workflow="axis"]').click();
    assert.ok(await page.locator('.configuration').evaluate(el=>el.getAnimations().length)>0,
      'Switching workflows should animate the incoming configuration');
    await page.locator('[data-workflow="orientation"]').click();
    await page.waitForTimeout(300);
    await heading.click();
    assert.equal(await page.getByLabel('search_limit',{exact:true}).inputValue(),'42');
    await page.locator('[data-workflow="orientation"]').click();
    assert.equal(await page.getByLabel('search_limit',{exact:true}).isVisible(),true,
      'Reselecting the current workflow should preserve the expanded settings');
    assert.equal(await page.locator('.configuration').evaluate(el=>el.getAnimations().length),0);
    await page.emulateMedia({reducedMotion:'reduce'});
    await heading.click();
    assert.equal(await page.locator('#advanced-fields').evaluate(el=>el.getAnimations({subtree:true}).length),0);
    await page.locator('[data-workflow="axis"]').click();
    assert.equal(await page.locator('.configuration').evaluate(el=>el.getAnimations().length),0);
    await page.emulateMedia({reducedMotion:'no-preference'});
    await heading.evaluate(el=>{el.click();el.click();el.click();});
    await page.waitForTimeout(300);
    assert.equal(await page.getByLabel('search_limit',{exact:true}).isVisible(),true);
    await page.screenshot({path:'/tmp/cryosparc-motion-desktop.png'});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:'/tmp/cryosparc-motion-mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('Browser motion checks passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
