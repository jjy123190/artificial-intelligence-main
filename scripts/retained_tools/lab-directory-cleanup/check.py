"""Read-only live acceptance for removing lab-directory hints and code locations."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
state=json.loads((root/'outputs/runtime/server.json').read_text(encoding='utf-8-sig'))
url=state['url']
assert state['running'] and url.startswith('http://127.0.0.1:')
checks,errors,captures=[],[],[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    context=browser.new_context(viewport={'width':1508,'height':768},reduced_motion='reduce')
    page=context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    for width,height,label in ((1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow')):
        page.set_viewport_size({'width':width,'height':height})
        for lab,count in (('lab01',4),('lab02',5)):
            page.goto(url+'/#/labs/'+lab)
            expect(page.locator('.module-row')).to_have_count(count,timeout=30000)
            expect(page.locator('.code-location,.lab-goal')).to_have_count(0)
            expect(page.get_by_text('代码位置',exact=False)).to_have_count(0)
            expect(page.get_by_text('每一部分都标明对应要求',exact=False)).to_have_count(0)
            expect(page.locator('.course-header p')).not_to_have_text('')
            assert page.locator('#main-content').count()==1
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            header=page.locator('.course-header').bounding_box()
            task_list=page.locator('.module-list').bounding_box()
            assert header and task_list
            assert 8<=task_list['y']-header['y']-header['height']<=48,{'header':header,'tasks':task_list}
            for row in page.locator('.module-row').all():
                assert row.get_attribute('href').startswith('#/labs/'+lab+'/')
            page.mouse.move(0,0)
            destination=out/(lab+'-'+label+'.png')
            page.screenshot(path=str(destination),full_page=True)
            captures.append(destination)
            page.evaluate('scrollTo(0,document.documentElement.scrollHeight)')
            page.wait_for_function("Math.abs(document.querySelector('.course-app-header').getBoundingClientRect().top)<=1")
            expect(page.locator('.course-app-header')).to_have_css('position','sticky')
            checks.append(lab+' '+label+': hint/code-location absent; summary, task links and sticky navigation preserved')
    page.goto(url+'/#/labs/lab01')
    expect(page.locator('.module-row')).to_have_count(4)
    page.get_by_role('link',name='知识表示与方法比较',exact=True).click()
    expect(page.get_by_role('heading',name='知识表示与方法比较',exact=True)).to_be_visible()
    expect(page.locator('.knowledge-canvas')).to_be_visible(timeout=30000)
    page.goto(url+'/#/labs/lab02')
    expect(page.locator('.module-row')).to_have_count(5)
    page.get_by_role('link',name='知识表示与建模',exact=True).click()
    expect(page.get_by_role('heading',name='知识表示与建模',exact=True)).to_be_visible()
    expect(page.get_by_role('button',name='构建当前案例的谓词事实',exact=True)).to_be_enabled(timeout=30000)
    checks.append('real task entry links for both experiments still open the correct pages')
    assert not errors,errors
    browser.close()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':url,'read_only':True,'checks':checks,'unexpected_browser_errors':errors,'screenshots':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in captures}}
(out/'browser-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for check in checks:print('PASS '+check)
