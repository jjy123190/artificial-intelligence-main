"""Read-only live regression for removing the lab-one task header badge."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
state=json.loads((ROOT/'outputs/runtime/server.json').read_text(encoding='utf-8-sig'))
url=state['url']
assert state['running'] and url.startswith('http://127.0.0.1:')
checks,errors,captures=[],[],[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    context=browser.new_context(reduced_motion='reduce')
    page=context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    lab=next(item for item in context.request.get(url+'/api/course').json()['labs'] if item['id']=='lab01')
    for width,height,label in ((1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow')):
        page.set_viewport_size({'width':width,'height':height})
        for module in lab['modules']:
            page.goto(url+'/#/labs/lab01/'+module['id'])
            expect(page.get_by_role('heading',name=module['title'],exact=True)).to_be_visible(timeout=30000)
            main=page.locator('#main-content')
            expect(main.locator('.course-header .requirement-label')).to_have_count(0)
            expect(main.get_by_text('核心要求',exact=True)).to_have_count(0)
            expect(main.locator('.task-requirement')).to_be_visible()
            expect(main.locator('.task-requirement')).to_contain_text('对应实验要求')
            expect(main.locator('.task-requirement p')).to_have_text(module['requirement'])
            expect(main.locator('.course-header p')).to_have_text(module['description'])
            assert main.count()==1
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(label,module['id'])
            if module['id']=='representation':
                expect(page.locator('.knowledge-canvas')).to_be_visible(timeout=30000)
                if label in ('laptop','mobile'):
                    destination=OUT/('representation-'+label+'.png')
                    page.mouse.move(0,0);page.screenshot(path=str(destination))
                    captures.append(destination)
            checks.append(label+' '+module['id']+': header badge removed; summary and requirement explanation preserved')
        page.goto(url+'/#/labs/lab01')
        expect(page.locator('.module-row')).to_have_count(4,timeout=30000)
        expect(page.locator('.module-row .requirement-label')).to_have_count(4)
        page.get_by_role('link',name='知识表示与方法比较',exact=True).click()
        expect(page.get_by_role('heading',name='知识表示与方法比较',exact=True)).to_be_visible()
    assert not errors,errors
    browser.close()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':url,'read_only':True,'checks':checks,'unexpected_browser_errors':errors,'screenshots':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in captures}}
(OUT/'browser-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for check in checks:print('PASS '+check)
