"""Read-only live acceptance for an experiment-only website; never edits case state."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
checks=[]
errors=[]
screenshots=[]
materials={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (root/'deliverables').rglob('*') if path.is_file()}
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    context=browser.new_context(viewport={'width':1508,'height':768},reduced_motion='reduce')
    page=context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    def capture(name):
        destination=out/name
        page.mouse.move(0,0)
        page.screenshot(path=str(destination),full_page=True)
        screenshots.append(destination)
    def experiment_only():
        expect(page.locator('.lab-entry code')).to_have_count(0)
        assert not any('experiments/' in text for text in page.locator('.lab-entry').all_text_contents())
        for footer in page.locator('.lab-entry-bottom').all():
            expect(footer).to_have_css('justify-content','flex-end')
        assert page.locator('a[href="#/materials"],a[href="#/labs/lab01/deliverables"],.materials-lab,.course-files,.app-footer').count()==0
        expect(page.get_by_text('成果与提交',exact=False)).to_have_count(0)
        expect(page.get_by_text('提交清单',exact=False)).to_have_count(0)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),page.viewport_size
    for width,height,label in ((1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow')):
        page.set_viewport_size({'width':width,'height':height})
        page.goto('http://127.0.0.1:8765/#/home')
        expect(page.get_by_role('heading',name='股权知识与智能推理实验平台',exact=True)).to_be_visible(timeout=30000)
        expect(page.locator('.lab-entry')).to_have_count(2)
        expect(page.locator('.lab-entry').first).to_contain_text('4 个任务')
        expect(page.locator('.lab-entry').last).to_contain_text('5 个任务')
        experiment_only()
        capture('home-'+label+'.png')
        for lab,count in (('lab01',4),('lab02',5)):
            page.evaluate('value=>location.hash=value','#/labs/'+lab)
            expect(page.locator('.module-row')).to_have_count(count)
            experiment_only()
            capture(lab+'-'+label+'.png')
        page.evaluate('scrollTo(0,document.documentElement.scrollHeight)')
        page.wait_for_function("Math.abs(document.querySelector('.course-app-header').getBoundingClientRect().top)<=1")
        expect(page.locator('.course-app-header')).to_have_css('position','sticky')
        page.locator('.research-menu > summary').click()
        expect(page.get_by_role('link',name='案例概览',exact=True)).to_be_visible()
        experiment_only()
        box=page.locator('.research-menu .action-menu-panel').bounding_box()
        assert box and box['x']>=-1 and box['x']+box['width']<=width+1,box
        page.screenshot(path=str(out/('menu-scrolled-'+label+'.png')))
        screenshots.append(out/('menu-scrolled-'+label+'.png'))
        page.keyboard.press('Escape')
        checks.append(label+' live home, 4/5 experiment directories and sticky tools menu passed')
    for fragment in ('#/materials','#/labs/lab01/deliverables'):
        page.evaluate('value=>location.hash=value',fragment)
        expect(page.get_by_role('heading',name='页面未找到',exact=True)).to_be_visible()
        experiment_only()
    checks.append('removed routes no longer render submission pages')
    assert materials=={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (root/'deliverables').rglob('*') if path.is_file()}
    checks.append('local deliverables remain byte-for-byte unchanged')
    assert not errors,errors
    browser.close()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':'http://127.0.0.1:8765','checks':checks,'unexpected_browser_errors':errors,'read_only':True,'local_materials_sha256':materials,'screenshots':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in screenshots}}
(out/'browser-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for check in checks:print('PASS '+check)
