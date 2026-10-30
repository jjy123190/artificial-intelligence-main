"""Read-only live measurements of return-navigation spacing; no case edits or saves."""
import argparse,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
state=json.loads((ROOT/'outputs/runtime/server.json').read_text(encoding='utf-8-sig'))
url=state['url']
assert state['running'] and url.startswith('http://127.0.0.1:')
parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');args=parser.parse_args()
baseline={} if args.before else json.loads((OUT/'before.json').read_text(encoding='utf-8'))['measurements']
measurements,errors,screenshots={},[],{}
routes=(('#/home','股权知识与智能推理实验平台','home'),('#/labs/lab01','实验01 · 知识表示、知识增强与推理','lab01'),('#/labs/lab02','实验02 · 确定性、不确定性与混合推理','lab02'),('#/labs/lab01/demo','知识增强与推理演示','demo'),('#/labs/lab02/modeling','知识表示与建模','modeling'),('#/case/C3','实体资料','source-records'),('#/labs/lab02/hybrid/overview','全案例分析','analysis'),('#/labs/lab02/modeling/rules','规则知识库','rules'))
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    context=browser.new_context(reduced_motion='reduce')
    page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
    for width,height,label in ((1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow')):
        page.set_viewport_size({'width':width,'height':height})
        for fragment,title,stem in routes:
            page.goto(url+'/'+fragment)
            expect(page.get_by_role('heading',name=title,exact=True)).to_be_visible(timeout=30000)
            if stem=='home':expect(page.locator('.lab-entry')).to_have_count(2)
            if stem=='lab01':expect(page.locator('.module-row')).to_have_count(4)
            if stem=='lab02':expect(page.locator('.module-row')).to_have_count(5)
            if stem=='analysis':expect(page.locator('.directory-row')).to_have_count(12,timeout=30000)
            main=page.locator('#main-content');expect(main).to_be_visible()
            assert main.count()==1
            crumb=main.locator(':scope > .course-breadcrumb,:scope > .page-breadcrumb')
            heading=page.get_by_role('heading',name=title,exact=True)
            topbar=page.locator('.course-app-header').bounding_box()
            titlebox=heading.bounding_box()
            assert topbar and titlebox
            box=crumb.bounding_box() if crumb.count() else None
            data={'header_bottom':topbar['y']+topbar['height'],'heading_y':titlebox['y'],'breadcrumb_y':box['y'] if box else None,'breadcrumb_height':box['height'] if box else None}
            key=label+':'+stem;measurements[key]=data
            if box:
                assert box['y']>=data['header_bottom']+8,(key,data)
                assert box['y']+box['height']<titlebox['y'],(key,data)
                link=crumb.locator('a').first
                linkbox=link.bounding_box();assert linkbox and linkbox['height']>=44,(key,linkbox)
                assert link.evaluate("el=>{const b=el.getBoundingClientRect();return el.contains(document.elementFromPoint(b.x+b.width/2,b.y+b.height/2));}"),key
                link.focus();expect(link).to_be_focused()
                page.evaluate('document.activeElement.blur()')
            if not args.before:
                old=baseline[key]
                if box:
                    gap=24 if width>760 else 16
                    previous_gap=old['heading_y']-old['breadcrumb_y']-old['breadcrumb_height']
                    actual_gap=data['heading_y']-data['breadcrumb_y']-data['breadcrumb_height']
                    assert abs(data['breadcrumb_y']-old['breadcrumb_y'])<=1,(key,'return row moved',old,data)
                    assert abs(actual_gap-gap)<=1,(key,'incorrect heading gap',actual_gap,gap)
                    assert actual_gap<previous_gap,(key,'gap did not shrink',previous_gap,actual_gap)
                    assert abs(old['heading_y']-data['heading_y']-(previous_gap-gap))<=1,(key,'unexpected content shift',old,data)
                    assert crumb.evaluate('el=>parseFloat(getComputedStyle(el).marginBottom)')==8,key
                else:
                    assert old['breadcrumb_y'] is None
                    assert abs(data['heading_y']-old['heading_y'])<=1,(key,'homepage moved',old,data)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),key
            if stem=='lab01' and label in ('laptop','mobile'):
                name=('before-' if args.before else 'after-')+stem+'-'+label+'.png'
                page.mouse.move(0,0);page.screenshot(path=str(OUT/name))
                screenshots[name]=hashlib.sha256((OUT/name).read_bytes()).hexdigest()
        page.evaluate('scrollTo(0,document.documentElement.scrollHeight)')
        page.wait_for_function("Math.abs(document.querySelector('.course-app-header').getBoundingClientRect().top)<=1")
        expect(page.locator('.course-app-header')).to_have_css('position','sticky')
        print('PASS '+label+': return navigation '+('baseline captured' if args.before else 'heading gap reduced, return row unchanged, hit targets clear'),flush=True)
    assert not errors,errors
    browser.close()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':url,'read_only':True,'measurements':measurements,'screenshots':screenshots,'unexpected_browser_errors':errors,'mode':'before' if args.before else 'after'}
(OUT/('before.json' if args.before else 'browser-checks.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
