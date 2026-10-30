from pathlib import Path
from datetime import datetime,timezone
import json
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/qa/sticky-header'
OUT.mkdir(parents=True,exist_ok=True)
url='http://127.0.0.1:8765'
checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    for width,height,name in [(1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow'),(720,500,'zoom200')]:
        page.set_viewport_size({'width':width,'height':height})
        page.goto(url+'/#/home')
        expect(page.locator('.lab-entry')).to_have_count(2,timeout=30000)
        expect(page.locator('.course-app-header')).to_be_visible()
        page.wait_for_function("() => Math.abs(parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--app-header-height')) - document.querySelector('.course-app-header').getBoundingClientRect().height) < 1.1")
        # A tall tablet may fit the entire home page; use a longer existing page to verify scrolling.
        if page.evaluate('document.documentElement.scrollHeight - innerHeight') <= 20:
            page.get_by_role('navigation',name='主导航',exact=True).get_by_role('link',name='实验二',exact=True).click()
            expect(page.locator('.module-row')).to_have_count(5)
            page.wait_for_function('() => window.scrollY<1')
        before=page.locator('.course-header h1').evaluate('e=>e.getBoundingClientRect().top')
        page.evaluate('window.scrollTo(0,document.documentElement.scrollHeight)')
        page.wait_for_function('() => window.scrollY>20')
        rect=page.locator('.course-app-header').evaluate('e=>({top:e.getBoundingClientRect().top,bottom:e.getBoundingClientRect().bottom,height:e.getBoundingClientRect().height,position:getComputedStyle(e).position})')
        assert abs(rect['top'])<1,rect
        assert rect['position']=='sticky',rect
        after=page.locator('.course-header h1').evaluate('e=>e.getBoundingClientRect().top')
        assert after<before-20,(before,after)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.locator('.research-menu > summary').click()
        expect(page.locator('.research-menu .action-menu-panel')).to_be_visible()
        panel=page.locator('.research-menu .action-menu-panel').evaluate('e=>({top:e.getBoundingClientRect().top,bottom:e.getBoundingClientRect().bottom,right:e.getBoundingClientRect().right,height:e.getBoundingClientRect().height,viewport:innerHeight})')
        assert panel['top']>=rect['bottom']-1,panel
        assert panel['bottom']<=height+1,panel
        expect(page.get_by_role('link',name='案例概览',exact=True)).to_be_visible()
        page.screenshot(path=str(OUT/(name+'-scrolled-menu.png')))
        page.keyboard.press('Escape')
        expect(page.locator('.research-menu > summary')).to_be_focused()
        page.get_by_role('navigation',name='主导航',exact=True).get_by_role('link',name='实验一',exact=True).click()
        expect(page.locator('.module-row')).to_have_count(5)
        page.wait_for_function('() => window.scrollY<1')
        h1top=page.locator('.course-header h1').evaluate('e=>e.getBoundingClientRect().top')
        headerBottom=page.locator('.course-app-header').evaluate('e=>e.getBoundingClientRect().bottom')
        assert h1top>=headerBottom-1,(h1top,headerBottom)
        page.evaluate('window.scrollTo(0,document.documentElement.scrollHeight)')
        assert abs(page.locator('.course-app-header').evaluate('e=>e.getBoundingClientRect().top'))<1
        page.locator('.module-row').last.evaluate("e=>e.scrollIntoView({block:'start'})")
        lasttop=page.locator('.module-row').last.evaluate('e=>e.getBoundingClientRect().top')
        headerBottom=page.locator('.course-app-header').evaluate('e=>e.getBoundingClientRect().bottom')
        assert lasttop>=headerBottom-1,(lasttop,headerBottom)
        page.locator('.skip-link').focus()
        expect(page.locator('.skip-link')).to_be_focused()
        page.keyboard.press('Enter')
        expect(page.locator('#main-content')).to_be_focused()
        assert abs(page.locator('.course-app-header').evaluate('e=>e.getBoundingClientRect().top'))<1
        page.screenshot(path=str(OUT/(name+'-navigation.png')))
        checks.append({'viewport':name,'width':width,'height':height,'header_height':rect['height'],'sticky_while_scrolling':True,'content_scrolls':True,'menu_visible':True,'route_header_not_obscured':True,'scroll_offset_and_skip_link':True})
        print('PASS '+name+' pinned navigation, scrolling content, menu, navigation and focus',flush=True)
    assert not errors,errors
    browser.close()
(OUT/'checks.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'successful':True,'checks':checks,'browser_errors':errors},ensure_ascii=False,indent=2),encoding='utf8')
print('5 sticky header viewport checks passed')
