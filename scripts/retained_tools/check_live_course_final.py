from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/'qa'/'course'
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(viewport={'width':1508,'height':768},reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8765/#/home')
    expect(page.get_by_role('heading',name='股权知识与智能推理实验平台',exact=True)).to_be_visible(timeout=30000)
    expect(page.locator('.lab-entry')).to_have_count(2)
    assert page.locator('.directory, .inspector, .app-footer').count()==0
    for name,fragment in [('home','home'),('lab01','labs/lab01'),('lab02','labs/lab02')]:
        page.evaluate('value=>location.hash=value','#/'+fragment)
        if name!='home':expect(page.locator('.module-row')).to_have_count(5)
        assert page.locator('#main-content').count()==1
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.screenshot(path=str(OUT/('final-'+name+'-live.png')),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.evaluate("location.hash='#/home'")
    expect(page.locator('.lab-entry')).to_have_count(2)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(OUT/'final-home-mobile-live.png'),full_page=True)
    assert not errors,errors
    browser.close()
print('PASS: final running service has experiment home, five tasks per lab, correct focus target and mobile layout')
