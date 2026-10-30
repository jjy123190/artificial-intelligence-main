from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/qa'
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(reduced_motion='reduce')
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    for width,height,label in [(1508,768,'desktop'),(390,844,'mobile')]:
        page.set_viewport_size({'width':width,'height':height})
        page.goto('http://127.0.0.1:8765/#/home')
        expect(page.locator('.lab-entry')).to_have_count(2,timeout=30000)
        expect(page.locator('.course-header > div > p')).to_have_text('统一场景：企业股权关系、结构线索与人工复核。')
        assert page.locator('.course-scene').count()==0
        expect(page.locator('.lab-entry').nth(0).locator(':scope > p')).to_contain_text('比较不同表示')
        expect(page.locator('.lab-entry').nth(1).locator(':scope > p')).to_contain_text('量化比较算法优化与混合策略')
        expect(page.locator('.lab-entry').nth(0).locator('.lab-task-preview > span')).to_have_count(5)
        expect(page.locator('.lab-entry').nth(1).locator('.lab-task-preview > span')).to_have_count(5)
        assert page.locator('.course-app-header').evaluate('el=>getComputedStyle(el).position')=='sticky'
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.screenshot(path=str(OUT/('home-scene-simplified-'+label+'.png')),full_page=True)
    page.get_by_role('link',name='进入实验01',exact=True).click()
    expect(page.locator('.module-row')).to_have_count(5)
    expect(page.locator('.course-header > div > p')).to_contain_text('用同一份股权知识')
    assert not errors,errors
    browser.close()
print('PASS: one shared scene introduction; duplicate block removed; lab-specific descriptions, modules and sticky navigation preserved on desktop/mobile')
