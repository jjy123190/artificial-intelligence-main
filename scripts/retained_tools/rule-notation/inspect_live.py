import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(viewport={'width':390,'height':844},reduced_motion='reduce')
    page.goto('http://127.0.0.1:8765/#/labs/lab02/modeling/rules')
    expect(page.locator('.rule-row')).to_have_count(27,timeout=30000)
    page.locator('.rule-notation summary').click()
    disclosure=page.locator('.rule-example-disclosure')
    print(json.dumps(disclosure.evaluate("el=>({text:el.textContent,rect:el.getBoundingClientRect().toJSON(),style:{display:getComputedStyle(el).display,visibility:getComputedStyle(el).visibility,opacity:getComputedStyle(el).opacity,color:getComputedStyle(el).color},aside:el.closest('aside').getBoundingClientRect().toJSON()})"),ensure_ascii=False,indent=2))
    expect(disclosure).to_be_visible()
    disclosure.scroll_into_view_if_needed()
    page.screenshot(path=str(Path(__file__).parent/'live-mobile-disclosure.png'))
    page.locator('.rule-notation').screenshot(path=str(Path(__file__).parent/'live-mobile-guide.png'))
    browser.close()
