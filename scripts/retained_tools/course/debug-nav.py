from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 b=p.chromium.launch(channel='chrome',headless=True)
 page=b.new_page(viewport={'width':390,'height':844})
 page.goto('http://127.0.0.1:8770/#/workbench');page.locator('.directory').wait_for()
 print(page.locator('.course-navigation > *').evaluate_all('els=>els.map(el=>({tag:el.tagName,cls:el.className,rect:{x:el.getBoundingClientRect().x,width:el.getBoundingClientRect().width,right:el.getBoundingClientRect().right},styles:{gap:getComputedStyle(el).gap,minWidth:getComputedStyle(el).minWidth,padding:getComputedStyle(el).padding,flex:getComputedStyle(el).flex}}))'))
 print(page.locator('.course-navigation').evaluate('el=>({width:el.clientWidth,gap:getComputedStyle(el).gap,display:getComputedStyle(el).display,padding:getComputedStyle(el).padding})'))
 page.screenshot(path='outputs/qa/course/nav-before.png',full_page=True)
 b.close()
