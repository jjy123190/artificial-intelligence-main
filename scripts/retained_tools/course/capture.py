from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).parent
with sync_playwright() as p:
 b=p.chromium.launch(channel='chrome',headless=True)
 page=b.new_page(viewport={'width':1440,'height':900},reduced_motion='reduce')
 for name,hash in [('home','home'),('lab01','labs/lab01'),('lab02','labs/lab02'),('representation','labs/lab01/representation'),('demo','labs/lab01/demo'),('deterministic','labs/lab02/deterministic'),('uncertain','labs/lab02/uncertain'),('optimization','labs/lab02/optimization'),('materials','materials')]:
  page.goto('http://127.0.0.1:8770/#/'+hash);page.locator('.course-header').wait_for();page.wait_for_timeout(1000);page.screenshot(path=str(OUT/(name+'.png')),full_page=True)
 b.close()
print('Captured course pages')
