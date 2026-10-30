from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import record_lab02_demo as base
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome')
    page = browser.new_page(viewport=base.VIEWPORT, locale='zh-CN')
    page.goto('http://127.0.0.1:8772/#/labs/lab02/deterministic')
    expect(page.locator('.header-status')).to_contain_text('已就绪')
    audit = {'live_results': {}}
    base.run_task(page, 'deterministic', audit)
    base.scroll_to(page, page.locator('.task-result'), 100)
    page.screenshot(path=str(WORK / 'probe-proof.png'))
    base.click(page, page.get_by_role('link', name='推导关系与证明演示', exact=True))
    expect(page.get_by_role('button', name='播放当前结论的证明')).to_be_enabled()
    print('GRAPH', page.url, page.locator('body').inner_text())
    base.scroll_to(page, page.locator('.graph-panel'), 90)
    page.screenshot(path=str(WORK / 'probe-graph.png'))
    base.click(page, page.get_by_role('button', name='播放当前结论的证明'))
    page.wait_for_timeout(2600)
    page.screenshot(path=str(WORK / 'probe-graph-playing.png'))
    page.goto('http://127.0.0.1:8772/#/labs/lab02/uncertain')
    expect(page.locator('.header-status')).to_contain_text('已就绪')
    print('MODELS', page.get_by_label('实验二参数来源').inner_text())
    print('TARGETS', page.get_by_label('实验二目标企业').inner_text())
    page.get_by_label('实验二目标企业').select_option('C3')
    base.run_task(page, 'uncertain', audit)
    base.scroll_to(page, page.locator('.task-result'), 90)
    page.screenshot(path=str(WORK / 'probe-uncertain.png'))
    print('UNCERTAIN',json.dumps(audit['live_results']['uncertain'],ensure_ascii=False)[:2500])
    page.goto('http://127.0.0.1:8772/#/labs/lab02/optimization')
    expect(page.get_by_role('button',name='确定性优化',exact=True)).to_be_visible()
    base.scroll_to(page,page.locator('.experiment-toolbar'),90)
    page.screenshot(path=str(WORK / 'probe-charts.png'))
    base.click(page,page.get_by_role('button',name='确定性优化',exact=True))
    base.scroll_to(page,page.locator('.performance-section'),90)
    page.screenshot(path=str(WORK / 'probe-performance.png'))
    base.click(page,page.get_by_role('button',name='评分与参数优化',exact=True))
    base.scroll_to(page,page.get_by_role('table',name='评分优化与消融实测'),150)
    page.screenshot(path=str(WORK / 'probe-scoring.png'))
    browser.close()
