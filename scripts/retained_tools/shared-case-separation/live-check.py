"""Read-only live UI acceptance. Uses in-memory inference only; never edits/saves case data."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
state=json.loads((root/'outputs/runtime/server.json').read_text(encoding='utf-8-sig'))
url=state['url']
assert url.startswith('http://127.0.0.1:') and state['running']
checks,errors,captures=[],[],[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    context=browser.new_context(viewport={'width':1508,'height':768},reduced_motion='reduce')
    page=context.new_page()
    page.on('pageerror',lambda error:errors.append(str(error)))
    requests=[]
    page.on('request',lambda request:requests.append(request.url))
    def go(fragment,title):
        page.evaluate('value=>location.hash=value',fragment)
        expect(page.get_by_role('heading',name=title,exact=True)).to_be_visible(timeout=30000)
    def capture(name,full=False):
        target=out/name
        page.mouse.move(0,0)
        page.screenshot(path=str(target),full_page=full)
        captures.append(target)
    def source():
        main=page.locator('#main-content')
        assert main.locator('.directory-filter,.directory-status,.score-row,.decision-status,.proof-tree,.proof-player,.engine-time,.workspace-change').count()==0
        expect(main.get_by_role('button',name='运行推理',exact=False)).to_have_count(0)
        expect(main.get_by_role('checkbox',name='推导控制关系',exact=True)).to_have_count(0)
        assert not any('推理完成' in value or '计算结果' in value for value in page.locator('.toast').all_text_contents())
        assert page.locator('#main-content').count()==1
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.goto(url+'/#/home')
    expect(page.locator('.lab-entry')).to_have_count(2,timeout=30000)
    expect(page.locator('.lab-entry').first).to_contain_text('4 个任务')
    expect(page.locator('.lab-entry').last).to_contain_text('5 个任务')
    expect(page.locator('.lab-entry code')).to_have_count(0)
    go('#/workbench','案例概览')
    expect(page.locator('.source-directory .directory-row')).to_have_count(12)
    expect(page.locator('.workspace-context')).to_contain_text('当前输入数据')
    source();capture('live-public-overview-laptop.png')
    page.locator('.research-menu > summary').click()
    expect(page.locator('.research-menu .menu-link')).to_have_count(2)
    assert page.locator('.research-menu .menu-link').evaluate_all("links=>links.map(link=>link.getAttribute('href'))")==['#/workbench','#/graph']
    capture('live-public-menu-laptop.png')
    page.keyboard.press('Escape')
    page.get_by_role('link',name='查看C3资料',exact=True).click()
    expect(page).to_have_url(url+'/#/case/C3')
    expect(page.get_by_role('table',name='实体原始关系记录')).to_contain_text('35%')
    source();capture('live-public-records-laptop.png')
    go('#/graph','关系图谱')
    expect(page.locator('.graph-canvas')).to_be_visible(timeout=30000)
    assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.nodes().length')==16
    assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.edges().length')==25
    source();capture('live-public-graph-laptop.png')
    assert not any(value.endswith('/api/analyze') for value in requests),requests
    checks.append('live home and public directory/records/graph contain no inference results or global analyze calls')
    go('#/labs/lab02/hybrid','混合推理与创新验证')
    page.get_by_role('link',name='查看全案例分析与复核结果',exact=True).click()
    expect(page).to_have_url(url+'/#/labs/lab02/hybrid/overview')
    expect(page.locator('.directory-row')).to_have_count(12,timeout=30000)
    expect(page.locator('.workspace-context')).to_contain_text('实验二结果',timeout=30000)
    expect(page.locator('.directory-columns')).to_contain_text('复核建议')
    expect(page.locator('.page-breadcrumb')).to_contain_text('混合推理与创新验证')
    capture('live-lab02-analysis-laptop.png')
    page.get_by_role('link',name='查看C3分析',exact=True).click()
    expect(page).to_have_url(url+'/#/labs/lab02/hybrid/entity/C3')
    page.get_by_role('link',name='推理证据',exact=True).click()
    expect(page).to_have_url(url+'/#/labs/lab02/deterministic/entity/C3/evidence')
    expect(page.locator('.proof-tree')).to_be_visible()
    page.get_by_role('link',name='在关系图谱中查看证明',exact=True).click()
    expect(page).to_have_url(url+'/#/labs/lab02/deterministic/graph/C3')
    expect(page.locator('.proof-player')).to_be_visible()
    expect(page.get_by_role('checkbox',name='推导控制关系',exact=True)).to_be_enabled()
    checks.append('live result and proof entries are placed in experiment-two hybrid/deterministic modules')
    for width,height,label in ((390,844,'mobile'),(320,740,'narrow')):
        page.set_viewport_size({'width':width,'height':height})
        for fragment,title,stem in (('#/workbench','案例概览','directory'),('#/case/C3','实体资料','records'),('#/graph','关系图谱','graph')):
            go(fragment,title)
            if stem=='graph':expect(page.locator('.graph-canvas')).to_be_visible()
            source();capture('live-public-'+stem+'-'+label+'.png',True)
        page.evaluate('scrollTo(0,document.documentElement.scrollHeight)')
        page.wait_for_function("Math.abs(document.querySelector('.course-app-header').getBoundingClientRect().top)<=1")
        page.locator('.research-menu > summary').click()
        expect(page.locator('.research-menu .menu-link')).to_have_count(2)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        page.keyboard.press('Escape')
        checks.append(label+' live source-only views and sticky menu fit without horizontal overflow')
    assert not errors,errors
    browser.close()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':url,'checks':checks,'read_only':True,'unexpected_browser_errors':errors,'screenshots':{file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in captures}}
(out/'live-browser-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for check in checks:print('PASS '+check)
