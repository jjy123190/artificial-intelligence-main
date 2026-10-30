"""Source-only shared case UI and experiment-two result ownership; isolated state."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.check_web import isolated_server, canonical_fragment
from playwright.sync_api import sync_playwright, expect
OUT=ROOT/'outputs'/'qa'/'shared-case-separation'

def run(url):
    OUT.mkdir(parents=True,exist_ok=True)
    checks,errors,screenshots=[],[],[]
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for folder in ('data/raw','deliverables') for path in (ROOT/folder).rglob('*') if path.is_file()}
    def passed(name):checks.append(name);print('PASS '+name,flush=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True)
        context=browser.new_context(viewport={'width':1508,'height':768},accept_downloads=True,reduced_motion='reduce')
        page=context.new_page()
        requests=[]
        page.on('request',lambda request:requests.append(request.url))
        page.on('pageerror',lambda error:errors.append(str(error)))
        def go(fragment,title=None,compat=False):
            page.evaluate('value=>location.hash=value',fragment)
            expect(page).to_have_url(url+'/'+(canonical_fragment(fragment) if compat else fragment))
            if title:expect(page.get_by_role('heading',name=title,exact=True)).to_be_visible(timeout=30000)
        def ready():expect(page.locator('.header-status')).to_contain_text('已就绪',timeout=30000)
        def no_overflow():
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),page.viewport_size
            assert page.locator('#main-content').count()==1
            assert page.locator('#main-content').is_visible()
        def source_only():
            main=page.locator('#main-content')
            assert main.locator('.directory-status,.directory-filter,.decision-status,.score-row,.proof-tree,.proof-player,.engine-time,.workspace-change').count()==0
            expect(main.get_by_role('button',name='运行推理',exact=False)).to_have_count(0)
            expect(main.get_by_role('checkbox',name='推导控制关系',exact=True)).to_have_count(0)
            expect(main.get_by_text('复核建议',exact=True)).to_have_count(0)
            assert not any('推理完成' in value or '计算结果' in value for value in page.locator('.toast').all_text_contents())
            no_overflow()
        def inference_count():return sum(value.endswith('/api/analyze') for value in requests)
        def capture(name,full=True):
            path=OUT/name
            page.mouse.move(0,0)
            page.screenshot(path=str(path),full_page=full)
            screenshots.append(path)
        page.goto(url+'/#/workbench')
        expect(page.locator('.source-directory .directory-row')).to_have_count(12,timeout=30000)
        ready();source_only()
        assert inference_count()==0,requests
        page.locator('.research-menu > summary').click()
        expect(page.locator('.research-menu .menu-link')).to_have_count(2)
        assert page.locator('.research-menu .menu-link').evaluate_all("links=>links.map(link=>link.getAttribute('href'))")==['#/workbench','#/graph']
        capture('public-menu-laptop.png',False)
        page.keyboard.press('Escape');capture('public-overview-laptop.png')
        page.get_by_label('搜索企业或自然人',exact=True).fill('恒通')
        expect(page.locator('.directory-row')).to_have_count(1)
        page.get_by_role('button',name='清空目录搜索',exact=True).click()
        page.get_by_role('link',name='查看C3资料',exact=True).click()
        expect(page).to_have_url(url+'/#/case/C3')
        expect(page.get_by_role('table',name='实体原始关系记录')).to_contain_text('35%')
        source_only();capture('public-entity-laptop.png')
        page.get_by_label('选择实体',exact=True).select_option('P1')
        expect(page).to_have_url(url+'/#/case/P1')
        expect(page.get_by_role('table',name='实体原始关系记录')).to_contain_text('80%')
        source_only()
        go('#/graph','关系图谱')
        expect(page.locator('.graph-canvas')).to_be_visible(timeout=30000)
        source_only()
        assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.nodes().length')==16
        assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.edges().length')==25
        page.get_by_label('选择实体',exact=True).select_option('C6')
        expect(page).to_have_url(url+'/#/graph')
        assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.nodes().length')==16
        count=page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.edges().length')
        page.get_by_role('button',name='持股',exact=True).click()
        assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.edges().length')<count
        page.get_by_role('button',name='持股',exact=True).click()
        before=page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.zoom()')
        page.locator('.graph-canvas').focus();page.keyboard.press('+')
        assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.zoom()')>before
        page.keyboard.press('0')
        capture('public-graph-laptop.png')
        assert inference_count()==0,requests
        passed('shared menu, searchable directory, company/person records and graph show source data only and never call experiment-two inference')

        go('#/workbench','案例概览')
        case=context.request.get(url+'/api/cases/base').json()['case']
        adjusted=json.loads(json.dumps(case))
        adjusted['label']='原始案例导入验收'
        next(row for row in adjusted['holdings'] if row['holder']=='C2' and row['target']=='C3')['ratio']=.25
        with page.expect_response(lambda response:response.url.endswith('/api/cases/validate')) as response:
            page.get_by_label('导入股权图JSON',exact=True).set_input_files({'name':'source.json','mimeType':'application/json','buffer':json.dumps({'case':adjusted,'target_id':'C3'},ensure_ascii=False).encode('utf-8')})
        assert response.value.ok,response.value.text()
        assert set(response.value.json())=={'id','case','completeness'}
        ready();source_only()
        expect(page.locator('.case-summary h2')).to_have_text('原始案例导入验收')
        go('#/case/C3','实体资料')
        expect(page.get_by_role('table',name='实体原始关系记录')).to_contain_text('25%')
        assert inference_count()==0,requests
        go('#/workbench','案例概览')
        invalid=json.loads(json.dumps(adjusted));invalid['holdings'][0]['target']='MISSING'
        with page.expect_response(lambda response:response.url.endswith('/api/cases/validate')) as response:
            page.get_by_label('导入股权图JSON',exact=True).set_input_files({'name':'invalid.json','mimeType':'application/json','buffer':json.dumps({'case':invalid}).encode('utf-8')})
        assert response.value.status==422
        expect(page.locator('.error-banner')).to_be_visible()
        expect(page.locator('.case-summary h2')).to_have_text('原始案例导入验收')
        page.get_by_role('button',name='关闭错误提示',exact=True).click()
        page.locator('#case-manager > summary').click()
        with page.expect_response(lambda response:response.url.endswith('/api/cases/base')):
            page.get_by_label('选择案例',exact=True).select_option('base')
        ready();expect(page.locator('.directory-row')).to_have_count(12)
        assert inference_count()==0,requests
        passed('public JSON import only validates source data; invalid references preserve the accepted case and changing cases does not infer')

        go('#/labs/lab01/demo','知识增强与推理演示')
        expect(page.get_by_role('button',name='运行实验一任务',exact=True)).to_be_enabled(timeout=30000)
        with page.expect_response(lambda response:response.url.endswith('/api/labs/lab01/run') and response.request.method=='POST') as response:
            page.get_by_role('button',name='运行实验一任务',exact=True).click()
        result=response.value.json()
        assert result['stats']['rules']==15 and result['stats']['total_facts']==135
        assert inference_count()==0,requests
        go('#/workbench','案例概览');source_only()
        passed('experiment one uses its original 15 rules and 135 facts without running or displaying experiment-two analysis')

        go('#/labs/lab02/modeling','知识表示与建模')
        expect(page.get_by_role('button',name='构建当前案例的谓词事实',exact=True)).to_be_enabled()
        with page.expect_response(lambda response:response.url.endswith('/api/labs/lab02/run') and response.request.method=='POST') as response:
            page.get_by_role('button',name='构建当前案例的谓词事实',exact=True).click()
        assert response.value.ok
        assert inference_count()==0,requests
        go('#/labs/lab02/hybrid','混合推理与创新验证')
        page.get_by_role('link',name='查看全案例分析与复核结果',exact=True).click()
        expect(page).to_have_url(url+'/#/labs/lab02/hybrid/overview')
        expect(page.locator('.directory-row')).to_have_count(12,timeout=30000)
        expect(page.locator('.workspace-context')).to_contain_text('实验二结果',timeout=30000)
        assert inference_count()==1,requests
        expect(page.locator('.directory-columns')).to_contain_text('复核建议')
        expect(page.locator('.course-navigation > a[href="#/labs/lab02"]')).to_have_attribute('aria-current','location')
        assert not page.locator('.research-menu').evaluate("el=>el.classList.contains('is-active')")
        expect(page.locator('.page-breadcrumb')).to_contain_text('混合推理与创新验证')
        capture('lab02-analysis-laptop.png')
        page.get_by_role('link',name='查看C3分析',exact=True).click()
        expect(page).to_have_url(url+'/#/labs/lab02/hybrid/entity/C3')
        expect(page.locator('.decision-status')).to_be_visible()
        page.get_by_role('link',name='查看确定性推理证据',exact=True).click()
        expect(page).to_have_url(url+'/#/labs/lab02/deterministic/entity/C3/evidence')
        expect(page.locator('.proof-tree')).to_be_visible()
        page.locator('.proof-node-label').first.click()
        page.get_by_role('link',name='在关系图谱中查看证明',exact=True).click()
        expect(page).to_have_url(url+'/#/labs/lab02/deterministic/graph/C3')
        expect(page.locator('.proof-player')).to_be_visible()
        expect(page.get_by_role('checkbox',name='推导控制关系',exact=True)).to_be_enabled()
        expect(page.locator('.page-breadcrumb')).to_contain_text('确定性推理')
        capture('lab02-proof-graph-laptop.png')
        go('#/graph','关系图谱');source_only()
        expect(page.locator('.graph-canvas')).to_be_visible()
        assert page.locator('.graph-canvas').evaluate('el=>el._cyreg.cy.edges().length')==25
        assert page.locator('.graph-canvas').evaluate("el=>el._cyreg.cy.edges('[kind=\"control\"]').length")==0
        assert page.locator('.graph-canvas').evaluate("el=>el._cyreg.cy.nodes('.evidence').length")==0
        passed('analysis, scores, review decisions, evidence and derived-control playback are scoped to experiment two; returning to public graph removes all proof data')

        go('#/labs/lab02/modeling/entity/C3/edit','调整企业数据')
        page.locator('#holding-value').fill('25')
        go('#/case/C3','实体资料')
        expect(page.get_by_role('table',name='实体原始关系记录')).to_contain_text('25%')
        source_only()
        expect(page.locator('.draft-state')).to_be_visible()
        page.locator('#case-manager > summary').click()
        page.get_by_label('选择案例',exact=True).select_option('extended')
        dialog=page.get_by_role('dialog')
        expect(dialog).to_be_visible()
        expect(dialog.get_by_role('button',name='保留草稿',exact=True)).to_be_focused()
        page.keyboard.press('Escape')
        expect(page.get_by_role('table',name='实体原始关系记录')).to_contain_text('25%')
        go('#/labs/lab02/hybrid/entity/C3','企业分析')
        expect(page.locator('.workspace-change')).to_be_visible()
        expect(page.locator('.decision-compare small')).to_have_count(2)
        go('#/labs/lab02/modeling/entity/C3/edit','调整企业数据')
        expect(page.locator('#holding-value')).to_have_value('25')
        passed('accepted input edits appear in public source records while experiment-two snapshots stay marked stale; cancelling case replacement preserves the draft')

        for width,height,label in ((1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow'),(720,500,'zoom200')):
            page.set_viewport_size({'width':width,'height':height})
            for fragment,title,stem in (('#/workbench','案例概览','overview'),('#/case/C3','实体资料','entity'),('#/graph','关系图谱','graph'),('#/labs/lab02/hybrid/overview','全案例分析','analysis')):
                go(fragment,title)
                if stem=='graph':expect(page.locator('.graph-canvas')).to_be_visible()
                if stem!='analysis':source_only()
                no_overflow()
                if label in ('mobile','narrow'):capture('split-'+stem+'-'+label+'.png')
            page.evaluate('scrollTo(0,document.documentElement.scrollHeight)')
            page.wait_for_function("Math.abs(document.querySelector('.course-app-header').getBoundingClientRect().top)<=1")
            page.locator('.research-menu > summary').click()
            expect(page.locator('.research-menu .menu-link')).to_have_count(2)
            no_overflow()
            box=page.locator('.research-menu .action-menu-panel').bounding_box()
            assert box and box['x']>=-1 and box['x']+box['width']<=width+1,box
            page.keyboard.press('Escape')
        passed('source and result pages stay readable at desktop, tablet, mobile, 320px and zoomed layouts with one main focus target and sticky navigation')
        go('#/entity/C3/evidence','推理证据',compat=True)
        expect(page.locator('.page-breadcrumb')).to_contain_text('确定性推理')
        go('#/settings','分析设置',compat=True)
        expect(page.locator('.page-breadcrumb')).to_contain_text('混合推理与创新验证')
        go('#/rules','规则知识库',compat=True)
        expect(page.locator('.course-breadcrumb')).to_contain_text('知识表示与建模')
        go('#/case/MISSING')
        expect(page.get_by_role('heading',name='当前案例没有这个实体',exact=True)).to_be_visible()
        passed('previous links resolve to experiment-two canonical pages and missing source entities recover safely')

        other=context.new_page()
        other.on('pageerror',lambda error:errors.append(str(error)))
        other.goto(url+'/#/workbench')
        expect(other.locator('.source-directory')).to_be_visible(timeout=30000)
        expect(other.locator('.header-status')).to_contain_text('已就绪',timeout=30000)
        other.route('**/api/analyze',lambda route:route.abort('failed'))
        other.evaluate("location.hash='#/labs/lab02/hybrid/overview'")
        expect(other.locator('.error-banner')).to_contain_text('无法连接本地引擎',timeout=30000)
        expect(other.get_by_role('heading',name='尚无实验二分析结果',exact=True)).to_be_visible()
        assert other.locator('.directory').count()==0
        other.evaluate("location.hash='#/workbench'")
        expect(other.locator('.source-directory .directory-row')).to_have_count(12)
        assert other.locator('.error-banner').count()==0
        expect(other.locator('.header-status')).to_contain_text('已就绪')
        other.keyboard.press('Control+Enter')
        assert other.locator('.error-banner').count()==0
        other.unroute('**/api/analyze')
        other.evaluate("location.hash='#/labs/lab02/hybrid/overview'")
        expect(other.get_by_role('button',name='运行实验二分析',exact=True)).to_be_enabled()
        with other.expect_response(lambda response:response.url.endswith('/api/analyze')) as response:
            other.get_by_role('button',name='运行实验二分析',exact=True).click()
        assert response.value.ok
        expect(other.locator('.directory-row')).to_have_count(12,timeout=30000)
        passed('first experiment-two inference failure has an explicit retry and does not contaminate or block shared source browsing')
        assert not errors,errors
        browser.close()
    after={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for folder in ('data/raw','deliverables') for path in (ROOT/folder).rglob('*') if path.is_file()}
    assert hashes==after,'Source data or local deliverables changed'
    passed('all raw data and retained report/PPT/video files remain byte-for-byte unchanged')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':url,'isolated':True,'checks':checks,'unexpected_browser_errors':errors,'source_materials_sha256':after,'screenshots':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in screenshots}}
    (OUT/'browser-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(len(checks))+' shared-case separation groups passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8765');parser.add_argument('--isolated',action='store_true');args=parser.parse_args()
    if not args.url.startswith(('http://127.0.0.1:','http://localhost:')):parser.error('Only local loopback servers are supported')
    if not args.isolated:parser.error('This regression edits in-memory drafts and must use --isolated')
    with isolated_server() as url:run(url)
