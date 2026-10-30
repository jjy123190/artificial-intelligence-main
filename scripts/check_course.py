"""Requirement-oriented course browser acceptance; uses isolated local state."""
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
OUTPUT=ROOT/'outputs'/'qa'/'course'

def run(url):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    checks,errors,captured_screenshots=[],[],[]
    material_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'deliverables').rglob('*') if path.is_file()}
    def passed(name):checks.append(name);print('PASS '+name,flush=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=True)
        context=browser.new_context(viewport={'width':1440,'height':900},accept_downloads=True,reduced_motion='reduce')
        page=context.new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        def go(fragment,title=None):
            fragment=canonical_fragment(fragment)
            page.evaluate('fragment=>location.hash=fragment',fragment)
            expect(page).to_have_url(url+'/'+fragment)
            if title:expect(page.get_by_role('heading',name=title,exact=True)).to_be_visible(timeout=30000)
        def run_task(endpoint,action):
            with page.expect_response(lambda r:r.url.endswith(endpoint) and r.request.method=='POST',timeout=30000) as response:action()
            assert response.value.ok,response.value.text()
            expect(page.locator('.task-result')).to_be_visible(timeout=20000)
            return response.value.json()
        def capture(name):
            destination=OUTPUT/name
            page.screenshot(path=str(destination),full_page=True)
            captured_screenshots.append(destination)
        def modeling_navigation():
            expect(page.locator('#main-content .task-links')).to_have_count(1)
            expect(page.locator('#main-content a[href="#/workbench"],#main-content a[href="#/graph"]')).to_have_count(0)
            expect(page.get_by_role('link',name='浏览与选择共享案例',exact=True)).to_have_count(0)
            expect(page.get_by_role('link',name='查看当前关系图谱',exact=True)).to_have_count(0)
            expect(page.get_by_role('link',name='查看完整规则库',exact=True)).to_have_count(0)
            expect(page.locator('#main-content a[href="#/labs/lab02/modeling/rules"]')).to_have_count(0)
            expect(page.locator('.task-bottom-nav a')).to_have_count(1)
            expect(page.get_by_role('button',name='调整目标企业输入',exact=True)).to_be_visible()
            expect(page.get_by_label('实验二建模编辑对象',exact=True)).to_be_visible()
            expect(page.get_by_role('link',name='扩展：证据与版本更新',exact=True)).to_have_attribute('href','#/labs/lab02/modeling/governance')
            expect(page.get_by_role('link',name='返回实验二任务目录',exact=True)).to_have_attribute('href','#/labs/lab02')
            expect(page.locator('.research-menu a[href="#/workbench"]')).to_have_count(1)
            expect(page.locator('.research-menu a[href="#/graph"]')).to_have_count(1)
            expect(page.get_by_role('button',name='构建当前案例的谓词事实',exact=True)).to_be_visible()
            expect(page.get_by_text('这些是有限知识库中的教学规则。',exact=False)).to_have_count(0)
            toggle=page.get_by_role('button',name='展开一阶规则知识库',exact=True)
            expect(toggle).to_have_attribute('aria-expanded','false')
            expect(page.locator('.rules-content')).to_be_hidden()
            expect(page.get_by_role('heading',name='一阶规则知识库',exact=True)).to_be_visible()
            expect(page.locator('.rules-header-actions .data-badge')).to_contain_text('27 条结构规则')
            toggle.focus();page.keyboard.press('Enter')
            expect(page.locator('.rules-content')).to_be_visible()
            toggle=page.get_by_role('button',name='收起一阶规则知识库',exact=True)
            expect(toggle).to_have_attribute('aria-expanded','true')
            toggle.focus();page.keyboard.press('Space')
            expect(page.locator('.rules-content')).to_be_hidden()
            expect(page.get_by_role('button',name='展开一阶规则知识库',exact=True)).to_have_attribute('aria-expanded','false')
        def experiment_only():
            if page.url.endswith('/#/labs/lab02/modeling'):
                modeling_navigation()
            if "#/labs/lab01/" in page.url and page.locator("#main-content.task-page").count():
                expect(page.locator("#main-content .course-header .requirement-label")).to_have_count(0)
                expect(page.locator("#main-content .task-requirement")).to_be_visible()
            expect(page.locator('.code-location,.lab-goal')).to_have_count(0)
            expect(page.get_by_text('代码位置',exact=False)).to_have_count(0)
            expect(page.get_by_text('每一部分都标明对应要求',exact=False)).to_have_count(0)
            expect(page.locator('.lab-entry code')).to_have_count(0)
            assert not any('experiments/' in text for text in page.locator('.lab-entry').all_text_contents())
            assert page.locator('a[href="#/materials"],a[href="#/labs/lab01/deliverables"]').count()==0
            assert page.locator('.course-files,.materials-lab,.submission-notes,.file-row').count()==0
            expect(page.get_by_text('成果与提交',exact=False)).to_have_count(0)
            expect(page.get_by_text('提交清单',exact=False)).to_have_count(0)
        def no_overflow():
            experiment_only()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),page.viewport_size
            assert page.locator('#main-content').count()==1
            assert page.locator('#main-content').is_visible()
        page.goto(url)
        expect(page.get_by_role('heading',name='股权知识与智能推理实验平台',exact=True)).to_be_visible(timeout=30000)
        expect(page.locator('.lab-entry')).to_have_count(2)
        assert page.locator('.directory,.inspector,.graph-canvas').count()==0
        assert page.locator('.app-footer').count()==0
        no_overflow();capture('home-desktop.png')
        passed('total home is experiment selection, not company scoring; two explicit lab entries')
        page.get_by_role('link',name='进入实验01',exact=True).click()
        expect(page.locator('.module-row')).to_have_count(4)
        expect(page.locator('.module-row').first).to_contain_text('核心任务（1）')
        page.get_by_role('link',name='知识表示与方法比较',exact=True).click()
        expect(page.locator('.knowledge-canvas')).to_be_visible(timeout=30000)
        expect(page.locator('.task-metrics')).to_contain_text('25 条原始关系')
        assert page.locator('.knowledge-canvas').evaluate("el=>el._cyreg.cy.nodes().length")==16
        page.get_by_role('button',name='产生式规则',exact=True).click()
        expect(page.get_by_role('table',name='实验一产生式规则').locator('tbody tr')).to_have_count(15)
        page.get_by_role('button',name='框架表示',exact=True).click()
        expect(page.get_by_role('table',name='实验一框架槽值')).to_contain_text('继承／缺省')
        expect(page.get_by_role('table',name='实验一框架槽值')).to_contain_text('必须公告披露')
        page.get_by_role('button',name='知识图谱',exact=True).click()
        page.get_by_label('实验一图谱关系',exact=True).select_option('all')
        assert page.locator('.knowledge-canvas').evaluate("el=>el._cyreg.cy.edges().length")==25
        page.get_by_label('实验一图谱实体',exact=True).select_option('P1')
        page.get_by_role('button',name='放大实验一图谱',exact=True).click()
        page.get_by_role('button',name='适配实验一图谱',exact=True).click()
        passed('lab01 exposes graph triples, original 15 rules, inherited frames and keyboard-compatible graph controls')
        go('#/labs/lab01/demo','知识增强与推理演示')
        expect(page.get_by_role('button',name='运行实验一任务',exact=True)).to_be_enabled(timeout=30000)
        result=run_task('/api/labs/lab01/run',lambda:page.get_by_role('button',name='运行实验一任务',exact=True).click())
        assert result['proved'] and result['stats']['total_facts']==135 and result['stats']['added_relations']==9
        expect(page.locator('.task-result')).to_contain_text('53 → 135')
        page.get_by_label('实验一查询主体',exact=True).select_option('P3')
        expect(page.locator('.task-result')).not_to_be_visible()
        negative=run_task('/api/labs/lab01/run',lambda:page.get_by_role('button',name='运行实验一任务',exact=True).click())
        assert not negative['proved'] and negative['proof']==[]
        go('#/labs/lab01')
        go('#/labs/lab01/demo','知识增强与推理演示')
        expect(page.get_by_label('实验一查询主体',exact=True)).to_have_value('P3')
        expect(page.locator('.task-result')).to_contain_text('暂不能证明')
        passed('lab01 real rule proof and negative query; 53-to-135 facts; input/result survives leaving the module')
        page.get_by_label('实验一演示任务',exact=True).select_option('penetration')
        page.get_by_label('实验一查询目标',exact=True).select_option('C3')
        equity=run_task('/api/labs/lab01/run',lambda:page.get_by_role('button',name='运行实验一任务',exact=True).click())
        assert abs(next(r for r in equity['owners'] if r['owner']=='P1')['ratio']-.342)<1e-12
        expect(page.get_by_role('table',name='实验一股权穿透结果')).to_contain_text('34.2000%')
        page.get_by_label('实验一穿透深度',exact=True).select_option('1')
        cutoff=run_task('/api/labs/lab01/run',lambda:page.get_by_role('button',name='运行实验一任务',exact=True).click())
        assert not any(r['beneficial'] for r in cutoff['owners'])
        page.get_by_label('实验一演示任务',exact=True).select_option('path')
        page.get_by_label('实验一查询主体',exact=True).select_option('C3');page.get_by_label('实验一查询目标',exact=True).select_option('C9')
        path=run_task('/api/labs/lab01/run',lambda:page.get_by_role('button',name='运行实验一任务',exact=True).click())
        assert path['path'][0][0]=='C3' and path['path'][-1][2]=='C9'
        page.get_by_label('实验一查询目标',exact=True).select_option('P1')
        page.get_by_label('实验一演示任务',exact=True).select_option('rules')
        expect(page.get_by_label('实验一查询目标',exact=True)).not_to_have_value('P1')
        passed('lab01 simple-path equity, depth cutoff and shortest relation path are actual computations, not lab02 replacements')
        go('#/labs/lab01/industry','行业案例与落地流程')
        expect(page.locator('.editorial-section')).to_contain_text('没有导入')
        expect(page.locator('.industry-process > div')).to_have_count(5)
        go('#/labs/lab01/limitations','局限与发展方向')
        expect(page.locator('.limits-list')).to_contain_text('简单路径')
        experiment_only()
        go('#/labs/lab01')
        expect(page.locator('.module-row')).to_have_count(4)
        expect(page.locator('.module-row').last).to_contain_text('局限与发展方向')
        no_overflow()
        passed('lab01 has four core experiment entries, industry process and honest limitations, without a submission task')
        go('#/labs/lab02')
        expect(page.locator('.module-row')).to_have_count(5)
        expect(page.locator('#main-content .lab-secondary')).to_have_count(0)
        expect(page.get_by_role('link',name='浏览共享案例数据',exact=True)).to_have_count(0)
        expect(page.get_by_role('link',name='扩展：证据与版本更新',exact=True)).to_have_count(0)
        passed('lab02 directory shows only five core tasks, without duplicate auxiliary links')
        page.locator('.research-menu > summary').click()
        expect(page.locator('.research-menu a[href="#/workbench"]')).to_be_visible()
        page.locator('.research-menu a[href="#/workbench"]').click()
        expect(page.get_by_role('heading',name='案例概览',exact=True)).to_be_visible(timeout=30000)
        page.locator('.research-menu > summary').click()
        page.locator('.research-menu a[href="#/graph"]').click()
        expect(page.get_by_role('heading',name='关系图谱',exact=True)).to_be_visible(timeout=30000)
        expect(page.locator('.graph-canvas')).to_be_visible(timeout=30000)
        go('#/labs/lab02')
        expect(page.locator('.module-row')).to_have_count(5)
        passed('shared case overview and graph remain accessible through the public-tools menu')
        page.get_by_role('link',name='知识表示与建模',exact=True).click()
        expect(page.locator('.rule-row')).to_have_count(27,timeout=30000)
        modeling_navigation()
        passed('lab02 modeling uses local inputs, inline rules and one evidence extension without duplicate destinations')
        expect(page.get_by_role('link',name='扩展：证据与版本更新',exact=True)).to_have_attribute('href','#/labs/lab02/modeling/governance')
        with page.expect_response(lambda r:r.url.endswith('/api/labs/lab02/run') and r.request.method=='POST',timeout=30000) as response:
            page.get_by_role('button',name='构建当前案例的谓词事实',exact=True).click()
        assert response.value.ok,response.value.text()
        expect(page.get_by_role('table',name='实验二一阶谓词事实')).to_contain_text('Holds(')
        passed('lab02 modeling builds sourced first-order facts and uses its own 27-rule knowledge base')
        go('#/labs/lab02/deterministic','确定性推理')
        proved=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert proved['proved'] and proved['equivalent']
        expect(page.locator('.task-details').filter(has_text='正向证明依据').first).to_be_visible()
        page.get_by_label('实验二推理方法',exact=True).select_option('backward')
        expect(page.locator('.workspace-change')).to_be_visible()
        expect(page.locator('.task-details').filter(has_text='正向证明依据').first).to_be_visible()
        backward=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert backward['proved'] and backward['backward_proof']
        expect(page.locator('.task-details').filter(has_text='反向证明路径').first).to_be_visible()
        page.get_by_label('实验二查询主体',exact=True).select_option('P3')
        unknown=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert not unknown['proved']
        page.get_by_label('实验二推理方法',exact=True).select_option('resolution')
        resolution=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert resolution['proved'] and resolution['status']=='proved' and resolution['proof']
        passed('lab02 forward/backward goals, no-proof case and independent resolution run; stale proof keeps old-method label')
        go('#/labs/lab02/uncertain','不确定性推理')
        score=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert score['bayes']['score']>0 and score['fuzzy']['score']>0
        expect(page.locator('.score-comparison')).to_contain_text('Bayes')
        expect(page.locator('.score-comparison')).to_contain_text('Mamdani')
        go('#/entity/C3/edit','调整企业数据')
        page.locator('#holding-value').fill('25')
        go('#/labs/lab02/uncertain','不确定性推理')
        expect(page.locator('.workspace-change')).to_be_visible()
        changed=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert changed['observation']['values']['related_trade'] == 0.0
        assert changed['task']=='uncertain'
        go('#/entity/C3/edit','调整企业数据');expect(page.locator('#holding-value')).to_have_value('25')
        passed('Bayes/Mamdani explain same indicators; accepted workspace edits persist and mark old task result stale')
        go('#/settings','分析设置');page.get_by_label('登记与测度资料',exact=True).uncheck()
        go('#/labs/lab02/uncertain','不确定性推理')
        partial=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert partial['observation']['values']['opacity'] is None
        expect(page.get_by_role('table',name='实验二输入指标')).to_contain_text('未知／未完整采集')
        passed('incomplete registry data reaches uncertain method as unknown instead of zero')
        go('#/settings','分析设置');page.get_by_label('登记与测度资料',exact=True).check()
        go('#/entity/C3/edit','调整企业数据');page.locator('#holding-value').fill('35')
        go('#/labs/lab02/hybrid','混合推理与创新验证')
        hybrid=run_task('/api/labs/lab02/run',lambda:page.get_by_role('button',name='运行本节实验',exact=True).click())
        assert len(hybrid['decisions'])==3 and len({r['score'] for r in hybrid['decisions']})==1
        assert all(r['proofs'] for r in hybrid['decisions'])
        expect(page.get_by_role('table',name='实验二混合策略对照').locator('tbody tr')).to_have_count(3)
        go('#/labs/lab02');go('#/labs/lab02/hybrid','混合推理与创新验证')
        expect(page.get_by_role('table',name='实验二混合策略对照')).to_be_visible()
        passed('three hybrid strategies preserve scores and proof sources; results survive task-directory navigation')
        go('#/labs/lab02/optimization','算法优化与对比实验')
        expect(page.locator('[aria-label="各方法的实测指标"] tbody tr')).to_have_count(11,timeout=30000)
        page.get_by_label('观测条件',exact=True).select_option('severe')
        assert page.locator('.recharts-bar-rectangle').count()>0
        page.get_by_role('button',name='确定性优化',exact=True).click()
        expect(page.locator('[aria-label="确定性优化实测"] tbody tr')).to_have_count(3)
        expect(page.get_by_role('table',name='目标切片实测').locator('tbody tr')).to_have_count(4)
        expect(page.get_by_role('table',name='归结搜索优化实测').locator('tbody tr')).to_have_count(2)
        page.get_by_role('button',name='评分与参数优化',exact=True).click()
        expect(page.get_by_role('table',name='评分优化与消融实测').locator('tbody tr')).to_have_count(14)
        page.get_by_text('查看先验、CPT与模糊参数敏感性原始实测',exact=True).click()
        expect(page.get_by_role('table',name='参数敏感性实测').locator('tbody tr')).to_have_count(45)
        page.get_by_role('button',name='扩展：证据冲突',exact=True).click()
        expect(page.get_by_role('table',name='证据冲突扩展实测').locator('tbody tr')).to_have_count(24)
        passed('optimization is a primary task and separates graph decisions, deterministic optimization, scoring/ablation and source conflicts')
        go('#/home','股权知识与智能推理实验平台')
        expect(page.locator('.lab-entry').nth(0)).to_contain_text('4 个任务')
        expect(page.locator('.lab-entry').nth(1)).to_contain_text('5 个任务')
        expect(page.locator('.course-utilities a')).to_have_count(1)
        page.locator('.research-menu > summary').click()
        expect(page.locator('.research-menu[open]')).to_be_visible()
        expect(page.get_by_role('link',name='案例概览',exact=True)).to_be_visible()
        experiment_only()
        page.keyboard.press('Escape')
        assert page.locator('.app-footer').count()==0
        passed('home and public menu show only experiment tools; no submission checklist or removed footer')
        for fragment in ('#/materials','#/labs/lab01/deliverables','#/labs/lab99','#/labs/__proto__','#/labs/constructor/demo','#/labs/lab01/nope','#/labs/lab02/hybrid/extra'):
            go(fragment,'页面未找到');no_overflow()
        page.get_by_role('link',name='返回案例概览',exact=True).last.click()
        expect(page.locator('.directory')).to_be_visible()
        passed('retired submission routes and strict invalid routes recover safely, including prototype-like path names')
        go('#/labs/lab01/demo','知识增强与推理演示')
        page.get_by_label('实验一查询主体',exact=True).select_option('P1');page.get_by_label('实验一查询目标',exact=True).select_option('C3')
        before=len(errors)
        page.route('**/api/labs/lab01/run',lambda route:route.abort('failed'))
        page.get_by_role('button',name='运行实验一任务',exact=True).click()
        expect(page.locator('.error-banner')).to_contain_text('无法连接本地引擎')
        page.unroute('**/api/labs/lab01/run')
        run_task('/api/labs/lab01/run',lambda:page.get_by_role('button',name='重试',exact=True).click())
        del errors[before:]
        passed('new task failure is visible and retry computes again without changing source data')
        for width,height,name in ((1508,768,'laptop'),(900,1000,'tablet'),(390,844,'mobile'),(320,740,'narrow'),(720,500,'zoom200')):
            page.set_viewport_size({'width':width,'height':height})
            for fragment,heading,stem in (('#/home','股权知识与智能推理实验平台','home'),('#/labs/lab01',None,'lab01'),('#/labs/lab02',None,'lab02'),('#/labs/lab01/representation','知识表示与方法比较','representation'),('#/labs/lab01/demo','知识增强与推理演示','demo'),('#/labs/lab02/modeling','知识表示与建模','modeling'),('#/labs/lab02/deterministic','确定性推理','deterministic'),('#/labs/lab02/uncertain','不确定性推理','uncertain'),('#/labs/lab02/optimization','算法优化与对比实验','optimization'),('#/labs/lab02/hybrid','混合推理与创新验证','hybrid')):
                go(fragment,heading)
                if stem in ('lab01','lab02'):expect(page.locator('.module-row')).to_have_count(4 if stem=='lab01' else 5)
                no_overflow();capture(stem+'-'+name+'.png')
            page.evaluate('scrollTo(0,document.documentElement.scrollHeight)')
            expect(page.locator('.course-app-header')).to_have_css('position','sticky')
            page.wait_for_function("Math.abs(document.querySelector('.course-app-header').getBoundingClientRect().top)<=1")
            geometry=page.locator('.course-app-header').bounding_box()
            assert geometry is not None and abs(geometry['y'])<=1,geometry
            page.locator('.research-menu > summary').click()
            expect(page.locator('.research-menu[open]')).to_be_visible()
            menu=page.locator('.research-menu .action-menu-panel').bounding_box()
            assert menu is not None and menu['y']>=geometry['height']-1,menu
            no_overflow()
            page.keyboard.press('Escape')
            page.evaluate('scrollTo(0,0)')
            passed(name+' course pages have one main focus target, no horizontal overflow and a sticky top navigation')
        page.set_viewport_size({'width':1440,'height':900});go('#/home')
        page.locator('.skip-link').focus();page.keyboard.press('Enter');expect(page.locator('#main-content')).to_be_focused()
        page.get_by_role('link',name='进入实验01',exact=True).focus();page.keyboard.press('Enter')
        expect(page.locator('.module-row')).to_have_count(4)
        assert page.evaluate("matchMedia('(prefers-reduced-motion:reduce)').matches")
        passed('keyboard skip-to-content and real experiment links work with reduced-motion mode')
        assert material_hashes=={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'deliverables').rglob('*') if path.is_file()}
        passed('all local deliverable files remain present and byte-for-byte unchanged')
        assert not errors,errors
        browser.close()
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'url':url,'checks':checks,'unexpected_browser_errors':errors,'isolated':True,'local_materials_sha256':material_hashes,'screenshots':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in captured_screenshots}}
    (OUTPUT/'browser-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(len(checks))+' course browser groups passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8765');parser.add_argument('--isolated',action='store_true');args=parser.parse_args()
    if not args.url.startswith(('http://127.0.0.1:','http://localhost:')):parser.error('Only local loopback servers are supported')
    if args.isolated:
        with isolated_server() as url:run(url)
    else:run(args.url.rstrip('/'))
