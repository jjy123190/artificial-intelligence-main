"""Record the real current lab03 UI, starting at the user's three-hop workbench.

Only a disposable browser tab receives presentation zoom and a visible cursor.
No application code, data, stored results, authentication, or network responses
are changed. All runs call the current workspace's actual localhost API.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from playwright.sync_api import sync_playwright, expect

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
URL = 'http://127.0.0.1:8766'
SIZE = {'width': 2560, 'height': 1440}
EXPECTED_ID = hashlib.sha256(str(ROOT).casefold().encode('utf-8')).hexdigest()[:16]
spec = importlib.util.spec_from_file_location('recording_capture', ROOT/'scripts/materials/record_lab01_demo.py')
cap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cap
spec.loader.exec_module(cap)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', default='take_01')
    args = parser.parse_args()
    folder = BASE/args.take
    folder.mkdir(exist_ok=False)
    audit = {'date':datetime.now().astimezone().isoformat(), 'base_url':URL,
             'expected_workspace_id':EXPECTED_ID, 'viewport':SIZE, 'presentation_zoom':1.6,
             'start':'/#/labs/lab03/search; shared-3, hybrid, budget 16, no previous run',
             'scope':'Lab03 search, comparison, GA/ML, LLM assistance and evaluation; not labs 1/2.',
             'operations':[], 'shots':[], 'runs':[], 'downloads':[], 'errors':[], 'http_errors':[]}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True,
                                     args=['--disable-background-timer-throttling','--disable-renderer-backgrounding'])
        context = browser.new_context(viewport=SIZE, device_scale_factor=1, locale='zh-CN', color_scheme='dark', accept_downloads=True)
        context.set_default_timeout(8000)
        health = context.request.get(URL+'/api/health').json()
        assert health['workspace_id']==EXPECTED_ID,health
        audit['health']=health
        served=context.request.get(URL+'/').body()
        assert served==(ROOT/'web/dist/index.html').read_bytes(),'Served build is not the current workspace'
        audit['served_index_sha256']=hashlib.sha256(served).hexdigest()
        page=context.new_page()
        page.on('pageerror',lambda error:audit['errors'].append(str(error)))
        start=[None]
        def response_log(response):
            if response.status>=400:audit['http_errors'].append({'url':response.url,'status':response.status})
            if response.url.endswith('/api/labs/lab03/run') and response.request.method=='POST':
                try:
                    audit['runs'].append({'at':None if start[0] is None else time.perf_counter()-start[0],
                                          'status':response.status,'request':response.request.post_data_json,'body':response.json()})
                except Exception as exc:audit['errors'].append('Response audit: '+str(exc))
        page.on('response',response_log)
        page.goto(URL+'/#/labs/lab03/search',wait_until='networkidle')
        expect(page.get_by_role('button',name='运行路径搜索',exact=True)).to_be_enabled()
        # Visit normal read-only modules before the take so initial data-loading
        # placeholders do not consume the user's one-minute demonstration.
        # React keeps each visited module's unmodified snapshot in memory.
        audit['prewarmed_modules'] = []
        for title, ready_heading in [
            ('遗传优化与路径学习','用验证图学习更好的搜索顺序'),
            ('LLM辅助与证据约束','模型评估行动，搜索程序产生答案'),
            ('对比实验与失败分析','比较成功率，也比较真实搜索开销'),
        ]:
            page.get_by_role('link',name=title,exact=True).click()
            expect(page.get_by_role('heading',name=ready_heading,exact=True)).to_be_visible()
            page.wait_for_load_state('networkidle')
            audit['prewarmed_modules'].append(title)
        page.get_by_role('link',name='多跳搜索工作台',exact=True).click()
        page.evaluate('window.scrollTo({top:0,behavior:"instant"})')
        expect(page.get_by_role('button',name='运行路径搜索',exact=True)).to_be_enabled()
        expect(page.get_by_label('实验三问题',exact=True)).to_have_value('shared-3')
        expect(page.get_by_label('实验三搜索方法',exact=True)).to_have_value('hybrid')
        expect(page.get_by_label('实验三扩展预算',exact=True)).to_have_value('16')
        assert page.get_by_role('button',name='导出本次结果',exact=True).count()==0
        page.evaluate('document.fonts.ready')
        page.evaluate(cap.PRESENTATION)
        page.mouse.move(1990,260)
        page.wait_for_timeout(500)
        capture=cap.Capture(context,page,folder)
        capture.start()
        page.wait_for_timeout(700)
        page.evaluate("document.getElementById('recording-sync').remove()")
        start[0]=time.perf_counter();audit['started_epoch']=time.time()

        def at(second):
            lag=time.perf_counter()-start[0]-second
            if lag>2.5:raise RuntimeError(f'Timeline late {lag:.3f}s at {second}s')
            if lag<0:page.wait_for_timeout(-lag*1000)

        def mark(name):
            row={'name':name,'seconds':round(time.perf_counter()-start[0],3),'url':page.url}
            audit['operations'].append(row)
            print(f'{row["seconds"]:6.2f}s {name}',flush=True)

        def shot(name):
            page.screenshot(path=str(folder/f'{name}.png'))
            audit['shots'].append({'name':name,'seconds':round(time.perf_counter()-start[0],3),'url':page.url})

        def move(locator):
            expect(locator).to_be_visible()
            box=locator.bounding_box();assert box
            if box['y']<88 or box['y']+box['height']>1430:
                scroll(locator)
                box=locator.bounding_box();assert box
            x,y=box['x']+box['width']/2,box['y']+box['height']/2
            origin=page.evaluate('window.recordingPointer || {x:1990,y:260}')
            for n in range(1,8):
                f=n/7;f=f*f*(3-2*f)
                page.mouse.move(origin['x']+(x-origin['x'])*f,origin['y']+(y-origin['y'])*f)
                page.wait_for_timeout(5)

        def click(locator):
            move(locator);locator.click();page.wait_for_timeout(65)

        def choose(label,value):
            locator=page.get_by_label(label,exact=True)
            move(locator);locator.focus();page.wait_for_timeout(70);locator.select_option(str(value));page.wait_for_timeout(100)

        def scroll(locator,top=190):
            locator.evaluate("(e,t)=>window.scrollTo({top:Math.max(0,e.getBoundingClientRect().top+scrollY-t),behavior:'smooth'})",top)
            page.wait_for_timeout(530)

        def section(title):
            click(page.get_by_role('link',name=title,exact=True))
            if title == '多跳搜索工作台':
                expect(page.get_by_role('button',name='运行路径搜索',exact=True)).to_be_enabled()
            else:
                heading = {'遗传优化与路径学习':'用验证图学习更好的搜索顺序',
                           'LLM辅助与证据约束':'模型评估行动，搜索程序产生答案',
                           '对比实验与失败分析':'比较成功率，也比较真实搜索开销'}[title]
                expect(page.get_by_role('heading',name=heading,exact=True)).to_be_visible()
            page.wait_for_timeout(100)

        def run():
            with page.expect_response(lambda r:r.url.endswith('/api/labs/lab03/run') and r.request.method=='POST') as result:
                click(page.get_by_role('button',name='运行路径搜索',exact=True))
            assert result.value.ok,result.value.text()
            body=result.value.json();assert body['result']['valid'],body['result']
            expect(page.get_by_role('button',name='导出本次结果',exact=True)).to_be_visible()
            return body

        def download(name):
            with page.expect_download() as event:click(page.get_by_role('button',name=name,exact=True))
            item=event.value
            target=folder/'downloads'/item.suggested_filename;target.parent.mkdir(exist_ok=True)
            item.save_as(str(target));data=json.loads(target.read_text(encoding='utf-8-sig'))
            audit['downloads'].append({'button':name,'file':target.relative_to(folder).as_posix(),'seconds':time.perf_counter()-start[0],
                                       'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'keys':list(data)})

        completed=False
        try:
            mark('01 原始三跳工作台：尚未运行');shot('01-start')
            at(1.3);click(page.get_by_role('button',name='2 跳示例',exact=True));mark('切换二跳问题建模')
            at(2.0);click(page.get_by_role('button',name='4 跳示例',exact=True));mark('切换四跳问题建模')
            at(2.8);click(page.get_by_role('button',name='3 跳示例',exact=True))
            expect(page.get_by_label('实验三问题',exact=True)).to_have_value('shared-3')
            at(3.4);click(page.get_by_text('状态、动作与终止条件',exact=True));mark('查看状态/动作/终止与专家约束')
            at(4.1);click(page.get_by_text('状态、动作与终止条件',exact=True))
            at(4.6);result=run();mark('三跳混合搜索返回真实可核验路径')
            at(5.5);choose('轨迹播放速度','300');click(page.get_by_role('button',name='播放轨迹',exact=True));mark('2倍速逐步播放搜索树')
            at(9.8);shot('02-search-tree')  # Playback reaches the final frame naturally.
            at(10.3);click(page.get_by_role('button',name='回看远航贸易有限公司，第 4 步',exact=True));mark('点击节点检查已证无解分支')
            at(11.0);click(page.get_by_role('button',name='最后一步',exact=True))
            at(11.4);click(page.get_by_role('button',name='步骤列表',exact=True));mark('切换步骤列表')
            at(12.2);click(page.get_by_role('button',name='搜索树',exact=True))
            at(12.6);download('导出本次结果');mark('导出真实搜索结果JSON')
            at(13.1);scroll(page.locator('.lab3-path-answer'));mark('查看路径解释、成本与逐边证据');shot('03-evidence')
            at(15.0);scroll(page.locator('.lab3-comparison'));click(page.get_by_role('button',name='对比四种算法',exact=True))
            expect(page.get_by_role('button',name='导出算法对比',exact=True)).to_be_visible();mark('同题实测BFS/固定A*/GA A*/MCTS');shot('04-comparison')
            at(19.3);download('导出算法对比');mark('导出四算法对比')

            at(20.0);section('遗传优化与路径学习')
            at(20.5);choose('遗传优化随机种子','19');mark('02 GA验证适应度曲线与种子切换');shot('05-ga')
            at(22.1);click(page.get_by_text('查看每一代的真实测量值',exact=True));scroll(page.get_by_role('table',name='GA逐代测量',exact=True));mark('查看逐代真实适应度与验证结果')
            at(23.9);scroll(page.get_by_role('heading',name='路径价值分类器与专家约束',exact=True));mark('03 路径分类器指标与专家规则');shot('06-learning')

            at(26.3);section('LLM辅助与证据约束');mark('04 LLM三项职责与同题配对结果');shot('07-llm-overview')
            at(28.9);click(page.get_by_text('新图与数值挑战：查看未改善与退化',exact=True));scroll(page.get_by_text('新图与数值挑战：查看未改善与退化',exact=True));mark('新图确认与数值挑战：如实展示局限')
            at(31.3);click(page.get_by_text('test-7-0-Q1 · 查看模型解释',exact=True));scroll(page.get_by_role('heading',name='实测解释与失败边界',exact=True));mark('对比模型幻觉解释与程序证据');shot('08-llm-failure')
            at(34.0);click(page.get_by_text('接入自己的远程模型',exact=True));scroll(page.get_by_text('接入自己的远程模型',exact=True));mark('展示可选远程模型接入说明，不调用接口')
            at(35.2);section('多跳搜索工作台');choose('实验三问题','test-7-0-Q0');choose('实验三搜索方法','hybrid_llm');choose('实验三扩展预算','8')
            result=run();assert result['llm']['mode']=='replay';mark('混合+LLM：真实历史评分重放（不是现场推断）')
            scroll(page.locator('.lab3-path-answer'));shot('09-llm-replay');mark('查看重放标识、专家拦截与答案证据')

            at(40.4);section('对比实验与失败分析');mark('05 全量3888次实验：成功率与搜索成本');shot('10-test')
            at(43.0);choose('实验三评估场景','ood');mark('切换分布变化图，检视泛化')
            at(44.8);choose('实验三对比预算','16');mark('切换扩展预算')
            at(46.0);choose('实验三对比预算','8');scroll(page.get_by_role('table',name='实验三算法对比结果',exact=True));mark('四项指标、边检查与无效答案率')
            at(48.3);scroll(page.get_by_role('heading',name='GA与学习引导是否真的有收益？',exact=True));mark('配对收益区间与过拟合边界')
            at(50.5);click(page.get_by_text('查看混合搜索的已证无解剪枝对比',exact=True));scroll(page.get_by_role('heading',name='无解问题：区分证明穷尽与预算耗尽',exact=True));mark('查看无解确认与证明剪枝')
            at(52.8);failure_section=page.get_by_role('heading',name='失败案例与改进边界',exact=True).locator('..')
            click(failure_section.locator('summary').nth(0));click(failure_section.locator('summary').nth(1));scroll(page.get_by_role('heading',name='失败案例与改进边界',exact=True));mark('06 失败案例：证据无效与预算耗尽');shot('11-failures')
            at(56.2);scroll(page.get_by_role('heading',name='比较成功率，也比较真实搜索开销',exact=True))
            at(57.1);download('导出当前评估');mark('导出当前评估数据')
            at(58.0);choose('实验三评估场景','test');mark('回到全量测试结果完成演示');shot('12-finish')
            at(60.7);page.screenshot(path=str(folder/'capture-final.png'));page.wait_for_timeout(120)
            audit['recorded_seconds']=time.perf_counter()-start[0]
            assert not audit['errors'] and not audit['http_errors'],audit
            assert len(audit['runs'])==6,len(audit['runs'])
            assert len(audit['downloads'])==3
            completed=True
        finally:
            capture.stop();audit['completed']=completed;write(folder/'audit.json',audit)
            browser.close()
    assert completed
    print('CAPTURE COMPLETE',folder,flush=True)


if __name__=='__main__':main()
