"""Re-record the unchanged submitted application; all working files stay here."""
from pathlib import Path
import sys, json, time, hashlib, subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright, expect

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import record_lab02_demo as b
b.OUTPUT = WORK
URL = 'http://127.0.0.1:8772'

POINTER = r"""() => {
 const style=document.createElement('style');
 style.textContent='*{cursor:none!important} #demo-pointer{position:fixed;left:0;top:0;width:27px;height:32px;z-index:2147483647;pointer-events:none;filter:drop-shadow(1px 2px 2px #0006)} .demo-click{position:fixed;width:42px;height:42px;margin:-21px;border:3px solid #ab4d26;border-radius:50%;pointer-events:none;z-index:2147483646;animation:demo-ring .6s ease-out forwards} @keyframes demo-ring{from{transform:scale(.35);opacity:.95}to{transform:scale(1.2);opacity:0}}';
 document.head.append(style);
 const pointer=document.createElement('div');pointer.id='demo-pointer';
 pointer.innerHTML='<svg viewBox="0 0 27 32"><path d="M3 2 L3 25 L9 19 L14 29 L18 27 L13 18 L23 18 Z" fill="white" stroke="#3b2c24" stroke-width="1.6" stroke-linejoin="round"/></svg>';
 document.body.append(pointer);
 document.addEventListener('mousemove',e=>{window.demoPointerPosition={x:e.clientX,y:e.clientY};pointer.style.transform='translate('+e.clientX+'px,'+e.clientY+'px)'});
 document.addEventListener('mousedown',e=>{const ring=document.createElement('div');ring.className='demo-click';ring.style.left=e.clientX+'px';ring.style.top=e.clientY+'px';document.body.append(ring);setTimeout(()=>ring.remove(),650)});
 const marker=document.createElement('div');marker.id='demo-sync';marker.style.cssText='position:fixed;left:0;top:0;width:24px;height:24px;background:rgb(255,0,255);z-index:2147483647';document.body.append(marker);
}"""

def choose(page, locator, value=None, label=None):
    b.move_to(page, locator)
    locator.focus()
    page.wait_for_timeout(160)
    locator.select_option(value=value) if value is not None else locator.select_option(label=label)
    page.wait_for_timeout(200)

def record():
    audit = {'source_session':'01a177dd-359c-7c12-b4c6-d8c9554bb65e',
             'url':URL,'live_results':{},'runs':[], 'shots':[],
             'browser_errors':[],'http_errors':[], 'composition':{
                 'website_fills_frame':True, 'audio':False, 'subtitles':False,
                 'footer_bar':False,'added_titles':False,'visible_pointer':True}}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome')
        context = browser.new_context(viewport=b.VIEWPORT, locale='zh-CN',
            device_scale_factor=1, record_video_dir=str(WORK/'raw'), record_video_size=b.VIEWPORT)
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on('pageerror', lambda e:audit['browser_errors'].append(str(e)))
        page.on('response',lambda r:audit['http_errors'].append({'status':r.status,'url':r.url}) if r.status>=400 else None)
        # Warm the real saved experiment response before the visible recording.
        page.goto(URL+'/#/labs/lab02/optimization')
        expect(page.get_by_role('button',name='确定性优化',exact=True)).to_be_visible()
        expect(page.locator('.chart-grid')).to_be_visible()
        page.locator('.course-navigation > a[href="#/labs/lab02"]').click()
        expect(page.locator('.module-row')).to_have_count(5)
        expect(page.locator('.header-status')).to_contain_text('已就绪')
        page.evaluate(POINTER)
        page.mouse.move(1430,145)
        page.wait_for_timeout(1000)
        page.evaluate("document.getElementById('demo-sync').remove()")
        started=time.perf_counter()
        def at(t): b.wait_until(page,started,t)
        def shot(name): b.capture(page,name,started,audit)
        def run(module,label):
            b.run_task(page,module,audit)
            audit['runs'].append({'label':label,'at_seconds':round(time.perf_counter()-started,3),
                                  'result':audit['live_results'][module]})
        shot('01-overview')
        at(2.5); b.enter(page,'知识表示与建模')
        at(4.0); run('modeling','谓词事实与真实来源')
        at(5.0); b.scroll_to(page,page.get_by_text('一阶谓词与来源（47 条）',exact=True),120)
        shot('02-predicate-facts')
        at(7.0)
        b.click(page,page.get_by_text('一阶谓词与来源（47 条）',exact=True))
        b.click(page,page.get_by_role('button',name='展开一阶规则知识库',exact=True))
        b.scroll_to(page,page.get_by_role('heading',name='一阶规则知识库',exact=True),95)
        shot('03-rule-library')

        at(11.0); b.directory(page); b.enter(page,'确定性推理')
        at(12.5); run('deterministic','正向推理 Controls(P1,C3)')
        b.scroll_to(page,page.locator('.task-result'),100); shot('04-forward-proof')
        at(17.0); b.scroll_to(page,page.locator('.lab-task-form'),135)
        choose(page,page.get_by_label('实验二推理方法'),'backward')
        run('deterministic','表格化反向推理 Controls(P1,C3)')
        b.scroll_to(page,page.locator('.task-result'),100); shot('05-backward-proof')
        at(22.5)
        b.click(page,page.get_by_role('link',name='推导关系与证明演示',exact=True))
        expect(page.get_by_role('button',name='播放当前结论的证明')).to_be_enabled()
        b.scroll_to(page,page.locator('.graph-panel'),90)
        at(25.3); b.click(page,page.get_by_role('button',name='播放当前结论的证明'))
        at(28.0); shot('06-proof-graph-inputs')
        at(32.5); shot('07-proof-graph-rules')
        at(37.2)
        expect(page.locator('.step-counter')).to_have_text('10 / 10')
        shot('08-proof-graph-complete')

        at(38.0); b.directory(page); b.enter(page,'不确定性推理')
        at(40.0); run('uncertain','教学预设 Bayes 与 Mamdani')
        b.scroll_to(page,page.locator('.task-result'),90); shot('09-manual-scores')
        at(44.0); b.scroll_to(page,page.locator('.lab-task-form'),130)
        choose(page,page.get_by_label('实验二参数来源'),label='图级训练 · 7 · 自动模糊拟合')
        run('uncertain','训练参数 Bayes 与自动拟合 Mamdani')
        assert audit['live_results']['uncertain']['fuzzy']['parameters']['training_count']>0
        b.scroll_to(page,page.locator('.task-result'),90); shot('10-trained-scores')

        at(49.5); b.directory(page); b.enter(page,'算法优化与对比实验')
        expect(page.locator('.chart-grid')).to_be_visible()
        at(51.0); b.scroll_to(page,page.locator('.experiment-toolbar'),90); shot('11-clean-comparison')
        at(54.0); choose(page,page.get_by_label('观测条件',exact=True),'value_noise'); shot('12-noise-comparison')
        at(56.5); b.click(page,page.get_by_role('button',name='确定性优化',exact=True))
        b.scroll_to(page,page.locator('.performance-section'),90)
        expect(page.get_by_role('region',name='确定性优化实测').locator('tbody tr')).to_have_count(3)
        shot('13-runtime-and-search-cost')
        at(61.0); b.click(page,page.get_by_role('button',name='评分与参数优化',exact=True))
        b.scroll_to(page,page.get_by_role('heading',name='评分算法优化与消融'),90); shot('14-scoring-ablation')
        at(65.0); choose(page,page.get_by_label('评分实验观测条件'),'noise_010'); shot('15-noisy-scoring-ablation')

        at(68.0); b.directory(page); b.enter(page,'混合推理与创新验证')
        at(70.0); choose(page,page.get_by_label('实验二参数来源'),'manual')
        run('hybrid','同分数的三种结构约束与校准策略')
        b.scroll_to(page,page.locator('.task-result'),110); shot('16-three-hybrid-strategies')
        assert len({x['score'] for x in audit['live_results']['hybrid']['decisions']})==1
        at(74.0); b.move_to(page,page.get_by_role('table',name='实验二混合策略对照').get_by_text('保留结构约束',exact=True))
        at(78.0); b.click(page,page.get_by_role('link',name='查看目标企业分析',exact=True))
        expect(page.get_by_role('heading',name='企业分析',exact=True)).to_be_visible()
        b.scroll_to(page,page.get_by_role('heading',name='恒通科技股份有限公司',exact=True),140)
        at(81.0); shot('17-company-analysis')
        at(84.0); b.move_to(page,page.get_by_text('结构约束对照',exact=True))
        at(87.0); shot('18-final-evidence-and-action')
        at(91.0)
        video=page.video
        context.close()
        raw=Path(video.path())
        browser.close()
    audit['raw_video']=str(raw)
    (WORK/'recording-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if audit['browser_errors'] or audit['http_errors']: raise RuntimeError('Browser or HTTP errors')
    return raw,audit

def render(raw,audit):
    start=b.find_start(raw)
    destination=WORK/'demo-90s.mp4'
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    command=[ffmpeg,'-hide_banner','-y','-ss',f'{start:.3f}','-i',str(raw),
        '-vf','fps=30,scale=1920:1080:flags=lanczos','-t','90','-an',
        '-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart','-metadata','title=实验二 90秒网站实际操作演示',str(destination)]
    result=subprocess.run(command,capture_output=True,encoding='utf-8',errors='replace')
    (WORK/'render.log').write_text(result.stderr,encoding='utf-8'); result.check_returncode()
    cap=cv2.VideoCapture(str(destination))
    width,height,frames,fps=[cap.get(k) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FPS]]
    assert (width,height,frames,fps)==(1920,1080,2700,30),(width,height,frames,fps)
    thumbs=[]
    samples=[1,6,10,15.5,21,29,34,37.5,42,47.5,52.5,55.5,59.5,63.5,66.5,73,82.5,88.5]
    for second in samples:
        cap.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,frame=cap.read();assert ok
        ok,png=cv2.imencode('.png',frame);assert ok;png.tofile(str(WORK/f'final-{second:04.1f}s.png'))
        thumb=cv2.resize(frame,(640,360));cv2.putText(thumb,f'{second:04.1f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2);thumbs.append(thumb)
    cap.release()
    sheet=np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0,len(thumbs),3)])
    cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(WORK/'contact-sheet.jpg'))
    result=subprocess.run([ffmpeg,'-hide_banner','-i',str(destination),'-f','null','-'],capture_output=True,encoding='utf-8',errors='replace')
    (WORK/'decode-check.log').write_text(result.stderr,encoding='utf-8');result.check_returncode()
    assert 'Audio:' not in result.stderr and 'Subtitle:' not in result.stderr
    audit['trim_start_seconds']=start
    audit['validation']={'width':int(width),'height':int(height),'fps':fps,'frames':int(frames),
        'duration_seconds':frames/fps,'audio_streams':0,'subtitle_streams':0,'full_decode_passed':True,
        'size_bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
    (WORK/'recording-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit['validation']),flush=True)

if __name__=='__main__':
    if '--render-only' in sys.argv:
        audit=json.loads((WORK/'recording-audit.json').read_text(encoding='utf-8'));render(Path(audit['raw_video']),audit)
    else:
        raw,audit=record();render(raw,audit)
