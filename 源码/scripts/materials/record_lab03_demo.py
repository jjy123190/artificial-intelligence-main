"""Record the actual lab-three website as a 60s 1440p silent demo."""
from pathlib import Path
import json,time,sys,base64,subprocess,hashlib,os,tempfile
import cv2,numpy as np,imageio_ffmpeg
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
if __package__:
    from . import record_lab01_demo as capture_tools
else:
    import record_lab01_demo as capture_tools
WORK=Path(tempfile.gettempdir())/'lab03-demo'
BASE_URL=os.environ.get('LAB03_DEMO_URL','http://127.0.0.1:8765').rstrip('/')
SIZE={'width':2560,'height':1440}


def main():
    work=WORK/time.strftime('%Y%m%d-%H%M%S');work.mkdir(parents=True)
    audit={'shots':[],'runs':[],'errors':[]}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='chrome',headless=True);context=browser.new_context(viewport=SIZE,device_scale_factor=1,locale='zh-CN');page=context.new_page();page.set_default_timeout(15000)
        page.on('pageerror',lambda e:audit['errors'].append(str(e)))
        page.goto(BASE_URL+'/#/labs/lab03/search')
        expect(page.get_by_role('button',name='运行路径搜索',exact=True)).to_be_enabled()
        page.evaluate(capture_tools.PRESENTATION)
        page.locator('.course-navigation > a[href="#/labs/lab03"]').click()
        expect(page.locator('.module-row')).to_have_count(4)
        page.mouse.move(2300,230)
        capture=capture_tools.Capture(context,page,work);capture.start();page.wait_for_timeout(700)
        page.evaluate("document.getElementById('recording-sync').remove()")
        started=time.perf_counter();audit['started_epoch']=time.time()
        def at(s):
            lag=time.perf_counter()-started-s
            if lag>7:raise RuntimeError(f'Timeline late {lag:.2f}s at {s}')
            if lag<0:page.wait_for_timeout(-lag*1000)
        def shot(name):
            page.screenshot(path=str(work/(name+'.png')));audit['shots'].append({'scene':name,'seconds':time.perf_counter()-started})
            print(name,round(time.perf_counter()-started,2),flush=True)
        def click(locator):capture_tools.click(page,locator)
        def scroll(locator,top=180):capture_tools.scroll(page,locator,top)
        def directory():click(page.locator('.course-navigation > a[href="#/labs/lab03"]'))
        def enter(name):click(page.get_by_role('link',name=name,exact=True))
        def run():
            with page.expect_response(lambda r:r.url.endswith('/api/labs/lab03/run') and r.request.method=='POST') as response:
                click(page.get_by_role('button',name='运行路径搜索',exact=True))
            assert response.value.ok,response.value.text()
            result=response.value.json();audit['runs'].append(result)
            expect(page.locator('.task-result')).to_be_visible()
        try:
            shot('01-directory')
            at(2);enter('多跳搜索工作台')
            at(4);run();scroll(page.locator('.task-result'),185);shot('02-real-search')
            at(8);scroll(page.locator('.lab3-playback'),190);click(page.get_by_role('button',name='播放轨迹',exact=True));shot('03-tree-start')
            at(16.5);click(page.get_by_role('button',name='最后一步',exact=True));shot('04-tree-path')
            at(18);directory();enter('遗传优化与路径学习');expect(page.locator('.lab3-curve')).to_be_visible();scroll(page.locator('.lab3-curve'),185);shot('05-ga-curve')
            at(24);directory();enter('LLM辅助与证据约束');expect(page.get_by_text('实测模型：',exact=False)).to_be_visible();scroll(page.locator('.lab3-method-notes'),185);shot('06-llm-protocol')
            at(29);scroll(page.get_by_role('heading',name='实测解释与失败边界',exact=True),185)
            details=page.locator('details').filter(has_text='test-7-0-Q1').first
            click(details.locator('summary'));shot('07-llm-failure')
            at(34);directory();enter('对比实验与失败分析');expect(page.get_by_role('table',name='实验三算法对比结果')).to_be_visible();scroll(page.locator('.lab3-bars'),185);shot('08-comparison')
            at(40);scroll(page.locator('.lab3-filters'),185);capture_tools.choose(page,page.get_by_label('实验三评估场景',exact=True),'ood');scroll(page.locator('.lab3-bars'),185);shot('09-distribution-shift')
            at(46);directory();enter('多跳搜索工作台');scroll(page.locator('.lab3-search-form'),180)
            capture_tools.choose(page,page.get_by_label('实验三问题',exact=True),'test-7-0-Q0')
            capture_tools.choose(page,page.get_by_label('实验三搜索方法',exact=True),'hybrid_llm')
            at(51);run();scroll(page.locator('.task-result'),185);shot('10-real-score-replay')
            at(58);shot('11-finish');at(60.7)
            page.screenshot(path=str(work/'last.png'));page.wait_for_timeout(150)
            audit['recorded_seconds']=time.perf_counter()-started
        finally:
            capture.stop();(work/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');browser.close()
    assert not audit['errors'] and len(audit['runs'])==2
    frames=sorted(json.loads((work/'capture-frames.json').read_text(encoding='utf-8')),key=lambda f:f['timestamp']);seen=False;begin=None
    for i,f in enumerate(frames):
        image=cv2.imdecode(np.fromfile(work/f['file'],dtype=np.uint8),cv2.IMREAD_COLOR);b,g,r=map(int,image[10,10]);marked=r>240 and b>240 and g<20
        if seen and not marked:begin=i;break
        seen|=marked
    assert begin is not None
    origin=frames[begin]['timestamp'];chosen=[f for f in frames[begin:] if f['timestamp']<origin+60]
    lines=['ffconcat version 1.0']
    for i,f in enumerate(chosen):
        end=chosen[i+1]['timestamp'] if i+1<len(chosen) else origin+60
        if end<=f['timestamp']:continue
        lines += [f"file '{f['file']}'",'option framerate 1000',f"duration {end-f['timestamp']:.9f}"]
    lines += [f"file '{chosen[-1]['file']}'",'option framerate 1000']
    (work/'frames.ffconcat').write_text('\n'.join(lines),encoding='utf-8')
    dest=ROOT/'deliverables/lab03/第4组_实验三_Demo_1分钟.mp4'
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    with (work/'encode.log').open('w',encoding='utf-8') as log:
        subprocess.run([ffmpeg,'-hide_banner','-y','-safe','0','-i','frames.ffconcat','-vf','setpts=PTS-STARTPTS,fps=30','-t','60','-frames:v','1800','-an','-c:v','libx264','-preset','medium','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],cwd=work,stderr=log,stdout=subprocess.DEVNULL,check=True)
    result=subprocess.run([ffmpeg,'-v','error','-i',str(dest),'-f','null','-'],capture_output=True,check=True)
    assert not result.stderr
    v=cv2.VideoCapture(str(dest));assert v.get(cv2.CAP_PROP_FRAME_COUNT)==1800 and v.get(cv2.CAP_PROP_FPS)==30 and v.get(cv2.CAP_PROP_FRAME_WIDTH)==2560
    thumbs=[]
    for second in [1,6,12,20,26,32,37,43,54]:
        v.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,im=v.read();assert ok
        thumb=cv2.resize(im,(640,360));cv2.putText(thumb,str(second)+'s',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1);thumbs.append(thumb)
    v.release();sheet=np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0,9,3)]);cv2.imencode('.jpg',sheet)[1].tofile(work/'contact-sheet.jpg')
    print('DELIVERED',dest,'60sec 2560x1440 30fps; contact-sheet:',work/'contact-sheet.jpg',flush=True)

if __name__=='__main__':main()