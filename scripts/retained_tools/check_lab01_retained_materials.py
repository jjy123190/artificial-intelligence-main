from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/qa'
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(viewport={'width':1508,'height':768},accept_downloads=True)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    url='http://127.0.0.1:8765'
    catalog=page.request.get(url+'/api/course').json()
    lab01=next(lab for lab in catalog['labs'] if lab['id']=='lab01')
    expected={'report','slides','presentation-video','benchmark'}
    assert {file['id'] for file in lab01['files']}==expected,lab01['files']
    assert all(file['available'] for file in lab01['files'])
    kept={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'deliverables/lab01').iterdir() if path.is_file()}
    assert set(kept)=={'课堂报告.docx','企业股权知识表示与风险推理.pptx','企业股权知识表示与风险推理.mp4'}
    for fragment in ('#/labs/lab01/deliverables','#/materials'):
        page.goto(url+'/'+fragment)
        section=page.get_by_label('实验01交付材料',exact=True)
        expect(section).to_be_visible(timeout=30000)
        expect(section.locator('.file-row')).to_have_count(4)
        expect(section).not_to_contain_text('待完成')
        for removed in ('系统运行演示视频','系统演示字幕','PPT讲解音频','PPT讲解字幕','原提交说明','PPT讲稿'):
            expect(section).not_to_contain_text(removed)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        if fragment=='#/materials':
            lab02=page.get_by_label('实验02交付材料',exact=True)
            expect(lab02).to_contain_text('待完成')
    for file in lab01['files']:
        response=page.request.get(url+file['href'])
        assert response.ok,response.text()
        assert response.body()==(ROOT/file['path']).read_bytes()
    assert page.request.get(url+'/api/course/files/lab01/submission').status==404
    assert page.request.get(url+'/api/course/files/lab01/presentation-audio').status==404
    section=page.get_by_label('实验01交付材料',exact=True)
    section.screenshot(path=str(OUT/'lab01-retained-materials.png'))
    assert not errors,errors
    browser.close()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'successful':True,'retained_submission_files':list(kept),'catalog_entries':[file['id'] for file in lab01['files']],'checks':['experiment-one task page and shared materials page show existing files only','deleted submissions have no pending placeholders','retained downloads match local bytes','missing-file downloads return 404','lab02 pending checklist unchanged'],'browser_errors':errors,'material_sha256':kept}
(OUT/'lab01-retained-materials-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: retained lab01 materials, both website entries, downloads and unchanged lab02 checklist')
