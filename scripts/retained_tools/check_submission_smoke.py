"""Verify a generated submission works after extraction, without the old workspace folders."""
from __future__ import annotations
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys,tempfile,time,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
archives=sorted((ROOT/'outputs/submissions').glob('course-project-lab02-*.zip'),key=lambda p:p.stat().st_mtime)
if not archives:raise SystemExit('No submission archive')
archive=archives[-1]
with tempfile.TemporaryDirectory(prefix='equity_submission_smoke_') as directory:
    temp=Path(directory).resolve()
    assert temp.parent==Path(tempfile.gettempdir()).resolve() and temp.name.startswith('equity_submission_smoke_')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            destination=(temp/name).resolve()
            if not destination.is_relative_to(temp):raise ValueError('Submission path traversal')
        z.extractall(temp)
        manifest=json.loads(z.read('SHA256.json'))
        for name,digest in manifest.items():assert hashlib.sha256((temp/name).read_bytes()).hexdigest()==digest
        assert not any(part in ('.state','node_modules','.venv','人工智能作业一','人工智能作业二') for name in z.namelist() for part in Path(name).parts)
    environment={**os.environ,'PYTHONPATH':str(temp),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'}
    tests=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','tests.lab01.test_project','tests.test_course','tests.test_packaging'],cwd=temp,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=180)
    if tests.returncode:raise AssertionError(tests.stdout+'\n'+tests.stderr)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    with (temp/'smoke-server.log').open('w',encoding='utf-8') as log:
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        process=subprocess.Popen([sys.executable,'-B','-X','utf8','main.py','web','--port',str(port)],cwd=temp,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        try:
            url='http://127.0.0.1:'+str(port)
            for _ in range(100):
                try:
                    with urllib.request.urlopen(url+'/api/health',timeout=1) as r:
                        health=json.loads(r.read());break
                except OSError:
                    if process.poll() is not None:raise AssertionError((temp/'smoke-server.log').read_text(encoding='utf-8'))
                    time.sleep(.2)
            else:raise AssertionError('Extracted server did not start')
            with urllib.request.urlopen(url+'/api/course') as r:catalog=json.loads(r.read())
            assert len(catalog['labs'])==2
            with urllib.request.urlopen(url+'/') as r:html=r.read()
            assert html==(temp/'web/dist/index.html').read_bytes()
            payload=json.dumps({'task':'rules','source':'P1','target':'C3'}).encode()
            req=urllib.request.Request(url+'/api/labs/lab01/run',data=payload,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=20) as r:lab01=json.loads(r.read())
            assert lab01['proved'] and lab01['stats']['total_facts']==135
            with urllib.request.urlopen(url+'/api/cases/base') as r:case=json.loads(r.read())
            payload=json.dumps({'case':case['case'],'completeness':case['completeness'],'task':'uncertain','source':'P1','target':'C3','method':'hybrid_locked','threshold':.5,'model_source':'manual'}).encode()
            req=urllib.request.Request(url+'/api/labs/lab02/run',data=payload,headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=20) as r:lab02=json.loads(r.read())
            assert lab02['bayes']['score']>0 and lab02['fuzzy']['score']>0
            with urllib.request.urlopen(url+'/api/course/files/lab01/report') as r:assert r.read()==(temp/'deliverables/lab01/课堂报告.docx').read_bytes()
            report={'created_utc':datetime.now(timezone.utc).isoformat(),'successful':True,'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'extracted_files_verified':len(manifest),'tests':'11 original lab01 + 18 course + 5 packaging tests','test_exit_code':tests.returncode,'independent_server':'started from extracted submission','old_directories_not_present':True,'lab01_rules':lab01['stats'],'lab02_scores':{'bayes':lab02['bayes']['score'],'fuzzy':lab02['fuzzy']['score']},'static_html_matches':True,'report_bytes_match':True}
            (ROOT/'outputs/qa/submission-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    assert temp.parent==Path(tempfile.gettempdir()).resolve() and temp.name.startswith('equity_submission_smoke_')
