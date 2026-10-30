"""Read-only final migration inventory: source files, immutable materials and persistent rows."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sqlite3,struct
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'outputs/qa/migration-files.json'
plan=json.loads(PLAN.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
errors=[]
checked=[]
by_source={row['source']:row for row in plan['records']}
for row in plan['records']:
    source=(ROOT/row['source']).resolve();target=(ROOT/row['target']).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not target.is_relative_to(ROOT.resolve()):errors.append('path-outside-workspace:'+row['source']);continue
    if not source.is_file() or not target.is_file():errors.append('missing-file:'+row['source']);continue
    source_hash,target_hash=sha(source),sha(target)
    if source_hash!=row['sha256']:errors.append('source-changed:'+row['source'])
    immutable=row['target'].startswith(('data/raw/','deliverables/')) or row['target'].startswith('outputs/lab01/benchmark')
    if immutable and source_hash!=target_hash:errors.append('immutable-file-mismatch:'+row['target'])
    checked.append({**row,'current_sha256':target_hash,'unchanged_bytes':source_hash==target_hash,'immutable':immutable})
ignored=[]
for old_name in ('人工智能作业一','人工智能作业二'):
    old=ROOT/old_name
    for directory,dirs,files in os.walk(old,followlinks=False):
        directory=Path(directory);relative=directory.relative_to(old)
        for name in list(dirs):
            child=directory/name
            if child.is_symlink():errors.append('unexpected-symlink:'+str(child));dirs.remove(name);continue
            if name in ('.venv','venv','node_modules','__pycache__','.pytest_cache') or relative.parts==('web',) and name=='dist' or relative.parts==('output',) and name=='web':
                ignored.append({'path':child.relative_to(ROOT).as_posix(),'reason':'installed dependencies/cache/rebuildable UI QA and runtime'});dirs.remove(name)
        for name in files:
            p=directory/name;key=p.relative_to(ROOT).as_posix()
            if key in by_source:continue
            if key=='人工智能作业二/.state/governance.sqlite3':ignored.append({'path':key,'reason':'persistent rows separately migrated and checked'});continue
            if name.endswith(('.pyc','.tsbuildinfo','.tmp')) or name in ('Thumbs.db','.DS_Store'):
                ignored.append({'path':key,'reason':'generated cache'});continue
            if old_name=='人工智能作业二' and relative.parts==() and name in ('启动网站.cmd','停止网站.cmd'):
                ignored.append({'path':key,'reason':'replaced by validated unified root launchers'});continue
            errors.append('unmapped-source-file:'+key)

def database_rows(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        tables=[row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {table:connection.execute('SELECT * FROM "'+table+'" ORDER BY 1,2').fetchall() for table in tables}
old_rows=database_rows(ROOT/'人工智能作业二/.state/governance.sqlite3')
new_rows=database_rows(ROOT/'.state/governance.sqlite3')
if old_rows!=new_rows:errors.append('persistent-state-rows-differ')

def video_duration(path):
    with path.open('rb') as f:
        total=path.stat().st_size
        def atoms(start,end):
            pos=start
            while pos+8<=end:
                f.seek(pos);size,kind=struct.unpack('>I4s',f.read(8));header=8
                if size==1:size=struct.unpack('>Q',f.read(8))[0];header=16
                if size==0:size=end-pos
                if size<header or pos+size>end:raise ValueError('Invalid MP4 atom')
                yield kind,pos+header,pos+size
                pos+=size
        for kind,start,end in atoms(0,total):
            if kind!=b'moov':continue
            for child,a,b in atoms(start,end):
                if child==b'mvhd':
                    f.seek(a);header=f.read(min(40,b-a));version=header[0]
                    scale=int.from_bytes(header[20:24] if version else header[12:16],'big')
                    duration=int.from_bytes(header[24:32] if version else header[16:20],'big')
                    return duration/scale
    raise ValueError('MP4 movie header missing')
media={}
for p in (ROOT/'deliverables/lab01').glob('*.mp4'):media[p.name]={'duration_seconds':video_duration(p),'sha256':sha(p),'note':'container duration only; material copied byte-for-byte'}
lecture=media.get('企业股权知识表示与风险推理.mp4',{})
if lecture.get('duration_seconds',float('inf'))>180:errors.append('presentation-video-longer-than-180s')
report={'created_utc':datetime.now(timezone.utc).isoformat(),'successful':not errors,'ready_for_old_directory_deletion':not errors,'checked_sources':len(checked),'files':checked,'ignored_sources':ignored,'errors':errors,'persistent_state':{'same_rows':old_rows==new_rows,'row_counts':{t:len(v) for t,v in new_rows.items()},'integrity':'ok'},'media':media}
(ROOT/'outputs/qa/migration-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'successful':report['successful'],'checked_sources':len(checked),'immutable_files':sum(r['immutable'] for r in checked),'errors':errors,'persistent_state':report['persistent_state'],'media':media},ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
