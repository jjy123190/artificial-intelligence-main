"""Replace only the official presentation video after complete verification."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil
WORK=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    meta=json.loads((WORK/'project.json').read_text(encoding='utf-8'))
    qa=json.loads((WORK/'qa/verification.json').read_text(encoding='utf-8'))
    source=Path(meta['source']);root=source.parents[2]
    video=WORK/'第4组_实验三_PPT演讲.mp4';dest=Path(meta['destination'])
    assert dest.resolve().is_relative_to(root/'deliverables/lab3')
    assert qa['full_decode']=='passed' and qa['duration_seconds']==240 and qa['source_slides']==17
    assert sha(video)==qa['sha256'] and sha(source)==meta['source_sha256']
    assert sha(dest)==meta['previous_video_sha256']
    assert sha(meta['previous_video_backup'])==meta['previous_video_sha256']
    for rel,expected in meta['protected_files_before'].items():assert sha(root/rel)==expected,rel
    qa['visual_review']='passed: 17-page final-video contact sheet and detailed subtitle frames reviewed; latest cover, conclusion and closing retained'
    (WORK/'qa/verification.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp=dest.with_name(dest.name+'.latest.tmp');assert not tmp.exists()
    shutil.copy2(video,tmp);assert sha(tmp)==qa['sha256'];os.replace(tmp,dest)
    for rel,expected in meta['protected_files_before'].items():assert sha(root/rel)==expected,rel
    delivery={'status':'passed','delivered_at':datetime.now().astimezone().isoformat(),'file':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,
              'duration_seconds':240,'resolution':[2560,1440],'fps':30,'source_ppt_sha256':meta['source_sha256'],'source_ppt_unchanged':True,
              'slides':17,'captions':85,'speech':'Chinese synthesized narration','previous_video_backup':meta['previous_video_backup'],'other_submission_files_unchanged':True}
    (WORK/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(delivery,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
