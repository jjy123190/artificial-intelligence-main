"""Generate WebVTT captions and small posters for verified current media."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]


def stamp(seconds):
    ms=round(seconds*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'


def build(output,captures,renders):
    output,captures,renders=map(Path,(output,captures,renders))
    demo=json.loads((output/'demo_evidence.json').read_text(encoding='utf-8'))
    presentation=json.loads((output/'presentation_evidence.json').read_text(encoding='utf-8'))
    if demo['artifact_revision']!=presentation['artifact_revision']:raise ValueError('Media evidence revisions differ')
    cues=[f"{stamp(r['start_seconds'])} --> {stamp(r['end_seconds'])}\n{r['label']}" for r in demo['steps']]
    demo_path=output/'实验四_当前页面与真实推理_Demo.vtt';demo_path.write_bytes(('WEBVTT\n\n'+'\n\n'.join(cues)+'\n').encode('utf-8'))
    cues=[]
    for row in presentation['slides']:
        phrases=[x.strip() for x in re.split(r'(?<=[。！？])',row['narration']) if x.strip()]
        weights=[len(x) for x in phrases];spoken=row['duration_seconds']-.7;start=row['start_seconds']
        for phrase,weight in zip(phrases,weights):
            duration=spoken*weight/sum(weights);cues.append(f'{stamp(start)} --> {stamp(start+duration)}\n{phrase}');start+=duration
    narration_path=output/'实验四_当前版本PPT讲解.vtt';narration_path.write_bytes(('WEBVTT\n\n'+'\n\n'.join(cues)+'\n').encode('utf-8'))
    for source,name in [(captures/'live-result.png','实验四_当前Demo预览.jpg'),(renders/'slide-01.png','实验四_当前PPT预览.jpg')]:
        image=Image.open(source).convert('RGB');image.thumbnail((1280,720));image.save(output/name,quality=85,optimize=True)
    generated=[demo_path,narration_path,output/'实验四_当前Demo预览.jpg',output/'实验四_当前PPT预览.jpg']
    evidence={'artifact_revision':demo['artifact_revision'],'caption_scope':'Demo labels follow recorded actions; narration phrase timings use proportional text length within measured per-slide SAPI audio, not forced word alignment.',
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in generated}}
    (output/'media_support_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(generated),'caption_scope':evidence['caption_scope']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'deliverables/lab4/当前版本'))
    p.add_argument('--captures',default=str(ROOT/'源码/.local/current-demo-20261125/captures'));p.add_argument('--renders',default=str(ROOT/'源码/.local/current-slides-render'))
    a=p.parse_args();build(a.output,a.captures,a.renders)
