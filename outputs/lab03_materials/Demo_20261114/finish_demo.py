"""Encode, verify and publish the new real-browser Lab03 demonstration."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
NAME = '第31组_实验三_Demo_1分钟.mp4'
DURATION = 60.0
FPS = 30


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render(work):
    audit = read(work/'audit.json')
    assert audit.get('completed'), 'Incomplete take cannot be rendered'
    assert not audit['errors'] and not audit['http_errors']
    assert len(audit['runs']) == 6 and len(audit['downloads']) == 3
    frames = sorted(read(work/'capture-frames.json'), key=lambda row: row['timestamp'])
    frames = list({row['timestamp']: row for row in frames}.values())
    seen_marker = False
    begin = None
    for i, frame in enumerate(frames):
        picture = cv2.imdecode(np.fromfile(work/frame['file'], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert picture.shape[:2] == (1440, 2560)
        b, g, r = map(int, picture[10, 10])
        marked = r > 240 and b > 240 and g < 20
        if seen_marker and not marked:
            begin = i
            break
        seen_marker |= marked
    assert begin is not None, 'Cannot find clean start after the recording synchronization marker'
    origin = frames[begin]['timestamp']
    assert audit['started_epoch']+audit['recorded_seconds'] >= origin+DURATION
    selected = [frame for frame in frames[begin:] if frame['timestamp'] < origin+DURATION]
    lines = ['ffconcat version 1.0']
    for i, frame in enumerate(selected):
        end = selected[i+1]['timestamp'] if i+1 < len(selected) else origin+DURATION
        assert end > frame['timestamp']
        lines += [f"file '{frame['file']}'", 'option framerate 1000', f"duration {end-frame['timestamp']:.9f}"]
    lines += [f"file '{selected[-1]['file']}'", 'option framerate 1000']
    (work/'frames.ffconcat').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    destination = work/NAME
    assert not destination.exists(), 'Do not overwrite a preexisting render'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-n', '-safe', '0', '-i', 'frames.ffconcat',
               '-vf', 'setpts=PTS-STARTPTS,fps=30', '-t', '60', '-frames:v', '1800', '-an',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-metadata', 'title=实验三网页版 Demo · 1分钟',
               '-metadata', 'comment=真实网页操作，2560x1440，原速无声；LLM为界面明确标注的历史评分重放。', str(destination)]
    write(work/'render-command.json', command)
    print('Encoding native 2560x1440, 60.000 seconds, 1800 frames...', flush=True)
    with (work/'render.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=work, stdout=subprocess.DEVNULL, stderr=log, check=True)
    write(work/'render.json', {'video':str(destination),'sha256':sha(destination),
                              'source_frames':len(selected),'capture_start_epoch':origin,
                              'start_offset_seconds':origin-audit['started_epoch'],
                              'duration_seconds':DURATION,'real_time_playback':True})
    print('ENCODED', destination, flush=True)


def verify(work):
    video = work/NAME
    audit = read(work/'audit.json')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print('Decoding all 1800 frames for verification...', flush=True)
    decode = subprocess.run([ffmpeg,'-v','error','-i',str(video),'-f','null','-'], capture_output=True)
    (work/'decode.log').write_bytes(decode.stderr)
    assert decode.returncode == 0 and not decode.stderr.strip(), decode.stderr.decode(errors='replace')
    probe = subprocess.run([ffmpeg,'-hide_banner','-i',str(video)], capture_output=True, encoding='utf-8',errors='replace')
    (work/'stream-info.txt').write_text(probe.stderr, encoding='utf-8')
    assert 'Duration: 00:01:00.00' in probe.stderr, probe.stderr
    assert 'Audio:' not in probe.stderr and 'Subtitle:' not in probe.stderr
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), video
    width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps,count = cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert (width,height,count)==(2560,1440,1800) and abs(fps-30)<1e-6,(width,height,count,fps)
    times = [0,2,7,10,14,18,22,25,28,32,38,42,46,51,54,59]
    samples = work/'verification-frames'; samples.mkdir(exist_ok=True)
    thumbs=[]
    frame_records=[]
    for second in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(second*FPS))
        ok, frame=cap.read(); assert ok,second
        if second==0:
            b,g,r=map(int,frame[10,10]); assert not (b>230 and r>230 and g<30),'Sync marker visible'
        filename=f'{second:02d}s.png'
        cv2.imencode('.png',frame)[1].tofile(samples/filename)
        thumb=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
        tile=np.full((396,640,3),26,dtype=np.uint8)
        tile[:360]=thumb
        cv2.putText(tile,f'{second:02d}s',(16,386),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),1,cv2.LINE_AA)
        thumbs.append(tile)
        frame_records.append({'seconds':second,'file':f'verification-frames/{filename}'})
    cap.release()
    sheet=np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0,len(thumbs),4)])
    cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(work/'contact-sheet.jpg')
    before=read(BASE/'protected_before.json')
    changes=[]
    for rel, expected in before.items():
        path=ROOT/rel
        actual=sha(path) if path.is_file() else None
        if actual!=expected:changes.append({'file':rel,'before':expected,'after':actual})
    runs=[]
    for run in audit['runs']:
        body=run['body'];result=body['result']
        runs.append({'seconds':run['at'],'request':run['request'],'http_status':run['status'],
                     'valid':result.get('valid'),'expanded':result.get('expanded'),
                     'llm_mode':body.get('llm',{}).get('mode')})
    report={'video':str(video),'width':width,'height':height,'fps':fps,'frame_count':count,
            'duration_seconds':count/fps,'bytes':video.stat().st_size,'sha256':sha(video),
            'full_decode':'passed','audio_streams':0,'subtitle_streams':0,
            'workspace_id':audit['health']['workspace_id'],'real_computations':runs,
            'download_count':len(audit['downloads']),'browser_errors':audit['errors'],'http_errors':audit['http_errors'],
            'protected_file_count':len(before),'protected_file_changes':changes,'samples':frame_records,
            'visual_review':'pending'}
    write(work/'verification.json',report)
    assert not changes,changes
    assert all(run['valid'] for run in runs)
    print(json.dumps({k:v for k,v in report.items() if k not in ['real_computations','samples']},ensure_ascii=False,indent=2),flush=True)


def publish(work):
    report=read(work/'verification.json')
    assert report['visual_review']=='passed','Manually review screenshots and set visual_review before publishing'
    assert report['full_decode']=='passed' and not report['protected_file_changes']
    source=work/NAME;assert sha(source)==report['sha256']
    destination=ROOT/'deliverables/lab3'/NAME
    assert destination.parent.is_dir()
    if destination.exists():
        assert sha(destination)==report['sha256'],'An existing user video must not be overwritten'
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src,dst)
        shutil.copystat(source,destination)
    assert sha(destination)==report['sha256']
    write(BASE/'delivery.json',{'path':str(destination),'sha256':report['sha256'],
                              'take':work.name,'date':datetime.now().astimezone().isoformat()})
    print('DELIVERED',destination,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['render','verify','publish','render-verify'])
    parser.add_argument('--take',required=True)
    args=parser.parse_args()
    work=(BASE/args.take).resolve()
    assert work.is_relative_to(BASE) and work.is_dir()
    if args.action in ['render','render-verify']:render(work)
    if args.action in ['verify','render-verify']:verify(work)
    if args.action=='publish':publish(work)


if __name__=='__main__':main()
