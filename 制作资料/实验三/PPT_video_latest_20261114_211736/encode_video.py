"""Encode the real, latest PowerPoint slide exports with synced narration and burned subtitles."""
from pathlib import Path
import hashlib,json,subprocess,wave
import imageio_ffmpeg

WORK=Path(__file__).resolve().parent
NAME='第4组_实验三_PPT演讲.mp4'

def main():
    meta=json.loads((WORK/'project.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()==meta['source_sha256'], 'PPT changed; use the latest saved version before encoding.'
    audio=WORK/'speech/narration-240s.wav'
    with wave.open(str(audio),'rb') as w:assert w.getnframes()==240*48000 and w.getframerate()==48000
    destination=WORK/NAME
    assert not destination.exists(), 'Do not overwrite an existing render'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-n','-safe','0','-i','slides.ffconcat',
             '-i','speech/narration-240s.wav','-map','0:v:0','-map','1:a:0',
             '-vf','setpts=PTS-STARTPTS,fps=30,ass=subtitles.ass,format=yuv420p',
             '-t','240','-frames:v','7200','-c:v','libx264','-preset','medium','-tune','stillimage','-crf','17',
             '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-ar','48000','-ac','1','-movflags','+faststart',
             '-metadata','title=实验三PPT演讲 · 四分钟配音字幕版',
             '-metadata','comment=根据用户最新17页PPT逐页演讲；中文合成配音，字幕直接叠加在PPT底部；原PPT未修改。',
             '-metadata:s:a:0','language=zho',str(destination)]
    (WORK/'qa/encode-command.json').write_text(json.dumps(command,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ENCODING 240s, 2560x1440, 30fps, 17 real PPT slides, Mandarin narration, 85 synced captions...',flush=True)
    with (WORK/'qa/encode.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=WORK,stdout=subprocess.DEVNULL,stderr=log,check=True)
    print('ENCODED',destination,'BYTES',destination.stat().st_size,flush=True)

if __name__=='__main__':main()
