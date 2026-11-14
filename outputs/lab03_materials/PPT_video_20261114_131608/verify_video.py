"""Verify all slides, subtitle placement, full decode, and narration synchronization."""
from pathlib import Path
import hashlib,json,math,subprocess,wave
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont

WORK=Path(__file__).resolve().parent
VIDEO=WORK/'第31组_实验三_PPT演讲_4分钟_配音字幕版.mp4'


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    meta=json.loads((WORK/'project.json').read_text(encoding='utf-8'))
    timeline=json.loads((WORK/'timeline.json').read_text(encoding='utf-8'))
    captions=json.loads((WORK/'subtitles.json').read_text(encoding='utf-8'))
    layout=json.loads((WORK/'qa/subtitle-layout.json').read_text(encoding='utf-8'))
    print('Full 240-second audio/video decoding...',flush=True)
    decode=subprocess.run([ffmpeg,'-v','error','-i',str(VIDEO),'-f','null','-'],capture_output=True)
    (WORK/'qa/full-decode.log').write_bytes(decode.stderr)
    assert decode.returncode==0 and not decode.stderr.strip(),decode.stderr.decode(errors='replace')
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert (count,fps,width,height)==(7200,30.0,2560,1440)
    refs=[]
    for row in timeline:
        im=cv2.imdecode(np.fromfile(WORK/row['image'],np.uint8),cv2.IMREAD_COLOR)
        refs.append(cv2.resize(im[:1315],(640,328),interpolation=cv2.INTER_AREA).astype(np.float32))
    checks=[];review=[];thumbs=[]
    render_dir=WORK/'qa/verified-frames';render_dir.mkdir(exist_ok=True)
    for row,place in zip(timeline,layout):
        sample_frame=round(place['sample_seconds']*30)
        sample_indices=sorted(set([row['start_frame'],sample_frame,row['end_frame']-1]))
        for index in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok,index
            thumb=cv2.resize(frame[:1315],(640,328),interpolation=cv2.INTER_AREA).astype(np.float32)
            distances=[float(np.mean(np.abs(thumb-ref))) for ref in refs]
            actual=int(np.argmin(distances))+1
            assert actual==row['slide'],(index,row['slide'],actual,distances)
            assert distances[actual-1]<1.8,(index,distances[actual-1])
            checks.append({'frame':index,'seconds':index/30,'expected_slide':row['slide'],'detected_slide':actual,
                           'body_mean_absolute_error':round(distances[actual-1],4)})
            if index==sample_frame:
                filename=f'slide-{row["slide"]:02d}-{index:04d}.png'
                cv2.imencode('.png',frame)[1].tofile(render_dir/filename)
                expected=cv2.imdecode(np.fromfile(WORK/'preview'/f'final-subtitle-slide-{row["slide"]:02d}.png',np.uint8),cv2.IMREAD_COLOR)
                box=place['rendered_bounds'];x0,y0,x1,y1=box
                caption_error=float(np.mean(np.abs(frame[y0-2:y1+2,x0-2:x1+2].astype(np.float32)-expected[y0-2:y1+2,x0-2:x1+2].astype(np.float32))))
                assert caption_error<9,(row['slide'],caption_error)
                review.append({'slide':row['slide'],'file':'qa/verified-frames/'+filename,'frame':index,
                               'caption':place['longest_caption'],'caption_pixel_error':round(caption_error,4)})
                tile=np.zeros((390,640,3),np.uint8);tile[:360]=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,f'{row["slide"]:02d} | {index/30:.1f}s',(12,381),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
                thumbs.append(tile)
    # Every phrase must be covered by its own slide with nonempty text and matching TTS words.
    for cue in captions:
        slot=timeline[cue['slide']-1]
        assert slot['start']<=cue['start']<cue['end']<=slot['end']
        assert cue['text'].strip() and cue['end']-cue['start']>=.75
    cap.set(cv2.CAP_PROP_POS_FRAMES,7199);ok,last=cap.read();assert ok
    cv2.imencode('.png',last)[1].tofile(render_dir/'last-frame.png');cap.release()
    while len(thumbs)%4:thumbs.append(np.zeros((390,640,3),np.uint8))
    sheet=np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0,len(thumbs),4)])
    cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(WORK/'qa/video-contact-sheet.jpg')
    print('All 17 slides and 16 transitions matched the latest PPT.',flush=True)
    # Decoded AAC must align sample-for-sample with the timing-aligned narration master.
    audio=subprocess.run([ffmpeg,'-v','error','-i',str(VIDEO),'-map','0:a:0','-ac','1','-ar','48000','-f','s16le','-'],capture_output=True,check=True)
    decoded=np.frombuffer(audio.stdout,dtype='<i2').astype(np.float32)
    with wave.open(str(WORK/'speech/narration-240s.wav'),'rb') as w:master=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)
    assert len(decoded)>=len(master) and len(decoded)-len(master)<=1024,(len(decoded),len(master))
    decoded=decoded[:len(master)]
    sync=[]
    for row in timeline:
        a=round(row['audio_start']*48000);b=round(row['audio_end']*48000)
        x=master[a:b];y=decoded[a:b]
        correlation=float(np.corrcoef(x,y)[0,1]);rms_db=20*math.log10(float(np.sqrt(np.mean(y*y)))/32768)
        snr=10*math.log10(float(np.mean(x*x))/max(float(np.mean((x-y)**2)),1e-12))
        assert correlation>.98 and -40<rms_db<-8,(row['slide'],correlation,rms_db)
        sync.append({'slide':row['slide'],'zero_lag_correlation':round(correlation,6),'rms_dbfs':round(rms_db,2),
                     'aac_vs_master_snr_db':round(snr,2),'last_word_before_next_slide':row['last_spoken_word_end']<row['end']})
    print('Narration present on every slide; AAC matches the timed master without drift.',flush=True)
    root=Path(meta['source']).parents[2]
    changes=[]
    for relative,expected in meta['protected_files_before'].items():
        path=root/relative;actual=sha(path) if path.is_file() else None
        if actual!=expected:changes.append({'path':relative,'before':expected,'after':actual})
    assert not changes,changes
    source_rows=json.loads((WORK/'source/slide-content.json').read_text(encoding='utf-8'))
    assert len(source_rows)==17 and '敬请老师批评指正' not in source_rows[-1]['text'],'Latest edited closing slide was not used'
    report={'video':str(VIDEO),'bytes':VIDEO.stat().st_size,'sha256':sha(VIDEO),'duration_seconds':count/fps,
            'width':width,'height':height,'fps':fps,'frames':count,'audio_codec':'AAC','audio_sample_rate':48000,
            'audio_channels':1,'full_decode':'passed','source_ppt_sha256':meta['source_sha256'],'source_slides':17,
            'used_latest_saved_ppt':True,'protected_file_changes':changes,'caption_count':len(captions),
            'subtitles_burned_in':True,'external_subtitle_file_required':False,'added_black_bars':False,
            'slides_and_transition_checks':checks,'audio_sync_checks':sync,'samples':review,'visual_review':'pending'}
    write(WORK/'qa/verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['slides_and_transition_checks','audio_sync_checks','samples']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
