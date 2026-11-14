"""Lay out the exact 240-second narrated PPT video and word-aligned subtitles."""
from __future__ import annotations
from pathlib import Path
import json,math,subprocess,unicodedata,wave
from datetime import datetime

from PIL import Image,ImageChops,ImageFont
import imageio_ffmpeg
import numpy as np

WORK=Path(__file__).resolve().parent
FPS=30;SR=48000;DURATION=240
FONT_PATH=Path('C:/Windows/Fonts/msyh.ttc')
FONT=ImageFont.truetype(str(FONT_PATH),56)


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def normalize(text):return ''.join(ch.lower() for ch in unicodedata.normalize('NFKC',text) if ch.isalnum())

def ass_time(second):
    cs=round(second*100);h,cs=divmod(cs,360000);m,cs=divmod(cs,6000);s,cs=divmod(cs,100)
    return f'{h}:{m:02d}:{s:02d}.{cs:02d}'

def srt_time(second):
    ms=round(second*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def pcm(ffmpeg,audio,filter_text=None):
    command=[ffmpeg,'-v','error','-i',str(audio)]
    if filter_text:command+=['-af',filter_text]
    command+=['-ac','1','-ar',str(SR),'-f','s16le','-']
    result=subprocess.run(command,capture_output=True,check=True)
    assert not result.stderr.strip(),result.stderr.decode(errors='replace')
    return np.frombuffer(result.stdout,dtype='<i2').copy()


def main():
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    narr=json.loads((WORK/'narration.json').read_text(encoding='utf-8'))
    manifest=json.loads((WORK/'speech/manifest.json').read_text(encoding='utf-8'))
    assert len(narr)==len(manifest)==17
    lead=[.30]+[.12]*15
    clips=[row['last_word_end_seconds']+.12 for row in manifest[:-1]]
    fixed=sum(lead)+.18*16+5.0
    tempo=sum(clips)/(DURATION-fixed)
    assert .97<tempo<1.06,tempo
    raw_slots=[lead[i]+clips[i]/tempo+.18 for i in range(16)]+[5.0]
    boundaries=[0];cumulative=0.
    for slot in raw_slots:
        cumulative+=slot;boundaries.append(round(cumulative*FPS))
    assert boundaries[-1]==7200 and boundaries[-2]==7050
    mix=np.zeros(DURATION*SR,dtype=np.int16)
    timeline=[];captions=[]
    for i,(row,meta) in enumerate(zip(narr,manifest)):
        slide=i+1;start=boundaries[i]/FPS;end=boundaries[i+1]/FPS
        words=[json.loads(l) for l in (WORK/meta['timing']).read_text(encoding='utf-8').splitlines() if l.strip()]
        assert normalize(row['text'])==normalize(''.join(word['text'] for word in words)),slide
        rate=tempo if slide<17 else 1.0
        clip_end=meta['last_word_end_seconds']+.12
        offset=.30 if slide==1 else (.15 if slide==17 else .12)
        sample_start=round((start+offset)*SR)
        segment=pcm(ffmpeg,WORK/meta['audio'],f'atrim=end={clip_end:.9f},asetpts=PTS-STARTPTS,atempo={rate:.12f}')
        assert sample_start+len(segment)<round(end*SR),(slide,sample_start+len(segment),round(end*SR))
        mix[sample_start:sample_start+len(segment)]=segment
        # Map all displayed phrases to the actual TTS word boundary stream.
        spans=[];cursor=0
        for word in words:
            clean=normalize(word['text']);assert clean
            spans.append({'char_start':cursor,'char_end':cursor+len(clean),'start':word['offset']/1e7,
                          'end':(word['offset']+word['duration'])/1e7,'text':word['text']})
            cursor+=len(clean)
        phrase_spans=[];cursor=0
        for text in row['captions']:
            clean=normalize(text);assert clean
            stop=cursor+len(clean)
            selected=[w for w in spans if w['char_end']>cursor and w['char_start']<stop]
            assert selected,(slide,text)
            phrase_spans.append({'text':text,'raw_start':selected[0]['start'],'raw_end':selected[-1]['end']})
            cursor=stop
        assert cursor==spans[-1]['char_end']
        for j,phrase in enumerate(phrase_spans):
            cue_start=start+offset+phrase['raw_start']/rate-.035
            cue_end=start+offset+phrase['raw_end']/rate+.12
            if j+1<len(phrase_spans):
                cue_end=min(cue_end,start+offset+phrase_spans[j+1]['raw_start']/rate-.04)
            cue_end=min(cue_end,end-.08)
            assert cue_end>cue_start,(slide,phrase)
            width=FONT.getlength(phrase['text'])
            assert width<=1250,(slide,width)
            # Cover and closing have names on the lower left; keep every caption to the right.
            if slide in [1,17]:assert 1280-width/2-4>800,(slide,width)  # Actual rendered bounds checked separately.
            captions.append({'index':len(captions)+1,'slide':slide,'start':round(cue_start,6),'end':round(cue_end,6),
                             'text':phrase['text'],'estimated_width_px':round(width,2),'position':[1280,1354],
                             'font':'Microsoft YaHei','font_size':56,'style':'Dark' if slide in [1,16,17] else 'Light',
                             'source_tts_start':phrase['raw_start'],'source_tts_end':phrase['raw_end']})
        timeline.append({'slide':slide,'start':start,'end':end,'duration':end-start,
                         'start_frame':boundaries[i],'end_frame':boundaries[i+1],'frames':boundaries[i+1]-boundaries[i],
                         'image':f'slides/slide-{slide:02d}.png','audio':meta['audio'],'audio_start':sample_start/SR,
                         'audio_end':(sample_start+len(segment))/SR,'raw_audio_duration':meta['duration'],
                         'last_spoken_word_end':start+offset+meta['last_word_end_seconds']/rate,
                         'trimmed_silence_only_after':clip_end,'tempo':rate,'audio_sample_count':len(segment)})
    assert all(a['end']<=b['start']+.002 for a,b in zip(captions,captions[1:])), 'Overlapping captions'
    wav=WORK/'speech/narration-240s-raw.wav'
    with wave.open(str(wav),'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(SR);out.writeframes(mix.tobytes())
    write(WORK/'timeline.json',timeline);write(WORK/'subtitles.json',captions)
    subtitle_header='''[Script Info]
Title: 实验三最新PPT演讲 · 四分钟
ScriptType: v4.00+
PlayResX: 2560
PlayResY: 1440
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dark,Microsoft YaHei,56,&H00EFF6F9,&H00EFF6F9,&H00372D17,&H80372D17,0,0,0,0,100,100,0,0,1,1.8,0.4,5,0,0,0,1
Style: Light,Microsoft YaHei,56,&H00372D17,&H00372D17,&H00EFF6F9,&H80EFF6F9,0,0,0,0,100,100,0,0,1,1.0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    events=[];srt=[]
    for cue in captions:
        events.append(f'Dialogue: 0,{ass_time(cue["start"])},{ass_time(cue["end"])},{cue["style"]},,0,0,0,,{{\\an5\\pos(1280,1354)}}{cue["text"]}')
        srt += [str(cue['index']),f'{srt_time(cue["start"])} --> {srt_time(cue["end"])}',cue['text'],'']
    (WORK/'subtitles.ass').write_text(subtitle_header+'\n'.join(events)+'\n',encoding='utf-8-sig')
    (WORK/'subtitles.srt').write_text('\n'.join(srt),encoding='utf-8-sig')
    concat=['ffconcat version 1.0']
    for row in timeline:
        concat += [f"file '{row['image']}'",'option framerate 30',f"duration {row['duration']:.9f}"]
    concat += [f"file '{timeline[-1]['image']}'",'option framerate 30']
    (WORK/'slides.ffconcat').write_text('\n'.join(concat)+'\n',encoding='utf-8')
    script=['# 实验三四分钟演讲稿\n','基于用户 2026-11-14 13:05:31 保存的17页PPT；逐句字幕与配音同文。\n']
    for row,slot in zip(narr,timeline):
        script += [f"## 第{row['slide']}页 · {slot['start']:.2f}—{slot['end']:.2f}秒\n",row['text']+'\n']
    (WORK/'演讲稿.md').write_text('\n'.join(script),encoding='utf-8')
    metrics={'duration_seconds':DURATION,'frames':7200,'sample_rate':SR,'audio_samples':len(mix),'tempo':tempo,
             'caption_count':len(captions),'min_caption_duration':min(c['end']-c['start'] for c in captions),
             'max_caption_width_px':max(c['estimated_width_px'] for c in captions),
             'trim_policy':'Only ~0.76 seconds of generated trailing silence per slide removed; every word retained.',
             'final_slide_start':timeline[-1]['start'],'raw_peak_dbfs':20*math.log10(max(abs(mix.astype(np.int32)))/32768)}
    write(WORK/'qa/timing.json',metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('SLIDE TIMELINE')
    for row in timeline:print(f"{row['slide']:02}  {row['start']:7.3f} - {row['end']:7.3f}  {row['duration']:6.3f}s  speech ends {row['last_spoken_word_end']:.3f}")


if __name__=='__main__':main()
