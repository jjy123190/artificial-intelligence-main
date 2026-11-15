"""Generate Mandarin narration from the latest PPT-based script, retaining word timing."""
from pathlib import Path
import asyncio,json,shutil,subprocess,time
import edge_tts,imageio_ffmpeg

WORK=Path(__file__).resolve().parent
VOICE='zh-CN-YunxiNeural'

async def main():
    rows=json.loads((WORK/'narration.json').read_text(encoding='utf-8'))
    semaphore=asyncio.Semaphore(3)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    async def one(row):
        async with semaphore:
            i=row['slide'];audio=WORK/'speech'/f'slide-{i:02d}.mp3';words=WORK/'speech'/f'slide-{i:02d}-boundaries.jsonl'
            if i==1 and not audio.exists():
                shutil.copyfile(WORK/'speech/probe.mp3',audio)
                shutil.copyfile(WORK/'speech/probe-boundaries.jsonl',words)
            if not audio.exists() or not words.exists():
                for attempt in range(1,4):
                    temp=WORK/'speech'/f'slide-{i:02d}-attempt-{attempt}.mp3'
                    temp_words=WORK/'speech'/f'slide-{i:02d}-attempt-{attempt}.jsonl'
                    try:
                        communicator=edge_tts.Communicate(row['text'],VOICE,rate='+0%',boundary='WordBoundary',
                                                          connect_timeout=20,receive_timeout=75)
                        await communicator.save(str(temp),str(temp_words))
                        assert temp.stat().st_size>1024 and temp_words.stat().st_size>30
                        shutil.copyfile(temp,audio);shutil.copyfile(temp_words,words)
                        break
                    except Exception as exc:
                        print(f'Voice slide {i} attempt {attempt}: {type(exc).__name__}',flush=True)
                        if attempt==3:raise
                        await asyncio.sleep(2*attempt)
            raw=await asyncio.to_thread(subprocess.run,[ffmpeg,'-v','error','-i',str(audio),'-ac','1','-ar','48000','-f','s16le','-'],capture_output=True,check=True)
            duration=len(raw.stdout)/96000
            tokens=[json.loads(line) for line in words.read_text(encoding='utf-8').splitlines() if line.strip()]
            record={'slide':i,'voice':VOICE,'rate':'+0%','duration':duration,'word_boundaries':len(tokens),
                    'audio':audio.relative_to(WORK).as_posix(),'timing':words.relative_to(WORK).as_posix(),
                    'first_word_seconds':tokens[0]['offset']/1e7,'last_word_end_seconds':(tokens[-1]['offset']+tokens[-1]['duration'])/1e7}
            print(f'Slide {i:02d}: {duration:.3f}s, {len(tokens)} timed words',flush=True)
            return record
    records=await asyncio.gather(*(one(row) for row in rows))
    (WORK/'speech/manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('TOTAL RAW AUDIO SECONDS',sum(row['duration'] for row in records),flush=True)

if __name__=='__main__':asyncio.run(main())
