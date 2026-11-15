"""Render subtitle previews and verify their true pixel bounds against slide text."""
from pathlib import Path
import json,subprocess,math
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

WORK=Path(__file__).resolve().parent

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def overlap(a,b,pad=6):
    return a[0]<b[2]+pad and a[2]>b[0]-pad and a[1]<b[3]+pad and a[3]>b[1]-pad

def main():
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    cues=json.loads((WORK/'subtitles.json').read_text(encoding='utf-8'))
    slides=json.loads((WORK/'slides/render.json').read_text(encoding='utf-8'))
    layout=[];tiles=[]
    for slide in slides:
        number=slide['number'];own=[c for c in cues if c['slide']==number]
        longest=max(own,key=lambda c:c['estimated_width_px'])
        when=(longest['start']+longest['end'])/2
        path=WORK/'preview'/f'final-subtitle-slide-{number:02d}.png'
        command=[ffmpeg,'-hide_banner','-loglevel','warning','-n','-loop','1','-framerate','30','-i',slide['png'],
                 '-vf',f'setpts=PTS+{when:.9f}/TB,ass=subtitles.ass','-frames:v','1','-update','1',str(path)]
        result=subprocess.run(command,cwd=WORK,capture_output=True)
        (WORK/'qa'/f'subtitle-{number:02d}.log').write_bytes(result.stderr)
        assert result.returncode==0,result.stderr.decode(errors='replace')
        im=Image.open(path).convert('RGB');ref=Image.open(WORK/slide['png']).convert('RGB')
        diff=np.max(np.abs(np.asarray(im).astype(np.int16)-np.asarray(ref).astype(np.int16)),axis=2)
        y,x=np.where(diff>8);assert len(x)>100,(number,'no visible subtitle')
        bounds=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        assert 25<=bounds[0]<bounds[2]<=2535 and 1318<=bounds[1]<bounds[3]<=1404,(number,bounds)
        text_overlaps=[]
        sx=2560/slide['slide_width_points'];sy=1440/slide['slide_height_points']
        for shape in slide['shapes']:
            if not shape.get('text','').strip():continue
            text_box=[shape['bound_x']*sx,shape['bound_y']*sy,(shape['bound_x']+shape['bound_width'])*sx,(shape['bound_y']+shape['bound_height'])*sy]
            if overlap(bounds,text_box):text_overlaps.append({'text':shape['text'],'bounds':text_box})
        assert not text_overlaps,(number,bounds,text_overlaps)
        layout.append({'slide':number,'longest_caption':longest['text'],'rendered_bounds':bounds,'sample_seconds':when,'not_over_body_or_footer_names':True})
        tile=Image.new('RGB',(650,395),'#d5dcdf');tile.paste(im.resize((640,360),Image.Resampling.LANCZOS),(5,5))
        ImageDraw.Draw(tile).text((12,369),f'Slide {number:02d} | Subtitles | {when:.2f}s',fill='#253441');tiles.append(tile)
        print(f'Slide {number:02d} captions within clear bottom area: {bounds}',flush=True)
    write(WORK/'qa/subtitle-layout.json',layout)
    for start in range(0,len(tiles),9):
        sheet=Image.new('RGB',(1950,1185),'#c8d1d5')
        for i,tile in enumerate(tiles[start:start+9]):sheet.paste(tile,((i%3)*650,(i//3)*395))
        sheet.save(WORK/'preview'/f'subtitle-contact-{start//9+1:02d}.jpg',quality=95)
    print('Subtitle layout verified for all 17 latest slides.',flush=True)

if __name__=='__main__':main()
