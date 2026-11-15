"""Render each page of the saved latest PPT with Microsoft PowerPoint, read-only."""
from pathlib import Path
from datetime import datetime
import hashlib,json,time
from PIL import Image,ImageDraw,ImageFont
import win32com.client

WORK=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    meta=json.loads((WORK/'project.json').read_text(encoding='utf-8'))
    source=Path(meta['source']);snapshot=Path(meta['snapshot'])
    assert sha(source)==sha(snapshot)==meta['source_sha256']
    app=pres=None;rows=[];started=time.monotonic()
    try:
        app=win32com.client.DispatchEx('PowerPoint.Application')
        app.DisplayAlerts=1
        pres=app.Presentations.Open(str(snapshot),ReadOnly=True,Untitled=False,WithWindow=False)
        assert pres.Slides.Count==meta['slides']==17
        w=float(pres.PageSetup.SlideWidth);h=float(pres.PageSetup.SlideHeight)
        assert abs(w/h-16/9)<1e-6
        for i in range(1,pres.Slides.Count+1):
            slide=pres.Slides.Item(i)
            filename=WORK/'slides'/f'slide-{i:02d}.png'
            assert not filename.exists()
            assert slide.TimeLine.MainSequence.Count==0, 'Animated slides require an animation-aware recording'
            slide.Export(str(filename),'PNG',2560,1440)
            with Image.open(filename) as im:assert im.size==(2560,1440)
            shapes=[]
            for j in range(1,slide.Shapes.Count+1):
                sh=slide.Shapes.Item(j)
                row={'name':str(sh.Name),'x':float(sh.Left),'y':float(sh.Top),'width':float(sh.Width),'height':float(sh.Height),'type':int(sh.Type)}
                if sh.HasTextFrame and sh.TextFrame.HasText:
                    tr=sh.TextFrame.TextRange
                    row.update(text=str(tr.Text),bound_x=float(tr.BoundLeft),bound_y=float(tr.BoundTop),bound_width=float(tr.BoundWidth),bound_height=float(tr.BoundHeight))
                shapes.append(row)
            rows.append({'number':i,'png':filename.relative_to(WORK).as_posix(),'sha256':sha(filename),'animation_count':0,'slide_width_points':w,'slide_height_points':h,
                         'hidden':bool(slide.SlideShowTransition.Hidden),'shapes':shapes})
            print(f'Exported latest slide {i:02d}/17: {filename.stat().st_size} bytes',flush=True)
        pres.Close();pres=None
    finally:
        if pres is not None:
            try:pres.Close()
            except Exception:pass
        if app is not None:
            try:
                if app.Presentations.Count==0:app.Quit()
            except Exception:pass
    assert sha(source)==sha(snapshot)==meta['source_sha256']
    write(WORK/'slides/render.json',rows)
    tiles=[]
    for row in rows:
        im=Image.open(WORK/row['png']).convert('RGB');im=im.resize((640,360),Image.Resampling.LANCZOS)
        tile=Image.new('RGB',(650,395),'#d5dcdf');tile.paste(im,(5,5))
        ImageDraw.Draw(tile).text((12,369),f"Latest PPT | Slide {row['number']:02d}",fill='#253441')
        tiles.append(tile)
    for start in range(0,len(tiles),9):
        sheet=Image.new('RGB',(1950,1185),'#c8d1d5')
        for i,tile in enumerate(tiles[start:start+9]):sheet.paste(tile,((i%3)*650,(i//3)*395))
        sheet.save(WORK/'preview'/f'latest-slides-{start//9+1:02d}.jpg',quality=94)
    print(json.dumps({'rendered':len(rows),'source_unchanged':True,'elapsed_seconds':round(time.monotonic()-started,2)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
