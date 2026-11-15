"""Export only the staged report using an isolated, hidden Word instance."""
from pathlib import Path
import hashlib,json
import win32com.client
import fitz
from PIL import Image,ImageDraw
OPS=Path(__file__).resolve().parent
SRC=OPS/'staged/第31组_实验三_实验报告.docx'
PDF=SRC.with_suffix('.pdf')
QA=OPS/'preview';QA.mkdir(exist_ok=True)
original=hashlib.sha256(SRC.read_bytes()).hexdigest()
app=doc=None
try:
    app=win32com.client.DispatchEx('Word.Application')
    app.Visible=False;app.DisplayAlerts=0
    doc=app.Documents.Open(str(SRC),ReadOnly=True,AddToRecentFiles=False,Visible=False)
    doc.Repaginate();doc.Fields.Update();doc.ExportAsFixedFormat(str(PDF),17)
    doc.Close(0);doc=None
finally:
    if doc is not None:
        try:doc.Close(0)
        except Exception:pass
    if app is not None:
        try:app.Quit(0)
        except Exception:pass
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==original,'Word export changed DOCX'
with fitz.open(PDF) as pdf:
    print('PDF PAGES',len(pdf),'bytes',PDF.stat().st_size)
    info=[]
    for i,p in enumerate(pdf):
        info.append({'page':i+1,'text':p.get_text(),'images':len(p.get_images())})
    (OPS/'pdf_pages.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    thumbs=[]
    for index in [0,1,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29]:
        if index>=len(pdf):continue
        p=pdf[index];pix=p.get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False)
        image=QA/f'page_{index+1:02}.png';pix.save(image)
        im=Image.open(image).convert('RGB');im.thumbnail((340,482))
        thumb=Image.new('RGB',(360,510),'#edf0f2');thumb.paste(im,((360-im.width)//2,5))
        ImageDraw.Draw(thumb).text((10,490),f'Page {index+1}',fill='#243741');thumbs.append(thumb)
    for start in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1080,1020),'#cdd3d8')
        for j,im in enumerate(thumbs[start:start+6]):sheet.paste(im,((j%3)*360,(j//3)*510))
        sheet.save(QA/f'contact_{start//6+1}.png')
