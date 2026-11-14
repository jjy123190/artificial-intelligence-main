"""Render the staged report with an isolated hidden Word instance, then QA previews."""
from pathlib import Path
import json,hashlib
import win32com.client
import fitz
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent
SRC=BASE/'draft/第31组_实验三_实验报告.docx'
PDF=SRC.with_suffix('.pdf')
QA=BASE/'qa'


def main():
    before=hashlib.sha256(SRC.read_bytes()).hexdigest()
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
    assert before==hashlib.sha256(SRC.read_bytes()).hexdigest(),'Export changed source DOCX'
    pdf=fitz.open(PDF);items=[];thumbs=[]
    for i,page in enumerate(pdf):
        text=page.get_text()
        items.append({'page':i+1,'characters':len(text),'text':text,'images':len(page.get_images()),'width':page.rect.width,'height':page.rect.height})
        pix=page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False)
        path=QA/f'page_{i+1:02}.png';pix.save(path)
        im=Image.open(path).convert('RGB');im.thumbnail((298,422))
        thumb=Image.new('RGB',(320,455),'#dfe6ed');thumb.paste(im,((320-im.width)//2,12))
        ImageDraw.Draw(thumb).text((10,435),f'Page {i+1:02} | {len(text)} chars',fill='#183249');thumbs.append(thumb)
    for start in range(0,len(thumbs),9):
        sheet=Image.new('RGB',(960,1365),'#cbd5e1')
        for j,im in enumerate(thumbs[start:start+9]):sheet.paste(im,((j%3)*320,(j//3)*455))
        sheet.save(QA/f'contact_{start//9+1}.png')
    (QA/'pdf_pages.json').write_text(json.dumps({'pages':len(pdf),'items':items},ensure_ascii=False,indent=2),encoding='utf-8')
    print('PDF',len(pdf),'pages',PDF.stat().st_size,'bytes')
    for item in items:print(item['page'],item['characters'],item['images'],item['text'][:150].replace('\n',' | '))


if __name__=='__main__':main()
