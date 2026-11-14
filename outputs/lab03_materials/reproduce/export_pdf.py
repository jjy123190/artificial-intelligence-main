"""Export a DOCX to PDF with a private hidden Word instance and render QA images."""
from pathlib import Path
import time,json,re,math
import win32com.client
import fitz
from PIL import Image,ImageOps,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parents[1]
src=BASE/'第31组_实验三_实验报告.docx';dst=src.with_suffix('.pdf');qa=BASE/'reproduce/qa';qa.mkdir(exist_ok=True)
app=doc=None
try:
 app=win32com.client.DispatchEx('Word.Application');app.Visible=False;app.DisplayAlerts=0
 doc=app.Documents.Open(str(src),ReadOnly=True,AddToRecentFiles=False)
 doc.Repaginate();doc.Fields.Update();doc.ExportAsFixedFormat(str(dst),17)
 doc.Close(0);doc=None
finally:
 if doc is not None:
  try:doc.Close(0)
  except Exception:pass
 if app is not None:
  try:app.Quit(0)
  except Exception:pass
pdf=fitz.open(dst);page_rows=[];thumbs=[]
# Delete only obsolete previews created by this exporter, after a successful export.
qa_root=qa.resolve()
for old in qa.iterdir():
 match=re.fullmatch(r'(page|contact)_(\d+)\.png',old.name)
 if not match or not old.is_file():continue
 limit=len(pdf) if match[1]=='page' else math.ceil(len(pdf)/9)
 if int(match[2])>limit:
  target=old.resolve();assert target.is_relative_to(qa_root) and target.parent==qa_root
  target.unlink()

for i,page in enumerate(pdf):
 text=page.get_text();blocks=page.get_text('blocks');page_rows.append({'page':i+1,'characters':len(text),'text':text,
 'images':len(page.get_images()),'width':page.rect.width,'height':page.rect.height})
 pix=page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False);path=qa/f'page_{i+1:02d}.png';pix.save(path)
 im=Image.open(path).convert('RGB');im.thumbnail((298,422));thumb=Image.new('RGB',(320,455),'#dfe6ed');thumb.paste(im,((320-im.width)//2,18))
 draw=ImageDraw.Draw(thumb);draw.text((10,432),f'Page {i+1:02d} | {len(text)} chars',fill='#183249');thumbs.append(thumb)
for start in range(0,len(thumbs),9):
 part=thumbs[start:start+9];sheet=Image.new('RGB',(320*3,455*3),'#cbd5e1')
 for j,im in enumerate(part):sheet.paste(im,((j%3)*320,(j//3)*455))
 sheet.save(qa/f'contact_{start//9+1}.png')
(qa/'pdf_pages.json').write_text(json.dumps({'pages':len(pdf),'items':page_rows},ensure_ascii=False,indent=2),encoding='utf-8')
print('PDF',len(pdf),'pages',dst.stat().st_size,'bytes')
for x in page_rows:print(x['page'],x['characters'],x['images'],x['text'][:85].replace('\n',' | '))