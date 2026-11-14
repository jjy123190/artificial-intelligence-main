"""Export the staged editable presentation and collect Office layout metrics."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
import win32com.client

BASE=Path(__file__).resolve().parent
SRC=BASE/"draft/第31组_实验三_汇报.pptx"
PDF=SRC.with_suffix(".pdf")
QA=BASE/"qa"


def main():
    before=sha256(SRC.read_bytes()).hexdigest()
    app=presentation=None
    geometry=[]
    try:
        app=win32com.client.DispatchEx("PowerPoint.Application")
        app.DisplayAlerts=1
        presentation=app.Presentations.Open(str(SRC),ReadOnly=True,Untitled=False,WithWindow=False)
        for i in range(1,presentation.Slides.Count+1):
            slide=presentation.Slides.Item(i)
            entries=[]
            for j in range(1,slide.Shapes.Count+1):
                sh=slide.Shapes.Item(j)
                row={"name":sh.Name,"x":sh.Left,"y":sh.Top,"width":sh.Width,"height":sh.Height,"type":sh.Type}
                if sh.HasTextFrame and sh.TextFrame.HasText:
                    tr=sh.TextFrame.TextRange
                    row.update(text=tr.Text,bound_x=tr.BoundLeft,bound_y=tr.BoundTop,
                               bound_width=tr.BoundWidth,bound_height=tr.BoundHeight,
                               font_size=tr.Font.Size)
                if sh.HasTable:
                    row["cells"]=[]
                    for ri in range(1,sh.Table.Rows.Count+1):
                        for ci in range(1,sh.Table.Columns.Count+1):
                            cell=sh.Table.Cell(ri,ci).Shape;tr=cell.TextFrame.TextRange
                            row["cells"].append({"row":ri,"column":ci,"text":tr.Text,
                                                 "x":sh.Left+sum(sh.Table.Columns.Item(k).Width for k in range(1,ci)),
                                                 "y":sh.Top+sum(sh.Table.Rows.Item(k).Height for k in range(1,ri)),
                                                 "width":sh.Table.Columns.Item(ci).Width,"height":sh.Table.Rows.Item(ri).Height,"bound_x":tr.BoundLeft,
                                                 "bound_y":tr.BoundTop,"bound_width":tr.BoundWidth,"bound_height":tr.BoundHeight})
                entries.append(row)
            geometry.append({"slide":i,"shapes":entries})
        # ppSaveAsPDF=32 avoids a pywin32 optional-PrintRange marshaling bug.
        presentation.SaveAs(str(PDF),32)
        presentation.Close();presentation=None
    finally:
        if presentation is not None:
            try:presentation.Close()
            except Exception:pass
        if app is not None:
            try:app.Quit()
            except Exception:pass
    assert sha256(SRC.read_bytes()).hexdigest()==before,"Export changed source PPTX"
    (QA/"office_geometry.json").write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding="utf-8")
    pdf=fitz.open(PDF);thumbs=[];pages=[]
    for i,page in enumerate(pdf):
        text=page.get_text()
        pages.append({"slide":i+1,"text":text,"characters":len(text),"width":page.rect.width,"height":page.rect.height})
        path=QA/f"slide_{i+1:02}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.30,1.30),alpha=False).save(path)
        im=Image.open(path).convert("RGB");im.thumbnail((480,270))
        thumb=Image.new("RGB",(500,300),"#d9dfdd");thumb.paste(im,((500-im.width)//2,7))
        ImageDraw.Draw(thumb).text((13,282),f"Slide {i+1:02}",fill="#22313a")
        thumbs.append(thumb)
    for start in range(0,len(thumbs),9):
        sheet=Image.new("RGB",(1500,900),"#c5cecd")
        for j,thumb in enumerate(thumbs[start:start+9]):sheet.paste(thumb,((j%3)*500,(j//3)*300))
        sheet.save(QA/f"contact_{start//9+1:02}.png")
    (QA/"ppt_pdf_pages.json").write_text(json.dumps({"exported_at":datetime.now().astimezone().isoformat(),"pages":pages},ensure_ascii=False,indent=2),encoding="utf-8")
    print("EXPORT",len(pdf),"slides",PDF.stat().st_size,"bytes")
    suspect=[]
    for slide in geometry:
        for sh in slide["shapes"]:
            if "text" in sh and (sh["bound_height"]>sh["height"]+3 or sh["bound_width"]>sh["width"]+3):
                suspect.append((slide["slide"],sh["name"],sh["text"][:70],round(sh["bound_height"]-sh["height"],1),round(sh["bound_width"]-sh["width"],1)))
            for cell in sh.get("cells",[]):
                if cell["bound_height"]>cell["height"]+2 or cell["bound_width"]>cell["width"]+2:
                    suspect.append((slide["slide"],"cell",cell["text"][:70],round(cell["bound_height"]-cell["height"],1),round(cell["bound_width"]-cell["width"],1)))
    print("OVERFLOW_CANDIDATES",len(suspect))
    for item in suspect:print(item)


if __name__=="__main__":main()
