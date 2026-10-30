"""Export through the local Office-compatible COM applications, with owned instances."""
from pathlib import Path
import json
import argparse
import pythoncom
import win32com.client

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[2]
OUT=ROOT/"deliverables/lab02"
GROUP=f"第{json.loads((WORK/'brief.json').read_text(encoding='utf-8'))['group']}组"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--only",choices=["all","docs","ppt"],default="all");args=parser.parse_args()
    pythoncom.CoInitialize()
    old=WORK/"office-export.json"
    result=json.loads(old.read_text(encoding="utf-8")) if old.exists() else {"documents":[],"presentations":[]}
    if args.only in ("all","docs"):
        result["documents"]=[]
        for path in sorted(OUT.glob(GROUP+"_*.docx")):
            word=win32com.client.DispatchEx("Word.Application")
            word.Visible=False;word.DisplayAlerts=0
            doc=None
            try:
                doc=word.Documents.Open(str(path),ReadOnly=False,AddToRecentFiles=False,Visible=False)
                doc.Repaginate();doc.Fields.Update();doc.Save()
                stats={"file":path.name,"word_count":int(doc.ComputeStatistics(0)),"pages":int(doc.ComputeStatistics(2))}
                if "实验报告" in path.name:
                    start=doc.Content.Duplicate;end=doc.Content.Duplicate
                    if start.Find.Execute("摘要") and end.Find.Execute("参考文献"):
                        stats["main_text_word_count"]=int(doc.Range(start.Start,end.Start).ComputeStatistics(0))
                pdf=path.with_suffix(".pdf")
                try:doc.ExportAsFixedFormat(str(pdf),17)
                except Exception:doc.SaveAs(str(pdf),17)
                stats["pdf"]=pdf.name
                result["documents"].append(stats)
                print(json.dumps(stats,ensure_ascii=True),flush=True)
            finally:
                if doc is not None:
                    try:doc.Close(False)
                    except Exception:pass
                try:
                    if word.Documents.Count==0:word.Quit()
                except Exception:pass
    ppt=None
    try:
        paths=list(OUT.glob(GROUP+"_*.pptx")) if args.only in ("all","ppt") else []
        if paths:
            result["presentations"]=[]
            ppt=win32com.client.DispatchEx("PowerPoint.Application")
            try:ppt.DisplayAlerts=1
            except Exception:pass
        for path in paths:
            presentation=None
            try:
                presentation=ppt.Presentations.Open(str(path),ReadOnly=True,Untitled=False,WithWindow=False)
                pdf=path.with_suffix(".pdf")
                presentation.SaveAs(str(pdf),32)
                render=WORK/"slide-renders";render.mkdir(exist_ok=True)
                checks=[]
                for i in range(1,presentation.Slides.Count+1):
                    slide=presentation.Slides(i)
                    slide.Export(str(render/f"slide-{i:02d}.png"),"PNG",1600,900)
                    for j in range(1,slide.Shapes.Count+1):
                        shape=slide.Shapes(j)
                        try:
                            if shape.HasTextFrame and shape.TextFrame.HasText:
                                tr=shape.TextFrame.TextRange
                                if tr.BoundHeight > shape.Height+4:
                                    checks.append({"slide":i,"shape":j,"text":tr.Text[:100],"height":shape.Height,"text_height":tr.BoundHeight})
                        except Exception:pass
                data={"file":path.name,"pdf":pdf.name,"slides":int(presentation.Slides.Count),"text_height_flags":checks}
                result["presentations"].append(data);print(json.dumps(data,ensure_ascii=True),flush=True)
            finally:
                if presentation is not None:
                    try:presentation.Close()
                    except Exception:pass
    finally:
        if ppt is not None:
            try:
                if ppt.Presentations.Count==0:ppt.Quit()
            except Exception:pass
        pythoncom.CoUninitialize()
    (WORK/"office-export.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
