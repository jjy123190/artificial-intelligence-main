"""Export only the rewritten report, and count its text in the installed Office engine."""
import json
from pathlib import Path
import pythoncom
import win32com.client

WORK = Path(__file__).resolve().parent
path = WORK / "第31组_人工智能推理实验报告.docx"
pythoncom.CoInitialize()
word = document = None
try:
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = word.Documents.Open(str(path), ReadOnly=False, AddToRecentFiles=False, Visible=False)
    document.Fields.Update()
    document.Repaginate()
    document.Save()
    start = document.Content.Duplicate
    end = document.Content.Duplicate
    assert start.Find.Execute("摘要") and end.Find.Execute("参考文献")
    report = {"file": path.name, "word_count": int(document.ComputeStatistics(0)),
              "main_text_word_count": int(document.Range(start.Start, end.Start).ComputeStatistics(0)),
              "pages": int(document.ComputeStatistics(2))}
    pdf = path.with_suffix(".pdf")
    try:
        document.ExportAsFixedFormat(str(pdf), 17)
    except Exception:
        document.SaveAs(str(pdf), 17)
    report["pdf"] = pdf.name
    (WORK / "office-export.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
finally:
    if document is not None:
        document.Close(False)
    if word is not None:
        try:
            if word.Documents.Count == 0:
                word.Quit()
        except Exception:
            pass  # WPS may already dispose its automation object after closing the last document.
    pythoncom.CoUninitialize()
