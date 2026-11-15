"""Read-only checks of editable slides, real chart numbers, and report preservation."""
from datetime import datetime
from hashlib import sha256
import csv
import json
from pathlib import Path
import re
import statistics
import unicodedata
from zipfile import ZipFile
import sys

from docx import Document
import fitz
from pptx import Presentation

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sys.path.insert(0, str(BASE.parent))
from verify_materials import historical_path, check_inventory


def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))


def hashfile(p):
    h=sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""):h.update(c)
    return h.hexdigest()


def norm(s):
    return re.sub(r"\s+","",unicodedata.normalize("NFKC",s)).replace("\u200b","")


def near(a,b):return abs(float(a)-float(b))<1e-9


def main():
    p=BASE/"draft/第31组_实验三_汇报.pptx";pdf_path=p.with_suffix(".pdf")
    spec=load(BASE/"source/deck_spec.json")
    ppt=Presentation(p);pdf=fitz.open(pdf_path)
    checks=[]
    def require(ok,label):
        if not ok:raise AssertionError(label)
    def passed(label,**kw):checks.append({"check":label,"passed":True,**kw})
    require(len(ppt.slides)==len(pdf)==18,"Slide count mismatch")
    require(near(ppt.slide_width/914400,16) and near(ppt.slide_height/914400,9),"Expected16:9")
    require(all(s.has_notes_slide and len(s.notes_slide.notes_text_frame.text)>100 for s in ppt.slides),"Speaker notes missing")
    require(all(near(pg.rect.width/pg.rect.height,16/9) for pg in pdf),"PDF ratio mismatch")
    passed("18_slides_16x9_with_speaker_notes",main_slides=16,appendix_slides=2)
    charts=[sh.chart for s in ppt.slides for sh in s.shapes if sh.has_chart]
    require(len(charts)==4,"Expected four native editable charts")
    require(sum(sh.shape_type==13 for s in ppt.slides for sh in s.shapes)==1,"Expected only current UI screenshot as raster")
    with ZipFile(p) as z:
        require(z.testzip() is None,"Corrupt PPTX zip")
        require(len([n for n in z.namelist() if n.startswith("ppt/embeddings/") and n.endswith(".xlsx")])==4,"Editable chart workbooks missing")
        require(b'TargetMode="External"' not in b"".join(z.read(n) for n in z.namelist() if n.endswith(".rels")),"External file dependency")
    passed("native_charts_tables_diagrams_and_embedded_image",charts=4,chart_workbooks=4)
    doc=Document(BASE/"source/第31组_实验三_实验报告.docx")
    tb=[[[c.text for c in row.cells] for row in t.rows] for t in doc.tables]
    for j,seed in enumerate(["7","19","43"]):
        true=load(BASE.parent/"evidence/raw/ga_history.json")[seed]["history"]
        require(all(near(x,y["fitness"]) for x,y in zip(charts[0].series[j].values,true)),"GA observations differ")
    for j,split in enumerate([8,9]):
        index={r[0]:r for r in tb[split][1:]}
        for x,method in zip(charts[1].series[j].values,["BFS","A* 固定","A* GA","MCTS","混合"]):
            expected=float(index[method][1].split("(")[1].rstrip(")%"))/100
            require(near(x,expected),"Success chart differs from report")
    with (BASE.parent/"evidence/raw/runs.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==3888,"Main run count changed")
    rs=[r for r in rows if r["split"]=="test" and int(r["budget"])==8]
    for chart,field in [(charts[2],"expanded"),(charts[3],"adjacency_edge_scans")]:
        for value,method in zip(chart.series[0].values,["bfs","astar_fixed","astar_ga","mcts"]):
            expected=round(statistics.mean(float(r[field]) for r in rs if r["method"]==method),3)
            require(near(value,expected),"Cost chart differs from raw CSV")
    rollout=round(statistics.mean(float(r["total_work"])-float(r["expanded"]) for r in rs if r["method"]=="mcts"),3)
    require(rollout==6.935,"MCTS rollout mismatch")
    passed("real_ga_success_and_cost_charts_match_report_and_csv",raw_rows=3888)
    success_text=norm(pdf[7].get_text())
    require(all(value in success_text for value in ["14.81%","9.26%","78.70%","75.93%","99.07%","98.15%","100.00%"]),"Percent labels lost their number format in Office")
    passed("success_chart_percent_labels_rendered")
    for slide_index,source_rows in [(11,spec["metrics"]["llm_table"]),(16,spec["metrics"]["full_metrics_table"])]:
        table=next(sh.table for sh in ppt.slides[slide_index].shapes if sh.has_table)
        require([[c.text for c in row.cells] for row in table.rows][1:]==source_rows,"Table changed from content source")
    require(spec["metrics"]["migration"]["positive"]==2 and spec["metrics"]["migration"]["negative"]==8,"Migration check count mismatch")
    members="".join(c.text for sh in ppt.slides[17].shapes if sh.has_table for row in sh.table.rows for c in row.cells)
    require(all(n in members for n in ["梁镇贤","陈宗波","温文辉","陈宇冲","刘亦达"]),"Member missing")
    require(not any(t in members for t in ["工作量","23%","18%","占比","贡献率"]),"Workload weight leaked")
    passed("llm_migration_and_team_text_consistent")
    text_items=0
    for i,slide in enumerate(ppt.slides):
        rendered=norm(pdf[i].get_text())
        for sh in slide.shapes:
            # Textboxes and native table cells should survive the Office export.
            values=[sh.text] if sh.has_text_frame else []
            if sh.has_table:values.extend(c.text for row in sh.table.rows for c in row.cells)
            for val in values:
                require(norm(val) in rendered,f"Text missing from PDF slide {i+1}: {val[:75]}")
                text_items+=bool(norm(val))
            if sh.shape_type!=9: # Connectors may point left or upwards.
                require(sh.left>=0 and sh.top>=0 and sh.left+sh.width<=ppt.slide_width+100 and sh.top+sh.height<=ppt.slide_height+100,
                        f"Shape outside slide {i+1}: {sh.name}")
        require("\ufffd" not in rendered,f"Replacement character on slide{i+1}")
    passed("ppt_pdf_all_text_and_table_cells_consistent",items=text_items)
    geometry=load(BASE/"qa/office_geometry.json")
    overflow=[]
    for row in geometry:
        for sh in row["shapes"]:
            if "text" in sh and (sh["bound_height"]>sh["height"]+3 or sh["bound_width"]>sh["width"]+3):
                overflow.append((row["slide"],sh["name"]))
            for cell in sh.get("cells",[]):
                if cell["bound_height"]>cell["height"]+2 or cell["bound_width"]>cell["width"]+2:
                    overflow.append((row["slide"],"cell",cell["row"],cell["column"]))
    require(not overflow,"Office text overflow: "+str(overflow))
    passed("powerpoint_text_boxes_and_cells_fit",overflow=0)
    check_inventory()  # Verify current files as well as the preserved historical baseline.
    baseline=load(BASE/"qa/protected_before.json")
    user_save=load(BASE/"qa/user_report_saved_during_build.json")
    changed=[]
    for rel,expected in baseline["protected_files"].items():
        path=historical_path(rel)
        if rel==user_save["changed_relative_path"]:
            require(expected==user_save["original_source_sha256"],"User-save old snapshot mismatch")
            require(hashfile(path)==user_save["latest_source_sha256"],"User edited report again; synchronize before publication")
            continue
        if not path.is_file() or hashfile(path)!=expected:changed.append(rel)
    require(not changed,"Pre-existing files changed: "+str(changed))
    require(hashfile(BASE/"source/第31组_实验三_实验报告.docx")==user_save["latest_source_sha256"],"Latest saved report source mismatch")
    passed("historical_evidence_preserved_and_current_material_inventory_verified",files=len(baseline["protected_files"]),independent_user_save=user_save["changes"])
    record={"status":"passed","checked_at":datetime.now().astimezone().isoformat(),"checks":checks,
            "pptx_sha256":hashfile(p),"pdf_sha256":hashfile(pdf_path),"slides":18,"main_slides":16,"appendix_slides":2,
            "scope":"Deck package, existing Office rendering, real numeric observations, preserved source snapshots and current relocated files; no Office export, algorithm test or benchmark rerun."}
    (BASE/"qa/ppt_validation_relocated.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print("VALIDATION",record["status"],len(checks),"checks")
    for c in checks:print("PASS",c["check"])


if __name__=="__main__":main()
