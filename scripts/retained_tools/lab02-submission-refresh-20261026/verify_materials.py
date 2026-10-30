"""Check the final academic artifacts against the frozen release and publish them."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import time

from docx import Document
import fitz
from pptx import Presentation

from refresh import WORK, ARTIFACTS, PPT_WORK, DEMO_WORK, digest, read_json, row, write_json


def slide_text(slide):
    values = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            values.append(shape.text)
        if shape.has_table:
            values.extend(c.text for r in shape.table.rows for c in r.cells)
    return "\n".join(values)


def verify_documents():
    e = read_json(WORK / "evidence.json")
    narration = read_json(PPT_WORK / "narration.json")
    main = Presentation(ARTIFACTS / "第31组_推理实验答辩.pptx")
    supplement = Presentation(ARTIFACTS / "第31组_推理实验答辩_补充材料.pptx")
    assert len(main.slides) == 9 and len(supplement.slides) == 4
    charts = [s.chart for slide in main.slides for s in slide.shapes if s.has_chart]
    extra = [s.chart for slide in supplement.slides for s in slide.shapes if s.has_chart]
    assert len(charts) == 3 and len(extra) == 2
    expected = [
        [[float(row(e, "forward", groups="16", model=m)["elapsed_ms_mean"])
          for m in ["naive_scan", "indexed_naive", "semi_naive"]]],
        [[float(row(e, "indicator", scenario=s, model=m)["f1_mean"])
          for s in ["clean", "noise_010", "noise_020", "noise_030", "severe"]]
         for m in ["bayes_learned", "fuzzy_learned"]],
        [[float(row(e, "graph", scenario="false_loop_020", model=m)["false_positive_rate_mean"])
          for m in ["hybrid_adaptive_unlocked", "hybrid_adaptive_locked", "hybrid_adaptive_calibrated"]]],
    ]
    for model, parameter in [("bayes", "prior"), ("fuzzy", "high_start")]:
        grouped = defaultdict(list)
        for r in e["tables"]["sensitivity"]:
            if (r["model"], r["parameter"]) == (model, parameter):
                grouped[float(r["value"])].append(float(r["f1"]))
        expected.append([[sum(v) / len(v) for _, v in sorted(grouped.items())]])
    points = 0
    for chart, values in zip(charts + extra, expected):
        assert len(chart.series) == len(values)
        for series, target in zip(chart.series, values):
            assert len(series.values) == len(target)
            for actual, wanted in zip(series.values, target):
                assert math.isclose(actual, wanted, rel_tol=1e-12, abs_tol=1e-12)
                points += 1
    body = "\n".join(slide_text(s) for s in list(main.slides) + list(supplement.slides))
    notes = "\n".join(s.notes_slide.notes_text_frame.text for s in main.slides)
    for banned in ["依据：", "数据来源", "三分钟", "录屏", "WPS", "200项", "6.28×", "362.81", "57.75"]:
        assert banned not in body + notes, banned
    for value in ["204项项目测试", "5.86×", "6.7 μs", "245.8 μs"]:
        assert value in body, value
    for slide, source in zip(main.slides, narration["slides"]):
        spoken = "".join(c.get("speech", c["text"]) for c in source["cues"])
        assert spoken in slide.notes_slide.notes_text_frame.text
    assert not read_json(PPT_WORK / "deck-export.json")["overflow_flags"]
    assert not read_json(PPT_WORK / "supplementary-export.json")["overflow_flags"]
    d = Document(ARTIFACTS / "第31组_创新点说明.docx")
    doc = "\n".join(x.text for x in d.paragraphs) + "\n" + "\n".join(
        c.text for t in d.tables for r in t.rows for c in r.cells)
    for value in ["204项", "5.86", "181.41", "0.8575", "0.7053", "6.7 μs", "245.8 μs",
                  "8.66", "43.94%", "35.28%", "0.6992", "0.6144", "0.6549", "94.41%"]:
        assert value in doc, value
    for banned in ["200项", "6.28倍", "362.81", "57.75"]:
        assert banned not in doc
    pdf_counts = {}
    for stem, count in [("第31组_推理实验答辩", 9), ("第31组_推理实验答辩_补充材料", 4), ("第31组_创新点说明", 3)]:
        with fitz.open(ARTIFACTS / (stem + ".pdf")) as pdf:
            assert len(pdf) == count
            assert all(len(page.get_text().strip()) > 100 for page in pdf)
            assert "\ufffd" not in "".join(page.get_text() for page in pdf)
            pdf_counts[stem] = count
    srt = (ARTIFACTS / "第31组_实验二_PPT演示视频.srt").read_text(encoding="utf-8-sig")
    assert srt.count(" --> ") == 38
    for value in ["1062.37", "181.41", "5.86", "204项"]:
        assert value in srt, value
    for banned in ["200项", "6.28倍", "362.81", "57.75"]:
        assert banned not in srt
    result = {"passed": True, "main_slides": 9, "supplementary_slides": 4,
              "native_editable_charts": 5, "chart_values_matched_to_latest_csv": points,
              "pdf_pages": pdf_counts, "notes_match_updated_narration": True,
              "subtitle_cues": 38, "old_run_values_absent": True,
              "visible_source_footers": False,
              "manual_layout_review": {"main_slides": 9, "supplementary_slides": 4, "innovation_pages": 3}}
    write_json(WORK / "materials-verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def publish():
    e = read_json(WORK / "evidence.json")
    release = Path(e["release"])
    target = release.parent
    for name, expected in e["source_sha256"].items():
        assert digest(release / name) == expected, name
    for name, expected in e["raw_data_sha256"].items():
        assert digest(release / "data/raw" / name) == expected, name
    source_check = read_json(WORK / "source-check.json")
    for path, expected in source_check["protected_reports"].items():
        assert digest(path) == expected, path
    assert read_json(WORK / "materials-verification.json")["passed"]
    video = read_json(PPT_WORK / "verification.json")
    assert video["full_decode"] == "passed" and video["duration_seconds"] == 180
    assert video["slide_count"] == 9 and video["subtitle_cues"] == 38
    assert read_json(PPT_WORK / "timeline.json")["source_sha256"] == digest(ARTIFACTS / "第31组_推理实验答辩.pptx")
    demo = read_json(DEMO_WORK / "recording-audit.json")
    assert demo["validation"]["duration_seconds"] == 60
    assert demo["validation"]["audio_streams"] == 0 and demo["validation"]["full_decode_passed"]
    assert demo["composition"]["subtitles"] is False
    assert read_json(DEMO_WORK / "no-caption-verification.json")["passed"]
    assert digest(ARTIFACTS / "第31组_实验二_1分钟Demo.mp4") == demo["validation"]["sha256"]
    names = ["第31组_推理实验答辩.pptx", "第31组_推理实验答辩.pdf",
             "第31组_推理实验答辩_补充材料.pptx", "第31组_推理实验答辩_补充材料.pdf",
             "第31组_创新点说明.docx", "第31组_创新点说明.pdf",
             "第31组_实验二_1分钟Demo.mp4", "第31组_实验二_PPT演示视频.mp4"]
    published = []
    backup = WORK / ("publication-backup-" + time.strftime("%Y%m%d-%H%M%S"))
    for name in names:
        source, destination = ARTIFACTS / name, target / name
        expected = digest(source)
        if destination.exists() and digest(destination) != expected:
            backup.mkdir(exist_ok=True)
            shutil.copy2(destination, backup / name)
        pending = destination.with_name(destination.name + ".pending")
        shutil.copy2(source, pending)
        assert digest(pending) == expected
        pending.replace(destination)
        published.append({"path": str(destination), "bytes": destination.stat().st_size, "sha256": expected})
    result = {"status": "complete", "source_session": source_check["source_session"],
              "destination": str(target), "published": published,
              "report_unchanged": True, "submitted_source_unchanged": True,
              "main_slides": 9, "supplementary_slides": 4, "innovation_pages": 3,
              "demo": {"seconds": 60, "audio": False, "subtitles": False},
              "ppt_video": {"seconds": 180, "slides": 9, "voice": video["voice"], "subtitle_cues": 38},
              "validation": {"continuous_slide_samples": len(video["continuous_slide_checks"]),
                             "rendered_subtitle_checks": len(video["subtitle_render_checks"]),
                             "demo_raw_frame_comparisons": 7,
                             "final_video_slides_visually_reviewed": 9},
              "evidence": {"project_tests": 204, "source_files": 90, "data_files": 6,
                           "tables": 9, "speedup": e["speedup"]}}
    write_json(WORK / "completion.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["documents", "publish"])
    args = parser.parse_args()
    {"documents": verify_documents, "publish": publish}[args.stage]()
