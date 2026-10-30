"""Match the encoded demo to real browser captures and save a sharpness comparison."""
from pathlib import Path
import json
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parent
DEMO = WORK / "demo-native"


def main():
    audit = json.loads((DEMO / "recording-audit.json").read_text(encoding="utf-8"))
    video = cv2.VideoCapture(audit["final_video"])
    checks = []
    for shot in audit["shots"]:
        source = cv2.imread(str(DEMO / f"recorded-{shot['scene']}.png"))
        assert source.shape[:2] == (1440, 2560)
        center = (audit["started_epoch"] + shot["at_seconds"] - audit["capture_start_epoch"]) / 1.5
        candidates = []
        for offset in [-.16, -.10, -.04, .02, .08, .14]:
            frame_number = max(0, min(1799, round((center + offset) * 30)))
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, encoded = video.read()
            assert ok
            mse = float(np.mean((source.astype(np.float32) - encoded.astype(np.float32)) ** 2))
            candidates.append((mse, frame_number, encoded))
        mse, frame_number, matched = min(candidates, key=lambda item: item[0])
        psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-12))
        if psnr < 26:
            raise RuntimeError(f"Scene not preserved clearly: {shot['scene']}, PSNR={psnr:.2f}dB")
        cv2.imencode(".png", matched)[1].tofile(str(DEMO / f"matched-{shot['scene']}.png"))
        checks.append({"scene": shot["scene"], "frame": frame_number, "second": frame_number / 30, "psnr_db": psnr})
    video.release()
    assert len(checks) == 18
    report = {"all_recorded_scenes_present": True, "native_resolution": [2560, 1440],
              "minimum_scene_psnr_db": min(item["psnr_db"] for item in checks), "scenes": checks}
    (DEMO / "quality-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compare the same methods and values at equal displayed size; only the old
    # 720p image is enlarged here. The actual delivered movie is native 1440p.
    before = Image.open(WORK / "demo-before-42.png").convert("RGB")
    after = Image.open(DEMO / "matched-14-scoring-ablation.png").convert("RGB")
    old_crop = before.crop((168, 72, 698, 392)).resize((1060, 640), Image.Resampling.LANCZOS)
    new_crop = after.crop((336, 144, 1396, 784))
    comparison = Image.new("RGB", (2160, 715), "#f8f5ef")
    comparison.paste(old_crop, (10, 65))
    comparison.paste(new_crop, (1090, 65))
    draw = ImageDraw.Draw(comparison)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    draw.text((20, 15), "原视频 · 720p（放大对比）", font=font, fill="#4c453c")
    draw.text((1100, 15), "重新录制 · 原生1440p", font=font, fill="#813f20")
    comparison.save(WORK / "demo-quality-comparison.png")
    print(json.dumps({"scenes": len(checks), "minimum_psnr_db": report["minimum_scene_psnr_db"],
                      "comparison": str(WORK / "demo-quality-comparison.png")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
