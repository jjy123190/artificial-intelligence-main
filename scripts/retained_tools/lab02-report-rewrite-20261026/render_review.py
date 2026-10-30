from pathlib import Path
import json
import math
import fitz
from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parent
pdf = fitz.open(WORK / "第31组_人工智能推理实验报告.pdf")
renders = WORK / "page-renders"
renders.mkdir(exist_ok=True)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
pages = []
for index, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
    pix.save(renders / f"page-{index+1:02}.png")
    text = page.get_text()
    lines = [line for line in text.splitlines() if line.strip()]
    outside = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                if x0 < 15 or y0 < 15 or x1 > page.rect.width - 15 or y1 > page.rect.height - 15:
                    outside.append(span["text"])
    pages.append({"page": index+1, "text_characters": len(text), "first_lines": lines[:3],
                  "last_lines": lines[-4:], "outside_page": outside})

for start in range(0, len(pdf), 6):
    sheet = Image.new("RGB", (1050, 2 * 530), "#dddcd8")
    draw = ImageDraw.Draw(sheet)
    for offset in range(min(6, len(pdf)-start)):
        image = Image.open(renders / f"page-{start+offset+1:02}.png").convert("RGB")
        image.thumbnail((340, 495))
        x, y = (offset % 3) * 350 + 5, (offset // 3) * 530
        draw.text((x+8, y+4), f"第{start+offset+1}页", font=font, fill="#222")
        sheet.paste(image, (x, y+29))
    sheet.save(WORK / f"contact-{start+1:02}.png")
(WORK / "layout-review.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(pages, ensure_ascii=False, indent=2))
