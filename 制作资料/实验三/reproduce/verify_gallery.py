"""Check the delivery's offline gallery and every SVG in a headless browser.
Only reads the project. Writes QA records/screenshots inside this delivery.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
from xml.etree import ElementTree as ET
import datetime
import hashlib
import json
import os

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parents[1]
QA = BASE / "reproduce/qa"
GALLERY = BASE / "图片与案例索引.html"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GalleryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.images = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        for attr in ("href", "src"):
            if attrs.get(attr):
                self.references.append(attrs[attr])
        if tag == "img":
            self.images.append(attrs)


def main():
    QA.mkdir(exist_ok=True)
    parser = GalleryParser()
    parser.feed(GALLERY.read_text(encoding="utf-8"))
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate HTML ids"
    assert len(parser.images) == 18
    for image in parser.images:
        assert image.get("alt"), image
    paths = set()
    for reference in parser.references:
        url = urlsplit(reference)
        assert not url.scheme and not url.netloc, ("Not an offline reference", reference)
        target = (BASE / unquote(url.path)).resolve() if url.path else GALLERY
        assert target.is_relative_to(BASE) and target.is_file(), ("Missing file", reference)
        paths.add(target)
        if url.fragment and not url.path:
            assert unquote(url.fragment) in parser.ids, ("Missing anchor", reference)

    layouts = []
    svg_issues = []
    svg_files = sorted((BASE / "figures").glob("*.svg"))
    assert len(svg_files) == 18
    with sync_playwright() as pw:
        chrome = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe"
        browser = pw.chromium.launch(headless=True, **({"executable_path": str(chrome)} if chrome.exists() else {}))
        try:
            for name, width, height in [("desktop", 1366, 900), ("mobile", 390, 844)]:
                # No app server, network or page JavaScript is required for navigation.
                context = browser.new_context(viewport={"width": width, "height": height}, offline=True, java_script_enabled=False)
                page = context.new_page()
                errors = []
                requests = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: requests.append(request.url))
                page.goto(GALLERY.as_uri(), wait_until="load")
                page.evaluate("""async () => {
                    for (const image of document.images) {image.loading='eager'; await image.decode();}
                    await document.fonts.ready;
                }""")
                layout = page.evaluate("""() => ({
                    width: innerWidth, scroll_width: document.documentElement.scrollWidth,
                    image_count: document.images.length,
                    broken_images: [...document.images].filter(i => !i.complete || !i.naturalWidth).map(i => i.src),
                    overflow: [...document.querySelectorAll('body *')].filter(e => {
                        const r=e.getBoundingClientRect(); return r.right > innerWidth+1 || r.left < -1;
                    }).map(e => ({tag:e.tagName, id:e.id}))
                })""")
                assert not errors and not layout["broken_images"] and not layout["overflow"], (name, errors, layout)
                assert layout["scroll_width"] == width and layout["image_count"] == 18
                assert all(not r.startswith(("http:", "https:")) for r in requests), requests
                page.locator('nav a[href="#09_tree_three_hop"]').click()
                assert page.evaluate("location.hash === '#09_tree_three_hop' && scrollY > 0")
                page.locator('[id="09_tree_three_hop"] a[href="#top"]').click()
                assert page.evaluate("location.hash === '#top' && scrollY <= 1")
                page.screenshot(path=str(QA / f"gallery_{name}.png"))
                layouts.append({"viewport": [width, height], "javascript_disabled": True,
                                "offline": True, "native_anchor_navigation": "passed", **layout})
                context.close()

            page = browser.new_page(viewport={"width": 1400, "height": 1100})
            for svg in svg_files:
                root = ET.parse(svg).getroot()
                assert root.tag == "{http://www.w3.org/2000/svg}svg"
                assert "viewBox" in root.attrib
                page.goto(svg.as_uri(), wait_until="load")
                page.evaluate("document.fonts.ready")
                issues = page.evaluate("""() => {
                    const svg=document.querySelector('svg'), v=svg.viewBox.baseVal;
                    const inverse=svg.getScreenCTM().inverse(), issues=[];
                    for (const text of svg.querySelectorAll('text')) {
                        const box=text.getBBox(), matrix=inverse.multiply(text.getScreenCTM());
                        const points=[[box.x,box.y],[box.x+box.width,box.y],[box.x,box.y+box.height],[box.x+box.width,box.y+box.height]]
                            .map(([x,y])=>new DOMPoint(x,y).matrixTransform(matrix));
                        const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
                        if(Math.min(...xs)<v.x-1 || Math.max(...xs)>v.x+v.width+1 || Math.min(...ys)<v.y-1 || Math.max(...ys)>v.y+v.height+1)
                            issues.push({text:text.textContent, bounds:[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)]});
                    }
                    return issues;
                }""")
                svg_issues.extend({"file": svg.name, **issue} for issue in issues)
            page.close()
        finally:
            browser.close()
    (QA / "svg_bounds.json").write_text(json.dumps(svg_issues, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert not svg_issues, svg_issues
    assets = [GALLERY] + sorted((BASE / "figures").glob("*.png")) + svg_files
    record = {
        "status": "passed", "verified_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "scope": "Offline delivery index only; original website is not started or modified.",
        "local_references_checked": len(parser.references), "unique_local_files": len(paths),
        "unique_anchors": len(parser.ids), "image_count": 18, "svg_files_checked": len(svg_files),
        "svg_text_out_of_bounds": 0, "layouts": layouts,
        "assets_sha256": {p.relative_to(BASE).as_posix(): sha(p) for p in assets},
    }
    (BASE / "evidence/gallery_validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("assets_sha256", "layouts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()