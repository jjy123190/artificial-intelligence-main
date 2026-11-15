"""Read-only course-deliverable checks; creates a local QA summary."""
from pathlib import Path
import json, re, zipfile, xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}

def audit():
    report = next((ROOT/'deliverables'/'lab01').glob('*.docx'))
    with zipfile.ZipFile(report) as archive:
        doc = ET.fromstring(archive.read('word/document.xml'))
        paragraphs = [''.join(x.text or '' for x in p.findall('.//w:t',NS)) for p in doc.findall('.//w:body/w:p',NS)]
    sections = {}
    heading = None
    for text in paragraphs:
        if re.match(r'^[一二三四五六七八九十]+、',text):
            heading=text; sections[heading]=[]
        elif heading: sections[heading].append(text)
    deck = next((ROOT/'deliverables'/'lab01').glob('*.pptx'))
    with zipfile.ZipFile(deck) as archive:
        paths=sorted([n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)], key=lambda n:int(re.search(r'(\d+)\.xml',n).group(1)))
        slides=[]
        for path in paths:
            text=[''.join(p.itertext()) for p in ET.fromstring(archive.read(path)).findall('.//a:t',NS)]
            slides.append({'slide':len(slides)+1,'text':text})
    ranges={'二、摘要':(270,330),'四、背景与目标':(300,400),'五、相关技术与工具准备':(400,500),'六、算法设计与实现':(600,800),'七、测试结果与分析':(400,500),'八、问题与改进思路':(300,400),'九、总结':(200,300)}
    counts={}
    for key,(low,high) in ranges.items():
        length=len(re.sub(r'\s+','',''.join(sections.get(key,[]))))
        counts[key]={'non_whitespace_characters':length,'suggested_range':[low,high],'within_range':low<=length<=high}
    result={'report':str(report.relative_to(ROOT)),'section_count':len(sections),'section_lengths':counts,'references':sections.get('十、参考文献',[]),'slides':slides,'note':'Character counts include captions/code; visual and semantic review still required.'}
    out=ROOT/'outputs/qa/lab01-deliverables';out.mkdir(parents=True,exist_ok=True)
    (out/'existing-deliverables-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'sections':len(sections),'slides':len(slides),'lengths':counts},ensure_ascii=False,indent=2))
    if len(sections)!=10 or not slides:raise SystemExit('Missing required report sections or slides')
if __name__=='__main__': audit()
