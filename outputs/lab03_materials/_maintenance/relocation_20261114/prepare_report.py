"""Prepare a staged, path-only report update; retain every previous file version."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from zipfile import ZipFile
from xml.sax.saxutils import escape
import json
import shutil
from docx import Document

OPS=Path(__file__).resolve().parent
BASE=OPS.parents[1]
ROOT=BASE.parents[1]
NAME='第31组_实验三_实验报告'
BEFORE=OPS/'before'/f'{NAME}.docx'
STAGE=OPS/'staged'
STAGE.mkdir(exist_ok=True)
source=ROOT/'deliverables/lab3'/f'{NAME}.docx'
assert source.read_bytes()==BEFORE.read_bytes(),'Report changed during preparation; do not overwrite user edits'
doc=Document(BEFORE)
updates={
 12: '本报告已整合搜索树可视化样例与失败案例分析。\nPNG、SVG、完整轨迹及核验脚本另存为制作资料。',
 16: doc.paragraphs[16].text.replace('附件JSON','归档JSON'),
 19: doc.paragraphs[19].text.replace('配套deliverables/lab3_备份目录','制作资料目录outputs/lab03_materials'),
 30: doc.paragraphs[30].text.replace('附件report_metrics.json','归档report_metrics.json'),
 121: doc.paragraphs[121].text.replace('报告附件复制冻结结果','制作资料保留冻结结果'),
 128: doc.paragraphs[128].text.replace('附件：','归档记录：'),
 131: doc.paragraphs[131].text.replace('另附','另存'),
 149: doc.paragraphs[149].text.replace('附件F01','归档F01'),
 162: doc.paragraphs[162].text.replace('完整GA树另附','完整GA树另存'),
 172: doc.paragraphs[172].text.replace('中性树另附','中性树另存'),
 180: doc.paragraphs[180].text.replace('报告附其原件','制作资料保留原件'),
 221: '样例核验：可选拓展_20261113中的verify_examples.py与example_verification.json，位于outputs/lab03_materials；运行方法见附录A。',
 232: '表28路径以outputs/lab03_materials/evidence为根，raw文件逐字节保留。模型权重清单见原项目data/lab03/llm_model_manifest.json；制作时的文件指纹见reproduce/qa/protected_before.json。第14章样例及历史验收另存于可选拓展_20261113；原验收记录保留，不代表本次重跑。',
 233: '附录 A 复现与资料使用说明',
 235: '在项目根目录打开PowerShell。下列命令检查当前制作资料及报告文件，并重放两个给定样例；不重建主实验、不训练、不调用LLM。历史验收清单保留原样，不作为当前版本重跑结果。',
 236: "Set-Location 'outputs/lab03_materials'\npython -B -X utf8 verify_materials.py\npython -B -X utf8 可选拓展_20261113/verify_examples.py\nSet-Location ../..",
 239: '本次未执行上述重建命令。重跑算法后应整套更新证据，不能混用新模型/权重与旧指标。reproduce/build_report.py仅对应旧版报告；可选拓展_20261113保留当时的增补源与核验材料。不要用历史生成器或lab03 report/all入口覆盖当前报告。',
 241: '表29 重点资料定位',
 242: doc.paragraphs[242].text.replace('均相对lab3_备份','均相对outputs/lab03_materials'),
 243: '正式提交文件集中在deliverables/lab3；制作源、图片、冻结证据与核验脚本独立存放于outputs/lab03_materials，不是另一套提交材料。成员职责见附录B；课程平台项目、组员关联与管理员提交仍由小组自行确认。',
}
# Replace only w:t content in the original package. Preserve every other ZIP part,
# paragraph/run property, embedded figure, table, header and footer byte-for-byte.
replacements=[]
for i,new in updates.items():
    paragraph=doc.paragraphs[i]
    if len(paragraph.runs)==1:
        replacements.append((paragraph.runs[0].text,new))
    else:
        old_lines=paragraph.text.split('\n'); new_lines=new.split('\n')
        assert len(old_lines)==len(new_lines),(i,old_lines,new_lines)
        replacements.extend((a,b) for a,b in zip(old_lines,new_lines) if a!=b)
with ZipFile(BEFORE) as z:
    raw=z.read('word/document.xml')
    for old,new in replacements:
        a=escape(old).encode('utf-8');b=escape(new).encode('utf-8')
        assert raw.count(a)==1,(old,raw.count(a))
        raw=raw.replace(a,b,1)
    assert 'lab3_备份'.encode('utf-8') not in raw
    target=STAGE/f'{NAME}.docx'
    with ZipFile(target,'w') as out:
        for info in z.infolist():out.writestr(info,raw if info.filename=='word/document.xml' else z.read(info.filename))
updated=Document(target)
assert len(doc.paragraphs)==len(updated.paragraphs)
for i,(a,b) in enumerate(zip(doc.paragraphs,updated.paragraphs)):
    assert b.text==updates.get(i,a.text),(i,a.text,b.text)
    if i not in updates:assert a._p.xml==b._p.xml,f'Unexpected paragraph formatting change {i}'
assert len(updated.tables)==len(doc.tables)==30
assert len(updated.inline_shapes)==len(doc.inline_shapes)==14
assert all(a._tbl.xml==b._tbl.xml for a,b in zip(doc.tables,updated.tables))
with ZipFile(BEFORE) as a, ZipFile(target) as b:
    assert a.namelist()==b.namelist()
    assert [n for n in a.namelist() if a.read(n)!=b.read(n)]==['word/document.xml']
record={'scope':'Only directory references and material/attachment wording; no experimental content or numeric tables changed.',
        'before_sha256':sha256(BEFORE.read_bytes()).hexdigest(), 'staged_sha256':sha256(target.read_bytes()).hexdigest(),
        'paragraph_updates':{str(i):{'before':doc.paragraphs[i].text,'after':v} for i,v in updates.items()},
        'unchanged_tables':30,'unchanged_embedded_figures':14}
(OPS/'report_update.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('STAGED report:',target)
print('Edited directory/material paragraphs:',len(updates),'tables unchanged: 30; figures unchanged: 14')
