"""Append one native, editable closing slide without rewriting the user's slides."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E
from pptx import Presentation

WORK = Path(__file__).resolve().parent
META = json.loads((WORK/'before.json').read_text(encoding='utf-8'))
SOURCE = Path(META['backup'])
DEST = WORK/'第31组_实验三_汇报_含结束页.pptx'
N = {
    'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel':'http://schemas.openxmlformats.org/package/2006/relationships',
    'ct':'http://schemas.openxmlformats.org/package/2006/content-types',
    'ep':'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties',
    'vt':'http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes',
}


def q(prefix, local): return '{'+N[prefix]+'}'+local

def serial(root): return E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)

def inch(value): return str(round(value*914400))


def main():
    with ZipFile(SOURCE) as zin:
        entries = {name:zin.read(name) for name in zin.namelist()}
        presentation = E.fromstring(entries['ppt/presentation.xml'])
        ids = presentation.find('p:sldIdLst',N)
        count = len(ids)
        assert count == 16, count
        assert not presentation.xpath('.//*[local-name()="sectionLst"]'), 'Review sections before appending'
        assert not any(name.startswith('_xmlsignatures/') for name in entries), 'Signed deck needs explicit handling'
        cover = E.fromstring(entries['ppt/slides/slide1.xml'])
        shape_tree = cover.find('p:cSld/p:spTree',N)
        templates={shape.find('.//p:cNvPr',N).get('name'):shape for shape in shape_tree
                   if shape.find('.//p:cNvPr',N) is not None}
        closing = deepcopy(cover)
        closing.find('p:cSld',N).set('name','谢谢聆听')
        tree = closing.find('p:cSld/p:spTree',N)
        for node in list(tree):
            if E.QName(node).localname not in ['nvGrpSpPr','grpSpPr']:
                tree.remove(node)

        def put(shape):
            shape.find('.//p:cNvPr',N).set('id',str(len(tree)))
            tree.append(shape)
            return shape

        def clone(name): return deepcopy(templates[name])

        def position(shape,x,y,w,h):
            transform=shape.find('p:spPr/a:xfrm',N)
            off=transform.find('a:off',N);off.set('x',inch(x));off.set('y',inch(y))
            ext=transform.find('a:ext',N);ext.set('cx',inch(w));ext.set('cy',inch(h))

        def text(name,value,x,y,w,h,pt,color,bold=False,font='微软雅黑',tracking=None):
            shape=clone('TextBox 33')
            shape.find('.//p:cNvPr',N).set('name',name)
            position(shape,x,y,w,h)
            body=shape.find('p:txBody',N)
            body_pr=body.find('a:bodyPr',N);body_pr.set('anchor','ctr');body_pr.set('wrap','none')
            for node in list(body):
                if E.QName(node).localname=='p':body.remove(node)
            paragraph=E.SubElement(body,q('a','p'))
            props=E.SubElement(paragraph,q('a','pPr'),algn='ctr')
            E.SubElement(E.SubElement(props,q('a','lnSpc')),q('a','spcPct'),val='100000')
            for spacing in ['spcBef','spcAft']:
                E.SubElement(E.SubElement(props,q('a',spacing)),q('a','spcPts'),val='0')
            run=E.SubElement(paragraph,q('a','r'))
            rp=E.SubElement(run,q('a','rPr'),lang='zh-CN',sz=str(round(pt*100)),b='1' if bold else '0')
            if tracking is not None:rp.set('spc',str(tracking))
            E.SubElement(E.SubElement(rp,q('a','solidFill')),q('a','srgbClr'),val=color)
            for node in ['latin','ea','cs']:E.SubElement(rp,q('a',node),typeface=font)
            E.SubElement(run,q('a','t')).text=value
            end=deepcopy(rp);end.tag=q('a','endParaRPr');paragraph.append(end)
            return put(shape)

        # Original visual language and footer are reused as native editable shapes.
        put(clone('Rectangle 1'))
        put(clone('TextBox 2'))
        text('S17-english','THANK YOU',4.5,2.16,7.0,0.42,18,'E58D5C',True,'Segoe UI',300)
        text('S17-thanks','谢谢聆听',2.5,2.96,11.0,1.3,68,'F9F6EF',True)
        rule=clone('Rectangle 1');rule.find('.//p:cNvPr',N).set('name','S17-accent-rule')
        position(rule,7.55,4.43,0.9,0.045);put(rule)
        text('S17-subtitle','敬请老师批评指正',3.0,4.85,10.0,0.62,26,'B9C8C9')
        put(clone('Connector 32'))
        put(clone('TextBox 33'))
        put(clone('TextBox 34'))
        number=text('S17-page-number','17',14.7,8.15,0.5,0.35,14,'B9C8C9')
        number.find('p:txBody/a:p/a:pPr',N).set('algn','r')

        slide_numbers=[int(name.removeprefix('ppt/slides/slide').removesuffix('.xml')) for name in entries
                       if name.startswith('ppt/slides/slide') and name.endswith('.xml')]
        new_number=max(slide_numbers)+1
        assert new_number == 17
        slide_name=f'ppt/slides/slide{new_number}.xml'
        rel_name=f'ppt/slides/_rels/slide{new_number}.xml.rels'
        relationships=E.fromstring(entries['ppt/_rels/presentation.xml.rels'])
        next_id=max(int(rel.get('Id')[3:]) for rel in relationships if rel.get('Id','').startswith('rId'))+1
        relationship_id='rId'+str(next_id)
        E.SubElement(relationships,q('rel','Relationship'),Id=relationship_id,
                     Type=N['r']+'/slide',Target=f'slides/slide{new_number}.xml')
        E.SubElement(ids,q('p','sldId'),id=str(max(int(i.get('id')) for i in ids)+1),attrib={q('r','id'):relationship_id})
        types=E.fromstring(entries['[Content_Types].xml'])
        E.SubElement(types,q('ct','Override'),PartName='/'+slide_name,
                     ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
        new_rels=E.fromstring(entries['ppt/slides/_rels/slide1.xml.rels'])
        for rel in list(new_rels):
            if not rel.get('Type').endswith('/slideLayout'):new_rels.remove(rel)
        assert len(new_rels)==1
        app=E.fromstring(entries['docProps/app.xml'])
        app.find('ep:Slides',N).text=str(count+1)
        heading=app.find('ep:HeadingPairs/vt:vector',N)
        for index,node in enumerate(heading):
            label=node.find('vt:lpstr',N)
            if label is not None and label.text in ['幻灯片标题','Slide Titles']:
                heading[index+1].find('vt:i4',N).text=str(count+1)
        titles=app.find('ep:TitlesOfParts/vt:vector',N)
        E.SubElement(titles,q('vt','lpstr')).text='谢谢聆听'
        titles.set('size',str(len(titles)))
        changes={'ppt/presentation.xml':serial(presentation),
                 'ppt/_rels/presentation.xml.rels':serial(relationships),
                 '[Content_Types].xml':serial(types),'docProps/app.xml':serial(app)}
        added={slide_name:serial(closing),rel_name:serial(new_rels)}
        assert not DEST.exists(),DEST
        with ZipFile(DEST,'x',compression=ZIP_DEFLATED) as zout:
            zout.comment=zin.comment
            for info in zin.infolist():zout.writestr(info,changes.get(info.filename,entries[info.filename]))
            for name,content in added.items():zout.writestr(name,content)
    with ZipFile(SOURCE) as old, ZipFile(DEST) as new:
        assert new.testzip() is None
        actual=[name for name in old.namelist() if old.read(name)!=new.read(name)]
        assert set(actual)==set(changes),(actual,list(changes))
        original_parts=[name for name in old.namelist() if name.startswith(('ppt/slides/','ppt/notesSlides/','ppt/media/','ppt/charts/','ppt/embeddings/','ppt/slideMasters/','ppt/slideLayouts/','ppt/theme/'))]
        assert all(old.read(name)==new.read(name) for name in original_parts)
        for name in new.namelist():
            if name.endswith(('.xml','.rels')):E.fromstring(new.read(name))
    prs=Presentation(DEST)
    assert len(prs.slides)==17
    original=Presentation(SOURCE)
    assert prs.slide_width==original.slide_width and prs.slide_height==original.slide_height
    for i in range(16):
        assert [s.text for s in prs.slides[i].shapes if s.has_text_frame] == [s.text for s in original.slides[i].shapes if s.has_text_frame]
    last_text=[s.text for s in prs.slides[-1].shapes if s.has_text_frame]
    assert '谢谢聆听' in last_text and '敬请老师批评指正' in last_text
    report={'source':META['source'],'backup':str(SOURCE),'candidate':str(DEST),'slides_before':16,'slides_after':17,
            'before_sha256':META['before_sha256'],'candidate_sha256':hashlib.sha256(DEST.read_bytes()).hexdigest(),
            'changed_existing_package_parts':actual,'added_parts':list(added),
            'unchanged_original_content_parts':len(original_parts),'original_slides_byte_identical':True,
            'editable_closing_slide':True,'closing_text':last_text,'visual_review':'pending'}
    (WORK/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
