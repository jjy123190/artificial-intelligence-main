"""Read-only verification for the migration-supplement report.

Default: verify the published DOCX/PDF, release manifest, original content, all
listed pre-existing source/evidence fingerprints, and the two given examples.
No lab03 search, training, model inference, document export, or resealing occurs.
Use --staged to review draft files before publication (not a release approval).
Use --write-record only to save this verification result in this extension's QA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn
import fitz
from lxml import etree

from verify_examples import run_checks

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
NAME='第31组_实验三_实验报告'


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def norm(text: str) -> str:
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).replace('\u200b','').replace('\ufeff','')


def c14n(element) -> bytes:
    return etree.tostring(element, method='c14n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged',action='store_true',help='Check the draft instead of the formal report; does not accept an unsealed release.')
    parser.add_argument('--write-record',action='store_true',help='Write only qa/staged_validation.json or qa/release_validation.json.')
    args=parser.parse_args()
    folder=BASE/'draft' if args.staged else ROOT/'deliverables/lab3'
    src=folder/(NAME+'.docx'); pdf_path=folder/(NAME+'.pdf')
    snapshot=BASE/'before'/(NAME+'.docx')
    baseline=load(BASE/'qa/protected_before.json')
    spec=load(BASE/'qa/update_spec.json')
    checks=[]

    def passed(name, **details):
        checks.append({'check':name,'passed':True,**details})

    # This snapshot is not re-created during verification.
    for rel,expected in baseline['formal_before'].items():
        path=BASE/'before'/Path(rel).name
        require(sha(path)==expected,f'Before snapshot changed: {path.name}')
    require(sha(snapshot)==spec['base_sha256'],'Report source snapshot mismatch')
    passed('pre_edit_snapshot_intact', files=2)

    changes=[]
    for rel,expected in baseline['protected_files'].items():
        p=(ROOT/rel).resolve()
        require(p.is_relative_to(ROOT),'Invalid baseline path')
        if not p.is_file() or sha(p)!=expected:
            changes.append(rel)
    require(not changes,f'Pre-existing project/backup files changed: {changes}')
    passed('pre_existing_source_data_output_backup_hashes_unchanged', files=len(baseline['protected_files']))

    with ZipFile(snapshot) as oldzip, ZipFile(src) as newzip:
        require(set(oldzip.namelist())==set(newzip.namelist()),'DOCX package entries changed')
        changed=[name for name in oldzip.namelist() if oldzip.read(name)!=newzip.read(name)]
        require(changed==['word/document.xml'],f'Unexpected DOCX package edits: {changed}')
        passed('only_body_xml_changed', preserved_parts=len(oldzip.namelist())-1)

    old=Document(snapshot); new=Document(src)
    for i,p in enumerate(old.paragraphs):
        after_i=spec['original_paragraph_after_index'][str(i)]
        q=new.paragraphs[after_i]
        expected=spec['paragraph_updates'].get(str(i),p.text)
        require(q.text==expected,f'Original paragraph mismatch: {i}')
        if str(i) not in spec['paragraph_updates']:
            require(c14n(p._p)==c14n(q._p),f'Original paragraph formatting changed: {i}')
    passed('all_original_paragraphs_preserved_except_documented_edits',original_count=len(old.paragraphs),documented_updates=len(spec['paragraph_updates']))

    require(len(new.tables)==30,'Expected 25 original + 5 new tables')
    for i,table in enumerate(old.tables):
        mapped=i if i<22 else i+5
        after=new.tables[mapped]
        if i==0:
            require(len(after.rows)==len(table.rows)+1,'Navigation table should gain one row')
            for r1,r2 in zip(table.rows,after.rows):
                require(c14n(r1._tr)==c14n(r2._tr),'Original navigation row changed')
            require([c.text for c in after.rows[-1].cells]==spec['nav_row_added'],'New navigation row mismatch')
        else:
            require(c14n(table._tbl)==c14n(after._tbl),f'Original table changed: {i}')
    require(len(new.inline_shapes)==len(old.inline_shapes)==14,'Embedded figure count changed')
    passed('all_original_tables_and_embedded_figures_preserved',original_tables=25,added_tables=5,images=14)

    ids=[e.get(qn('w14:paraId')) for e in new._element.iter() if e.get(qn('w14:paraId'))]
    require(not [k for k,v in Counter(ids).items() if v>1],'Duplicate Word paragraph identifiers')
    captions=[int(re.match(r'表(\d+)\s',p.text).group(1)) for p in new.paragraphs if p.style.name=='Caption' and re.match(r'表(\d+)\s',p.text)]
    require(captions==list(range(1,31)),f'Table captions out of sequence: {captions}')
    passed('table_numbering_and_word_identifiers',numbered_tables=30)

    # The exact text and boundaries of the supplement are part of the checked source.
    content=load(BASE/'extension_content.json')
    for item in content:
        if item['kind']=='table':continue
        require(any(p.text==item['text'] for p in new.paragraphs),'Missing source paragraph: '+item['text'][:60])
    new_tables=[item for item in content if item['kind']=='table']
    for table,item in zip(new.tables[22:27],new_tables):
        require([[c.text for c in row.cells] for row in table.rows]==[item['headers']]+item['rows'],'Supplement table disagrees with its source')
    passed('supplement_matches_editable_source')

    actual=run_checks();stored=load(BASE/'example_verification.json')
    require(actual==stored,'Stored example evidence differs from deterministic replay')
    require(actual['positive_examples_accepted']==2 and actual['negative_cases_rejected']==8,'Example count mismatch')
    proof_rows=new_tables[1]['rows']
    for row,trace in zip(proof_rows,actual['theorem']['trace']):
        goal='；'.join(a+'='+b for a,b in trace['open_goals']) or '∅（全部关闭）'
        require(row[0]==str(trace['step']) and row[2]==goal,'Reported proof obligation differs from checker')
    short={'ontable':'T','on':'O','clear':'C','holding':'H','handempty':'E'}
    def compact(fact):
        head=fact.split('(',1)[0]
        return short[head]+(fact[len(head):] if '(' in fact else '')
    for row,trace in zip(new_tables[3]['rows'],actual['planning']['trace'][1:]):
        expected=[str(trace['step']),trace['action'],'{'+', '.join(compact(x) for x in trace['state'])+'}',str(trace['g'])]
        require(row==expected,'Reported planning state differs from checker')
    passed('example_certificates_replayed_and_match_report',positive_examples=2,negative_cases=8,plan_cost=4)

    run_counts={}
    for rel in ['outputs/lab03/runs.csv','deliverables/lab3_备份/evidence/raw/runs.csv']:
        with (ROOT/rel).open(encoding='utf-8-sig',newline='') as stream:
            count=sum(1 for _ in csv.DictReader(stream))
        require(count==3888,f'Main experiment row count changed: {rel}={count}')
        run_counts[rel]=count
    passed('original_3888_main_runs_not_changed',row_counts=run_counts)

    pdf=fitz.open(pdf_path)
    require(len(pdf)==30,f'Expected 30 pages, got {len(pdf)}')
    pages=[norm(p.get_text()) for p in pdf];full=''.join(pages)
    checked_text_items=0
    for p in new.paragraphs:
        if norm(p.text):
            require(norm(p.text) in full,'DOCX paragraph missing from PDF: '+p.text[:70]);checked_text_items+=1
    for table in new.tables:
        for row in table.rows:
            for cell in row.cells:
                if norm(cell.text):
                    require(norm(cell.text) in full,'DOCX table text missing from PDF: '+cell.text[:70]);checked_text_items+=1
    require('\ufffd' not in full,'PDF contains a Unicode replacement character')
    passed('docx_pdf_text_consistency',checked_text_items=checked_text_items,pdf_pages=len(pdf))

    expected_headings={24:'14 可选拓展：数学定理证明与规划迁移',25:'14.2 样例一',26:'14.3 样例二',27:'14.4 验证指标',28:'参考文献与证据来源',29:'附录 A 复现',30:'附录 B 成员分工'}
    for page_num,text in expected_headings.items():
        require(norm(text) in pages[page_num-1],f'Unexpected heading pagination: {text}')
    for page_num,item in zip([24,25,26,26,27],new_tables):
        require(norm(item['caption']) in pages[page_num-1],'Caption separated from its table')
        for row in [item['headers']]+item['rows']:
            for text in row:
                require(norm(text) in pages[page_num-1],'New table split across pages')
    for page_num in range(24,31):
        page=pdf[page_num-1]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if 55<s['bbox'][1]<785]
        require(all(60<=s['bbox'][0] and s['bbox'][2]<=535 and s['bbox'][3]<=785 for s in spans),f'New page text overflow: {page_num}')
    passed('supplement_pages_and_tables_not_split_or_outside_body',supplement_pages=[24,25,26,27])

    # Rendered preservation of the 21 untouched pages, not just their text.
    baseline_pdf=fitz.open(BASE/'qa/before_userdoc.pdf')
    untouched=[1]+list(range(3,23))
    for page_num in untouched:
        a=baseline_pdf[page_num-1].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        b=pdf[page_num-1].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        require((a.width,a.height,a.samples)==(b.width,b.height,b.samples),f'Unedited page rendering changed: {page_num}')
    passed('untouched_original_pages_pixel_identical',pages=untouched)
    appendix=pages[29]
    require(not any(term in appendix for term in ['工作量','百分比','23%','18%','占比']),'Workload weights leaked into member appendix')
    require('deliverables/lab3/reproduce/' not in full and 'deliverables/lab3/evidence/' not in full,'Outdated attachment root remains')
    passed('member_assignments_and_attachment_paths')

    refs=load(BASE/'reference_checks.json')['records']
    require(len(refs)==2 and all(r['status_code']==200 for r in refs),'Reference check evidence missing')
    require(all(term in refs[0]['checked_excerpts'] for term in ['Eq.symm','Eq.trans']),'Equality reference support missing')
    require(refs[1]['metadata']['DOI']=='10.1016/0004-3702(71)90010-5','STRIPS DOI mismatch')
    passed('new_external_reference_records_present',sources=2)

    if not args.staged:
        manifest=load(BASE/'release_manifest.json')
        for rel,expected in manifest['report_files'].items():
            path=(ROOT/rel).resolve()
            require(path.is_relative_to(ROOT),'Invalid release manifest path')
            require(sha(path)==expected,f'Release report hash mismatch: {rel}')
        for rel,expected in manifest['support_files'].items():
            path=(BASE/rel).resolve()
            require(path.is_relative_to(BASE),'Invalid support manifest path')
            require(sha(path)==expected,f'Release support hash mismatch: {rel}')
        user_files=[p.name for p in folder.iterdir() if not p.name.startswith('~$')]
        require(sorted(user_files)==sorted([NAME+'.docx',NAME+'.pdf']),f'Unexpected formal delivery items: {user_files}')
        passed('published_release_hashes_and_two_file_layout',support_files=len(manifest['support_files']))
    else:
        passed('staged_review_only_not_yet_published')

    record={'status':'passed','checked_at':datetime.now().astimezone().isoformat(),'mode':'staged' if args.staged else 'published','report_files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [src,pdf_path]},'checks':checks,'scope':'Document/evidence preservation plus replay of two given examples. No full cross-domain solver benchmark or new lab03 acceptance run.'}
    if args.write_record:
        dest=BASE/'qa'/('staged_validation.json' if args.staged else 'release_validation.json')
        dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','mode':record['mode'],'checks':len(checks),'pages':30,'positive_examples':2,'negative_cases':8,'protected_files':len(baseline['protected_files'])},ensure_ascii=False,indent=2))
    for check in checks:print('PASS',check['check'])


if __name__=='__main__':main()
