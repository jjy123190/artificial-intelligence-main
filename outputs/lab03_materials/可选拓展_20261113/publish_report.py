"""Publish the reviewed supplement once, without losing an open Word document.

Refuses unsaved user edits or any changed on-disk baseline. Closes only the exact
saved target (saving any last-moment user edits), checks hashes again, and reopens
it in the same application. It never kills Word or closes other user documents.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from shutil import copy2
import uuid

import win32com.client

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
FORMAL=(ROOT/'deliverables/lab3').resolve()
DRAFT=(BASE/'draft').resolve()
NAME='第31组_实验三_实验报告'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p: Path,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def checked_replace(temp: Path,target: Path) -> None:
    # Two explicit files only, no recursive move/delete or computed directory operation.
    if temp.resolve().parent!=DRAFT or target.resolve().parent!=FORMAL:
        raise ValueError('Unexpected replacement path outside named directories')
    if target.name not in {NAME+'.docx',NAME+'.pdf'} or not temp.is_file():
        raise ValueError('Unexpected target/temp file')
    os.replace(temp,target)


def main():
    baseline=load(BASE/'qa/protected_before.json')
    staged=load(BASE/'qa/staged_validation.json')
    if staged['status']!='passed' or staged['mode']!='staged':
        raise RuntimeError('Staged verification is missing or failed')
    for rel,expected in staged['report_files'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError('Staged report changed since QA')
    target=(FORMAL/(NAME+'.docx')).resolve()
    def check_current():
        for rel,expected in baseline['formal_before'].items():
            if sha(ROOT/rel)!=expected:
                raise RuntimeError('Current report changed since snapshot; rebase before publishing: '+rel)
    check_current()
    app=None;opened_doc=None;closed=False;changed=[];temp_files=[]
    current_active=None;was_active=False;saved_before=None
    reopened=False;published=False
    try:
        try:app=win32com.client.GetActiveObject('Word.Application')
        except Exception:app=None
        if app is not None:
            try:current_active=str(app.ActiveDocument.FullName)
            except Exception:current_active=None
            for i in range(1,app.Documents.Count+1):
                candidate=app.Documents.Item(i)
                if Path(str(candidate.FullName)).resolve()==target:
                    if opened_doc is not None:raise RuntimeError('Target open twice; stop for manual review')
                    opened_doc=candidate
            if opened_doc is not None:
                saved_before=bool(opened_doc.Saved)
                if not saved_before:raise RuntimeError('Open target has unsaved edits; nothing has been replaced')
                if opened_doc.ReadOnly:raise RuntimeError('Open target is read-only; not replacing it')
                was_active=current_active is not None and Path(current_active).resolve()==target
        lock=FORMAL/('~$'+target.name[2:])
        if opened_doc is None and lock.exists():
            raise RuntimeError('Target has an Office lock but is not found in the active instance; close it manually first')
        if opened_doc is not None:
            if not opened_doc.Saved:raise RuntimeError('Document was edited while preparing publication')
            # SaveChanges=-1 preserves any input that arrived between Saved and Close.
            # If this alters the disk baseline, the post-close check aborts safely.
            opened_doc.Close(SaveChanges=-1)
            opened_doc=None;closed=True
        check_current()
        for suffix in ['.docx','.pdf']:
            source=DRAFT/(NAME+suffix)
            temp=DRAFT/(NAME+suffix+'.publish-'+uuid.uuid4().hex+'.tmp')
            copy2(source,temp);temp_files.append(temp)
            if sha(source)!=sha(temp):raise RuntimeError('Staging copy failed')
            checked_replace(temp,FORMAL/(NAME+suffix));changed.append(suffix)
        for suffix in ['.docx','.pdf']:
            if sha(DRAFT/(NAME+suffix))!=sha(FORMAL/(NAME+suffix)):
                raise RuntimeError('Published file hash mismatch')
        support=[
            'README_本次增补.md','verify_examples.py','example_verification.json',
            'verify_report.py','build_extension.py','export_staged.py','publish_report.py',
            'extension_content.json','第14章_可选迁移拓展.md','reference_checks.json',
            'before/'+NAME+'.docx','before/'+NAME+'.pdf',
            'draft/'+NAME+'.docx','draft/'+NAME+'.pdf',
            'qa/protected_before.json','qa/document_before.json','qa/update_spec.json',
            'qa/before_userdoc.pdf','qa/pdf_pages.json','qa/staged_validation.json','qa/visual_review.json',
        ]
        manifest={
            'created_at':datetime.now().astimezone().isoformat(),
            'version':'30-page-report-with-chapter-14',
            'algorithm':'sha256',
            'report_files':{(FORMAL/(NAME+suffix)).relative_to(ROOT).as_posix():sha(FORMAL/(NAME+suffix)) for suffix in ['.docx','.pdf']},
            'support_files':{rel:sha(BASE/rel) for rel in support},
            'exclusions':'Manifest itself, mutable QA release/publish records, and generated PNG previews. Old backup files are separately protected by qa/protected_before.json.',
            'scope':'Only report DOCX/PDF published; original code/data/output and pre-existing backup kept intact.',
        }
        write_json(BASE/'release_manifest.json',manifest)
        published=True
    except Exception:
        # Restore only files replaced by this operation; current user edits were
        # checked before replacement and have never been discarded.
        for suffix in reversed(changed):
            original=BASE/'before'/(NAME+suffix)
            temp=DRAFT/(NAME+suffix+'.restore-'+uuid.uuid4().hex+'.tmp')
            copy2(original,temp);temp_files.append(temp)
            checked_replace(temp,FORMAL/(NAME+suffix))
        raise
    finally:
        if closed and app is not None:
            try:
                newdoc=app.Documents.Open(str(target),ReadOnly=False,AddToRecentFiles=False,Visible=bool(app.Visible))
                reopened=True
                # Do not steal focus from another document that was active before.
                if not was_active and current_active:
                    for d in app.Documents:
                        if str(d.FullName)==current_active:
                            d.Activate();break
            except Exception as exc:
                print('NOTICE: Report is saved; automatic reopen failed:',str(exc))
        for temp in temp_files:
            if temp.exists():
                if temp.resolve().parent!=DRAFT:raise RuntimeError('Refusing cleanup outside staging directory')
                temp.unlink()
        write_json(BASE/'qa/publish_result.json',{
            'time':datetime.now().astimezone().isoformat(),'published':published,
            'open_target_saved_before':saved_before,'only_target_temporarily_closed':closed,
            'target_reopened_in_same_application':reopened,'other_documents_closed':0,
        })
    print(json.dumps({'published':published,'pages':30,'target_was_saved':saved_before,'target_reopened':reopened,'formal_files':[NAME+'.docx',NAME+'.pdf']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
