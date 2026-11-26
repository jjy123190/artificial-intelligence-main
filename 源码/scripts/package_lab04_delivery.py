"""Create one runnable lab04 delivery including source and versioned materials.

Keeps the source checkout at 源码/ beside deliverables/ so approved download
paths work after extraction. All files are explicitly checksummed; no .local,
training data, installed dependencies, private state or nested source ZIPs.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from .package_project import ROOT, build_package

ALLOWED={'.md','.docx','.pdf','.pptx','.mp4','.srt','.vtt','.jpg','.json'}
CURRENT_REQUIRED=(
 '实验四_小样本分类与特征校准_当前整合报告.md','实验四_小样本分类与特征校准_当前整合报告.docx','实验四_小样本分类与特征校准_当前整合报告.pdf',
 '实验四_小样本分类与特征校准_当前汇报.pptx','实验四_当前页面与真实推理_Demo.mp4','实验四_当前版本PPT讲解.mp4',
 '实验四_当前页面与真实推理_Demo.vtt','实验四_当前版本PPT讲解.vtt','实验四_当前Demo预览.jpg','实验四_当前PPT预览.jpg',
 'report_evidence.json','slides_evidence.json','demo_evidence.json','presentation_evidence.json','media_support_evidence.json')


def hash_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def material_files(repo):
    repo=Path(repo).resolve();root=repo/'deliverables/lab4';current=root/'当前版本'
    if not root.is_dir():raise FileNotFoundError('Root deliverables/lab4 is required for the full bundle')
    for name in CURRENT_REQUIRED:
        if not(current/name).is_file():raise FileNotFoundError('Current material missing: '+name)
    files=[]
    for path in root.rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Unsafe material path')
        if path.is_file() and path.suffix.lower() in ALLOWED and not path.name.startswith('~$'):files.append(path)
    return sorted(files)


def verify_materials(repo):
    root=Path(repo)/'deliverables/lab4/当前版本'
    report=json.loads((root/'report_evidence.json').read_text(encoding='utf-8'))
    slides=json.loads((root/'slides_evidence.json').read_text(encoding='utf-8'))
    demo=json.loads((root/'demo_evidence.json').read_text(encoding='utf-8'))
    presentation=json.loads((root/'presentation_evidence.json').read_text(encoding='utf-8'))
    support=json.loads((root/'media_support_evidence.json').read_text(encoding='utf-8'))
    revisions={v['artifact_revision'] for v in (report,slides,demo,presentation,support)}
    if len(revisions)!=1:raise ValueError('Materials mix publication revisions')
    support_names={'实验四_当前页面与真实推理_Demo.vtt','实验四_当前版本PPT讲解.vtt','实验四_当前Demo预览.jpg','实验四_当前PPT预览.jpg'}
    if set(support.get('files',{}))!=support_names:raise ValueError('Incomplete caption/poster identity')
    checks=[('实验四_小样本分类与特征校准_当前整合报告.md',report['report_sha256']),
            ('实验四_小样本分类与特征校准_当前汇报.pptx',slides['slide_sha256']),
            ('实验四_当前版本PPT讲解.mp4',presentation['video_sha256']),*support['files'].items()]
    for name,digest in checks:
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or hash_file(path)!=digest:raise ValueError('Material checksum mismatch: '+name)
    if demo.get('video_sha256')!=hash_file(root/'实验四_当前页面与真实推理_Demo.mp4'):raise ValueError('Demo video differs from recorded evidence')
    if presentation['slides_sha256']!=slides['slide_sha256'] or len(demo['runs'])!=4 or demo['console_errors']:
        raise ValueError('Slides/video or real-demo evidence incomplete')
    runs=demo['runs']
    for run in runs:
        if run.get('artifact_revision')!=report['artifact_revision'] or run.get('total')!=75 or run.get('shots') not in (1,5):raise ValueError('Demo run belongs to another evidence revision')
        if type(run.get('correct')) is not int or not 0<=run['correct']<=75 or abs(run['accuracy']-run['correct']/75)>1e-9:raise ValueError('Invalid demo run score')
        if len(run['support_ids'])!=5*run['shots'] or len(set(run['query_ids']))!=75 or set(run['support_ids'])&set(run['query_ids']):raise ValueError('Invalid demo task sample identities')
    if {run['method'] for run in runs[:3]}!={'baseline','maml','calibrated'} or any(run['support_ids']!=runs[0]['support_ids'] or run['query_ids']!=runs[0]['query_ids'] for run in runs[:3]):raise ValueError('Demo three-method comparison is not paired')
    return revisions.pop()


def build_delivery(repo=None):
    repo=Path(repo).resolve() if repo else ROOT.parent
    source=repo/'源码'
    if not source.is_dir():raise FileNotFoundError('Bundle expects repo/源码')
    files=material_files(repo);revision=verify_materials(repo)
    from experiments.lab04.artifact_audit import audit_publication
    if audit_publication(source/'outputs/lab04')['artifact_revision']!=revision:raise ValueError('Materials do not describe current published model')
    source_package=build_package('lab04',source)
    directory=source/'outputs/submissions';destination=directory/('lab04-complete-delivery-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.zip')
    temporary=destination.with_suffix('.zip.tmp');hashes={}
    note=('实验四完整交付包\n\n先双击根目录“打开实验四.cmd”。源码位于源码目录，报告/幻灯片/视频位于deliverables/lab4/当前版本。\n'
          '本包描述当前已发布模型，未发布候选不作为成绩；v1和各补充记录另存历史。\n'
          'PPT讲解使用本机标准合成语音；Demo为实际浏览器操作、无旁白，附中文字幕。\n'
          '依赖未安装时首次启动需要按源码README准备；完整训练数据、.local、node_modules和私人状态不在包中。\n'
          '课程组员关联、项目创建和管理员提交需在课程平台实际确认；本包不等于已经提交。\n'
          '全文件哈希见SHA256.json，源码内另有文件校验清单。\n').encode('utf-8')
    launcher='@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\ncall "%~dp0源码\\experiments\\lab04\\打开网站.cmd" %*\r\nexit /b %errorlevel%\r\n'.encode('utf-8')
    try:
        with zipfile.ZipFile(source_package['archive']) as src,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            for info in src.infolist():
                if info.is_dir():continue
                if '..' in Path(info.filename).parts or Path(info.filename).is_absolute():raise ValueError('Unsafe source ZIP member')
                content=src.read(info.filename);name='源码/'+info.filename;archive.writestr(name,content);hashes[name]=hashlib.sha256(content).hexdigest()
            for path in files:
                name=path.relative_to(repo).as_posix();archive.write(path,name);hashes[name]=hash_file(path)
            materials_source=repo/'制作资料/实验四'
            if materials_source.is_dir():
                for path in sorted(materials_source.iterdir()):
                    if path.is_file() and not path.is_symlink() and path.suffix in ('.py','.mjs','.ps1','.json','.md'):
                        name=path.relative_to(repo).as_posix();archive.write(path,name);hashes[name]=hash_file(path)
            for name,content in (('交付说明.txt',note),('打开实验四.cmd',launcher)):
                archive.writestr(name,content);hashes[name]=hashlib.sha256(content).hexdigest()
            archive.writestr('SHA256.json',json.dumps(hashes,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('Full delivery ZIP corruption')
            if set(archive.namelist())!=set(hashes)|{'SHA256.json'}:raise ValueError('Full delivery inventory mismatch')
            for name,digest in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Full delivery checksum mismatch: '+name)
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return {'archive':str(destination),'bytes':destination.stat().st_size,'sha256':hash_file(destination),'files':len(hashes),
            'artifact_revision':revision,'source_archive':source_package['archive'],'current_and_historical_materials':True,'all_hashes_match':True,'private_state_excluded':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo');args=parser.parse_args()
    print(json.dumps(build_delivery(args.repo),ensure_ascii=False,indent=2))
