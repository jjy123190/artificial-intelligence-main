"""Package the whole course project; exclude private state and installed dependencies."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.state','.venv','venv','node_modules','__pycache__','.pytest_cache','人工智能作业一','人工智能作业二'}
ALLOWED={'.py','.ps1','.cmd','.md','.txt','.json','.csv','.html','.css','.ts','.tsx','.mjs','.js','.svg','.png','.jpg','.jpeg','.webp','.docx','.pptx','.pdf','.mp4','.mp3','.srt','.toml'}

def project_files(root:Path=ROOT):
    root=root.resolve()
    result=[]
    for directory, dirs, names in os.walk(root, followlinks=False):
        current=Path(directory)
        relative=current.relative_to(root)
        for name in list(dirs):
            child=current/name
            if name in EXCLUDED or (relative.parts and relative.parts[0]=='outputs' and name in ('qa','runtime','submissions')):
                dirs.remove(name)
            elif child.is_symlink():
                raise ValueError('Symlinks must not be packaged: '+child.relative_to(root).as_posix())
        for name in sorted(names):
            path=current/name
            if path.is_symlink():raise ValueError('Symlinks must not be packaged: '+path.relative_to(root).as_posix())
            if not path.is_file() or not path.resolve().is_relative_to(root):continue
            if name.startswith('~$') or name.endswith(('.pyc','.tsbuildinfo','.tmp')):continue
            if path.suffix.lower() in ALLOWED or name=='.gitignore':result.append(path)
    return sorted(result)

def build_package(experiment:str,root:Path=ROOT):
    root=root.resolve()
    if experiment not in ('lab01','lab02'):raise ValueError('Unknown experiment')
    directory=root/'outputs'/'submissions';directory.mkdir(parents=True,exist_ok=True)
    destination=directory/('course-project-'+experiment+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.zip')
    temporary=destination.with_suffix('.zip.tmp')
    files=project_files(root)
    if not (root/'web'/'dist'/'index.html').is_file():raise FileNotFoundError('Build web assets before packaging: cd web; npm run build')
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    note=('本包提交完整课程项目。\n本次实验：'+experiment+'\n总目录：双击“打开网站.cmd”选择实验。\n本次直达入口：experiments/'+experiment+'/打开网站.cmd\n网页地址片段：#/labs/'+experiment+'\n实现位置：experiments/'+experiment+'/\n公共能力：core/、backend/、web/。\n运行与复现见README.md。\n报告、PPT和视频按课程要求另行核对；文件存在不等于已提交。\n不含node_modules、虚拟环境、本机案例数据库、运行记录或验收截图。\n').encode('utf-8')
    hashes['本次实验提交说明.txt']=hashlib.sha256(note).hexdigest()
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in files:archive.write(path,path.relative_to(root).as_posix())
            archive.writestr('本次实验提交说明.txt',note)
            archive.writestr('SHA256.json',json.dumps(hashes,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP integrity check failed')
            if set(archive.namelist()) != set(hashes)|{'SHA256.json'}:raise ValueError('ZIP inventory mismatch')
            for name,digest in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('File checksum mismatch: '+name)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'archive':str(destination),'experiment':experiment,'files':len(hashes),'bytes':destination.stat().st_size,'all_hashes_match':True,'private_state_excluded':True}

def main(argv=None):
    parser=argparse.ArgumentParser(description='提交完整项目，附上本次实验入口；不打包私人状态与依赖。')
    parser.add_argument('--experiment',choices=['lab01','lab02'],required=True)
    args=parser.parse_args(argv)
    print(json.dumps(build_package(args.experiment),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
