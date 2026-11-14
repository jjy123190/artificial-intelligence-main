"""Preserved third-stage code/data for historical audits, not live algorithm selection."""
from __future__ import annotations
import hashlib,json,sys,types,zipfile
from .dataset import ROOT
FOLDER=ROOT/'outputs/lab03/reliability'

def stage_three_bytes(path):
    meta=json.loads((FOLDER/'baseline.json').read_text(encoding='utf-8-sig'))
    archive=FOLDER/'stage_three_baseline.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['archive_sha256'], '第三阶段归档损坏'
    with zipfile.ZipFile(archive) as z:data=z.read(path)
    assert hashlib.sha256(data).hexdigest()==meta['files'][path], ('第三阶段历史文件失配',path)
    return data

def stage_three_json(path):return json.loads(stage_three_bytes(path).decode('utf-8-sig'))

def stage_three_file(name):
    path=FOLDER/('previous_'+name)
    assert path.read_bytes()==stage_three_bytes('outputs/lab03/'+name), ('第三阶段历史副本失配',name)
    return path

def stage_three_sources():return stage_three_json('outputs/lab03/manifest.json')['source_sha256']

def stage_three_search():
    package='lab03_preserved_stage_three'
    sources={name:stage_three_bytes('experiments/lab03/'+name+'.py') for name in ('domain','dataset','learning','search')}
    if package+'.search' in sys.modules:return sys.modules[package+'.search']
    module=types.ModuleType(package);module.__path__=[];module.__package__=package;sys.modules[package]=module
    artifacts=types.ModuleType(package+'.artifacts');artifacts.provenance=lambda:{};sys.modules[artifacts.__name__]=artifacts
    for name,data in sources.items():
        module=types.ModuleType(package+'.'+name);module.__package__=package;module.__file__=str(ROOT/'experiments/lab03'/f'{name}.py')
        sys.modules[module.__name__]=module
        exec(compile(data.decode('utf-8-sig'),module.__file__+' [stage three]','exec'),module.__dict__)
    return sys.modules[package+'.search']
