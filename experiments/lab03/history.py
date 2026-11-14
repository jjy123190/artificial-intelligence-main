"""Hash-checked historical stages, never silently rebound to current code."""
from __future__ import annotations
import hashlib,json,sys,types,zipfile
from .dataset import ROOT
FOLDER=ROOT/'outputs/lab03/structure'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def stage_two_bytes(path):
    metadata=read_json(FOLDER/'baseline.json')
    archive=FOLDER/'stage_two_baseline.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==metadata['archive_sha256'], '第二阶段归档损坏'
    with zipfile.ZipFile(archive) as z:data=z.read(path)
    assert hashlib.sha256(data).hexdigest()==metadata['files'][path], ('历史文件不一致',path)
    return data

def stage_two_json(path):
    return json.loads(stage_two_bytes(path).decode('utf-8-sig'))

def stage_two_file(name):
    path=FOLDER/('previous_'+name)
    assert path.read_bytes()==stage_two_bytes('outputs/lab03/'+name), ('历史副本失配',name)
    return path

def stage_two_source_hashes():
    return stage_two_json('outputs/lab03/manifest.json')['source_sha256']

def stage_two_search():
    package='lab03_preserved_stage_two'
    # Validate source identity before reusing module objects as well.
    sources={name:stage_two_bytes('experiments/lab03/'+name+'.py') for name in ('domain','dataset','learning','search')}
    if package+'.search' in sys.modules:return sys.modules[package+'.search']
    module=types.ModuleType(package);module.__path__=[];module.__package__=package
    sys.modules[package]=module
    artifacts=types.ModuleType(package+'.artifacts');artifacts.provenance=lambda:{}
    sys.modules[artifacts.__name__]=artifacts
    for name,source in sources.items():
        full=package+'.'+name;module=types.ModuleType(full);module.__package__=package
        module.__file__=str(ROOT/'experiments/lab03'/f'{name}.py');sys.modules[full]=module
        exec(compile(source.decode('utf-8-sig'),module.__file__+' [stage two]', 'exec'),module.__dict__)
    return sys.modules[package+'.search']
