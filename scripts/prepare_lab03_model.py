"""Download pinned Qwen weights to user cache; never into the submission folder."""
from pathlib import Path
import os,urllib.request,hashlib
MODEL='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
EXPECTED_SHA256='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'

def main():
    root=Path(os.environ.get('LAB03_LOCAL_MODEL',str(Path.home()/'.cache/lab03-models/Qwen2.5-0.5B-Instruct'))).resolve()
    root.mkdir(parents=True,exist_ok=True)
    base=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/'
    for name in ['config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors','LICENSE']:
        path=root/name
        if path.exists():continue
        temporary=path.with_suffix(path.suffix+'.partial')
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(base+name,timeout=120) as response, temporary.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
        temporary.replace(path)
    with (root/'model.safetensors').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=EXPECTED_SHA256:raise RuntimeError('模型权重哈希不符，请检查下载')
    print('Model ready:',root,'SHA256:',digest)
    return 0

if __name__=='__main__':raise SystemExit(main())