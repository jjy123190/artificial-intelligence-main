"""Render self-contained diagram SVGs at 2x; no changes outside this delivery."""
from pathlib import Path
import json, subprocess, shutil, os
BASE=Path(__file__).resolve().parents[1]
items=json.loads((BASE/'evidence/figure_index.json').read_text(encoding='utf-8'))
runner=Path.home()/'.codex/skills/baoyu-diagram/scripts/main.ts'
if runner.exists() and shutil.which('bun'):
 for item in items:
  svg=BASE/'figures'/(item['name']+'.svg');png=svg.with_suffix('.png')
  cmd=[shutil.which('bun'),str(runner),str(svg),'-s','2','-o',str(png)]
  if shutil.which('rtk'):cmd=[shutil.which('rtk'),'proxy']+cmd
  subprocess.run(cmd,check=True,capture_output=True)
else:
 from playwright.sync_api import sync_playwright
 with sync_playwright() as pw:
  chrome=Path(os.environ.get('ProgramFiles','C:/Program Files'))/'Google/Chrome/Application/chrome.exe'
  browser=pw.chromium.launch(headless=True,**({'executable_path':str(chrome)} if chrome.exists() else {}))
  for item in items:
   w,h=item['viewBox'];page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=2)
   svg=BASE/'figures'/(item['name']+'.svg')
   page.set_content('<style>body{margin:0}svg{display:block}</style>'+svg.read_text(encoding='utf-8'))
   page.screenshot(path=str(svg.with_suffix('.png')));page.close()
  browser.close()
print('Rendered',len(items),'diagrams at 2x')