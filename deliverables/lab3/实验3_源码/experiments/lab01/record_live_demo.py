"""Record the actual step-mode Python process in a dedicated window, not a log replay."""
from pathlib import Path
import os,sys,subprocess,threading,queue,time,json,ctypes,tkinter as tk
try:
 ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
 ctypes.windll.user32.SetProcessDPIAware()
from PIL import Image,ImageTk
from imageio_ffmpeg import get_ffmpeg_exe
CODE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[2]
QA=ROOT/'outputs/qa/lab01-recording';QA.mkdir(parents=True,exist_ok=True)
OUT=QA/'system_live_recording.mp4'
TITLE='Equity KG Live Execution'
class Demo:
 def __init__(self):
  self.root=tk.Tk();self.root.tk.call('tk','scaling',1.333333);self.root.title(TITLE);self.root.geometry('1280x720+20+20');self.root.configure(bg='#101c2c');self.root.attributes('-topmost',True)
  tk.Label(self.root,text='企业股权知识表示与风险推理｜实际程序运行',bg='#101c2c',fg='#76decd',font=('Microsoft YaHei',21,'bold')).pack(pady=(16,4))
  self.status=tk.StringVar(value='独立 Python 进程 · 自动分步操作')
  tk.Label(self.root,textvariable=self.status,bg='#101c2c',fg='#a7b5ca',font=('Microsoft YaHei',12)).pack(pady=4)
  self.text=tk.Text(self.root,bg='#101c2c',fg='#e4ebf4',font=('Microsoft YaHei',13),wrap='word',borderwidth=0,padx=24,pady=12)
  self.text.pack(fill='both',expand=True)
  self.q=queue.Queue();self.lines=[];self.waiting=False;self.proc=None;self.rec=None;self.done=False
  self.root.after(1200,self.start);self.root.after(100,self.poll)
  self.root.protocol('WM_DELETE_WINDOW',self.finish)
 def start(self):
  self.log=(QA/'live-recorder.log').open('wb')
  self.rec=subprocess.Popen([get_ffmpeg_exe(),'-y','-f','gdigrab','-framerate','15','-draw_mouse','0','-i','title='+TITLE,'-vf','scale=1280:720,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','23','-movflags','+faststart',str(OUT)],stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,creationflags=subprocess.CREATE_NO_WINDOW)
  self.text.insert('end','> python -X utf8 -u main.py lab01 --step\n\n')
  env=dict(os.environ,PYTHONUTF8='1',MPLCONFIGDIR=str(QA/'matplotlib'))
  self.proc=subprocess.Popen([sys.executable,'-X','utf8','-u',str(ROOT/'main.py'),'lab01','--step'],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
  threading.Thread(target=self.read,daemon=True).start()
 def read(self):
  while True:
   ch=self.proc.stdout.read(1)
   if not ch:break
   self.q.put(ch)
  self.q.put(None)
 def poll(self):
  buf=''
  while not self.q.empty():
   item=self.q.get()
   if item is None:
    self.done=True;self.root.after(3500,self.graph);break
   buf+=item
  if buf:
   self.lines.append(buf);self.text.insert('end',buf);self.text.see('end');self.root.update_idletasks()
   tail=''.join(self.lines)[-220:]
   if not self.waiting and ('回车' in tail or 'Enter' in tail):
    self.waiting=True;self.root.after(6500,self.advance)
  if not self.done:self.root.after(80,self.poll)
 def advance(self):
  if self.proc and self.proc.poll() is None:
   self.proc.stdin.write('\n');self.proc.stdin.flush()
  self.waiting=False;self.lines.append('\n'+' '*230)
 def graph(self):
  code=self.proc.wait(timeout=10)
  self.status.set(f'实际运行结束：exit code {code}｜9条控制关系回写，6家企业风险线索（非法律认定）')
  self.text.pack_forget()
  image=Image.open(ROOT/'outputs/lab01/kg_reasoned.png');image.thumbnail((1230,595))
  self.photo=ImageTk.PhotoImage(image);tk.Label(self.root,image=self.photo,bg='#ffffff').pack(pady=5)
  self.root.after(10000,self.finish)
 def finish(self):
  if self.proc and self.proc.poll() is None:self.proc.terminate()
  if self.rec and self.rec.poll() is None:
   self.rec.stdin.write(b'q\n');self.rec.stdin.flush();self.rec.wait(timeout=30)
  if hasattr(self,'log'):self.log.close()
  (QA/'live-process-output.txt').write_text(''.join(self.lines),encoding='utf-8')
  (QA/'live-recording-evidence.json').write_text(json.dumps({'command':[sys.executable,'main.py','lab01','--step'],'process_exit_code':self.proc.returncode if self.proc else None,'recorder_exit_code':self.rec.returncode if self.rec else None,'capture':'FFmpeg gdigrab capture of dedicated Tk window displaying live subprocess stdout; automated Enter events','output':str(OUT)},ensure_ascii=False,indent=2),encoding='utf-8')
  self.root.destroy()
Demo().root.mainloop()
