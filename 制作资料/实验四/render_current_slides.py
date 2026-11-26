"""Render current PPT with hidden PowerPoint and reject overflowing text."""
import argparse,json
from pathlib import Path
import pythoncom
import win32com.client
ROOT=Path(__file__).resolve().parents[2]

def render(source,output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    pythoncom.CoInitialize();app=presentation=None
    try:
        app=win32com.client.DispatchEx('PowerPoint.Application')
        presentation=app.Presentations.Open(str(source),True,False,False)
        observations=[]
        for i in range(1,presentation.Slides.Count+1):
            slide=presentation.Slides(i);slide.Export(str(output/f'slide-{i:02}.png'),'PNG',1600,900)
            for j in range(1,slide.Shapes.Count+1):
                shape=slide.Shapes(j)
                if shape.HasTextFrame and shape.TextFrame.HasText:
                    value=shape.TextFrame.TextRange
                    observations.append({'slide':i,'text':value.Text,'width':shape.Width,'height':shape.Height,'bound_width':value.BoundWidth,'bound_height':value.BoundHeight})
        (output/'text-fit.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2),encoding='utf-8')
        bad=[r for r in observations if r['bound_width']>r['width']+3 or r['bound_height']>r['height']+2]
        print(json.dumps({'slides':presentation.Slides.Count,'overflow':bad},ensure_ascii=False,indent=2))
        if bad:raise ValueError('Text exceeds slide text boxes')
    finally:
        if presentation:presentation.Close()
        if app:app.Quit()
        pythoncom.CoUninitialize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(ROOT/'deliverables/lab4/当前版本/实验四_小样本分类与特征校准_当前汇报.pptx'))
    p.add_argument('--output',default=str(ROOT/'源码/.local/current-slides-render'))
    args=p.parse_args();render(args.source,args.output)
