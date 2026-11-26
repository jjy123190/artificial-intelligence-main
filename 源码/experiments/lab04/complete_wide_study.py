"""Wait for the existing validation PID, then evaluate and verify one locked study.

Never starts/restarts training or publishes weights. Only a fully completed
validation selection triggers the fixed test; all logs remain explicit.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--created',type=float,required=True);p.add_argument('--candidate',default='.local/lab04-v3-wide')
    p.add_argument('--output',default='.local/lab04-wide-test');p.add_argument('--data-root',default='.local/mini-imagenet')
    args=p.parse_args()
    while True:
        try:
            process=psutil.Process(args.pid)
            if abs(process.create_time()-args.created)>.01 or not process.is_running():break
            if not any('after_training' in x or 'validate_v3' in x for x in process.cmdline()):break
        except psutil.NoSuchProcess:break
        time.sleep(30)
    candidate=Path(args.candidate);selection=candidate/'validation_selection.json'
    if not selection.exists() or json.loads(selection.read_text()).get('status')!='complete' or not(candidate/'selection_lock.json').is_file():
        print('Validation did not complete; no test command has been started.',flush=True);return 1
    result=Path(args.output)/'wide_results.json'
    if result.exists() and json.loads(result.read_text()).get('status')=='complete':
        print('Wide test already complete; verifying without repeating.',flush=True)
    else:
        print('Validation locked. Starting the fixed paired test, not publishing.',flush=True)
        code=subprocess.call([sys.executable,'-B','-u','-X','utf8','-m','experiments.lab04.evaluate_wide','--candidate',args.candidate,'--output',args.output,'--data-root',args.data_root])
        if code:return code
    return subprocess.call([sys.executable,'-B','-X','utf8','-m','experiments.lab04.verify_wide','--artifacts',args.output,'--candidate',args.candidate])

if __name__=='__main__':raise SystemExit(main())
