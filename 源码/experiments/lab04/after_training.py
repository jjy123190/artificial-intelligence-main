"""Wait on a verified child PID, then validate its completed candidate once.

A PID without the expected creation time is terminal; never re-run training.
This helper only starts validation, and only after training.json says complete.
"""
from __future__ import annotations
import argparse,json,subprocess,time,sys
from pathlib import Path
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--created',type=float,required=True);args=p.parse_args()
    output=Path(args.candidate)
    while True:
        try:
            process=psutil.Process(args.pid)
            if abs(process.create_time()-args.created)>.01:break
            if 'experiments.lab04.train_v3' not in process.cmdline():break
            if not process.is_running():break
        except psutil.NoSuchProcess:break
        time.sleep(30)
    progress=output/'training.json'
    if not progress.exists() or json.loads(progress.read_text())['status']!='complete':
        print('Training stopped without completion. No new process or test evaluation started.',flush=True);return 1
    command=[sys.executable,'-B','-u','-X','utf8','-m','experiments.lab04.validate_v3',
             '--candidate',str(output),'--data-root',args.data_root]
    print('Training completed; starting validation-only selection.',flush=True)
    return subprocess.call(command)

if __name__=='__main__':raise SystemExit(main())
