"""Wait for the verified continuation process, then validate once; never test/publish."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil


def is_training_live(pid,created):
    try:
        process=psutil.Process(pid)
        return process.is_running() and abs(process.create_time()-created)<.01 and 'experiments.lab04.train_continuation' in process.cmdline()
    except psutil.NoSuchProcess:return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True);p.add_argument('--created',type=float,required=True)
    p.add_argument('--candidate',default='.local/lab04-continuation-20261125');p.add_argument('--output',default='.local/lab04-continuation-validation-20261125')
    p.add_argument('--data-root',default='.local/mini-imagenet');args=p.parse_args()
    while is_training_live(args.pid,args.created):time.sleep(30)
    training=Path(args.candidate)/'training.json'
    if not training.is_file() or json.loads(training.read_text(encoding='utf-8')).get('status')!='complete':
        print('Verified training process ended without completed record; no validation, test or restart was started.',flush=True);return 1
    print('Training complete. Running the preregistered validation only; no test/publish command.',flush=True)
    return subprocess.call([sys.executable,'-B','-u','-X','utf8','-m','experiments.lab04.validate_continuation',
                            '--candidate',args.candidate,'--output',args.output,'--data-root',args.data_root])


if __name__=='__main__':raise SystemExit(main())
