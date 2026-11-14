"""Capture short diagnostic traces; timing tables come from unprofiled runs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXT='.exe' if os.name=='nt' else ''

def clean(text):return text.replace(str(ROOT),'$ROOT').replace(str(ROOT).replace('\\','/'),'$ROOT')
def invoke(argv,path,timeout=180):
    result=subprocess.run(list(map(str,argv)),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    path.write_text(clean(result.stdout+result.stderr),encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=HERE/'build/profiles')
    p.add_argument('--results-dir',type=Path,default=HERE/'results')
    p.add_argument('--include-unsafe-history',action='store_true',help='explicitly include the historical unsynchronized binary in the diagnostic audit')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.results_dir.mkdir(parents=True,exist_ok=True)
    needed=['seq','seq_workspace','improved','improved_workspace','ssm']
    if a.include_unsafe_history: needed.append('ssm_unsafe_history')
    for name in needed:
        if not (HERE/'build'/(name+EXT)).is_file(): raise RuntimeError(f'missing binary {name}; run build with the matching options first')
    report={'diagnostic_only':True,'nsys':[]}
    nsys=shutil.which('nsys')
    if not nsys:raise RuntimeError('nsys not installed')
    for name in ('seq','seq_workspace','improved','improved_workspace'):
        stem=a.output_dir/name
        argv=[nsys,'profile','--trace=cuda','--sample=none','--force-overwrite=true','--output='+str(stem),
              HERE/'build'/(name+EXT),'1','2048','64','64','--iters','3','--warmup','1']
        r=invoke(argv,a.output_dir/(name+'.txt'))
        if r.returncode:raise RuntimeError(f'nsys profile failed: {name}')
        database=stem.with_suffix('.sqlite')
        r=invoke([nsys,'export','--type','sqlite','--force-overwrite=true','--output='+str(database),stem.with_suffix('.nsys-rep')],a.output_dir/(name+'-export.txt'))
        if r.returncode:raise RuntimeError(f'nsys export failed: {name}')
        with sqlite3.connect(database) as db:
            names=dict(db.execute('select id,value from StringIds'))
            counts=Counter(names[x[0]] for x in db.execute('select nameId from CUPTI_ACTIVITY_KIND_RUNTIME'))
        report['nsys'].append({'variant':name,'shape':[1,2048,64,64],'correctness_calls':2,'warmup':1,'timed_calls':3,
                              'api_calls':dict(sorted(counts.items()))})
        print('traced',name,flush=True)
    # Record the historical racecheck outcome, whether or not this build exposes it.
    sanitizer=shutil.which('compute-sanitizer.exe')
    if not sanitizer:
        cuda=os.environ.get('CUDA_PATH_V13_1') or os.environ.get('CUDA_PATH')
        sanitizer=str(Path(cuda)/'compute-sanitizer/compute-sanitizer.exe') if cuda else None
    if sanitizer and a.include_unsafe_history:
        r=invoke([sanitizer,'--tool','racecheck','--racecheck-report','hazard','--error-exitcode','9',
                  HERE/'build'/('ssm_unsafe_history'+EXT),'1','33','7','32','--check-only'],a.output_dir/'unsafe-racecheck.txt')
        log=clean(r.stdout+r.stderr)
        summary=re.search(r'RACECHECK SUMMARY:[^\n]+',log)
        report['unsafe_history_racecheck']={'returncode':r.returncode,
            'summary':summary.group(0) if summary else 'missing',
            'first_hazard_excerpt':'\n'.join(log.splitlines()[:8])}
    # ncu.bat is a launcher on Windows; call the real executable when available.
    ncu=shutil.which('ncu.exe') or shutil.which('ncu')
    if ncu and ncu.lower().endswith('.bat'):
        ncu=str(Path(ncu).parent/'target/windows-desktop-win7-x64/ncu.exe')
    if ncu:
        r=invoke([ncu,'--set','basic','--kernel-name','regex:.*selective_scan_warp_shared_kernel.*','--launch-count','1',
                  '--csv',HERE/'build'/('ssm'+EXT),'1','1024','64','64','--check-only'],a.output_dir/'ncu.txt')
        text=r.stdout+r.stderr
        report['ncu']={'returncode':r.returncode,'counter_permission_denied':'ERR_NVGPUCTRPERM' in text,
                       'log':clean(text)}
    (a.results_dir/'profiling.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
