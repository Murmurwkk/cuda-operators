"""Build and serially run the scan experiments; Python standard library only.

Works in the private lab (op_mur/) and public cuda-operators (selective_scan/).
Use a CUDA-capable Windows developer shell or a Linux CUDA toolchain.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = ROOT / ('op_mur' if (ROOT / 'op_mur').is_dir() else 'selective_scan')
BUILD = HERE / 'build'
RESULTS = HERE / 'results'
VARIANTS = {
    'ssm':'ssm.cu', 'seq':'ssm_seq.cu', 'rewrite':'selective_scan_rewrite.cu',
    'improved':'selective_scan_improved.cu', 'seq_parallel':'selective_scan_seq_parallel.cu',
    'streaming_seq_parallel_fixed':'selective_scan_streaming_seq_parallel_fixed.cu',
}
MATRIX = [(1,l,64,64) for l in (256,512,1024,2048,4096,8192)] + [
    (1,1024,128,64),(1,1024,256,64),(1,1024,64,32),(1,1024,64,128),
    (8,512,64,64),(4,1024,64,64)]
EXT = '.exe' if os.name == 'nt' else ''

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tool(name):
    # Match the experiment's CUDA 13.1 before a stale CUDA_PATH on this host.
    cuda = os.environ.get('CUDA_PATH_V13_1') or os.environ.get('CUDA_PATH')
    if cuda:
        candidate = Path(cuda)/'bin'/(name + ('.exe' if os.name=='nt' else ''))
        if candidate.exists(): return str(candidate)
    found = shutil.which(name)
    if not found: raise RuntimeError(f'{name} not found')
    return found

def clean(s):
    return str(s).replace(str(ROOT), '$ROOT').replace(str(ROOT).replace('\\','/'), '$ROOT')

def command(args, *, env=None, timeout=300):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', env=env, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {clean(args)}\n{clean(result.stdout)}\n{clean(result.stderr)}')
    return result.stdout

def msvc_env():
    env = os.environ.copy()
    if os.name != 'nt' or shutil.which('cl'): return env
    # VsDevCmd can exist even when vswhere has no registered installation.
    candidates=[]
    for base in ('ProgramFiles(x86)','ProgramFiles'):
        top=Path(os.environ.get(base,''))/'Microsoft Visual Studio'
        if top.is_dir(): candidates += list(top.glob('*/BuildTools/Common7/Tools/VsDevCmd.bat'))
        if top.is_dir(): candidates += list(top.glob('*/*/Common7/Tools/VsDevCmd.bat'))
    if not candidates: raise RuntimeError('Run from an x64 Visual Studio developer shell')
    dev=sorted(set(candidates),reverse=True)[0]
    # Pass cmd its own quoted command line; list2cmdline's backslash quoting
    # is for C argv parsing and is not cmd.exe's quote syntax.
    setup=f'cmd.exe /d /s /c "call "{dev}" -no_logo -arch=x64 -host_arch=x64 >nul && set"'
    initialized=subprocess.run(setup,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if initialized.returncode: raise RuntimeError(initialized.stderr)
    result=initialized.stdout
    for line in result.splitlines():
        if '=' in line:
            k,v=line.split('=',1)
            if k: env[k.upper()]=v
    return env

def build(args):
    BUILD.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    env=msvc_env()
    flags=['-std=c++17','-O3','-lineinfo',f'-arch={args.arch}']
    if os.name=='nt': flags+=['-allow-unsupported-compiler','-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH','-Xcompiler','/O2']
    nvcc=tool('nvcc')
    variants={k:SOURCES/v for k,v in VARIANTS.items()}
    variants.update(seq_workspace=HERE/'seq_workspace.cu',improved_workspace=HERE/'improved_workspace.cu')
    if args.include_unsafe_history: variants['ssm_unsafe_history']=HERE/'historical/ssm_before_sync.cu'
    manifest={'schema':1,'arch':args.arch,'flags':flags,'nvcc_version':command([nvcc,'--version']),
              'gpu':command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']).strip(),
              'protocol':{'oracle':'FP64 recurrence, twice before timing','atol':2e-5,'rtol':2e-4,
                          'seed':17,'warmup':6,'iterations':30,'rounds':args.rounds},'variants':{}}
    if os.name=='nt':
        compiler_path=shutil.which('cl',path=env.get('PATH',''))
        if not compiler_path: raise RuntimeError('cl not found in developer environment')
        compiler=subprocess.run([compiler_path,'/Bv'],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
        manifest['msvc_versions']=sorted(set(re.findall(r'\b(?:14|19)\.\d+\.\d+(?:\.\d+)?\b',compiler.stdout+compiler.stderr)))
    manifest['runner_sha256']=sha(__file__)
    for name,source in variants.items():
        obj=BUILD/(name+'.obj' if os.name=='nt' else name+'.o')
        exe=BUILD/(name+EXT)
        defs=['-DUSE_WORKSPACE_ADAPTER'] if name.endswith('_workspace') else []
        compile_cmd=[nvcc,*flags,'-I',SOURCES,*defs,'-c',source,'-o',obj]
        link_cmd=[nvcc,*flags,*defs,HERE/'bench.cpp',obj,'-o',exe]
        logs=command(compile_cmd,env=env)+command(link_cmd,env=env)
        (BUILD/(name+'.log')).write_text(clean(logs),encoding='utf-8')
        manifest['variants'][name]={'source':source.name,'source_sha256':sha(source),
              'harness_sha256':sha(HERE/'bench.cpp'),'binary_sha256':sha(exe),
              'compile':clean(' '.join(map(str,compile_cmd))),'link':clean(' '.join(map(str,link_cmd)))}
        if name.endswith('_workspace'):
            base='ssm_seq.cu' if name=='seq_workspace' else 'selective_scan_improved.cu'
            manifest['variants'][name]['included_source_sha256']=sha(SOURCES/base)
        print('built',name,flush=True)
    (RESULTS/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def run_case(name, shape, *, seed=17, check_only=False, iters=30, warmup=6):
    exe=BUILD/(name+EXT)
    if not exe.is_file(): raise RuntimeError(f'missing binary: {name}; run build first')
    cmd=[exe,*map(str,shape),'--seed',str(seed),'--iters',str(iters),'--warmup',str(warmup)]
    if check_only: cmd+=['--check-only']
    text=command(cmd)
    row=json.loads(text.strip().splitlines()[-1])
    if not row.get('correct'): raise RuntimeError('executable did not confirm correctness')
    return dict(variant=name,**row)

def save_csv(name, rows):
    RESULTS.mkdir(exist_ok=True)
    keys=list(rows[0])
    with (RESULTS/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def correctness(args):
    RESULTS.mkdir(exist_ok=True)
    print(command([BUILD/('ssm'+EXT),'--self-test']).strip(),flush=True)
    shapes=[(1,l,7,n) for l,n in [(1,3),(2,4),(31,31),(32,32),(33,33),(255,63),(256,64),(257,65),(1023,64),(1024,64),(1025,64)]]
    shapes += [(2,33,7,128),(1,2,1,1)]
    rows=[]
    for shape in shapes:
      for seed in (17,29,101):
        for name in VARIANTS:
          rows.append(run_case(name,shape,seed=seed,check_only=True))
      save_csv('correctness.csv',rows)
      print('correct',shape,len(rows),'cases',flush=True)

def sanitize(args):
    RESULTS.mkdir(exist_ok=True)
    rows=[]
    for check in ('memcheck','racecheck','synccheck'):
      for shape in ((1,33,7,32),(1,33,7,64),(1,33,7,65),(1,33,7,128)):
        argv=[tool('compute-sanitizer'),'--tool',check,'--error-exitcode','9',
              BUILD/('ssm'+EXT),*map(str,shape),'--check-only']
        result=subprocess.run(list(map(str,argv)),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        filename=f'{check}-N{shape[3]}.txt'
        (RESULTS/filename).write_text(clean(result.stdout+result.stderr),encoding='utf-8')
        report=result.stdout+result.stderr
        summary_ok=(bool(re.search(r'RACECHECK SUMMARY:\s*0 hazards displayed \(0 errors, 0 warnings\)',report))
                    if check=='racecheck' else bool(re.search(r'ERROR SUMMARY:\s*0 errors',report)))
        rows.append({'tool':check,'B':shape[0],'L':shape[1],'D':shape[2],'N':shape[3],
                     'returncode':result.returncode,'summary_ok':summary_ok,'log':filename})
        (RESULTS/'sanitizer.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print('sanitize',check,shape,result.returncode,flush=True)
        if result.returncode or not summary_ok: raise RuntimeError(f'sanitizer failed or incomplete: {filename}')

def measure(args):
    suites={
      'matrix':[(s,list(VARIANTS)) for s in MATRIX],
      'workspace':[((1,l,64,64),['seq','seq_workspace','improved','improved_workspace']) for l in (2048,4096,8192)],
      'chains':[((1,l,d,64),list(VARIANTS)) for l in (1024,8192) for d in (1,2,4,8,16,32,64)],
    }
    if args.include_unsafe_history:
      suites['sync_comparison']=[(s,['ssm','ssm_unsafe_history']) for s in MATRIX]
    if args.suite and args.suite not in suites:
      raise RuntimeError('sync_comparison requires explicit --include-unsafe-history')
    if args.suite: suites={k:v for k,v in suites.items() if k==args.suite}
    for suite,cases in suites.items():
      rows=[]
      for r in range(args.rounds):
        for shape,names in cases:
          order=names.copy();random.Random(1700+r).shuffle(order)
          for name in order:
            row=run_case(name,shape);row['round']=r+1;rows.append(row)
          save_csv(suite+'_raw.csv',rows)
          print('measured',suite,r+1,shape,flush=True)
      groups={}
      for row in rows: groups.setdefault((row['variant'],row['B'],row['L'],row['D'],row['N']),[]).append(row)
      summary=[]
      for (name,b,l,d,n),rr in groups.items():
        vv=[r['ms'] for r in rr]
        summary.append(dict(variant=name,B=b,L=l,D=d,N=n,rounds=len(rr),ms_median=statistics.median(vv),
            ms_min=min(vv),ms_max=max(vv),wall_ms_median=statistics.median(r['wall_ms'] for r in rr),
            max_abs_error=max(r['max_abs_error'] for r in rr)))
      save_csv(suite+'_summary.csv',summary)

def sass(args):
    records=[]
    for name in VARIANTS:
      raw=command([tool('cuobjdump'),'--dump-sass',BUILD/(name+EXT)])
      (BUILD/(name+'.sass')).write_text(raw,encoding='utf-8')
      for part in re.split(r'Function\s*:\s*',raw)[1:]:
        lines=part.splitlines();symbol=lines[0].strip()
        instructions=[]
        for line in lines[1:]:
          match=re.search(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b',line)
          if match: instructions.append(match.group(1))
        row={'variant':name,'kernel':symbol,'instructions':len(instructions)}
        for op in ('SHFL','ATOM','RED','LDG','STG','BAR','WARPSYNC','MUFU','FFMA','FADD','FMUL','LDS'):
          row[op]=sum(i.startswith(op) for i in instructions)
        records.append(row)
    save_csv('sass.csv',records)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','correctness','sanitize','measure','sass'))
    p.add_argument('--arch',default='sm_120');p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--include-unsafe-history',action='store_true',help='explicit opt-in to the historical pre-sync audit baseline')
    p.add_argument('--suite',choices=('matrix','workspace','chains','sync_comparison'))
    p.add_argument('--results-dir',type=Path,default=RESULTS,help='use a separate directory for a new run; bundled reference data live in results/')
    args=p.parse_args()
    if args.rounds<3: p.error('at least three independent rounds are required')
    globals()['RESULTS']=args.results_dir.resolve()
    globals()[args.command](args)

if __name__=='__main__': main()
