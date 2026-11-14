"""Check saved coverage, correctness, reductions, and reported scan statistics."""
import argparse
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
MATRIX={(1,l,64,64) for l in (256,512,1024,2048,4096,8192)}|{
    (1,1024,128,64),(1,1024,256,64),(1,1024,64,32),(1,1024,64,128),
    (8,512,64,64),(4,1024,64,64)}
VARIANTS={'ssm','seq','rewrite','improved','seq_parallel','streaming_seq_parallel_fixed'}

def read(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def shape(r):return tuple(int(r[k]) for k in ('B','L','D','N'))
def key(r):return r['variant'],shape(r)

def verify(directory, require_history=False):
    counts={}
    for suite,names,shapes in [
      ('matrix',VARIANTS,MATRIX),
      ('sync_comparison',{'ssm','ssm_unsafe_history'},MATRIX),
      ('workspace',{'seq','seq_workspace','improved','improved_workspace'},{(1,l,64,64) for l in (2048,4096,8192)}),
      ('chains',VARIANTS,{(1,l,d,64) for l in (1024,8192) for d in (1,2,4,8,16,32,64)})]:
        if suite=='sync_comparison':
            raw_exists=(directory/(suite+'_raw.csv')).exists()
            summary_exists=(directory/(suite+'_summary.csv')).exists()
            if not raw_exists and not summary_exists:
                if require_history: raise ValueError('historical comparison is required but both CSV files are missing')
                continue
            if raw_exists != summary_exists:
                raise ValueError('historical comparison is incomplete: raw and summary CSV files must both exist')
        rows=read(directory/(suite+'_raw.csv'));summaries=read(directory/(suite+'_summary.csv'))
        expected={(n,s) for n in names for s in shapes}
        assert {key(r) for r in rows}==expected,(suite,'raw coverage')
        assert {key(r) for r in summaries}==expected,(suite,'summary coverage')
        assert len(summaries)==len(expected),(suite,'duplicate summary')
        for r in rows:
            assert r['correct']=='True'
            assert all(isfinite(float(r[k])) for k in ('ms','wall_ms','max_abs_error','max_normalized_error'))
            assert float(r['ms'])>0 and float(r['wall_ms'])>0
            assert float(r['max_normalized_error'])<=1
        for summary in summaries:
            selected=[r for r in rows if key(r)==key(summary)]
            assert len(selected)==int(summary['rounds'])>=3
            assert len({int(r['round']) for r in selected})==len(selected)
            values=[float(r['ms']) for r in selected]
            for name,value in [('ms_median',statistics.median(values)),('ms_min',min(values)),('ms_max',max(values))]:
                assert abs(float(summary[name])-value)<1e-10,(suite,key(summary),name)
        counts[suite]=len(rows)
    correct=read(directory/'correctness.csv')
    assert len(correct)==234
    assert all(r['correct']=='True' and float(r['max_normalized_error'])<=1 for r in correct)
    sanitizer=json.loads((directory/'sanitizer.json').read_text())
    assert len(sanitizer)==12 and all(r['returncode']==0 and r['summary_ok'] for r in sanitizer)
    rows=read(directory/'matrix_summary.csv')
    wins=[];largest=0
    for s in sorted(MATRIX):
        by_name={r['variant']:float(r['ms_median']) for r in rows if shape(r)==s}
        wins.append({'shape':list(s),'winner':min(by_name,key=by_name.get)})
        largest=max(largest,max(by_name.values())/by_name['ssm'])
    report={'raw_row_counts':counts,'correctness_cases':len(correct),'sanitizer_cases':len(sanitizer),
            'ssm_wins':sum(r['winner']=='ssm' for r in wins),'winners':wins,
            'largest_slowdown_relative_to_ssm':largest,
            'matrix_max_abs_error':max(float(r['max_abs_error']) for r in rows)}
    report['figure_inputs_sha256']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ['matrix_summary.csv','workspace_summary.csv','chains_summary.csv']}
    report['plot_source_sha256']=hashlib.sha256((HERE/'plot_timings.py').read_bytes()).hexdigest()
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=HERE/'results');p.add_argument('--out',type=Path)
    p.add_argument('--require-history',action='store_true',help='require both historical comparison CSV files as well as the three default suites')
    a=p.parse_args();report=verify(a.results,require_history=a.require_history);text=json.dumps(report,indent=2)
    if a.out:a.out.write_text(text,encoding='utf-8')
    print(text)
