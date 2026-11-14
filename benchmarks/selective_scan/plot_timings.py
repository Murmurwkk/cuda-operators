"""Draw scan measurements from the saved CSV summaries, never inline numbers."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PALETTE={"ssm":"#2a78d6","seq":"#eb6834","rewrite":"#1baf7a","improved":"#eda100","seq_parallel":"#e87ba4","streaming_seq_parallel_fixed":"#008300"}
LABELS={"ssm":"ssm.cu","seq":"ssm_seq.cu","rewrite":"selective_scan_rewrite.cu","improved":"selective_scan_improved.cu","seq_parallel":"selective_scan_seq_parallel.cu","streaming_seq_parallel_fixed":"selective_scan_streaming_seq_parallel_fixed.cu"}

def read(path):
    with path.open(newline="",encoding="utf-8") as f:
        return [{k: (v if k=="variant" else float(v)) for k,v in row.items()} for row in csv.DictReader(f)]

def axis(ax,xlabel):
    ax.set_xscale("log",base=2);ax.set_yscale("log")
    ax.set_xlabel(xlabel);ax.set_ylabel("Time per solve() call (ms)")
    ax.grid(True,which="major",alpha=.25)
    ax.spines[["top","right"]].set_visible(False)

def line(ax,rows,x,label,color):
    rows=sorted(rows,key=lambda r:r[x])
    xx=[r[x] for r in rows];med=[r["ms_median"] for r in rows]
    ax.plot(xx,med,"o-",label=label,color=color,linewidth=1.8,markersize=5)
    ax.fill_between(xx,[r["ms_min"] for r in rows],[r["ms_max"] for r in rows],color=color,alpha=.12)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--results",type=Path,default=HERE/"results")
p.add_argument("--out-dir",type=Path,default=(ROOT/"blog_mur/src/assets/selective-scan" if (ROOT/"blog_mur").exists() else HERE/"figures"))
a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
rows=read(a.results/"matrix_summary.csv")
fig,ax=plt.subplots(figsize=(9.2,6.4),layout="constrained")
for name,color in PALETTE.items():
    line(ax,[r for r in rows if r["variant"]==name and r["B"]==1 and r["D"]==64 and r["N"]==64],"L",LABELS[name],color)
axis(ax,"Sequence length L (B=1, D=N=64)")
ax.set_xticks([256,512,1024,2048,4096,8192],labels=["256","512","1024","2048","4096","8192"])
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,fontsize=8)
fig.savefig(a.out_dir/"time-vs-length.png",dpi=150);plt.close(fig)
if (a.results/"chains_summary.csv").exists():
    rows=read(a.results/"chains_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout="constrained")
    for ax,length in zip(axes,[1024,8192]):
        for name,color in PALETTE.items(): line(ax,[r for r in rows if r["variant"]==name and r["L"]==length],"D",LABELS[name],color)
        axis(ax,"Independent chains D (B=1, N=64)");ax.set_title(f"L = {length:,}")
        ax.set_xticks([1,2,4,8,16,32,64],labels=["1","2","4","8","16","32","64"])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=8)
    fig.savefig(a.out_dir/"independent-chains.png",dpi=150);plt.close(fig)
if (a.results/"workspace_summary.csv").exists():
    rows=read(a.results/"workspace_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout="constrained")
    for ax,base,title in zip(axes,["seq","improved"],["ssm_seq.cu","selective_scan_improved.cu"]):
        line(ax,[r for r in rows if r["variant"]==base],"L","Allocate per call",PALETTE[base])
        line(ax,[r for r in rows if r["variant"]==base+"_workspace"],"L","Caller-owned workspace","#222222")
        axis(ax,"Sequence length L (B=1, D=N=64)");ax.set_title(title)
        ax.set_xticks([2048,4096,8192],labels=["2048","4096","8192"]);ax.legend(frameon=False)
    fig.savefig(a.out_dir/"workspace-ablation.png",dpi=150);plt.close(fig)
print("Figures written to",a.out_dir)
