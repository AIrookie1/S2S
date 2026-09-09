#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
HERE = Path(__file__).resolve().parent
COLOR = "#0072B2"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=HERE/"weight_kappa_mc1.json"); p.add_argument("--output-prefix",type=Path,default=HERE/"fig5_weight_kappa_mc1"); a=p.parse_args()
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":18,"axes.linewidth":1.5,"lines.linewidth":2.5,"lines.markersize":12.5,"xtick.direction":"out","ytick.direction":"out","xtick.major.width":1.5,"ytick.major.width":1.5,"pdf.fonttype":42,"svg.fonttype":"none","savefig.bbox":"tight"})
    data=json.loads(a.input.read_text()); rows=sorted(data["results"],key=lambda r:float(r["weight_kappa"])); x=[float(r["weight_kappa"]) for r in rows];y=[float(r["MC1"]) for r in rows]
    fig,ax=plt.subplots(figsize=(10.45 ,3.15)); ax.plot(x,y,color=COLOR,marker="o",markerfacecolor="white",markeredgecolor=COLOR,markeredgewidth=1.5); ax.set_xscale("log"); ax.set_xlabel("WEIGHT_KAPPA"); ax.set_ylabel("MC1"); ax.set_xticks(x)
    labels=["1" if v==1 else f"1e{int(round(math.log10(v)))}" for v in x]; ax.set_xticklabels(labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6)); ax.tick_params(top=False,right=False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.margins(x=.04,y=.10); a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in (".png",".pdf",".svg"): fig.savefig(a.output_prefix.with_suffix(ext),dpi=600)
    plt.close(fig)
    print(f"Saved: {a.output_prefix.with_suffix('.png')}")
if __name__=="__main__": main()
