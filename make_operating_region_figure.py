"""Plot the complete prespecified 20 dB geometry sweep; no fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    r = json.loads(Path('results/operating_region/summary.json').read_text())
    out = Path('manuscript/figures')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.7,
        'lines.linewidth':1.2,'lines.markersize':4,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
    fig, axes = plt.subplots(2,2,figsize=(6.7,4.65))
    fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.93,wspace=.31,hspace=.43)
    colors=['#2D628A','#C77927','#4D8B75','#97588E']
    styles=['-','--','-.',':']
    plotted = {'source':'results/operating_region/summary.json','analytic':[], 'monte_carlo':[]}
    for row,m in enumerate((16,32)):
        for j,length in enumerate(range(12,16)):
            vals = [p for p in r['analytic'] if p['m']==m and p['length']==length and p['snr_db']==20]
            vals.sort(key=lambda p:p['delta'])
            plotted['analytic'].extend(vals)
            x=[p['delta'] for p in vals]
            axes[row,0].plot(x,[p['normalized_mse']['16'] for p in vals],
                color=colors[j],ls=styles[j],label=f'$L={length}$')
            axes[row,1].plot(x[1:],[p['crossover_blocks'] for p in vals[1:]],
                color=colors[j],ls=styles[j])
            if length in (12,15):
                mc = [p for p in r['monte_carlo'] if p['m']==m and p['length']==length and p['blocks']==16]
                plotted['monte_carlo'].extend(mc)
                axes[row,0].plot([p['delta'] for p in mc],[p['normalized_mse'] for p in mc],
                    'o',ms=4,mfc='white',mec=colors[j],mew=.8)
        axes[row,0].set(yscale='log',ylabel=r'Calibration MSE / $v^2$',
            title=f'({"ac"[row]}) {m} pilots: error at $T=16$')
        axes[row,1].set(ylabel=r'Crossover, $T_\star$ (blocks)',
            title=f'({"bd"[row]}) {m} pilots: crossover',ylim=(0,12.5 if row==0 else 3.0))
        axes[row,1].axhline(1,color='#888888',lw=.7,ls='--',zorder=0)
        axes[row,1].text(.03,.96,r'$\delta=0$: no bias floor',transform=axes[row,1].transAxes,
            fontsize=7,va='top')
        for ax in axes[row]:
            ax.set(xlabel=r'Fractional offset, $\delta$ (samples)',xlim=(-.02,.92),xticks=[0,.3,.6,.9])
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',color='#dddddd',lw=.45)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.53,.015))
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'operating_region.{ext}',facecolor='white')
    plt.close(fig)
    (out/'operating_region_points.json').write_text(json.dumps(plotted,indent=2),encoding='utf-8')
    print('Operating-region figure exported as PDF/SVG/PNG.')


if __name__=='__main__':run()
