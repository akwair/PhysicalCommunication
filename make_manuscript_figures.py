"""Rebuild manuscript figures from recorded numerical results; matplotlib only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_study import stage3_profiles, PILOTS
from waveform_model import projection_noise_bias

OUT=Path('manuscript/figures')
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
    'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
    'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,
    'lines.linewidth':1.2,'lines.markersize':4,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
COLORS=['#2D628A','#C77927','#4D8B75','#97588E','#5B6574','#222222']
POINTS={}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def row(rows,**kwargs):
    matches=[r for r in rows if all(r.get(k)==v for k,v in kwargs.items())]
    assert len(matches)==1,(kwargs,len(matches))
    return matches[0]
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#dddddd',linewidth=.45,zorder=0)
def save(fig,name):
    for extension in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{extension}',facecolor='white')
    plt.close(fig)

def calibration():
    summary=read('results/stage3_pool_summary/summary.json')
    audit=read('results/stage3_correlation_audit.json')
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.65))
    fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.84,wspace=.32)
    for idx,profile in enumerate(['exp2','fractional_sparse','close_fractional']):
        delays,power=stage3_profiles()[profile]
        bias=projection_noise_bias(delays,power,PILOTS,12)
        snrs=np.linspace(0,20,100)
        axes[0].plot(snrs,1+bias*10**(snrs/10),c=COLORS[idx],alpha=.85)
        values=[row(summary['noise'],profile=profile,snr_db=s,source='residual_16')['mean_ratio'] for s in [0,10,20]]
        axes[0].plot([0,10,20],values,'o',mfc='white',c=COLORS[idx],
             label={'exp2':'Matched (Exp-2)','fractional_sparse':'Separated fractional','close_fractional':'Close fractional'}[profile])
        POINTS['calibration_mean_'+profile]=values
    axes[0].set(xlabel='Nominal SNR (dB)',ylabel=r'Mean noise estimate / true variance',
                title='(a) Mean: 16 calibration blocks',xticks=[0,10,20])
    axes[0].legend(loc='upper left',frameon=False)
    for idx,rho in enumerate([0,.9,.99,1.]):
        rs=[row(audit['results'],rho=rho,blocks=t) for t in [1,4,16]]
        axes[1].plot([1,4,16],[r['predicted_relative_std'] for r in rs],c=COLORS[idx],label=fr'$\rho={rho:g}$')
        axes[1].plot([1,4,16],[r['relative_std'] for r in rs],'o',mfc='white',c=COLORS[idx])
    axes[1].set(xlabel='Calibration blocks, T',ylabel=r'Standard deviation / true variance',
                title='(b) Variability: separated, 20 dB',xticks=[1,4,16],ylim=(0,9))
    axes[1].legend(frameon=False,ncol=2,loc='lower left')
    for ax in axes: style(ax)
    save(fig,'calibration_bias')

def reconstruction():
    summary=read('results/stage3_offgrid_summary/summary.json')
    methods=[('eb_residual_16','Integer EB, residual'),
        ('cyclic_extended_eb_residual16','Extended EB, residual'),
        ('grid_residual16','Grid search, residual'),
        ('refined_residual16','Refined, residual'),
        ('refined_true_noise_ref','Refined, known noise'),
        ('oracle_lmmse_ref','Oracle covariance LMMSE')]
    fig,axes=plt.subplots(2,2,figsize=(6.7,4.55))
    fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.92,wspace=.30,hspace=.43)
    for column,profile in enumerate(['fractional_sparse','close_fractional']):
        for i,(method,label) in enumerate(methods):
            rows=[row(summary['results'],profile=profile,snr_db=s,method=method) for s in [0,10,20]]
            values={key:[r[key] for r in rows] for key in ['nmse_db','ber']}
            POINTS[f'{profile}_{method}']=values
            for axis,key in zip(axes[:,column],['nmse_db','ber']):
                axis.plot([0,10,20],values[key],color=COLORS[i],marker=['o','s','^','D','v','x'][i],
                          linestyle='--' if i in [4,5] else '-',label=label,mfc='white')
        axes[0,column].set(title=['(a) Separated fractional','(b) Close fractional'][column],
            ylabel='Full-CFR NMSE (dB)',xticks=[0,10,20],ylim=(-30,1))
        axes[1,column].set(title=['(c) Separated fractional','(d) Close fractional'][column],
            xlabel='Nominal SNR (dB)',ylabel='Uncoded QPSK BER',xticks=[0,10,20],yscale='log',ylim=(.004,.5))
        for axis in axes[:,column]: style(axis)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.53,.005))
    save(fig,'reconstruction')

def confirmation():
    summary=read('results/cp_confirmation_summary/summary.json')
    cells=[('fractional_sparse',10),('fractional_sparse',20),('close_fractional',10),('close_fractional',20)]
    labels=['Sep.\n10 dB','Sep.\n20 dB','Close\n10 dB','Close\n20 dB']
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.65))
    fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.85,wspace=.32)
    x=np.arange(4)
    for idx,source in enumerate(['residual16','cp16']):
        values=[row(summary['noise'],profile=p,snr_db=s,source=source)['mean_ratio'] for p,s in cells]
        axes[0].plot(x,values,['o','s'][idx],c=COLORS[idx],label=['Residual, T=16','CP difference, T=16'][idx])
        POINTS['cp_noise_'+source]=values
    axes[0].axhline(1,c='#777777',ls='--',lw=.8,label='True variance')
    axes[0].set(title='(a) Calibration mean',ylabel='Mean estimate / true variance',xticks=x,xticklabels=labels,ylim=(0,12))
    axes[0].legend(frameon=False,loc='upper left',fontsize=7)
    rs=[row(summary['comparisons'],profile=p,snr_db=s,candidate='refined_cp16',reference='refined_residual16') for p,s in cells]
    gains=np.array([r['gain_db'] for r in rs]);cis=np.array([r['ci95'] for r in rs])
    axes[1].errorbar(x,gains,yerr=np.stack([gains-cis[:,0],cis[:,1]-gains]),fmt='o',c=COLORS[0],capsize=3)
    axes[1].axhline(0,c='#777777',ls='--',lw=.8)
    axes[1].set(title='(b) CP calibration gain in refined receiver',ylabel='Paired NMSE gain (dB)',
        xticks=x,xticklabels=labels,ylim=(-.2,5))
    POINTS['cp_gains']=rs
    for ax in axes: style(ax)
    save(fig,'cp_confirmation')


if __name__=='__main__':
    calibration();reconstruction();confirmation()
    (OUT/'plotted_points.json').write_text(json.dumps(POINTS,indent=2),encoding='utf-8')
    print('Three manuscript figures exported as PDF, SVG, PNG.')
