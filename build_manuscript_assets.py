"""Generate manuscript numerical tables directly from recorded results."""
import json
from pathlib import Path

def reconstruction_table():
    doc=json.loads(Path('results/stage3_offgrid_summary/summary.json').read_text())
    methods=['eb_residual_16','cyclic_extended_eb_residual16','grid_residual16',
             'refined_residual16','refined_true_noise_ref','oracle_lmmse_ref']
    profiles={'exp2':'Exp-2','exp8':'Exp-8','sparse_integer':'Sparse integer',
              'fractional_sparse':'Separated fractional','close_fractional':'Close fractional'}
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Full-channel NMSE (dB) at 20~dB nominal SNR; 900 paired groups per profile. Integer EB, extended EB, and grid/refined residual controls use 16-block calibration. The known-noise and oracle columns receive extra statistical information.}',
        r'\label{tab:reconstruction}',r'\small\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'& Integer & Extended & Grid & Refined & Refined & Oracle \\',
        r'Profile & EB & EB & residual & residual & known $v$ & LMMSE \\',r'\midrule']
    for profile,label in profiles.items():
        values=[next(r['nmse_db'] for r in doc['results'] if r['profile']==profile and
                     r['snr_db']==20 and r['method']==method) for method in methods]
        lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    Path('manuscript/table_reconstruction.tex').write_text('\n'.join(lines),encoding='utf-8')

def valse_table():
    doc=json.loads(Path('results/valse_check_summary/summary.json').read_text())
    profiles={'sparse_integer':'Sparse integer','fractional_sparse':'Separated fractional','close_fractional':'Close fractional'}
    methods=['eb_residual16','refined_residual16','refined_known_noise','valse_equation']
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Pilot-domain reconstruction and noise inference; 450 groups per cell. Columns 3--6 give NMSE in dB. VALSE learns noise from one block with circular frequency support; refined controls use the delay bound $[0,12]$. The comparison concerns the observed pilot grid.}',
        r'\label{tab:valse}',r'\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',
        r'\toprule',r'& SNR & Integer & Refined & Refined & VALSE & VALSE \\',
        r'Profile & (dB) & EB & residual & known $v$ & & $\mathbb E[\widehat v]/v$ \\',r'\midrule']
    for profile,label in profiles.items():
        for snr in [10,20]:
            rows=[next(r for r in doc['results'] if r['profile']==profile and r['snr_db']==snr and r['method']==m) for m in methods]
            values=[r['pilot_nmse_db'] for r in rows]+[rows[-1]['mean_noise_ratio']]
            lines.append(label+f' & {snr} & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    Path('manuscript/table_valse.tex').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    reconstruction_table()
    valse_table()
