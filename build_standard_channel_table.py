"""Build manuscript tables directly from the audited EPA/EVA summary."""
import json
from pathlib import Path


def run():
    report = json.loads(Path('results/standard_channel_summary/summary.json').read_text())
    metrics = {(r['profile'], r['snr_db'], r['method']): r for r in report['results']}
    noise = {(r['profile'], r['snr_db'], r['source']): r for r in report['noise']}
    gains = {(r['profile'], r['snr_db']): r for r in report['comparisons'] if r['candidate'] == 'refined_cp16'}
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Standard EPA/EVA power-delay profiles: noise calibration and continuous-delay receiver NMSE (dB), with 900 paired groups per condition. R and CP use 16-block residual and CP calibration; K uses known noise. Noise ratios are empirical means divided by the thermal variance. All receiver settings are fixed.}',
        r'\label{tab:standard_nmse}', r'\small', r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'PDP & SNR & \multicolumn{2}{c}{Noise ratio} & \multicolumn{3}{c}{NMSE (dB)} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-7}',
        r' & (dB) & R & CP & R & CP & K \\', r'\midrule']
    for profile in ('EPA', 'EVA'):
        for snr in (0, 10, 20):
            nr = noise[profile, snr, 'residual16']['mean_ratio']
            nc = noise[profile, snr, 'cp16']['mean_ratio']
            nmse = [metrics[profile, snr, 'refined_'+s]['nmse_db'] for s in ('residual16', 'cp16', 'known_noise')]
            lines.append(f'{profile} & {snr} & {nr:.3f} & {nc:.3f} & '+ ' & '.join(f'{x:.2f}' for x in nmse)+r' \\')
        if profile == 'EPA': lines.append(r'\midrule')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', '', r'\begin{table}[t]', r'\centering',
        r'\caption{Paired NMSE gains from CP calibration and uncoded BER for the same standard-profile experiment. Positive gain favors CP; brackets give pointwise 95\% paired bootstrap intervals, stratified by seed. BER columns use the continuous-delay receiver with residual (R), CP, and known-noise (K) calibration.}',
        r'\label{tab:standard_gain}', r'\small', r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        r'PDP & SNR & NMSE gain (dB) & \multicolumn{3}{c}{BER} \\',
        r'\cmidrule(lr){4-6}', r' & (dB) & CP over R [95\% CI] & R & CP & K \\', r'\midrule'])
    for profile in ('EPA', 'EVA'):
        for snr in (0, 10, 20):
            gain = gains[profile, snr]
            ber = [metrics[profile, snr, 'refined_'+s]['ber'] for s in ('residual16', 'cp16', 'known_noise')]
            lines.append(f"{profile} & {snr} & {gain['gain_db']:.2f} [{gain['ci95'][0]:.2f}, {gain['ci95'][1]:.2f}] & "+
                         ' & '.join(f'{x:.4f}' for x in ber)+r' \\')
        if profile == 'EPA': lines.append(r'\midrule')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    Path('manuscript/table_standard_channels.tex').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
