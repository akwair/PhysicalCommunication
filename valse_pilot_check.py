"""Exploratory pilot-only VALSE implementation check; frozen protocol settings.

No full-CFR or BER prediction is constructed from compressed VALSE moments.
All methods use the same target pilot observation. The residual calibration
uses sixteen blocks; VALSE learns its own variance from the target block.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits
from stage3_study import (FREQ, M, PILOTS, stage3_profiles, generate,
                          projected_statistics, response_matrix)
from offgrid_baseline import estimate_offgrid
from valse_python import estimate_valse

PROFILES = ['sparse_integer','fractional_sparse','close_fractional']
METHODS = ['eb_residual16','refined_residual16','refined_known_noise','valse_equation']
FILES = ['valse_pilot_check.py','valse_python.py','test_valse_python.py',
         'offgrid_baseline.py','stage3_study.py','noise_study.py','pilot_ofdm.py',
         'confirmation_protocol.txt']


def encode(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path,document):
    path.write_text(json.dumps(document,default=encode,indent=2,allow_nan=False),encoding='utf-8')


def source_hashes():
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}


def pilot_metric(estimate,truth):
    return np.stack([np.mean(abs(estimate-truth)**2,axis=1),
                     np.mean(abs(truth)**2,axis=1)])


def mapping_audit():
    frequencies = FREQ[PILOTS][np.argsort(FREQ[PILOTS])]
    assert np.allclose(frequencies, -.5+np.arange(M)/16)
    errors = {}
    for name in PROFILES:
        delays = stage3_profiles()[name][0]
        actual = response_matrix(delays)[PILOTS][np.argsort(FREQ[PILOTS])]
        compressed = np.exp(-2j*np.pi*np.outer(np.arange(M),delays)/16)*np.exp(1j*np.pi*delays)
        errors[name] = float(np.max(abs(actual-compressed)))
    assert max(errors.values())<1e-12
    return errors


def run(args):
    start_hashes = source_hashes()
    rng = np.random.default_rng(args.seed)
    order = np.argsort(FREQ[PILOTS])
    signed = FREQ[PILOTS][order]
    raw,rows,diagnostics = {},[],[]
    began = time.perf_counter()
    for profile in PROFILES:
        delays,power = stage3_profiles()[profile]
        for snr in [10,20]:
            variance = 10**(-snr/10)
            truth,yp,_,_,_ = generate(rng,args.frames,delays,power,variance)
            target = yp[:,0][:,order]
            pilot_truth = truth[:,PILOTS][:,order]
            z,vhat,basis = projected_statistics(yp)
            residual_noise = vhat.mean(axis=1)
            prefix = f'{profile}_{snr}'
            raw[f'{prefix}_residual16_noise'] = residual_noise
            eb = (z[:,0]*np.maximum(0,1-(residual_noise[:,None]/M)/
                                    np.maximum(abs(z[:,0])**2,1e-300)))@basis[PILOTS][order].T
            raw[f'{prefix}_eb_residual16'] = pilot_metric(eb,pilot_truth)
            times = {}
            method_diagnostics = {}
            for name,supplied in [('refined_residual16',residual_noise),
                                  ('refined_known_noise',np.full(args.frames,variance))]:
                predicted,calls = [],[]
                start = time.perf_counter()
                for i in range(args.frames):
                    fit = estimate_offgrid(target[i],signed,signed,max(supplied[i],1e-12),
                                           max_delay=12.,grid_step=.25,max_paths=6,refine=True)
                    predicted.append(fit['response'])
                    calls.append({k:v for k,v in fit.items() if k not in ('response','gains','delays')})
                times[name] = time.perf_counter()-start
                raw[f'{prefix}_{name}'] = pilot_metric(np.asarray(predicted),pilot_truth)
                raw[f'{prefix}_{name}_orders'] = np.array([fit['selected_order'] for fit in calls])
                method_diagnostics[name] = calls
            predictions,valse_calls = [],[]
            start = time.perf_counter()
            for i in range(args.frames):
                fit = estimate_valse(target[i],np.arange(M),max_iter=500,heuristic=2,
                                     covariance_index='equation')
                predictions.append(fit['y_estimate'])
                valse_calls.append({k:fit[k] for k in ('noise_var','iterations','converged',
                                    'support','support_history','stability','kappa','freqs')})
            times['valse_equation'] = time.perf_counter()-start
            raw[f'{prefix}_valse_equation'] = pilot_metric(np.asarray(predictions),pilot_truth)
            for field in ['noise_var','iterations','converged','support']:
                raw[f'{prefix}_valse_{field}'] = np.array([fit[field] for fit in valse_calls])
            diagnostics.append(dict(profile=profile,snr_db=snr,
                                    refined=method_diagnostics,valse=valse_calls))
            for name in METHODS:
                values = raw[f'{prefix}_{name}']
                row = dict(profile=profile,snr_db=snr,method=name,groups=args.frames,
                           pilot_nmse_db=float(10*np.log10(values[0].sum()/values[1].sum())),
                           total_seconds=times.get(name))
                if name.startswith('refined'):
                    calls = method_diagnostics[name]
                    row.update(mean_order=float(np.mean([c['selected_order'] for c in calls])),
                               selected_optimizer_failures=sum(not c['refinement_success'][c['selected_order']] for c in calls),
                               candidate_optimizer_failures=sum(sum(not ok for ok in c['refinement_success']) for c in calls))
                elif name=='valse_equation':
                    interventions={}
                    for c in valse_calls:
                        for key,value in c['stability'].items():
                            interventions[key] = interventions.get(key,0)+value
                    row.update(mean_order=float(np.mean([sum(c['support']) for c in valse_calls])),
                               mean_noise_ratio=float(np.mean([c['noise_var'] for c in valse_calls])/variance),
                               converged_groups=sum(c['converged'] for c in valse_calls),
                               max_iterations=max(c['iterations'] for c in valse_calls),
                               mean_iterations=float(np.mean([c['iterations'] for c in valse_calls])),
                               stability=interventions)
                rows.append(row)
            print(f'{args.out.name}: {profile} {snr} dB complete; VALSE '
                  f'{rows[-1]["pilot_nmse_db"]:.3f} dB; '
                  f'{rows[-1]["converged_groups"]}/{args.frames} converged',flush=True)
    end_hashes = source_hashes()
    if start_hashes!=end_hashes:
        raise RuntimeError('Source files changed during campaign')
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'frame_metrics.npz',**raw)
    record = dict(status='exploratory_python_paper_equation_implementation_check',
        author_matlab_numerical_equivalence_validated=False,
        seed=args.seed,groups_per_cell=args.frames,results=rows,diagnostics=diagnostics,
        mapping_audit_max_error=mapping_audit(),source_hashes=start_hashes,
        numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        total_seconds=time.perf_counter()-began,threadpools=threadpool_info(),
        metric='ratio of summed per-group mean pilot squared error to summed pilot truth energy',
        assumptions={'pilot_order':'ascending signed frequency [-0.5,...,0.4375]',
                     'valse_indices':list(range(M)), 'valse_heuristic':2,
                     'valse_covariance_index':'equation','valse_max_iter':500,
                     'residual_calibration_blocks':16,'valse_observation_blocks':1,
                     'full_cfr_extrapolation':False,'ber_comparison':False,
                     'refined_delay_support':[0.,12.],
                     'valse_frequency_support':'unrestricted circle; delay modulo 16',
                     'known_noise_reference_has_oracle_noise':True})
    write_json(args.out/'results.json',record)


def aggregate(paths,out):
    out.mkdir(parents=True,exist_ok=True)
    documents=[json.loads((path/'results.json').read_text(encoding='utf-8')) for path in paths]
    files=[np.load(path/'frame_metrics.npz') for path in paths]
    seeds=[d['seed'] for d in documents]
    if len(set(seeds))!=len(seeds):
        raise ValueError('Seeds must be independent and distinct')
    if any(d['source_hashes']!=documents[0]['source_hashes'] for d in documents):
        raise ValueError('Campaigns used different source files')
    merged={key:np.concatenate([f[key] for f in files],axis=1 if key.endswith(tuple(METHODS)) else 0)
            for key in files[0].files}
    np.savez_compressed(out/'frame_metrics.npz',**merged)
    rows,comparisons=[],[]
    rng=np.random.default_rng(1320261113)
    for profile in PROFILES:
        for snr in [10,20]:
            prefix=f'{profile}_{snr}'
            variance=10**(-snr/10)
            count=len(merged[f'{prefix}_valse_noise_var'])
            index=rng.integers(0,count,size=(2000,count))
            for name in METHODS:
                values=merged[f'{prefix}_{name}']
                originals=[r for d in documents for r in d['results']
                           if r['profile']==profile and r['snr_db']==snr and r['method']==name]
                row=dict(profile=profile,snr_db=snr,method=name,groups=count,
                    pilot_nmse_db=float(10*np.log10(values[0].sum()/values[1].sum())))
                if name.startswith('refined'):
                    row.update(mean_order=float(merged[f'{prefix}_{name}_orders'].mean()),
                        selected_optimizer_failures=sum(r['selected_optimizer_failures'] for r in originals),
                        candidate_optimizer_failures=sum(r['candidate_optimizer_failures'] for r in originals))
                elif name=='valse_equation':
                    stability={}
                    for r in originals:
                        for key,value in r['stability'].items():
                            stability[key]=stability.get(key,0)+value
                    row.update(mean_noise_ratio=float(merged[f'{prefix}_valse_noise_var'].mean()/variance),
                        median_noise_ratio=float(np.median(merged[f'{prefix}_valse_noise_var'])/variance),
                        mean_order=float(merged[f'{prefix}_valse_support'].sum(axis=1).mean()),
                        converged_groups=int(merged[f'{prefix}_valse_converged'].sum()),
                        nonconverged_groups=int((~merged[f'{prefix}_valse_converged']).sum()),
                        mean_iterations=float(merged[f'{prefix}_valse_iterations'].mean()),
                        max_iterations=int(merged[f'{prefix}_valse_iterations'].max()),stability=stability)
                if name!='eb_residual16':
                    row['total_seconds']=sum(r['total_seconds'] for r in originals)
                rows.append(row)
            baseline=merged[f'{prefix}_valse_equation'][0]
            for comparator in ['refined_residual16','refined_known_noise']:
                reference=merged[f'{prefix}_{comparator}'][0]
                change=10*np.log10(baseline[index].sum(axis=1)/reference[index].sum(axis=1))
                comparisons.append(dict(profile=profile,snr_db=snr,reference=comparator,
                    contrast='VALSE pilot NMSE dB minus comparator pilot NMSE dB; negative favors VALSE',
                    difference_db=float(10*np.log10(baseline.sum()/reference.sum())),
                    percentile_95_ci_db=np.quantile(change,[.025,.975]),groups=count,resamples=2000))
    summary=dict(status='exploratory_implementation_check_not_author_runtime_equivalence',
        source_directories=[str(path) for path in paths],seeds=seeds,
        groups_per_cell=sum(d['groups_per_cell'] for d in documents),
        source_hashes=documents[0]['source_hashes'],results=rows,paired_bootstrap=comparisons,
        bootstrap_seed=1320261113,bootstrap_unit='independent calibration group with paired methods',
        multiplicity_adjusted=False,assumptions=documents[0]['assumptions'],
        artifact_hashes={str(path/name):hashlib.sha256((path/name).read_bytes()).hexdigest()
                         for path in paths for name in ['results.json','frame_metrics.npz']})
    write_json(out/'summary.json',summary)
    write_json(out/'paired_bootstrap.json',comparisons)
    for f in files:
        f.close()
    print(json.dumps(rows,default=encode,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    run_parser=sub.add_parser('run')
    run_parser.add_argument('--frames',type=int,default=150)
    run_parser.add_argument('--seed',type=int,default=1220261113)
    run_parser.add_argument('--out',type=Path,default=Path('results/valse_check1'))
    summary_parser=sub.add_parser('aggregate')
    summary_parser.add_argument('paths',nargs='+',type=Path)
    summary_parser.add_argument('--out',type=Path,default=Path('results/valse_check_summary'))
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command=='run':
            if args.frames<1:
                parser.error('frames must be positive')
            run(args)
        else:
            aggregate(args.paths,args.out)
