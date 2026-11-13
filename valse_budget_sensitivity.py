"""Replay all capped VALSE cases at 500 then 5000 updates; preserve old data."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits
from stage3_study import FREQ, M, PILOTS, stage3_profiles, generate
from valse_pilot_check import PROFILES, source_hashes, write_json
from valse_python import estimate_valse


def fit_summary(fit, truth):
    return dict(pilot_error=float(np.mean(abs(fit['y_estimate']-truth)**2)),
                noise_var=float(fit['noise_var']),
                iterations=int(fit['iterations']), converged=bool(fit['converged']),
                order=int(fit['support'].sum()), support=fit['support'],
                support_history=fit['support_history'], freqs=fit['freqs'],
                kappa=fit['kappa'], stability=fit['stability'])


def check_replay(fit, truth, old, saved_error):
    """Check direct saved observables and the entire deterministic trajectory."""
    error = float(np.mean(abs(fit['y_estimate']-truth)**2))
    assert_allclose(error, saved_error, rtol=1e-12, atol=1e-14)
    for key in ('noise_var', 'freqs', 'kappa'):
        assert_allclose(fit[key], old[key], rtol=1e-12, atol=1e-14)
    for key in ('iterations', 'converged', 'support', 'support_history'):
        assert_array_equal(fit[key], old[key])
    assert fit['stability'] == old['stability']
    return dict(pilot_error_absolute_difference=abs(error-float(saved_error)),
                noise_var_absolute_difference=abs(float(fit['noise_var'])-old['noise_var']),
                support_history_exact_match=True,
                max_frequency_absolute_difference=float(np.max(abs(fit['freqs']-old['freqs']),initial=0)),
                max_kappa_absolute_difference=float(np.max(abs(fit['kappa']-old['kappa']),initial=0)))


def aggregate_cell(error, energy, noise, order, converged, iterations, variance):
    return dict(groups=len(error),
                pilot_nmse_db=float(10*np.log10(np.sum(error)/np.sum(energy))),
                mean_noise_ratio=float(np.mean(noise)/variance),
                mean_order=float(np.mean(order)),
                converged_groups=int(np.sum(converged)),
                capped_groups=int(np.sum(~np.asarray(converged,dtype=bool))),
                mean_iterations=float(np.mean(iterations)),
                max_iterations=int(np.max(iterations)))


def main():
    start = time.perf_counter()
    paths = [Path(f'results/valse_check{i}') for i in (1,2,3)]
    docs = [json.loads((p/'results.json').read_text(encoding='utf-8')) for p in paths]
    current_hashes = source_hashes()
    for doc in docs:
        if doc['source_hashes'] != current_hashes:
            raise RuntimeError('Current source differs from recorded campaign source')
    artifact_hashes = {str(p/name):hashlib.sha256((p/name).read_bytes()).hexdigest()
                      for p in paths for name in ('frame_metrics.npz','results.json')}
    raw = [{key:data[key].copy() for key in data.files}
           for data in [np.load(p/'frame_metrics.npz') for p in paths]]
    replacements = [{key:val.copy() for key,val in data.items()} for data in raw]
    selected, cells = [], []
    order = np.argsort(FREQ[PILOTS])
    archive = {}
    truth_checks = 0
    for campaign,(doc,data) in enumerate(zip(docs,raw)):
        rng = np.random.default_rng(doc['seed'])
        frames = doc['groups_per_cell']
        for profile in PROFILES:
            delays,power = stage3_profiles()[profile]
            for snr in (10,20):
                variance = 10**(-snr/10)
                truth,yp,_,_,_ = generate(rng,frames,delays,power,variance)
                target = yp[:,0][:,order]
                pilot_truth = truth[:,PILOTS][:,order]
                prefix = f'{profile}_{snr}'
                truth_energy = np.mean(abs(pilot_truth)**2,axis=1)
                assert_allclose(truth_energy,data[f'{prefix}_valse_equation'][1],rtol=0,atol=0)
                truth_checks += frames
                diag = next(c for c in doc['diagnostics'] if c['profile']==profile and c['snr_db']==snr)
                for index in np.flatnonzero(~data[f'{prefix}_valse_converged']):
                    old = diag['valse'][int(index)]
                    assert old['iterations']==500 and not old['converged']
                    key = f'campaign{campaign+1}_{prefix}_frame{index}'
                    archive[key+'_target'] = target[index]
                    archive[key+'_truth'] = pilot_truth[index]
                    selected.append(dict(campaign=campaign,seed=doc['seed'],profile=profile,
                                         snr_db=snr,index=int(index),prefix=prefix,key=key,
                                         old=old,target=target[index],truth=pilot_truth[index]))
    assert len(selected)==12 and truth_checks==2700
    np.savez_compressed('results/valse_budget_inputs.npz',**archive)
    cases = []
    for seq,case in enumerate(selected,1):
        campaign,prefix,index = case['campaign'],case['prefix'],case['index']
        began = time.perf_counter()
        replay = estimate_valse(case['target'],np.arange(M),max_iter=500,
                                heuristic=2,covariance_index='equation')
        replay_check = check_replay(replay,case['truth'],case['old'],
                                   raw[campaign][f'{prefix}_valse_equation'][0,index])
        replay_seconds = time.perf_counter()-began
        began = time.perf_counter()
        fit = estimate_valse(case['target'],np.arange(M),max_iter=5000,
                             heuristic=2,covariance_index='equation')
        extended_seconds = time.perf_counter()-began
        before,after = fit_summary(replay,case['truth']),fit_summary(fit,case['truth'])
        row = {k:case[k] for k in ('campaign','seed','profile','snr_db','index','key')}
        row.update(before=before,after=after,replay_check=replay_check,
                   replay_seconds=replay_seconds,extended_seconds=extended_seconds,
                   execution_failure=None)
        cases.append(row)
        new = replacements[campaign]
        new[f'{prefix}_valse_equation'][0,index] = after['pilot_error']
        for field in ('noise_var','iterations','converged','support'):
            new[f'{prefix}_valse_{field}'][index] = fit[field]
        print(f'{seq}/12 {case["key"]}: {after["iterations"]} iterations; '
              f'converged={after["converged"]}; error '
              f'{before["pilot_error"]:.8g}->{after["pilot_error"]:.8g}',flush=True)
    for profile in PROFILES:
        for snr in (10,20):
            prefix = f'{profile}_{snr}'
            row = dict(profile=profile,snr_db=snr)
            for label,group in [('original_500',raw),('capped_replayed_5000',replacements)]:
                values = np.concatenate([g[f'{prefix}_valse_equation'] for g in group],axis=1)
                joined = lambda field: np.concatenate([g[f'{prefix}_valse_{field}'] for g in group])
                row[label] = aggregate_cell(values[0],values[1],joined('noise_var'),
                    joined('support').sum(axis=1),joined('converged'),joined('iterations'),10**(-snr/10))
            row['nmse_change_db'] = row['capped_replayed_5000']['pilot_nmse_db']-row['original_500']['pilot_nmse_db']
            row['mean_noise_ratio_change'] = row['capped_replayed_5000']['mean_noise_ratio']-row['original_500']['mean_noise_ratio']
            cells.append(row)
    if current_hashes != source_hashes():
        raise RuntimeError('Source files changed during audit')
    for name,sha in artifact_hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    report = dict(status='completed_local_python_iteration_budget_sensitivity',
        author_matlab_runtime_equivalence=False,iteration_budgets=[500,5000],
        protocol='valse_budget_protocol.txt',source_hashes=current_hashes,
        audit_source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                            for name in ('valse_budget_protocol.txt','valse_budget_sensitivity.py')},
        original_artifact_hashes=artifact_hashes,original_artifacts_unchanged=True,
        reconstructed_truth_energy_exact_matches=truth_checks,
        total_original_fits=2700,capped_cases_replayed=len(cases),
        converged_before=2688,converged_after=2688+sum(c['after']['converged'] for c in cases),
        all_cases_retained=True,execution_failures=0,cases=cases,per_cell=cells,
        selected_inputs_artifact='results/valse_budget_inputs.npz',
        selected_inputs_sha256=hashlib.sha256(Path('results/valse_budget_inputs.npz').read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        total_seconds=time.perf_counter()-start)
    write_json(Path('results/valse_budget_sensitivity.json'),report)
    print(json.dumps(cells,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
