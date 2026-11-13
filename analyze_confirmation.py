"""Aggregate the frozen CP control experiment; group-level paired bootstrap."""
import hashlib
import json
from pathlib import Path
import numpy as np


def run():
    paths=[Path(f'results/cp_confirmation{i}') for i in [1,2,3]]
    docs=[json.loads((p/'results.json').read_text()) for p in paths]
    assert len({d['seed'] for d in docs})==3
    assert all(d['source_hashes']==docs[0]['source_hashes'] for d in docs)
    for path,digest in docs[0]['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    arrays=[np.load(p/'frame_metrics.npz') for p in paths]
    rng=np.random.default_rng(1320261113)
    rows,noise,comparisons,diagnostics=[],[],[],[]
    for profile in ['fractional_sparse','close_fractional']:
        for snr in [10,20]:
            prefix=f'{profile}_{snr}_'
            values={}
            for method in ['eb_residual16','eb_cp16','refined_residual16','refined_cp16','refined_known_noise']:
                a=np.concatenate([data[prefix+method] for data in arrays],axis=1)
                values[method]=a
                rows.append(dict(profile=profile,snr_db=snr,method=method,groups=a.shape[1],
                    nmse_db=float(10*np.log10(a[0].sum()/a[1].sum())),ber=float(a[2].sum()/(96*a.shape[1]))))
            for source in ['residual16','cp16','known_noise']:
                v=np.concatenate([a[prefix+source+'_noise'] for a in arrays])*10**(snr/10)
                noise.append(dict(profile=profile,snr_db=snr,source=source,
                    mean_ratio=float(v.mean()),relative_std=float(v.std())))
                calls=[call for d in docs for cell in d['diagnostics'] if
                       cell['profile']==profile and cell['snr_db']==snr and cell['source']==source
                       for call in cell['calls']]
                diagnostics.append(dict(profile=profile,snr_db=snr,source=source,calls=len(calls),
                    selected_failure=sum(not c['refinement_success'][c['selected_order']] for c in calls),
                    any_candidate_failure=sum(not all(c['refinement_success']) for c in calls),
                    mean_selected_order=float(np.mean([c['selected_order'] for c in calls])),
                    mean_nfev=float(np.mean([c['nfev'] for c in calls]))))
            for candidate,reference in [('refined_cp16','refined_residual16'),
                    ('refined_known_noise','refined_cp16'),('eb_cp16','eb_residual16')]:
                ca,re=values[candidate],values[reference]
                gain,ber=[],[]
                for _ in range(20):
                    indices=rng.integers(ca.shape[1],size=(100,ca.shape[1]))
                    gain.extend(10*np.log10(re[0,indices].sum(axis=1)/ca[0,indices].sum(axis=1)))
                    ber.extend((re[2,indices]-ca[2,indices]).mean(axis=1)/96)
                comparisons.append(dict(profile=profile,snr_db=snr,candidate=candidate,reference=reference,
                    gain_db=float(10*np.log10(re[0].sum()/ca[0].sum())),ci95=np.quantile(gain,[.025,.975]).tolist(),
                    ber_reduction=float((re[2]-ca[2]).mean()/96),ber_ci95=np.quantile(ber,[.025,.975]).tolist()))
    output=dict(status='frozen_independent_cp_control_no_multiplicity_adjustment',
        groups_per_cell=sum(d['groups_per_cell'] for d in docs),seeds=[d['seed'] for d in docs],
        bootstrap_replicates=2000,results=rows,noise=noise,comparisons=comparisons,diagnostics=diagnostics,
        maximum_cp_signal_difference=max(max(d['signal_cancellation_audit'].values()) for d in docs))
    out=Path('results/cp_confirmation_summary')
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for c in comparisons:
        print(c)


if __name__=='__main__': run()
