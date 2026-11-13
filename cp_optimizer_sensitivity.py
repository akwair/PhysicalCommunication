"""Post-confirmation optimization diagnostic; frozen sources/results are unchanged.

Reconstruct all cells in their exact RNG order, then rerun only calls with any
candidate failure at max_nfev=60. Budget 300 replaces those calls in an in-memory
copy. A second diagnostic uses 1000 only for selected failures remaining at 300.
Neither success flags nor this selective audit establish global optimality.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import offgrid_baseline as baseline
from cp_confirmation import generate_cp, cp_signal_audit
from stage3_study import FREQ, PILOTS, projected_statistics, stage3_profiles, metrics

SOURCES = ("residual16", "cp16", "known_noise")
PROFILES = ("fractional_sparse", "close_fractional")
FROZEN = ("cp_confirmation.py", "offgrid_baseline.py", "stage3_study.py",
          "confirmation_protocol.txt", "waveform_model.py", "pilot_ofdm.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_summary(fit):
    failures = [k for k, success in enumerate(fit["refinement_success"])
                if k and not success]
    return dict(selected_order=int(fit["selected_order"]), failed_orders=failures,
                selected_failure=fit["selected_order"] in failures,
                nfev=int(fit["nfev"]), nfev_history=fit["nfev_history"],
                selected_score=float(fit["selected_score"]),
                finite=bool(all(np.all(np.isfinite(fit[k])) for k in
                                ("response", "gains", "delays", "rss", "selected_score"))))


def one_seed(folder):
    folder = Path(folder)
    doc = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    for name, digest in doc["source_hashes"].items():
        if sha(name) != digest:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    with np.load(folder / "frame_metrics.npz") as data:
        original = {key: data[key].copy() for key in data.files}
    at300 = {k: v.copy() for k, v in original.items()}
    at1000 = {k: v.copy() for k, v in original.items()}
    lookup = {(c["profile"], c["snr_db"], c["source"]): c["calls"]
              for c in doc["diagnostics"]}
    rng = np.random.default_rng(doc["seed"])
    frames = doc["groups_per_cell"]
    records, cells = [], []
    optimizer = baseline.least_squares

    def budget_fit(args, budget):
        def override(*a, **kw):
            kw["max_nfev"] = budget
            return optimizer(*a, **kw)
        try:
            baseline.least_squares = override
            return baseline.estimate_offgrid(*args, max_delay=12., grid_step=.25,
                                             max_paths=6, refine=True)
        finally:
            baseline.least_squares = optimizer

    for profile in PROFILES:
        delays, power = stage3_profiles()[profile]
        for snr in (10, 20):
            variance = 10 ** (-snr / 10)
            truth, pilots, bits, received, cp = generate_cp(rng, frames, delays, power, variance)
            _, residual, _ = projected_statistics(pilots)
            variances = dict(residual16=residual.mean(axis=1), cp16=cp,
                             known_noise=np.full(frames, variance))
            for source in SOURCES:
                prefix = f"{profile}_{snr}_"
                key = prefix + "refined_" + source
                supplied = variances[source]
                if not np.array_equal(supplied, original[prefix + source + "_noise"]):
                    raise RuntimeError(f"Noise replay not exact: {doc['seed']}, {key}")
                calls = lookup[(profile, snr, source)]
                flagged = [i for i, call in enumerate(calls)
                           if not all(call["refinement_success"])]
                local = []
                for frame in flagged:
                    args = (pilots[frame, 0], FREQ[PILOTS], FREQ,
                            max(supplied[frame], 1e-12))
                    low = budget_fit(args, 60)
                    saved = calls[frame]
                    if (low["selected_order"] != saved["selected_order"] or
                            low["nfev"] != saved["nfev"] or
                            low["refinement_success"] != saved["refinement_success"] or
                            not np.allclose(low["score_history"], saved["score_history"],
                                            atol=1e-10, rtol=1e-10)):
                        raise RuntimeError(f"Original fit replay failed: {doc['seed']} {key} {frame}")
                    def score(fit):
                        return metrics(fit["response"][None, :], truth[frame:frame+1],
                                       bits[frame:frame+1], received[frame:frame+1])[:, 0]
                    low_metrics = score(low)
                    if not np.allclose(low_metrics, original[key][:, frame], atol=1e-12, rtol=1e-10):
                        raise RuntimeError(f"Original metric replay failed: {key} {frame}")
                    high = budget_fit(args, 300)
                    high_metrics = score(high)
                    at300[key][:, frame] = high_metrics
                    at300[key + "_orders"][frame] = high["selected_order"]
                    record = dict(seed=doc["seed"], profile=profile, snr_db=snr,
                                  source=source, frame=frame, original_reproduced=True,
                                  original=fit_summary(low), budget300=fit_summary(high),
                                  metrics_original=low_metrics.tolist(),
                                  metrics300=high_metrics.tolist())
                    if record["budget300"]["selected_failure"]:
                        highest = budget_fit(args, 1000)
                        final_metrics = score(highest)
                        record["budget1000"] = fit_summary(highest)
                        record["metrics1000"] = final_metrics.tolist()
                        at1000[key][:, frame] = final_metrics
                        at1000[key + "_orders"][frame] = highest["selected_order"]
                    else:
                        at1000[key][:, frame] = high_metrics
                        at1000[key + "_orders"][frame] = high["selected_order"]
                    records.append(record)
                    local.append(record)
                unflagged = np.ones(frames, dtype=bool)
                unflagged[flagged] = False
                for amended in (at300, at1000):
                    if not np.array_equal(amended[key][:, unflagged], original[key][:, unflagged]):
                        raise RuntimeError("Unflagged metrics changed")
                cells.append(dict(seed=doc["seed"], profile=profile, snr_db=snr,
                                  source=source, total_calls=frames,
                                  flagged_calls=len(flagged),
                                  original_selected_failures=sum(r["original"]["selected_failure"] for r in local),
                                  selected_failures300=sum(r["budget300"]["selected_failure"] for r in local),
                                  any_failures300=sum(bool(r["budget300"]["failed_orders"]) for r in local),
                                  calls1000=sum("budget1000" in r for r in local),
                                  selected_failures_final=sum(r.get("budget1000", r["budget300"])["selected_failure"] for r in local),
                                  any_failures_final=sum(bool(r.get("budget1000", r["budget300"])["failed_orders"]) for r in local)))
            print(f"Sensitivity seed {doc['seed']} {profile} {snr} dB complete", flush=True)
    if baseline.least_squares is not optimizer:
        raise RuntimeError("Optimizer override was not restored")
    selected_keys = [f"{p}_{s}_refined_{n}" for p in PROFILES for s in (10,20) for n in SOURCES]
    return dict(seed=doc["seed"], records=records, cells=cells,
                original={k: original[k] for k in selected_keys},
                budget300={k: at300[k] for k in selected_keys},
                budget1000selective={k: at1000[k] for k in selected_keys})


def run(workers=3):
    folders = [f"results/cp_confirmation{i}" for i in (1,2,3)]
    evidence_paths = [f"{folder}/{name}" for folder in folders
                      for name in ("results.json", "frame_metrics.npz")]
    hashes = {name: sha(name) for name in (*FROZEN, *evidence_paths)}
    if workers == 1:
        parts = [one_seed(folder) for folder in folders]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(one_seed, folders))
    records = [r for p in parts for r in p["records"]]
    cells = [r for p in parts for r in p["cells"]]
    summaries, contrasts = [], []
    bootstrap_rng = np.random.default_rng(1520261113)
    for profile in PROFILES:
        for snr in (10,20):
            n = sum(p["original"][f"{profile}_{snr}_refined_cp16"].shape[1] for p in parts)
            indices = bootstrap_rng.integers(0, n, size=(2000,n))
            for budget in ("original", "budget300", "budget1000selective"):
                arrays = {}
                for source in SOURCES:
                    key = f"{profile}_{snr}_refined_{source}"
                    array = np.concatenate([p[budget][key] for p in parts], axis=1)
                    arrays[source] = array
                    summaries.append(dict(profile=profile,snr_db=snr, budget=budget, source=source,
                                          groups=n, nmse_db=float(10*np.log10(array[0].sum()/array[1].sum())),
                                          ber=float(array[2].sum()/(96*n))))
                a, b = arrays["residual16"][0], arrays["cp16"][0]
                gains = 10*np.log10(a[indices].sum(axis=1)/b[indices].sum(axis=1))
                contrasts.append(dict(profile=profile,snr_db=snr,budget=budget,groups=n,
                                      cp_gain_db=float(10*np.log10(a.sum()/b.sum())),
                                      percentile95_ci_db=np.quantile(gains,[.025,.975]).tolist()))
    if {name: sha(name) for name in hashes} != hashes:
        raise RuntimeError("Frozen source or evidence changed during diagnostic")
    doc = dict(status="posthoc_failure_flagged_sensitivity_not_global_convergence",
               source_and_evidence_hashes=hashes, diagnostic_script_sha256=sha(__file__),
               frozen_sources_and_evidence_unchanged=True,
               original_budgets=60, first_diagnostic_budget=300, selected_failure_budget=1000,
               seeds=[p["seed"] for p in parts], all_noise_replays_bit_exact=True,
               all_flagged_original_fits_and_metrics_reproduced=True,
               all_unflagged_metrics_bit_exact=True,
               bootstrap=dict(replicates=2000,seed=1520261113,unit="paired groups pooled across seeds",
                              multiplicity_adjusted=False, role="diagnostic percentile interval"),
               scope="300 reruns only original calls with any failed candidate; 1000 reruns only selected failures remaining at300. Other calls are unchanged. Success does not establish global optimality; unflagged local minima and all original-success calls were not budget-stressed. No primary confirmatory result is overwritten.",
               cp_model_audit=dict(signal_cancellation_max_errors=cp_signal_audit(),
                   expected_noise_ratio=1., relative_variance=1/64,
                   formula="vhat=sum|n_cp-n_tail|^2/(2*64); 64 independent complex differences with variance2v",
                   clean_pairs="CP indices12..15 paired with useful60..63, maxdelay<=12, CP16",
                   dependence="CP statistic shares useful-tail receiver noise with pilot/data FFT observations; not independent of them",
                   assumptions="Ideal synchronized block-static point-path channel, white receiver noise, no extra front-end filter memory. Gains can differ across calibration blocks; all64 noise differences remain independent under this model."),
               overall=dict(total_calls=10800,flagged_calls=len(records),
                   original_selected_failures=sum(r["original"]["selected_failure"] for r in records),
                   any_failures300=sum(bool(r["budget300"]["failed_orders"]) for r in records),
                   selected_failures300=sum(r["budget300"]["selected_failure"] for r in records),
                   calls1000=sum("budget1000" in r for r in records),
                   selected_failures_final=sum(r.get("budget1000",r["budget300"])["selected_failure"] for r in records),
                   any_failures_final=sum(bool(r.get("budget1000",r["budget300"])["failed_orders"]) for r in records),
                   nonfinite_final=sum(not r.get("budget1000",r["budget300"])["finite"] for r in records)),
               results=summaries,contrasts=contrasts,cells=cells,records=records)
    Path("results/cp_optimizer_sensitivity.json").write_text(json.dumps(doc,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(doc["overall"],indent=2),flush=True)
    for row in contrasts:
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    run(parser.parse_args().workers)
