"""Separate sampled convergence audit; never substitutes for main-run logs.

Uses 20 independent groups per profile/SNR, a distinct fixed random seed, and
the unchanged stage-3 generator and estimator defaults. No timing conclusions
are drawn because the main study may run concurrently.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from offgrid_baseline import estimate_offgrid
from stage3_study import (FREQ, PILOTS, generate, pool_noise,
                          projected_statistics, stage3_profiles)


SEED = 820261113
FRAMES = 20
SOURCE_NAMES = ["stage3_study.py", "offgrid_baseline.py", "noise_study.py",
                "pilot_ofdm.py", "stage3_optimizer_audit.py"]


def hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in SOURCE_NAMES}


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)),
        "maximum": float(values.max()),
    }


def summarize(records):
    return {
        "calls": len(records),
        "visited_refined_candidates": sum(r["visited_candidates"] for r in records),
        "failed_visited_candidates": sum(len(r["failed_orders"]) for r in records),
        "calls_with_any_refinement_failure": sum(bool(r["failed_orders"]) for r in records),
        "calls_with_selected_refinement_failure": sum(r["selected_failed"] for r in records),
        "calls_with_nonfinite_output": sum(not r["finite_output"] for r in records),
        "total_nfev_distribution": distribution([r["nfev"] for r in records]),
        "selected_order_distribution": distribution([r["selected_order"] for r in records]),
    }


def run():
    starting_hashes = hashes()
    rng = np.random.default_rng(SEED)
    records, cells = [], []
    for profile, (delays, power) in stage3_profiles().items():
        for snr in (0, 10, 20):
            variance = 10 ** (-snr / 10)
            _, pilots, _, _, _ = generate(rng, FRAMES, delays, power, variance)
            _, residual_variances, _ = projected_statistics(pilots)
            sources = {
                "residual_16": pool_noise(residual_variances, 16),
                "true_noise_ref": np.full(FRAMES, variance),
            }
            for source, supplied_variances in sources.items():
                cell_records = []
                for frame in range(FRAMES):
                    fit = estimate_offgrid(
                        pilots[frame, 0], FREQ[PILOTS], FREQ,
                        max(supplied_variances[frame], 1e-12),
                        max_delay=12., grid_step=.25, max_paths=6, refine=True,
                    )
                    failed_orders = [k for k, ok in enumerate(fit["refinement_success"])
                                     if k > 0 and not ok]
                    finite = all(np.all(np.isfinite(fit[key])) for key in
                                 ("response", "delays", "gains", "rss", "selected_score"))
                    record = {
                        "profile": profile, "snr_db": snr, "source": source,
                        "frame": frame, "selected_order": fit["selected_order"],
                        "visited_candidates": len(fit["refinement_success"]) - 1,
                        "failed_orders": failed_orders,
                        "selected_failed": fit["selected_order"] in failed_orders,
                        "finite_output": bool(finite), "nfev": fit["nfev"],
                        "nfev_history": fit["nfev_history"],
                        "selected_score": float(fit["selected_score"]),
                        "score_history": [float(value) for value in fit["score_history"]],
                        "stopping_reason": fit["stopping_reason"],
                        "supplied_noise_ratio": float(supplied_variances[frame] / variance),
                    }
                    records.append(record)
                    cell_records.append(record)
                cells.append(dict(profile=profile, snr_db=snr, source=source,
                                  **summarize(cell_records)))
            print(f"Audited {profile} {snr} dB", flush=True)
    ending_hashes = hashes()
    if starting_hashes != ending_hashes:
        raise RuntimeError("Source files changed during the optimizer audit")
    document = {
        "status": "separate_sampled_optimizer_diagnostic_not_main_run_failure_frequency",
        "seed": SEED, "frames_per_profile_snr": FRAMES,
        "noise_sources": ["residual_16", "true_noise_ref"],
        "source_hashes": starting_hashes, "source_unchanged_during_audit": True,
        "estimator_configuration": dict(max_delay=12., grid_step=.25,
                                        max_paths=6, refine=True),
        "interpretation": {
            "failure_means": "SciPy least_squares success=False for a visited candidate",
            "selected_failure_means": "BIC-selected order had success=False",
            "finite_does_not_prove_global_optimality": True,
            "does_not_recover_main_run_convergence_logs": True,
            "wall_time_not_interpreted": True,
        },
        "overall": summarize(records), "cells": cells,
        "failures": [r for r in records if r["failed_orders"] or not r["finite_output"]],
        "records": records,
    }
    output = Path("results/stage3_optimizer_audit.json")
    output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(document["overall"], indent=2), flush=True)


def convergence_sensitivity():
    """Reconstruct only selected-failure cases; preserve the original audit."""
    import offgrid_baseline as baseline

    output = Path("results/stage3_optimizer_audit.json")
    document = json.loads(output.read_text(encoding="utf-8"))
    starting_hashes = hashes()
    for name, original_hash in document["source_hashes"].items():
        if name != "stage3_optimizer_audit.py" and starting_hashes[name] != original_hash:
            raise RuntimeError(f"Original estimator/generator source changed: {name}")
    affected = {(r["profile"], r["snr_db"], r["source"], r["frame"]): r
                for r in document["records"] if r["selected_failed"]}
    rng = np.random.default_rng(document["seed"])
    records, cells = [], []
    original_optimizer = baseline.least_squares

    def larger_budget(*args, **kwargs):
        kwargs["max_nfev"] = 300
        return original_optimizer(*args, **kwargs)

    for profile, (delays, power) in stage3_profiles().items():
        for snr in (0, 10, 20):
            variance = 10 ** (-snr / 10)
            truth, pilots, _, _, _ = generate(
                rng, document["frames_per_profile_snr"], delays, power, variance)
            _, residual_variances, _ = projected_statistics(pilots)
            sources = {
                "residual_16": pool_noise(residual_variances, 16),
                "true_noise_ref": np.full(len(truth), variance),
            }
            energies = np.mean(abs(truth) ** 2, axis=1)
            for source, supplied_variances in sources.items():
                cell_records = []
                for frame in range(len(truth)):
                    key = (profile, snr, source, frame)
                    if key not in affected:
                        continue
                    saved = affected[key]
                    arguments = (pilots[frame, 0], FREQ[PILOTS], FREQ,
                                 max(supplied_variances[frame], 1e-12))
                    # Original responses were not saved. Reconstruct only the
                    # affected baseline calls and verify against saved evidence.
                    original = baseline.estimate_offgrid(*arguments)
                    if (original["selected_order"] != saved["selected_order"]
                            or original["nfev"] != saved["nfev"]
                            or not np.allclose(original["score_history"],
                                               saved["score_history"], rtol=1e-10, atol=1e-10)):
                        raise RuntimeError(f"Original diagnostic call did not reproduce: {key}")
                    try:
                        baseline.least_squares = larger_budget
                        higher = baseline.estimate_offgrid(*arguments)
                    finally:
                        baseline.least_squares = original_optimizer
                    original_mse = float(np.mean(abs(original["response"] - truth[frame]) ** 2))
                    higher_mse = float(np.mean(abs(higher["response"] - truth[frame]) ** 2))
                    record = {
                        "profile": profile, "snr_db": snr, "source": source, "frame": frame,
                        "original_record_reproduced": True,
                        "original_selected_order": original["selected_order"],
                        "higher_selected_order": higher["selected_order"],
                        "selected_order_changed": original["selected_order"] != higher["selected_order"],
                        "higher_selected_converged": bool(higher["refinement_success"][higher["selected_order"]]),
                        "higher_failed_orders": [k for k, ok in enumerate(higher["refinement_success"])
                                                 if k > 0 and not ok],
                        "original_nfev": original["nfev"], "higher_nfev": higher["nfev"],
                        "higher_nfev_history": higher["nfev_history"],
                        "original_selected_score": float(original["selected_score"]),
                        "higher_selected_score": float(higher["selected_score"]),
                        "original_mse": original_mse, "higher_mse": higher_mse,
                        "truth_energy": float(energies[frame]),
                        "original_nmse_db": float(10 * np.log10(original_mse / energies[frame])),
                        "higher_nmse_db": float(10 * np.log10(higher_mse / energies[frame])),
                        "nmse_gain_db": float(10 * np.log10(original_mse / higher_mse)),
                        "response_difference_nmse": float(np.mean(abs(higher["response"] - original["response"]) ** 2)
                                                          / energies[frame]),
                        "higher_finite_output": bool(all(np.all(np.isfinite(higher[k])) for k in
                                                          ("response", "gains", "delays", "rss", "selected_score"))),
                    }
                    records.append(record)
                    cell_records.append(record)
                if cell_records:
                    original_sum = sum(r["original_mse"] for r in cell_records)
                    higher_sum = sum(r["higher_mse"] for r in cell_records)
                    cells.append({
                        "profile": profile, "snr_db": snr, "source": source,
                        "affected_calls": len(cell_records),
                        "selected_converged_at_300": sum(r["higher_selected_converged"] for r in cell_records),
                        "selected_order_changes": sum(r["selected_order_changed"] for r in cell_records),
                        "affected_subset_pooled_nmse_gain_db": float(10 * np.log10(original_sum / higher_sum)),
                        "whole_audit_cell_linear_nmse_change": float((higher_sum - original_sum) / energies.sum()),
                        "whole_audit_cell_frames": len(truth),
                    })
    if len(records) != len(affected) or baseline.least_squares is not original_optimizer:
        raise RuntimeError("Sensitivity audit incomplete or optimizer not restored")
    ending_hashes = hashes()
    if ending_hashes != starting_hashes:
        raise RuntimeError("Source files changed during sensitivity audit")
    overall = {
        "affected_calls": len(records),
        "original_calls_reproduced": sum(r["original_record_reproduced"] for r in records),
        "selected_converged_at_300": sum(r["higher_selected_converged"] for r in records),
        "selected_order_changes": sum(r["selected_order_changed"] for r in records),
        "finite_higher_outputs": sum(r["higher_finite_output"] for r in records),
        "calls_with_improved_response_mse": sum(r["higher_mse"] < r["original_mse"] for r in records),
        "per_call_nmse_gain_db": distribution([r["nmse_gain_db"] for r in records]),
        "higher_total_nfev_distribution": distribution([r["higher_nfev"] for r in records]),
    }
    document["convergence_sensitivity"] = {
        "status": "selected_failure_subset_only_not_main_run_rankings",
        "source_hashes": starting_hashes, "source_unchanged_during_audit": True,
        "original_max_nfev_per_candidate": 60, "higher_max_nfev_per_candidate": 300,
        "original_results_preserved": True,
        "interpretation": "Positive nmse_gain_db favors the 300-evaluation budget; effects are conditional on original selected failure. Whole-cell change replaces only these affected audit calls, leaving other calls unchanged. This cannot establish main-sweep rankings or full convergence at 300.",
        "overall": overall, "cells": cells, "records": records,
    }
    output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)
    for cell in cells:
        print(json.dumps(cell), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    if args.sensitivity:
        convergence_sensitivity()
    else:
        run()
