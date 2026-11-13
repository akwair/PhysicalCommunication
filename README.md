# Residual-noise calibration in off-grid OFDM

Python code and simulation results for OFDM noise calibration and channel
estimation. The experiments compare pilot-residual and cyclic-prefix noise
estimates using controlled channel profiles and standard EPA/EVA power-delay
profiles.

## Setup

Python 3.13.9 was used for the recorded experiments.

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

## Results

| Experiment | Saved results |
| --- | --- |
| Residual averaging | `results/stage3_pool_summary/` |
| Channel reconstruction | `results/stage3_offgrid_summary/` |
| Paired CP calibration | `results/cp_confirmation_summary/` |
| Bias and variance across channel geometries | `results/operating_region/` |
| EPA/EVA validation | `results/standard_channel_summary/` |
| Pilot-domain VALSE comparison | `results/valse_check_summary/` |

NPZ files contain per-group metrics and noise estimates. JSON and CSV files
contain aggregate results. Observations can be regenerated from the recorded
seeds. The EPA/EVA experiments use standard power-delay profiles in a
block-static OFDM model.

## Reproduce

Run commands from the repository root. On PowerShell:

```powershell
./reproduce.ps1 -Mode Verify
./reproduce.ps1 -Mode Figures
./reproduce.ps1 -Mode StandardChannels
```

Figures are written to `figures/`. The standard-channel rerun writes to
`results/standard_channel_reproduction/`, leaving the archived results intact.
See [README_REPRODUCE.txt](README_REPRODUCE.txt) for seeds, data definitions,
and individual commands.

## Sources

- [EPA/EVA parameters](references/standard_channels/SOURCE.json)
- [VALSE source provenance](references/valse_original/PROVENANCE.json)
- [VALSE implementation notes](valse_implementation_notes.txt)
