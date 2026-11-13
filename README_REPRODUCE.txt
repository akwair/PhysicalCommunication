Reproducing the OFDM calibration experiments

Environment
Python 3.13.9, with package versions in requirements.txt. CPU execution.
Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before
running Python. The PowerShell entry point sets these automatically.

Verify
python -m pytest -q
python verify_standard_channel_results.py

Data
Observations are simulated. NPZ files store per-group errors, true channel
energy, bit-error counts, noise estimates and selected diagnostics. Complete
received waveforms are regenerated from seeds. Inputs needed for optimizer
replays are retained separately.

NMSE is the ratio of summed error energy to summed true channel energy.
BER counts 96 bits per target block. Positive CP-versus-residual NMSE gain
favors CP calibration. Bootstrap resampling uses paired groups; the reported
95% intervals are pointwise. VALSE comparisons use pilot-domain errors.

Campaigns
- stage3_pool1..3: 5000 groups per seed/cell; five profiles, three SNRs.
  Seeds: 520261113, 520261114, 520261115.
- stage3_offgrid1..3: 300 groups per seed/cell; the same 15 cells.
  Seeds: 820261113, 820261114, 820261115.
- cp_confirmation1..3: 300 groups per seed/cell; two fractional-delay
  profiles at 10 and 20 dB. Seeds: 1120261113, 1120261114, 1120261115.
- valse_check1..3: 150 groups per seed/cell; six pilot-domain cells.
  Seeds: 1220261113, 1220261114, 1220261115.
- operating_region: 160 analytical cells and 16 Monte Carlo cells with
  20000 groups each. SeedSequence root: 1320261113; nested T=1,4,16.
- standard_channel1..3: 300 groups per seed/cell; EPA/EVA at 0,10,20 dB.
  Seeds: 1520261113, 1520261114, 1520261115; 5400 groups in total.

Rebuild summaries
python analyze_stage3.py --mode pooling
python analyze_stage3.py --mode offgrid
python analyze_confirmation.py
python valse_pilot_check.py aggregate results/valse_check1 results/valse_check2 results/valse_check3
python analyze_standard_channels.py

Plot saved results
python plot_results.py
python make_operating_region_figure.py
Outputs: figures/*.pdf, *.svg, *.png, and plotted-value JSON files.

Rerun EPA/EVA
./reproduce.ps1 -Mode StandardChannels
Outputs go to results/standard_channel_reproduction/seed1..3 and summary/.
Pass -StandardOutputRoot with a new path for another run. Completed runs
are not overwritten by the standard-channel generator.

The EPA/EVA delays and relative powers come from 3GPP TS 36.101. All physical
paths are retained at a 250 ns sampling interval. Gains follow the same
block-static Gaussian model as the controlled experiments. Receiver settings
remain fixed, including the six-component continuous-delay fitting cap.
Parameter provenance: references/standard_channels/SOURCE.json.
Design: standard_channel_protocol.txt.

Optimization checks
results/cp_optimizer_sensitivity.json contains the CP-control budget study.
results/standard_channel_budget/ contains the EPA/EVA replay at budget 300.
results/valse_budget_sensitivity.json contains the VALSE budget replay.
The scripts retain the original metrics and identify the calls rerun.
For a new complete campaign, use a separate copy with fresh result folders;
some legacy experiment scripts overwrite their named output files.
./reproduce.ps1 -Mode Experiments runs the campaigns listed in that script.

Source integrity
Several summaries verify the hashes of the experiment code used to generate
their inputs. Keep those files unchanged when checking the archived runs.
The operating-region design is retained at its recorded path:
revisions/adversarial_20261113/operating_region_protocol.txt.

Model and implementation notes
The waveform model is synchronized, unfiltered, block-static CP-OFDM.
CP and FFT noise retain their physical dependence. EPA/EVA validation uses
standard power-delay profiles, without standardized Doppler trajectories.
See waveform_audit.txt for the finite-burst check.

valse_python.py implements the cited VALSE equations and Heuristic 2.
See valse_implementation_notes.txt for conventions and numerical settings.
The original MATLAB demo is not redistributed; its source and hashes are
recorded in references/valse_original/PROVENANCE.json.
