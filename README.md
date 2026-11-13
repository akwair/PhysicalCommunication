# Residual-noise calibration in off-grid OFDM

Research code, simulated-result arrays and manuscript for **Residual-noise
calibration in off-grid OFDM: Leakage bias, temporal correlation, and
cyclic-prefix controls**.

## Manuscript and submission files

- [Current manuscript PDF](output/submission/PhysicalCommunication.pdf)
- [Flat LaTeX source archive](output/submission/PhysicalCommunication_LaTeX.zip)
- [Cover letter](output/submission/Cover_letter.txt)
- [Highlights](output/submission/Highlights.txt)
- [Format and upload verification](manuscript/journal_check.txt)

The submission directory is the current delivery. Repository upload is not
journal submission. Publisher-wide requirements have been checked; the
current journal-specific guide was inaccessible on 2026-11-13, as documented
in the verification record.

## Reproduction

See [README_REPRODUCE.txt](README_REPRODUCE.txt) for dependencies, saved result
definitions, random seeds, experimental limitations, and build commands.
Install the versions in `requirements.txt` and run:

```powershell
./reproduce_manuscript.ps1 -Mode Verify
```

`results/` contains simulated data and numerical summaries. `manuscript/`
contains editable LaTeX, references and figures. The operating-region
experiment protocol is retained in `revisions/adversarial_20261113/`.

The original third-party VALSE MATLAB demo is excluded. Its provenance is
recorded in `references/valse_original/PROVENANCE.json`. Temporary files,
compilation logs, old manuscript copies, internal review history, author
administrative records and duplicate delivery archives are excluded from Git.
