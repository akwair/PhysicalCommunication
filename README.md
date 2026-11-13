# Residual-noise calibration in off-grid OFDM

Research code, simulated-result arrays and manuscript for **Residual-noise
calibration in off-grid OFDM: Leakage bias, temporal correlation, and
cyclic-prefix controls**.

Authors: Yuxiao Kuang (corresponding author) and Shengming Ni.

## Manuscript and submission files

- [Current manuscript PDF](output/submission/PhysicalCommunication.pdf)
- [Flat LaTeX source archive](output/submission/PhysicalCommunication_LaTeX.zip)
- [Cover letter](output/submission/Cover_letter.docx)
- [Highlights](output/submission/Highlights.docx)
- [Declaration of competing interests](output/submission/Declaration_of_competing_interests.docx)
- [Format and upload verification](manuscript/journal_check.txt)

The submission directory is the current delivery. Repository upload is not
journal submission. Journal-specific requirements were checked against the
official Guide for Authors PDF supplied on 2026-11-13. Both authors' photographs
are supplied separately in `output/submission/author_photos/`. Both author
biographies remain to be supplied. Repository access for editors and reviewers must be arranged:
the unauthenticated repository URL returned HTTP 404 during checking.

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
