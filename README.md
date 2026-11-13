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
- [Author biographies](output/submission/Author_biographies.docx)
- [Declaration of competing interests](output/submission/Declaration_of_competing_interests.docx)
- [Format and upload verification](manuscript/journal_check.txt)

The submission directory is the current delivery. Repository upload is not
journal submission. Journal-specific requirements were checked against the
official Guide for Authors PDF supplied on 2026-11-13. Both authors' photographs
are supplied separately in `output/submission/author_photos/`. Both author
biographies are supplied in `output/submission/Author_biographies.docx`. Repository access for editors and reviewers must be arranged:
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

## Standard power-delay-profile validation

The additional EPA/EVA campaign preserves the delays and relative powers in
3GPP TS 36.101 and uses the paper's block-static OFDM model. It contains 5400
paired groups across two profiles and SNRs 0, 10 and 20 dB. At 20 dB, changing
residual to CP calibration improves the same receiver's NMSE by 5.75 dB (EPA)
and 6.12 dB (EVA). Full six-cell results, including the low-SNR outcomes, are
in [the summary](results/standard_channel_summary/summary.json).

See [the protocol](standard_channel_protocol.txt),
[parameter provenance](references/standard_channels/SOURCE.json), and
[reproduction guide](README_REPRODUCE.txt). The post-run optimizer-budget audit
retains the primary results and changes all six NMSE gains by less than
0.0015 dB. This validation is simulation with standard PDPs, not measured data
or a full LTE conformance test.
