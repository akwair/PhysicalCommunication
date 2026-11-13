$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest test_noise_study.py test_stage3_study.py test_offgrid_baseline.py test_waveform_model.py -q
if ($LASTEXITCODE -ne 0) { throw 'Scientific checks failed.' }
python pooling_correlation_audit.py
if ($LASTEXITCODE -ne 0) { throw 'Pooling correlation audit failed.' }
foreach ($runIndex in 1..3) {
    $runSeed = 520261112 + $runIndex
    python stage3_study.py --frames 5000 --seed $runSeed --out "results/stage3_pool$runIndex"
    if ($LASTEXITCODE -ne 0) { throw "Pooling run $runIndex failed." }
}
python analyze_stage3.py --mode pooling
if ($LASTEXITCODE -ne 0) { throw 'Pooling analysis failed.' }
foreach ($runIndex in 1..3) {
    $runSeed = 820261112 + $runIndex
    python stage3_study.py --frames 300 --seed $runSeed --offgrid --out "results/stage3_offgrid$runIndex"
    if ($LASTEXITCODE -ne 0) { throw "Off-grid run $runIndex failed." }
}
python analyze_stage3.py --mode offgrid
if ($LASTEXITCODE -ne 0) { throw 'Off-grid analysis failed.' }
