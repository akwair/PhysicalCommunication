$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest test_noise_study.py -q
if ($LASTEXITCODE -ne 0) { throw 'Scientific tests failed.' }
python audit_numerics.py
if ($LASTEXITCODE -ne 0) { throw 'Numerical audit failed.' }
foreach ($runIndex in 1..3) {
    $runSeed = 420261112 + $runIndex
    python noise_study.py --frames 5000 --seed $runSeed --out "results/noise_rep$runIndex"
    if ($LASTEXITCODE -ne 0) { throw "Experiment $runIndex failed." }
}
python analyze_noise_study.py
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed.' }
