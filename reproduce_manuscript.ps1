param([ValidateSet('Verify','Figures','Experiments')][string]$Mode='Verify')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
function Run-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $Arguments" }
}
if ($Mode -eq 'Verify') {
    Run-Python -m pytest test_noise_study.py test_stage3_study.py test_offgrid_baseline.py test_waveform_model.py test_cp_confirmation.py test_valse_python.py test_operating_region_study.py test_valse_budget_sensitivity.py -q
    exit 0
}
if ($Mode -eq 'Experiments') {
    for ($i=0; $i -lt 3; $i++) {
        $rep=$i+1
        Run-Python stage3_study.py --frames 5000 --seed (520261113+$i) --out "results/stage3_pool$rep"
        Run-Python stage3_study.py --frames 300 --seed (820261113+$i) --offgrid --out "results/stage3_offgrid$rep"
        Run-Python cp_confirmation.py --frames 300 --seed (1120261113+$i) --out "results/cp_confirmation$rep"
        Run-Python valse_pilot_check.py run --frames 150 --seed (1220261113+$i) --out "results/valse_check$rep"
    }
    Run-Python operating_region_study.py
    Run-Python valse_budget_sensitivity.py
}
Run-Python analyze_stage3.py --mode pooling
Run-Python analyze_stage3.py --mode offgrid
Run-Python analyze_confirmation.py
Run-Python valse_pilot_check.py aggregate results/valse_check1 results/valse_check2 results/valse_check3
Run-Python make_manuscript_figures.py
Run-Python build_manuscript_assets.py
Run-Python make_operating_region_figure.py
