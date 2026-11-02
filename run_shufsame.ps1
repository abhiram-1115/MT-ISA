# =============================================================================
# H6 control: aux0.3_all_gv with hints shuffled WITHIN each gold label, seeds 12-17,
# then the rate report.
# Usage (project root, OLD venv active): powershell -ExecutionPolicy Bypass -File .\run_shufsame.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$All = [System.Diagnostics.Stopwatch]::StartNew()

powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Configs "aux0.3_all_gv_shufsame" -SeedList "12,13,14,15,16,17"

$log = "logs\aux0.3_all_gv_shufsame_lr1e-4_s12.log"
if (Test-Path $log) {
    $hit = Select-String -Path $log -Pattern "shuffle-within-label:" -SimpleMatch | Select-Object -First 1
    if ($hit) { Write-Host "[check] $($hit.Line)" } else { Write-Host "[check] no 'shuffle-within-label:' line in $log" }
} else {
    Write-Host "[check] $log not found"
}

python analyze_rate.py --out results\rate_report_h6.txt

Write-Host ("Total time: {0:N1} min" -f $All.Elapsed.TotalMinutes)
Write-Host "Report: results\rate_report_h6.txt"
