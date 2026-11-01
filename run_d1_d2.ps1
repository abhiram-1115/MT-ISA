# =============================================================================
# D1 / D2 diagnostics, run back to back (old venv / Flan-T5).
#   a. prepare_mams.py                         (MAMS-ATSA -> data/external/mams_{train,val}.json)
#   b. ce_all_gv, ce_all_gv_imp50/25/0          seeds 6-11 (finished ce_all_gv runs are skipped)
#   c. mams_only_ce                             seeds 6-8
#   d. compare_runs.py vs ce_all_gv          -> results/compare_d1_d2.txt
#   e. analyze_d1_d2.py                      -> results/d1_d2_report.txt
# Usage (project root, old venv active):
#   powershell -ExecutionPolicy Bypass -File .\run_d1_d2.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
$All = [System.Diagnostics.Stopwatch]::StartNew()

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=================== [stage] $Name   $(Get-Date) ==================="
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    & $Body
    Write-Host ("[stage done] {0} in {1:N1} min" -f $Name, $t.Elapsed.TotalMinutes)
}

Invoke-Stage "a. prepare_mams.py" {
    python prepare_mams.py
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "data/external/mams_train.json") -or -not (Test-Path "data/external/mams_val.json")) {
        Write-Host "ERROR: prepare_mams.py failed (exit code $LASTEXITCODE) or did not write data/external/mams_train.json / mams_val.json. Stopping."
        exit 1
    }
}

# run_experiments.ps1 runs in its own process: its preflight 'exit' must not end this script
Invoke-Stage "b. D1 configs (seeds 6-11)" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Configs "ce_all_gv,ce_all_gv_imp50,ce_all_gv_imp25,ce_all_gv_imp0" -SeedList "6,7,8,9,10,11"
}

Invoke-Stage "c. mams_only_ce (seeds 6-8)" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Configs "mams_only_ce" -SeedList "6,7,8"
}

Invoke-Stage "d. compare_runs.py" {
    python compare_runs.py models --baseline ce_all_gv --lr 1e-4 | Out-File -Encoding utf8 results\compare_d1_d2.txt
    Get-Content results\compare_d1_d2.txt
}

Invoke-Stage "e. analyze_d1_d2.py" {
    python analyze_d1_d2.py
}

Write-Host ""
Write-Host ("Total time: {0:N1} min" -f $All.Elapsed.TotalMinutes)
Write-Host "Reports: results\d1_d2_report.txt and results\compare_d1_d2.txt"
