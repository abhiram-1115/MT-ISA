# =============================================================================
# Night run (project root, OLD venv active):
#   a) restaurant: aux0.3_all_gv_labelword seeds 12-17, H7 report
#   b) laptop: polonly_all_gv, aux0.3_all_gv, aux0.3_all_gv_shuf, aux0.3_all_gv_shufsame seeds 12-17
#      (run_experiments.ps1 loops seed-outer: every config gets seed 12 before any gets 13, so an
#       early stop leaves all configs with the same finished seeds)
#   c) laptop rate report (only if b finished)
# Usage: powershell -ExecutionPolicy Bypass -File .\run_night.ps1
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

Invoke-Stage "a) restaurant aux0.3_all_gv_labelword (seeds 12-17) + H7 report" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Configs "aux0.3_all_gv_labelword" -SeedList "12,13,14,15,16,17"
    $log = "logs\aux0.3_all_gv_labelword_lr1e-4_s12.log"
    if (Test-Path $log) {
        $hit = Select-String -Path $log -Pattern "opinion-target=label:" -SimpleMatch | Select-Object -First 1
        if ($hit) { Write-Host "[check] $($hit.Line)" } else { Write-Host "[check] no 'opinion-target=label:' line in $log" }
    } else {
        Write-Host "[check] $log not found"
    }
    python analyze_rate.py --out results\rate_report_h7.txt
}

$LapAux = "data\auxiliary\laptop14_train_all_aux_qwen7b_full.json"
if (-not (Test-Path $LapAux)) {
    Write-Host ""
    Write-Host "STOP: $LapAux is missing - the laptop runs (stage b) and the laptop report (stage c) are skipped."
    Write-Host "      Generate it first (auxiliary_generator.py on data/processed/laptop14_train_all_gen.json) and re-run this script; stage a will be skipped as done."
    Write-Host ("Total time: {0:N1} min" -f $All.Elapsed.TotalMinutes)
    exit 1
}

$LapDone = $false
Invoke-Stage "b) laptop configs (seeds 12-17)" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Dataset laptop -Configs "polonly_all_gv,aux0.3_all_gv,aux0.3_all_gv_shuf,aux0.3_all_gv_shufsame" -SeedList "12,13,14,15,16,17"
    $script:LapDone = ($LASTEXITCODE -eq 0)
    if (-not $script:LapDone) { Write-Host "laptop run_experiments.ps1 exited with code $LASTEXITCODE" }
}

if ($LapDone) {
    Invoke-Stage "c) laptop rate report" {
        python analyze_rate.py --dataset laptop --out results\rate_report_laptop.txt
    }
} else {
    Write-Host "stage c skipped (stage b did not finish cleanly)"
}

Write-Host ""
Write-Host ("Total time: {0:N1} min" -f $All.Elapsed.TotalMinutes)
Write-Host "Reports: results\rate_report_h7.txt, results\rate_report_laptop.txt"
