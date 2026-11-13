# =============================================================================
# Night run 2 (project root, OLD venv active). Each stage is seed-outer and writes its report.
#   a) laptop     aux0.3_all_gv_labelword  seeds 12-17 -> results\rate_report_laptop.txt
#   b) restaurant aux0.3_all_gv_nollm      seeds 12-17 -> log check + results\rate_report_h8.txt
#   c) laptop     aux0.3_all_gv_nollm      seeds 12-17 -> results\rate_report_laptop.txt
# Usage: powershell -ExecutionPolicy Bypass -File .\run_night2.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path "results" | Out-Null
$All = [System.Diagnostics.Stopwatch]::StartNew()
$Seeds = "12,13,14,15,16,17"

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=================== [stage] $Name   $(Get-Date) ==================="
    $t = [System.Diagnostics.Stopwatch]::StartNew()
    & $Body
    Write-Host ("[stage done] {0} in {1:N1} min" -f $Name, $t.Elapsed.TotalMinutes)
}

function Show-LogLine {
    param([string]$Log, [string]$Pattern)
    if (-not (Test-Path $Log)) { Write-Host "[check] $Log not found"; return }
    $hit = Select-String -Path $Log -Pattern $Pattern -SimpleMatch | Select-Object -First 1
    if ($hit) { Write-Host "[check] $($hit.Line)" } else { Write-Host "[check] no '$Pattern' line in $Log" }
}

Invoke-Stage "a) laptop aux0.3_all_gv_labelword" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Dataset laptop -Configs "aux0.3_all_gv_labelword" -SeedList $Seeds
    Show-LogLine "logs_laptop\aux0.3_all_gv_labelword_lr1e-4_s12.log" "opinion-target=label:"
    python analyze_rate.py --dataset laptop --out results\rate_report_laptop.txt
}

Invoke-Stage "b) restaurant aux0.3_all_gv_nollm" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Configs "aux0.3_all_gv_nollm" -SeedList $Seeds
    Show-LogLine "logs\aux0.3_all_gv_nollm_lr1e-4_s12.log" "aspect-target=target:"
    Show-LogLine "logs\aux0.3_all_gv_nollm_lr1e-4_s12.log" "opinion-target=label:"
    python analyze_rate.py --out results\rate_report_h8.txt
}

Invoke-Stage "c) laptop aux0.3_all_gv_nollm" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Dataset laptop -Configs "aux0.3_all_gv_nollm" -SeedList $Seeds
    Show-LogLine "logs_laptop\aux0.3_all_gv_nollm_lr1e-4_s12.log" "aspect-target=target:"
    python analyze_rate.py --dataset laptop --out results\rate_report_laptop.txt
}

Write-Host ""
Write-Host ("Total time: {0:N1} min" -f $All.Elapsed.TotalMinutes)
Write-Host "Reports: results\rate_report_laptop.txt, results\rate_report_h8.txt"
