# =============================================================================
# Rate-controlled confirmation run, back to back (project root, OLD venv active).
#   a. T5 configs polonly_all_gv, ce_all_gv, aux0.3_all_gv, aux0.3_all_gv_nullneu,
#      mtisa_in_all_gv on seeds 12-17 (run_experiments.ps1, own process)
#   b. LoRA probabilities: .\venv_lora\Scripts\python.exe rescore_lora_probs.py
#   c. python analyze_rate.py  -> results\rate_report.txt
# Usage: powershell -ExecutionPolicy Bypass -File .\run_rate.ps1
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

Invoke-Stage "a. T5 configs, seeds 12-17" {
    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Configs "polonly_all_gv,ce_all_gv,aux0.3_all_gv,aux0.3_all_gv_nullneu,mtisa_in_all_gv" -SeedList "12,13,14,15,16,17"
    # informational: the first nullneu run's '--null-neutral-opinion changed N rows' line
    $nl = Get-ChildItem "logs" -Filter "aux0.3_all_gv_nullneu_lr1e-4_s*.log" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -First 1
    if ($nl) {
        $hit = Select-String -Path $nl.FullName -Pattern "--null-neutral-opinion changed" -SimpleMatch | Select-Object -First 1
        if ($hit) { Write-Host "[check] $($nl.Name): $($hit.Line)" } else { Write-Host "[check] $($nl.Name): no '--null-neutral-opinion changed' line found" }
    } else {
        Write-Host "[check] no aux0.3_all_gv_nullneu log found"
    }
}

Invoke-Stage "b. LoRA probabilities (venv_lora)" {
    if (Test-Path ".\venv_lora\Scripts\python.exe") {
        & .\venv_lora\Scripts\python.exe rescore_lora_probs.py
        if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: rescore_lora_probs.py exited with code $LASTEXITCODE (see output above)" }
    } else {
        Write-Host "WARNING: .\venv_lora\Scripts\python.exe not found - LoRA configs will be n/a in the report"
    }
}

Invoke-Stage "c. analyze_rate.py" {
    python analyze_rate.py
}

Write-Host ""
Write-Host ("Total time: {0:N1} min" -f $All.Elapsed.TotalMinutes)
Write-Host "Report: results\rate_report.txt"
