# =============================================================================
# Zip the small result files (*.json: test_predictions, val_predictions, test_metrics, history, ...)
# of every run folder exp_*_s12 .. exp_*_s17 (folder names kept):
#   models\        -> results\bundle_restaurant.zip
#   models_laptop\ -> results\bundle_laptop.zip
# Usage: powershell -ExecutionPolicy Bypass -File .\bundle_results.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "results" | Out-Null

function New-Bundle {
    param([string]$ModelsDir, [string]$Zip)
    if (-not (Test-Path $ModelsDir)) { Write-Host "$ModelsDir not found - skipped"; return }
    # -Filter cannot express [2-7]; match the folder name with a regex instead
    $runs = @(Get-ChildItem $ModelsDir -Directory | Where-Object { $_.Name -like "exp_*" -and $_.Name -match "_s1[2-7]$" })
    if ($runs.Count -eq 0) { Write-Host "no exp_*_s12..s17 folders in $ModelsDir - skipped"; return }
    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ("bundle_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    $nFiles = 0
    foreach ($r in $runs) {
        $files = @(Get-ChildItem $r.FullName -File -Filter "*.json")
        if ($files.Count -eq 0) { continue }
        $dest = Join-Path $stage $r.Name
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        foreach ($f in $files) { Copy-Item $f.FullName $dest; $nFiles++ }
    }
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $Zip -Force
    Remove-Item $stage -Recurse -Force
    $mb = (Get-Item $Zip).Length / 1MB
    Write-Host ("{0}: {1} run folders, {2} json files -> {3} ({4:N2} MB)" -f $ModelsDir, $runs.Count, $nFiles, $Zip, $mb)
}

New-Bundle "models" "results\bundle_restaurant.zip"
New-Bundle "models_laptop" "results\bundle_laptop.zip"
