# =============================================================================
# Generate auxiliary data with several small Ollama models and compare them.
#
# Run from the project root (where auxiliary_generator.py lives):
#
#   powershell -ExecutionPolicy Bypass -File .\run_generation.ps1 -Smoke   # quick test (~30 instances/model)
#   powershell -ExecutionPolicy Bypass -File .\run_generation.ps1           # full run
#
# Edit $Models below to add/remove generators. Each model writes its own
# output file, so nothing overwrites another model's data.
#
# Re-running is safe: pass -Resume to continue an interrupted model instead
# of starting it over (per-model; the smoke run always starts fresh).
# =============================================================================
param(
    [switch]$Smoke,
    [switch]$Resume,
    [string]$Input = "data/processed/restaurant14_train_implicit.json",
    [int]$MaxEpochs = 10,
    [int]$Workers = 1
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

# ---- edit this list to change which generators you compare -----------------
# key = short label used in file names and the comparison table
# value = the exact Ollama model tag (must already be `ollama pull`-ed)
$Models = [ordered]@{
    "llama1b"  = "llama3.2:1b"
    "qwen3b"   = "qwen2.5:3b"
    "qwen7b"   = "qwen2.5:7b"
    "mistral7b" = "mistral"
}
# ------------------------------------------------------------------------------

$OutDir = "data/auxiliary"
New-Item -ItemType Directory -Force -Path $OutDir, "logs" | Out-Null

$MaxInstances = $null
if ($Smoke) {
    $MaxInstances = 30
    Write-Host "[SMOKE MODE] $MaxInstances instances per model, for a quick pipeline check"
}

Write-Host "Models to run: $($Models.Keys -join ', ')"
Write-Host "Start: $(Get-Date)"
$AllTimer = [System.Diagnostics.Stopwatch]::StartNew()

$FileArgs = @()

foreach ($name in $Models.Keys) {
    $tag = $Models[$name]
    $suffix = if ($Smoke) { "smoke" } else { "full" }
    $outJson = "$OutDir/restaurant14_train_implicit_aux_${name}_${suffix}.json"

    Write-Host ""
    Write-Host "=================== [generate] $name ($tag)   $(Get-Date) ==================="
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    $a = @("auxiliary_generator.py",
           "--input", $Input, "--output", $outJson,
           "--model", $tag, "--max-epochs", "$MaxEpochs", "--workers", "$Workers")
    if ($MaxInstances) { $a += @("--max-instances", "$MaxInstances") }
    if ($Resume -and -not $Smoke) { $a += "--resume" }

    & python @a 2>&1 | Tee-Object -FilePath "logs/gen_${name}_${suffix}.log"

    if (Test-Path $outJson) {
        Write-Host ("[done] $name in {0:N0}s -> $outJson" -f $timer.Elapsed.TotalSeconds)
        $FileArgs += "--file"
        $FileArgs += "$name=$outJson"
    } else {
        Write-Host "[FAILED] $name - see logs/gen_${name}_${suffix}.log"
    }
}

Write-Host ""
Write-Host "=================== COMPARISON ==================="
if ($FileArgs.Count -gt 0) {
    & python compare_aux.py @FileArgs | Tee-Object -FilePath "logs/comparison_${suffix}.txt"
} else {
    Write-Host "No models finished successfully - nothing to compare."
}

Write-Host ""
Write-Host ("Total time: {0:N0} min" -f $AllTimer.Elapsed.TotalMinutes)
Write-Host "Next: pick the generator(s) with the best neu% (neutral convergence), then train on that file with train_mt_isa.py"
