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
    [string]$TrainFile = "data/processed/restaurant14_train_implicit.json",
    [int]$MaxEpochs = 10,
    [int]$Workers = 1,
    [string]$Models = "",   # comma-separated subset/order, e.g. "qwen7b,llama1b" - default runs all, in priority order
    [int]$SmokeInstances = 8   # instances per model in -Smoke mode; keep this small, it's just a wiring check
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

# ---- edit this list to change which generators you compare -----------------
# key = short label used in file names and the comparison table
# value = the exact Ollama model tag (must already be `ollama pull`-ed)
$AllModels = [ordered]@{
    "qwen7b"    = "qwen2.5:7b"    # best in the smoke test: highest conv%, best on neutral, fewest iters
    "llama1b"   = "llama3.2:1b"   # extreme-small contrast point
    "qwen3b"    = "qwen2.5:3b"
    "mistral7b" = "mistral"
}
# ------------------------------------------------------------------------------

if ($Models) {
    $ModelsToRun = [ordered]@{}
    foreach ($key in ($Models -split ",").Trim()) {
        if ($AllModels.Contains($key)) {
            $ModelsToRun[$key] = $AllModels[$key]
        } else {
            Write-Host "WARNING: unknown model key '$key' (known: $($AllModels.Keys -join ', ')) - skipping"
        }
    }
    if ($ModelsToRun.Count -eq 0) { Write-Host "ERROR: no valid models in -Models '$Models'"; exit 1 }
} else {
    $ModelsToRun = $AllModels
}

$OutDir = "data/auxiliary"
New-Item -ItemType Directory -Force -Path $OutDir, "logs" | Out-Null

$MaxInstances = $null
if ($Smoke) {
    $MaxInstances = $SmokeInstances
    Write-Host "[SMOKE MODE] $MaxInstances instances per model, for a quick pipeline check"
}

Write-Host "Models to run: $($ModelsToRun.Keys -join ', ')"
Write-Host "Start: $(Get-Date)"
$AllTimer = [System.Diagnostics.Stopwatch]::StartNew()

$FileArgs = @()

foreach ($name in $ModelsToRun.Keys) {
    $tag = $ModelsToRun[$name]
    $suffix = if ($Smoke) { "smoke" } else { "full" }
    $outJson = "$OutDir/restaurant14_train_implicit_aux_${name}_${suffix}.json"

    Write-Host ""
    Write-Host "=================== [generate] $name ($tag)   $(Get-Date) ==================="
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    $a = @("auxiliary_generator.py",
           "--input", $TrainFile, "--output", $outJson,
           "--model", $tag, "--max-epochs", "$MaxEpochs", "--workers", "$Workers")
    if ($MaxInstances) { $a += @("--max-instances", "$MaxInstances") }
    if (-not $Smoke) { $a += "--resume" }   # always resumable for full runs: safe no-op if starting fresh

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
