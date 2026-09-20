# =============================================================================
# MT-ISA experiment runner (PowerShell version of run_experiments.sh)
#
# Runs every stage sequentially, picks the learning rate and the best D-AWL
# variant by VALIDATION macro-F1 (never by test), and writes a summary.
#
# Run from the project root (where train_mt_isa.py lives), in the SAME
# environment where torch.cuda.is_available() is True:
#
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Smoke   # test the script (~3 min)
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1           # the real run
#
# (or, inside an open PowerShell window:  .\run_experiments.ps1 )
#
# Switches:  -Smoke      1 epoch, 64 instances, outputs go to models_smoke\
#            -NoBf16     disable bf16 (auto-enabled when CUDA is available)
#            -KeepCkpt   keep best.pt of every run (~1 GB each; deleted by default)
# Options:   -Epochs 20 -BatchSize 32 -Accum 1 -Train ... -Aux ... -Test ... -Model ...
#
# Re-running is safe: finished runs (test_metrics.json exists) are skipped.
# =============================================================================
param(
    [switch]$Smoke,
    [switch]$NoBf16,
    [switch]$KeepCkpt,
    [string]$Train = "data/processed/restaurant14_train_implicit.json",
    [string]$Aux   = "data/auxiliary/restaurants_train_implicit_aux.json",
    [string]$Test  = "data/processed/restaurant14_test_implicit.json",
    [string]$Model = "google/flan-t5-base",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$Accum = 1
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$ModelsDir  = "models"
$ExtraFlags = @()
if ($Smoke) {
    $Epochs = 1
    $ModelsDir = "models_smoke"
    $ExtraFlags = @("--max-instances", "64")
    Write-Host "[SMOKE MODE] 1 epoch, 64 instances, outputs in $ModelsDir/"
}

New-Item -ItemType Directory -Force -Path "logs", "results", $ModelsDir | Out-Null
Start-Transcript -Path "logs_master.txt" -Append | Out-Null

# ---- pre-flight checks ------------------------------------------------------
foreach ($f in @($Train, $Aux, $Test, "train_mt_isa.py", "mt_isa_model.py", "summarize.py")) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file $f"
        Stop-Transcript | Out-Null
        exit 1
    }
}

$Bf16 = $false
if (-not $NoBf16) {
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $Bf16 = $true }
}
if (-not $Bf16) {
    Write-Host "NOTE: bf16 is OFF (no CUDA torch found, or -NoBf16). If you expected the GPU, stop and fix the environment first."
}

Write-Host "Config: model=$Model epochs=$Epochs batch=${BatchSize}x${Accum} bf16=$Bf16"
Write-Host "Start: $(Get-Date)"
$AllTimer = [System.Diagnostics.Stopwatch]::StartNew()

# ---- one training run -------------------------------------------------------
function Invoke-Train {
    param([string]$Tag, [int]$Bs, [int]$Acc, [string]$Lr, [int]$Seed, [string[]]$Extra)
    $a = @("train_mt_isa.py",
           "--train-data", $Train, "--aux-data", $Aux, "--test-data", $Test,
           "--model-name", $Model,
           "--batch-size", "$Bs", "--grad-accum", "$Acc",
           "--num-epochs", "$Epochs", "--lr", $Lr, "--seed", "$Seed",
           "--output-dir", "$ModelsDir/exp_$Tag")
    if ($Bf16) { $a += "--bf16" }
    if ($ExtraFlags.Count -gt 0) { $a += $ExtraFlags }
    if ($Extra -and $Extra.Count -gt 0) { $a += $Extra }
    & python @a 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "logs/$Tag.log" -Append
}

function Invoke-Run {
    param([string]$Name, [string]$Lr, [int]$Seed, [string[]]$Extra)
    $tag = "${Name}_lr${Lr}_s${Seed}"
    $out = "$ModelsDir/exp_$tag"
    $metrics = "$out/test_metrics.json"
    if (Test-Path $metrics) {
        Write-Host "[skip] $tag (already done)"
        return
    }
    Write-Host ""
    Write-Host "=================== [run] $tag   $(Get-Date) ==================="
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-Train -Tag $tag -Bs $BatchSize -Acc $Accum -Lr $Lr -Seed $Seed -Extra $Extra
    if (-not (Test-Path $metrics)) {
        Write-Host "[retry] $tag failed; retrying with batch 16 x accum 2 (possible OOM)"
        Invoke-Train -Tag $tag -Bs 16 -Acc 2 -Lr $Lr -Seed $Seed -Extra $Extra
    }
    if (Test-Path $metrics) {
        Write-Host ("[done] $tag in {0:N0}s" -f $timer.Elapsed.TotalSeconds)
        if (-not $KeepCkpt) { Remove-Item "$out/best.pt" -ErrorAction SilentlyContinue }
    } else {
        Write-Host "[FAILED] $tag - see logs/$tag.log"
    }
}

# returns the directory with the highest val_f1 among the given directories
function Get-BestByVal {
    param([string[]]$Dirs)
    $best = $null
    $bestV = [double]::NegativeInfinity
    foreach ($d in $Dirs) {
        $f = "$d/test_metrics.json"
        if (Test-Path $f) {
            $v = [double]((Get-Content $f -Raw | ConvertFrom-Json).val_f1)
            if ($v -gt $bestV) { $bestV = $v; $best = $d }
        }
    }
    return $best
}

# =============================================================================
# STAGE 0: pick the learning rate (polarity-only baseline, seed 42)
# =============================================================================
Invoke-Run "polonly" "1e-5" 42 @("--polarity-only")
Invoke-Run "polonly" "1e-4" 42 @("--polarity-only")
$best = Get-BestByVal @("$ModelsDir/exp_polonly_lr1e-5_s42", "$ModelsDir/exp_polonly_lr1e-4_s42")
$BestLr = "1e-5"
if ($best -like "*lr1e-4*") { $BestLr = "1e-4" }
Write-Host ""
Write-Host ">>> Stage 0 picked LR=$BestLr (by validation macro-F1)"

# =============================================================================
# STAGE 1: the comparison (seed 42)
# =============================================================================
Invoke-Run "mtl_equal"   $BestLr 42 @("--d-awl-strategy", "none", "--freeze-t-awl")
Invoke-Run "mtl_tawl"    $BestLr 42 @("--d-awl-strategy", "none")
Invoke-Run "in_stored"   $BestLr 42 @("--d-awl-strategy", "input")
Invoke-Run "out_stored"  $BestLr 42 @("--d-awl-strategy", "output")
Invoke-Run "in_conv"     $BestLr 42 @("--d-awl-strategy", "input",  "--conf-source", "convergence")
Invoke-Run "out_conv"    $BestLr 42 @("--d-awl-strategy", "output", "--conf-source", "convergence")
Invoke-Run "tawl_dropnc" $BestLr 42 @("--d-awl-strategy", "none", "--drop-nonconverged")

$bestD = Get-BestByVal @(
    "$ModelsDir/exp_in_stored_lr${BestLr}_s42",
    "$ModelsDir/exp_out_stored_lr${BestLr}_s42",
    "$ModelsDir/exp_in_conv_lr${BestLr}_s42",
    "$ModelsDir/exp_out_conv_lr${BestLr}_s42")

$BestName = ""
$BestFlags = @()
if     ($bestD -like "*exp_in_stored*")  { $BestName = "in_stored";  $BestFlags = @("--d-awl-strategy", "input") }
elseif ($bestD -like "*exp_out_stored*") { $BestName = "out_stored"; $BestFlags = @("--d-awl-strategy", "output") }
elseif ($bestD -like "*exp_in_conv*")    { $BestName = "in_conv";    $BestFlags = @("--d-awl-strategy", "input",  "--conf-source", "convergence") }
elseif ($bestD -like "*exp_out_conv*")   { $BestName = "out_conv";   $BestFlags = @("--d-awl-strategy", "output", "--conf-source", "convergence") }
if ($BestName -eq "") { $shown = "none" } else { $shown = $BestName }
Write-Host ""
Write-Host ">>> Stage 1 best D-AWL variant by validation: $shown"

# =============================================================================
# STAGE 2: extra seeds for the key comparison (baseline, T-AWL only, best D-AWL)
# =============================================================================
foreach ($s in 1, 2) {
    Invoke-Run "polonly"  $BestLr $s @("--polarity-only")
    Invoke-Run "mtl_tawl" $BestLr $s @("--d-awl-strategy", "none")
    if ($BestName -ne "") {
        Invoke-Run $BestName $BestLr $s $BestFlags
    }
}

# =============================================================================
# SUMMARY
# =============================================================================
Write-Host ""
Write-Host "=================== SUMMARY ==================="
python summarize.py $ModelsDir results

# bundle the small files (logs + metrics) into results\bundle.zip
$stage = "results/bundle"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item "logs" "$stage/logs" -Recurse -ErrorAction SilentlyContinue
Copy-Item "results/summary.txt", "results/summary.csv" $stage -ErrorAction SilentlyContinue
Get-ChildItem "$ModelsDir" -Directory -Filter "exp_*" | ForEach-Object {
    $dest = Join-Path $stage $_.Name
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    foreach ($n in "test_metrics.json", "history.json", "test_predictions.json") {
        $p = Join-Path $_.FullName $n
        if (Test-Path $p) { Copy-Item $p $dest }
    }
}
Compress-Archive -Path "$stage/*" -DestinationPath "results/bundle.zip" -Force
Remove-Item $stage -Recurse -Force

Write-Host ""
Write-Host ("Total time: {0:N0} min" -f $AllTimer.Elapsed.TotalMinutes)
Write-Host "Send me: results\summary.txt  (and results\bundle.zip for the full logs)"
Stop-Transcript | Out-Null
