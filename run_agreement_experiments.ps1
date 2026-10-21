# Compare raw confidence, random, agreement-only, and combined weights.
# Validation is created inside train_mt_isa.py from the same training split.
param(
    [switch]$Smoke,
    [switch]$TuneAlpha,
    [switch]$Bf16,
    [string]$Train = "data/processed/restaurant14_train_implicit.json",
    [string]$Aux = "data/auxiliary/restaurant14_train_implicit_aux_mistral7b_full.json",
    [string]$MistralAux = "data/auxiliary/restaurant14_train_implicit_aux_mistral7b_full.json",
    [string]$QwenAux = "data/auxiliary/restaurant14_train_implicit_aux_qwen7b_full.json",
    [string]$Test = "data/processed/restaurant14_test_implicit.json",
    [string]$Model = "google/flan-t5-base",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$Accum = 1,
    [string]$OutputDir = "models_agreement"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
if ($Smoke) { $Epochs = 1; $OutputDir = "models_agreement_smoke" }
$seeds = @(1, 2, 42)
$modes = @(
    @{ Name = "A_confidence"; Source = "confidence"; Alpha = $null },
    @{ Name = "B_random"; Source = "random"; Alpha = $null },
    @{ Name = "C_agreement"; Source = "agreement"; Alpha = 0.0 }
)
if (-not $TuneAlpha) { $modes += @{ Name = "D_combined"; Source = "combined"; Alpha = 0.5 } }

foreach ($path in @($Train, $Aux, $MistralAux, $QwenAux, $Test)) {
    if (-not (Test-Path $path)) { throw "Missing input: $path" }
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Invoke-Experiment($mode, $seed, $alpha, $includeTest = $true) {
    $tag = "$($mode.Name)_s$seed"
    $out = Join-Path $OutputDir $tag
    if (Test-Path (Join-Path $out "test_metrics.json")) {
        Write-Host "[skip] $tag"
        return
    }
    $args = @("train_mt_isa.py", "--train-data", $Train, "--aux-data", $Aux,
        "--model-name", $Model, "--batch-size", "$BatchSize",
        "--grad-accum", "$Accum", "--num-epochs", "$Epochs", "--seed", "$seed",
        "--output-dir", $out, "--weight-source", $mode.Source,
        "--mistral-aux-data", $MistralAux, "--qwen-aux-data", $QwenAux)
    if ($Bf16) { $args += "--bf16" }
    if ($includeTest) { $args += @("--test-data", $Test) }
    if ($null -ne $alpha) { $args += @("--agreement-alpha", "$alpha") }
    if ($Smoke) { $args += @("--max-instances", "64") }
    Write-Host "[run] $tag"
    & python @args
}

foreach ($mode in $modes) {
    foreach ($seed in $seeds) { Invoke-Experiment $mode $seed $mode.Alpha }
}

if ($TuneAlpha) {
    foreach ($alpha in @(0.25, 0.5, 0.75)) {
        $mode = @{ Name = "D_tune_alpha_$alpha"; Source = "combined" }
        Invoke-Experiment $mode 42 $alpha $false
    }
    $bestAlpha = $null
    $bestValAcc = [double]::NegativeInfinity
    foreach ($alpha in @(0.25, 0.5, 0.75)) {
        $historyPath = Join-Path $OutputDir "D_tune_alpha_$alpha\history.json"
        if (Test-Path $historyPath) {
            $history = Get-Content $historyPath -Raw | ConvertFrom-Json
            $best = $history | Sort-Object val_acc -Descending | Select-Object -First 1
            if ($best.val_acc -gt $bestValAcc) { $bestValAcc = $best.val_acc; $bestAlpha = $alpha }
        }
    }
    if ($null -eq $bestAlpha) { throw "No alpha tuning run produced validation history." }
    Write-Host "Selected alpha=$bestAlpha by validation accuracy=$bestValAcc; test runs use this alpha only."
    $selected = @{ Name = "D_combined"; Source = "combined" }
    foreach ($seed in $seeds) { Invoke-Experiment $selected $seed $bestAlpha $true }
}

$summary = @()
Get-ChildItem $OutputDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Name -match "^(A_confidence|B_random|C_agreement|D_combined)_s") {
        $metricsPath = Join-Path $_.FullName "test_metrics.json"
        if (Test-Path $metricsPath) {
            $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
            $summary += [pscustomobject]@{ Mode = $Matches[1]; F1 = [double]$metrics.macro_f1; Acc = [double]$metrics.acc }
        }
    }
}
Write-Host "`nMode                 Test macro-F1 (mean +/- std)       Test accuracy (mean +/- std)"
foreach ($group in ($summary | Group-Object Mode)) {
    $f1 = @($group.Group | ForEach-Object F1)
    $acc = @($group.Group | ForEach-Object Acc)
    $f1Mean = ($f1 | Measure-Object -Average).Average
    $accMean = ($acc | Measure-Object -Average).Average
    $f1Std = if ($f1.Count -gt 1) { [math]::Sqrt((($f1 | ForEach-Object { ($_ - $f1Mean) * ($_ - $f1Mean) } | Measure-Object -Sum).Sum) / ($f1.Count - 1)) } else { 0 }
    $accStd = if ($acc.Count -gt 1) { [math]::Sqrt((($acc | ForEach-Object { ($_ - $accMean) * ($_ - $accMean) } | Measure-Object -Sum).Sum) / ($acc.Count - 1)) } else { 0 }
    "{0,-20} {1:F4} +/- {2:F4}                  {3:F4} +/- {4:F4}" -f $group.Name, $f1Mean, $f1Std, $accMean, $accStd
}