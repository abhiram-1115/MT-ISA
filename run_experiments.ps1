# =============================================================================
# MT-ISA experiment runner (PowerShell version of run_experiments.sh)
#
# Runs every config for every seed sequentially and writes a summary.
#
# Grid: configs x seeds 1..10 (see $ConfigTable below).
#
# Run from the project root (where train_mt_isa.py lives), in the SAME
# environment where torch.cuda.is_available() is True:
#
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Smoke   # test the script (~3 min)
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1           # the real run
#
# (or, inside an open PowerShell window:  .\run_experiments.ps1 )
#
# Switches:  -Smoke      1 epoch, 64 instances, seed 1 only, outputs go to models_smoke\
#            -NoBf16     disable bf16 (auto-enabled when CUDA is available)
#            -KeepCkpt   keep best.pt of every run (~1 GB each; deleted by default)
#            -Dataset laptop   laptop14 (only configs with --train-all, e.g. polonly_all_gv,
#                              mtisa_in_all_gv); outputs go to models_laptop\, logs_laptop\,
#                              results_laptop\ so restaurant results are never mixed in.
#            -AllAux PATH (alias of -TrainAllAux) overrides the train-all aux file of mtisa_in_all_gv,
#            aux0.3_all_gv, mtisa_ce_all_gv, lora_mtl*; mtisa4o_* always use the GPT-4o-mini aux file
#            data/auxiliary/<dataset>14_train_all_aux_gpt4omini_full.json.
#            kd_all_gv uses -TeacherProbs (default data/teacher/<dataset>14_train_all_teacher_probs.json,
#            made by teacher_folds.py in venv_lora).
#            lora7b_pol = Qwen2.5-7B-Instruct in 4-bit (needs bitsandbytes), batch 4 x 4.
#            lora_pol / lora_mtl / lora_mtl_shuf call train_lora_cls.py (Qwen2.5-3B LoRA, needs
#            peft and transformers >= 4.37; lr 2e-4, 3 epochs, batch 8 x 2); their folders are
#            exp_<name>_lr2e-4_s<seed>.
# Options:   -Lr 1e-4 -SeedList "1,2345,2" -Configs "grad_align,polonly" -Epochs 20 -BatchSize 32 -Accum 1 -Train ... -Aux ... -Test ... -Model ...
#
# Re-running is safe: finished runs (test_metrics.json exists) are skipped.
# =============================================================================
param(
    [switch]$Smoke,
    [switch]$NoBf16,
    [switch]$KeepCkpt,
    [string]$Train = "data/processed/restaurant14_train_implicit.json",
    [string]$Aux   = "data/auxiliary/restaurant14_train_implicit_aux_qwen7b_full.json",
    [string]$Test  = "data/processed/restaurant14_test_implicit.json",
    [string]$Model = "google/flan-t5-base",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$Accum = 1,
    [string]$Lr = "1e-4",
    [string]$SeedList = "1,2,3,4,5,6,7,8,9,10",
    [string]$Configs = "",
    [ValidateSet("restaurant", "laptop")][string]$Dataset = "restaurant",
    [string]$TrainAll = "",
    [string]$TestAll = "",
    [Alias("AllAux")][string]$TrainAllAux = "",
    [string]$TeacherProbs = ""
)

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# ---- dataset-dependent paths (explicit -Train/-Aux/-Test/... still win) -----
$Suffix = ""
if ($Dataset -eq "laptop") {
    $Suffix = "_laptop"
    if (-not $PSBoundParameters.ContainsKey("Train")) { $Train = "data/processed/laptop14_train_all.json" }
    if (-not $PSBoundParameters.ContainsKey("Test"))  { $Test  = "data/processed/laptop14_test_implicit.json" }
    if (-not $PSBoundParameters.ContainsKey("Aux"))   { $Aux   = "" }   # no implicit-only aux for laptop
    if ($TrainAll -eq "")    { $TrainAll    = "data/processed/laptop14_train_all.json" }
    if ($TestAll -eq "")     { $TestAll     = "data/processed/laptop14_test_all.json" }
    if ($TrainAllAux -eq "") { $TrainAllAux = "data/auxiliary/laptop14_train_all_aux_qwen7b_full.json" }
    if ($TeacherProbs -eq "") { $TeacherProbs = "data/teacher/laptop14_train_all_teacher_probs.json" }
    $Aux4o = "data/auxiliary/laptop14_train_all_aux_gpt4omini_full.json"
} else {
    if ($TrainAll -eq "")    { $TrainAll    = "data/processed/restaurant14_train_all.json" }
    if ($TestAll -eq "")     { $TestAll     = "data/processed/restaurant14_test_all.json" }
    if ($TrainAllAux -eq "") { $TrainAllAux = "data/auxiliary/restaurant14_train_all_aux_qwen7b_full.json" }
    if ($TeacherProbs -eq "") { $TeacherProbs = "data/teacher/restaurant14_train_all_teacher_probs.json" }
    $Aux4o = "data/auxiliary/restaurant14_train_all_aux_gpt4omini_full.json"
}

$ModelsDir  = "models$Suffix"
$LogsDir    = "logs$Suffix"
$ResultsDir = "results$Suffix"
$ExtraFlags = @()
if ($Smoke) {
    $Epochs = 1
    $ModelsDir = "models_smoke$Suffix"
    $ExtraFlags = @("--max-instances", "64")
    Write-Host "[SMOKE MODE] 1 epoch, 64 instances, outputs in $ModelsDir/"
}

New-Item -ItemType Directory -Force -Path $LogsDir, $ResultsDir, "results", $ModelsDir | Out-Null
Start-Transcript -Path "logs_master$Suffix.txt" -Append | Out-Null

# ---- pre-flight checks ------------------------------------------------------
$PreFiles = @($Test, "train_mt_isa.py", "mt_isa_model.py", "summarize.py")
if ($Dataset -eq "restaurant") { $PreFiles += @($Train, $Aux) }
foreach ($f in $PreFiles) {
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
    param([string]$Tag, [int]$Bs, [int]$Acc, [string]$Lr, [int]$Seed, [string[]]$Extra, [bool]$Lora = $false)
    if ($Lora) {
        # LoRA runs (train_lora_cls.py): Qwen2.5-3B, 3 epochs (1 in -Smoke), lr 2e-4, bf16 built in
        $ep = 3
        if ($Smoke) { $ep = 1 }
        $a = @("train_lora_cls.py", "--dataset", $Dataset,
               "--test-data", $Test, "--train-all-data", $TrainAll, "--test-all-data", $TestAll,
               "--batch-size", "$Bs", "--grad-accum", "$Acc",
               "--num-epochs", "$ep", "--lr", $Lr, "--seed", "$Seed",
               "--output-dir", "$ModelsDir/exp_$Tag")
    } else {
        $a = @("train_mt_isa.py",
               "--train-data", $Train, "--test-data", $Test,
               "--train-all-data", $TrainAll, "--test-all-data", $TestAll,
               "--model-name", $Model,
               "--batch-size", "$Bs", "--grad-accum", "$Acc",
               "--num-epochs", "$Epochs", "--lr", $Lr, "--seed", "$Seed",
               "--output-dir", "$ModelsDir/exp_$Tag")
        if ($Aux -ne "") { $a += @("--aux-data", $Aux) }
        if ($Bf16) { $a += "--bf16" }
    }
    if ($ExtraFlags.Count -gt 0) { $a += $ExtraFlags }
    if ($Extra -and $Extra.Count -gt 0) { $a += $Extra }
    & python @a 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "$LogsDir/$Tag.log" -Append
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
    $isLora = $LoraConfigs -contains $Name
    $bs1 = $BatchSize; $ac1 = $Accum; $bs2 = 16; $ac2 = 2
    if ($isLora) { $bs1 = 8; $ac1 = 2; $bs2 = 4; $ac2 = 4 }
    if ($Name -like "lora7b*") { $bs1 = 4; $ac1 = 4; $bs2 = 2; $ac2 = 8 }   # 7B in 4-bit
    Invoke-Train -Tag $tag -Bs $bs1 -Acc $ac1 -Lr $Lr -Seed $Seed -Extra $Extra -Lora $isLora
    if (-not (Test-Path $metrics)) {
        Write-Host "[retry] $tag failed; retrying with batch $bs2 x accum $ac2 (possible OOM)"
        Invoke-Train -Tag $tag -Bs $bs2 -Acc $ac2 -Lr $Lr -Seed $Seed -Extra $Extra -Lora $isLora
    }
    if (Test-Path $metrics) {
        Write-Host ("[done] $tag in {0:N0}s" -f $timer.Elapsed.TotalSeconds)
        # LoRA adapters are small and prior_correct.py needs them -> only T5 checkpoints are deleted
        if (-not $KeepCkpt -and -not $isLora) { Remove-Item "$out/best.pt" -ErrorAction SilentlyContinue }
    } else {
        Write-Host "[FAILED] $tag - see $LogsDir/$tag.log"
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
# Main grid: fixed LR, seeds 1..10, every config
# Folder naming stays exp_<name>_lr<lr>_s<seed> so summarize.py still works.
# =============================================================================
# configs that call train_lora_cls.py (Qwen2.5-3B LoRA; folder tag uses lr 2e-4)
$LoraConfigs = @("lora_pol", "lora_mtl", "lora_mtl_shuf", "lora7b_pol")
$ConfigTable = [ordered]@{
    "polonly"    = @("--polarity-only")
    "mtl_tawl"   = @("--d-awl-strategy", "none")
    "in_stored"  = @("--d-awl-strategy", "input")
    "out_stored" = @("--d-awl-strategy", "output")
    "in_shuffle" = @("--d-awl-strategy", "input",  "--conf-source", "shuffle")
    "out_shuffle"= @("--d-awl-strategy", "output", "--conf-source", "shuffle")
    "in_const"   = @("--d-awl-strategy", "input",  "--conf-source", "constant", "--const-conf", "0.9")
    "out_const"  = @("--d-awl-strategy", "output", "--conf-source", "constant", "--const-conf", "0.9")
    "aux0.1"     = @("--d-awl-strategy", "none", "--aux-weight", "0.1")
    "aux0.3"     = @("--d-awl-strategy", "none", "--aux-weight", "0.3")
    "aux1.0"     = @("--d-awl-strategy", "none", "--aux-weight", "1.0")
    "aux_conv0.3" = @("--d-awl-strategy", "none", "--aux-mask-nonconverged", "--aux-weight", "0.3")
    "aux0.3_shufhints" = @("--d-awl-strategy", "none", "--aux-weight", "0.3", "--shuffle-hints")
    "lf0.1"      = @("--d-awl-strategy", "none", "--label-first-seq", "0.1")
    "lf0.3"      = @("--d-awl-strategy", "none", "--label-first-seq", "0.3")
    "lf1.0"      = @("--d-awl-strategy", "none", "--label-first-seq", "1.0")
    "lf0.3_shuf" = @("--d-awl-strategy", "none", "--label-first-seq", "0.3", "--shuffle-hints")
    "polonly_gv" = @("--polarity-only", "--val-group-by-sentence")
    "lf1.0_gv"   = @("--d-awl-strategy", "none", "--label-first-seq", "1.0", "--val-group-by-sentence")
    "lf1.0_shuf_gv" = @("--d-awl-strategy", "none", "--label-first-seq", "1.0", "--shuffle-hints", "--val-group-by-sentence")
    "polonly_all_gv" = @("--polarity-only", "--train-all", "--val-group-by-sentence")
    "mtisa_in_all_gv" = @("--d-awl-strategy", "input", "--train-all", "--val-group-by-sentence", "--train-all-aux", $TrainAllAux)
    "mtisa4o_all_gv"      = @("--d-awl-strategy", "input", "--train-all", "--val-group-by-sentence", "--train-all-aux", $Aux4o)
    "mtisa4o_shuf_all_gv" = @("--d-awl-strategy", "input", "--train-all", "--val-group-by-sentence", "--train-all-aux", $Aux4o, "--shuffle-hints")
    "mtisa_ce_all_gv" = @("--d-awl-strategy", "input", "--train-all", "--val-group-by-sentence", "--train-all-aux", $TrainAllAux, "--polarity-ce3")
    "aux0.3_all_gv" = @("--d-awl-strategy", "none", "--aux-weight", "0.3", "--train-all", "--val-group-by-sentence", "--train-all-aux", $TrainAllAux)
    "lora_pol"      = @("--tasks", "pol")
    "lora_mtl"      = @("--tasks", "mtl", "--train-all-aux", $TrainAllAux)
    "lora_mtl_shuf" = @("--tasks", "mtl", "--train-all-aux", $TrainAllAux, "--shuffle-hints")
    "lora7b_pol"    = @("--tasks", "pol", "--model-name", "Qwen/Qwen2.5-7B-Instruct", "--load-4bit", "--eval-batch-size", "8")
    "ce_all_gv"  = @("--polarity-only", "--train-all", "--val-group-by-sentence", "--polarity-ce3")
    "kd_all_gv"  = @("--polarity-only", "--train-all", "--val-group-by-sentence", "--kd-probs", $TeacherProbs)
    "ls_all_gv"  = @("--polarity-only", "--train-all", "--val-group-by-sentence", "--label-smoothing", "0.1")
    "grad_align" = @("--d-awl-strategy", "grad_align")
}
$Selected = @($ConfigTable.Keys)
if ($Configs.Trim() -ne "") {
    $Selected = @($Configs -split "[,\s]+" | Where-Object { $_ -ne "" })
    $unknown = @($Selected | Where-Object { -not $ConfigTable.Contains($_) })
    if ($unknown.Count -gt 0) {
        Write-Host "ERROR: unknown config(s): $($unknown -join ', '). Valid: $($ConfigTable.Keys -join ', ')"
        Stop-Transcript | Out-Null
        exit 1
    }
}
# configs that read the *_all files need those files (and laptop supports only such configs)
$needAll = @($Selected | Where-Object { ($ConfigTable[$_] -contains "--train-all") -or ($LoraConfigs -contains $_) })
if (@($Selected | Where-Object { $LoraConfigs -contains $_ }).Count -gt 0 -and -not (Test-Path "train_lora_cls.py")) {
    Write-Host "ERROR: missing file train_lora_cls.py"
    Stop-Transcript | Out-Null
    exit 1
}
if ($Dataset -eq "laptop" -and $needAll.Count -ne $Selected.Count) {
    $bad = @($Selected | Where-Object { $needAll -notcontains $_ })
    Write-Host "ERROR: -Dataset laptop supports only configs with --train-all (e.g. polonly_all_gv, mtisa_in_all_gv, aux0.3_all_gv). Not supported: $($bad -join ', ')"
    Stop-Transcript | Out-Null
    exit 1
}
$chk = @()
if ($needAll.Count -gt 0) { $chk += @($TrainAll, $TestAll) }
foreach ($name in $Selected) {   # every aux file a selected config passes via --train-all-aux
    $cfgArgs = @($ConfigTable[$name])
    $ix = [array]::IndexOf($cfgArgs, "--train-all-aux")
    if ($ix -ge 0 -and $ix + 1 -lt $cfgArgs.Count -and $chk -notcontains $cfgArgs[$ix + 1]) { $chk += $cfgArgs[$ix + 1] }
}
if (@($Selected | Where-Object { $ConfigTable[$_] -contains "--kd-probs" }).Count -gt 0) { $chk += $TeacherProbs }
foreach ($f in $chk) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file $f"
        Stop-Transcript | Out-Null
        exit 1
    }
}
Write-Host "Dataset: $Dataset | models: $ModelsDir | logs: $LogsDir | results: $ResultsDir"
$Seeds = @($SeedList -split "[,\s]+" | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
if ($Smoke) { $Seeds = @(1) }
Write-Host "Configs: $($Selected -join ', ') | seeds: $($Seeds -join ',')"

# seed-major order: every config gets seed 1 before any gets seed 2
foreach ($s in $Seeds) {
    foreach ($name in $Selected) {
        $lrUse = $Lr
        if ($LoraConfigs -contains $name) { $lrUse = "2e-4" }
        Invoke-Run $name $lrUse $s $ConfigTable[$name]
    }
}


# =============================================================================
# SUMMARY
# =============================================================================
Write-Host ""
Write-Host "=================== SUMMARY ==================="
python summarize.py $ModelsDir $ResultsDir

# bundle the small files (logs + metrics) into results\bundle.zip
$stage = "$ResultsDir/bundle"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item $LogsDir "$stage/logs" -Recurse -ErrorAction SilentlyContinue
Copy-Item "$ResultsDir/summary.txt", "$ResultsDir/summary.csv" $stage -ErrorAction SilentlyContinue
Get-ChildItem "$ModelsDir" -Directory -Filter "exp_*" | ForEach-Object {
    $dest = Join-Path $stage $_.Name
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    foreach ($n in "test_metrics.json", "history.json", "test_predictions.json") {
        $p = Join-Path $_.FullName $n
        if (Test-Path $p) { Copy-Item $p $dest }
    }
}
Compress-Archive -Path "$stage/*" -DestinationPath "$ResultsDir/bundle.zip" -Force
Remove-Item $stage -Recurse -Force

Write-Host ""
Write-Host ("Total time: {0:N0} min" -f $AllTimer.Elapsed.TotalMinutes)
Write-Host "Send me: $ResultsDir\summary.txt  (and $ResultsDir\bundle.zip for the full logs)"
Stop-Transcript | Out-Null
