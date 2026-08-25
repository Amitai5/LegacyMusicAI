[CmdletBinding()]
param(
    [switch]$SkipAceStep,
    [switch]$SkipSoulX,
    [switch]$SkipWeights
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AceRepository = Join-Path $ProjectRoot "vendor\ACE-Step-1.5"
$SoulXRepository = Join-Path $ProjectRoot "vendor\SoulX-Singer"
$AceModels = Join-Path $ProjectRoot "models\shared\ace-step"
$SoulXModels = Join-Path $ProjectRoot "models\shared\soulx"
$CacheRoot = Join-Path $ProjectRoot "models\cache"
$AceTrainingPatch = Join-Path $ProjectRoot "config\patches\ace-step-windows-training.patch"
$AceCommit = "14c0211d5a0653b0f63e27686f4c3f151b4d8629"
$SoulXCommit = "81aeb3ae772c70093c3de74dc23c92d983801ae4"

function Sync-PinnedRepository {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Commit,
        [string]$KnownPatch = ""
    )

    if (-not (Test-Path (Join-Path $Path ".git"))) {
        git clone $Url $Path
    }
    $Origin = (git -C $Path remote get-url origin).Trim()
    if ($Origin -ne $Url) {
        throw "Unexpected upstream origin at ${Path}: $Origin"
    }
    if (git -C $Path status --porcelain) {
        if (-not $KnownPatch -or -not (Test-Path -LiteralPath $KnownPatch)) {
            throw "Refusing to update a modified upstream checkout: $Path"
        }

        git -C $Path apply --unidiff-zero --reverse --check $KnownPatch 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Refusing to update an upstream checkout with changes beyond the registered compatibility patch: $Path"
        }
        git -C $Path apply --unidiff-zero --reverse $KnownPatch
        if ($LASTEXITCODE -ne 0 -or (git -C $Path status --porcelain)) {
            git -C $Path apply --unidiff-zero $KnownPatch 2>$null
            throw "Refusing to update an upstream checkout with changes beyond the registered compatibility patch: $Path"
        }
    }
    git -C $Path fetch origin $Commit --depth 1
    git -C $Path checkout --detach $Commit
    $Actual = (git -C $Path rev-parse HEAD).Trim()
    if ($Actual -ne $Commit) {
        throw "Pinned checkout verification failed at $Path"
    }

    if ($KnownPatch) {
        git -C $Path apply --unidiff-zero --check $KnownPatch
        if ($LASTEXITCODE -ne 0) {
            throw "Compatibility patch does not apply cleanly to pinned checkout: $KnownPatch"
        }
        git -C $Path apply --unidiff-zero $KnownPatch
        if ($LASTEXITCODE -ne 0) {
            throw "Compatibility patch failed: $KnownPatch"
        }
    }
}

function Find-CondaExecutable {
    $Command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        return $Command.Source
    }
    foreach ($Candidate in @(
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }
    throw "Conda is required for the isolated SoulX-Singer runtime."
}

function Find-SoulXPython {
    param([Parameter(Mandatory)] [string]$Conda)

    $Info = (& $Conda info --json | ConvertFrom-Json)
    $Environment = $Info.envs | Where-Object { (Split-Path $_ -Leaf) -eq "soulxsinger" } | Select-Object -First 1
    if ($null -eq $Environment) {
        return $null
    }
    $Python = Join-Path $Environment "python.exe"
    if (Test-Path $Python) {
        return $Python
    }
    return $null
}

New-Item -ItemType Directory -Force -Path $AceModels, $SoulXModels, $CacheRoot | Out-Null

Push-Location $ProjectRoot
try {
    uv sync --locked --group dev
}
finally {
    Pop-Location
}

if (-not $SkipAceStep) {
    Sync-PinnedRepository -Path $AceRepository -Url "https://github.com/ACE-Step/ACE-Step-1.5.git" -Commit $AceCommit -KnownPatch $AceTrainingPatch
    Push-Location $AceRepository
    try {
        uv sync --locked --no-dev
        if (-not $SkipWeights) {
            & ".\.venv\Scripts\hf.exe" download ACE-Step/Ace-Step1.5 --local-dir $AceModels
        }
    }
    finally {
        Pop-Location
    }

    $CheckpointLink = Join-Path $AceRepository "checkpoints"
    if (-not (Test-Path $CheckpointLink)) {
        New-Item -ItemType Junction -Path $CheckpointLink -Target $AceModels | Out-Null
    }
    else {
        $Item = Get-Item -Force $CheckpointLink
        $ActualTarget = [System.IO.Path]::GetFullPath([string]($Item.Target | Select-Object -First 1))
        $ExpectedTarget = (Resolve-Path $AceModels).Path
        if ($Item.LinkType -ne "Junction" -or $ActualTarget -ne $ExpectedTarget) {
            throw "ACE-Step checkpoints exists but is not the expected model junction: $CheckpointLink"
        }
    }
}

if (-not $SkipSoulX) {
    Sync-PinnedRepository -Path $SoulXRepository -Url "https://github.com/Soul-AILab/SoulX-Singer.git" -Commit $SoulXCommit
    $Conda = Find-CondaExecutable
    $SoulXPython = Find-SoulXPython -Conda $Conda
    if ($null -eq $SoulXPython) {
        & $Conda create -n soulxsinger -y python=3.10
        $SoulXPython = Find-SoulXPython -Conda $Conda
    }
    if ($null -eq $SoulXPython) {
        throw "The soulxsinger Conda environment was not created successfully."
    }

    & $SoulXPython -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0+cu121 torchaudio==2.2.0+cu121
    & $SoulXPython -m pip install -r (Join-Path $ProjectRoot "environments\soulx\requirements-runtime.txt")

    if (-not $SkipWeights) {
        $env:HF_HOME = Join-Path $CacheRoot "huggingface"
        $Hf = Join-Path (Split-Path $SoulXPython) "Scripts\hf.exe"
        & $Hf download Soul-AILab/SoulX-Singer model-svc.pt --local-dir (Join-Path $SoulXModels "SoulX-Singer")
        & $Hf download Soul-AILab/SoulX-Singer-Preprocess --include "mel-band-roformer-karaoke/*" --include "rmvpe/*" --local-dir (Join-Path $SoulXModels "SoulX-Singer-Preprocess")
        & $Hf download openai/whisper-base
    }
}

Write-Host "Model runtimes are installed and pinned. Run: uv run legacy-music doctor --strict"
