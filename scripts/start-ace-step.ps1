[CmdletBinding()]
param(
    [ValidateRange(1, 65535)] [int]$Port = 8001,
    [string]$ApiKey = "",
    [switch]$InitializeLanguageModel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Repository = Join-Path $ProjectRoot "vendor\ACE-Step-1.5"
$Executable = Join-Path $Repository ".venv\Scripts\acestep-api.exe"
$Models = Join-Path $ProjectRoot "models\shared\ace-step"
$CheckpointLink = Join-Path $Repository "checkpoints"

if (-not (Test-Path $Executable)) {
    throw "ACE-Step is not installed. Run scripts\install-models.ps1 first."
}
if (-not (Test-Path (Join-Path $Models "acestep-v15-turbo\model.safetensors"))) {
    throw "ACE-Step turbo weights are missing. Run scripts\install-models.ps1 first."
}
if (-not (Test-Path $CheckpointLink)) {
    New-Item -ItemType Junction -Path $CheckpointLink -Target $Models | Out-Null
}
else {
    $Item = Get-Item -Force $CheckpointLink
    $ActualTarget = [System.IO.Path]::GetFullPath([string]($Item.Target | Select-Object -First 1))
    if ($Item.LinkType -ne "Junction" -or $ActualTarget -ne (Resolve-Path $Models).Path) {
        throw "ACE-Step checkpoints is not the expected shared-model junction."
    }
}

$env:ACESTEP_API_HOST = "127.0.0.1"
$env:ACESTEP_API_PORT = $Port.ToString()
$env:ACESTEP_PROJECT_ROOT = $Repository
$env:ACESTEP_CONFIG_PATH = "acestep-v15-turbo"
$env:ACESTEP_DEVICE = "cuda"
$env:ACESTEP_OFFLOAD_TO_CPU = "true"
$env:ACESTEP_COMPILE_MODEL = "false"
$env:ACESTEP_NO_INIT = "false"
$env:ACESTEP_INIT_LLM = if ($InitializeLanguageModel) { "true" } else { "false" }
$env:HF_HOME = Join-Path $ProjectRoot "models\cache\huggingface"
$env:NUMBA_CACHE_DIR = Join-Path $ProjectRoot "models\cache\numba"
$env:MPLCONFIGDIR = Join-Path $ProjectRoot "models\cache\matplotlib"
if ($ApiKey) {
    $env:ACESTEP_API_KEY = $ApiKey
}

Push-Location $ProjectRoot
try {
    & $Executable --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
