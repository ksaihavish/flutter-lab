<#
.SYNOPSIS
  One-command setup and launch for Flutter Lab (Windows).

.EXAMPLE
  .\start.ps1              # set up (first run only) and open the web app
  .\start.ps1 -Test        # run the test suite
  .\start.ps1 -Evaluate    # regenerate outputs/results.json and all paper figures
  .\start.ps1 -Train       # (re)train the neural policy (installs PyTorch, ~5 min)
#>
param(
    [switch]$Train,
    [switch]$Evaluate,
    [switch]$Test,
    [int]$Port = 8000,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

# The virtual environment lives outside the project (~1 GB with PyTorch; never synced or committed).
if ($env:FLUTTER_AI_VENV) { $Venv = $env:FLUTTER_AI_VENV }
else { $Venv = Join-Path $env:LOCALAPPDATA "flutter-ai\venv" }
$Py = Join-Path $Venv "Scripts\python.exe"

function Find-Python {
    foreach ($v in @("3.14", "3.13", "3.12", "3.11", "3.10")) {
        try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { return @("py", "-$v") } } catch {}
    }
    try { & python -c "import sys; assert sys.version_info >= (3, 10)" 2>$null; if ($LASTEXITCODE -eq 0) { return @("python") } } catch {}
    throw "Python 3.10+ not found. Install it from https://www.python.org/downloads/ and re-run."
}

if (-not (Test-Path $Py)) {
    Write-Host "Creating virtual environment at $Venv ..." -ForegroundColor Cyan
    $base = @(Find-Python)
    if ($base.Length -gt 1) { & $base[0] $base[1] -m venv $Venv } else { & $base[0] -m venv $Venv }
    & $Py -m pip install --upgrade pip --quiet
}

# (Re)install when pyproject.toml changes
$stamp = Join-Path $Venv ".flutter-ai-stamp"
$hash = (Get-FileHash (Join-Path $Root "pyproject.toml")).Hash
$extras = "llm,dev"
$policy = Join-Path $Root "models\neural_policy_uav_flap.npz"
if ($Train -or -not (Test-Path $policy)) { $extras = "llm,dev,train" }
$want = "$hash|$extras"
$have = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { "" }
# Reinstall if pyproject changed, or if we now need extras (train) that are not installed yet.
$upToDate = ($have -eq $want) -or ($have -eq "$hash|llm,dev,train")
if (-not $upToDate) {
    Write-Host "Installing dependencies ($extras) ..." -ForegroundColor Cyan
    if ($extras -like "*train*") {
        & $Py -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    }
    & $Py -m pip install -e ".[$extras]" --quiet
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    Set-Content $stamp $want
}

if ($Test)     { & $Py -m pytest; exit $LASTEXITCODE }
if ($Train -or -not (Test-Path $policy)) {
    Write-Host "Training the neural policies (about 5 minutes each on a laptop CPU) ..." -ForegroundColor Cyan
    & $Py -m flutter_ai train --preset uav_flap
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }
    & $Py -m flutter_ai train --preset uav_section
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }
}
if ($Evaluate) { & $Py -m flutter_ai evaluate; exit $LASTEXITCODE }

$serveArgs = @("-m", "flutter_ai", "serve", "--port", "$Port")
if ($NoBrowser) { $serveArgs += "--no-browser" }
& $Py @serveArgs
