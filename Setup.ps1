param([switch]$SkipUI)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.runtime\python'
$env:UV_PYTHON_BIN_DIR = Join-Path $PSScriptRoot '.runtime\bin'
$env:PYTHONUTF8 = '1'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    $binDir = Join-Path $PSScriptRoot '.runtime\bin'
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    $archive = Join-Path $binDir 'uv.zip'
    Write-Host 'Downloading uv from its official release...'
    Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/download/0.11.7/uv-x86_64-pc-windows-msvc.zip' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $binDir -Force
    $uvExe = Join-Path $binDir 'uv.exe'
} else { $uvExe = $uvCommand.Source }
& $uvExe python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python setup failed.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $uvExe venv --relocatable --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating the isolated runtime failed.' }
}
Write-Host 'Installing locked runtime dependencies...'
& $uvExe pip sync --python .venv\Scripts\python.exe requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed.' }
& $uvExe pip install --python .venv\Scripts\python.exe --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Installing Subtitle Studio failed.' }
if (-not $SkipUI) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Install Node.js 20.19+ to build the interface, then run Setup again.' }
    Push-Location frontend
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
& .venv\Scripts\python.exe -c "from subtitle_studio.models import discover_existing; print('Existing compatible models:', ', '.join(discover_existing()) or 'none yet')"
Write-Host 'Setup complete. Open Start.bat, then use Models & settings to install missing models.'
