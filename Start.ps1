param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$dataDirectory = if ($env:SUBTITLE_STUDIO_DATA) { $env:SUBTITLE_STUDIO_DATA } else { Join-Path $PSScriptRoot 'data' }
$serverFile = Join-Path $dataDirectory 'server.json'
if (Test-Path -LiteralPath $serverFile) {
    try {
        $serverInfo = Get-Content -LiteralPath $serverFile -Raw | ConvertFrom-Json
        $baseUrl = 'http://127.0.0.1:' + $serverInfo.port
        $health = Invoke-RestMethod -Uri ($baseUrl + '/health') -TimeoutSec 2
        if ($health.app -eq 'Subtitle Studio') { if (-not $NoBrowser) { Start-Process $baseUrl }; exit 0 }
    } catch { }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe') -or -not (Test-Path -LiteralPath 'web\index.html')) {
    & (Join-Path $PSScriptRoot 'Setup.ps1')
}
$logDir = Join-Path $dataDirectory 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONUTF8 = '1'
Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -ArgumentList @('-m','subtitle_studio','--no-browser') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'server-output.log') -RedirectStandardError (Join-Path $logDir 'server-error.log')
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    Start-Sleep -Milliseconds 500
    if (Test-Path -LiteralPath $serverFile) {
        try {
            $serverInfo = Get-Content -LiteralPath $serverFile -Raw | ConvertFrom-Json
            $baseUrl = 'http://127.0.0.1:' + $serverInfo.port
            $health = Invoke-RestMethod -Uri ($baseUrl + '/health') -TimeoutSec 1
            if ($health.app -eq 'Subtitle Studio') { if (-not $NoBrowser) { Start-Process $baseUrl }; exit 0 }
        } catch { }
    }
}
throw 'Startup failed. See data\logs\server-error.log.'
