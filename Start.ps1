param([switch]$NoBrowser, [ValidateRange(0,65535)][int]$Port = 0)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$dataDirectory = if ($env:SUBTITLE_STUDIO_DATA) { $env:SUBTITLE_STUDIO_DATA } else { Join-Path $PSScriptRoot 'data' }
$serverFile = Join-Path $dataDirectory 'server.json'
$serverInfo = $null
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
$launchPort = $Port
if (-not $launchPort -and $serverInfo -and $serverInfo.port -gt 0 -and $serverInfo.port -le 65535) {
    $portProbe = $null
    try {
        $portProbe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, [int]$serverInfo.port)
        $portProbe.ExclusiveAddressUse = $true
        $portProbe.Start()
        $launchPort = [int]$serverInfo.port
    } catch { } finally { if ($portProbe) { $portProbe.Stop() } }
}
Start-Process -FilePath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -ArgumentList @('-m','subtitle_studio','--port',[string]$launchPort,'--no-browser') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'server-output.log') -RedirectStandardError (Join-Path $logDir 'server-error.log')
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
