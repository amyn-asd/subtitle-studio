$ErrorActionPreference = 'Stop'
$dataDirectory = if ($env:SUBTITLE_STUDIO_DATA) { $env:SUBTITLE_STUDIO_DATA } else { Join-Path $PSScriptRoot 'data' }
$serverFile = Join-Path $dataDirectory 'server.json'
if (-not (Test-Path -LiteralPath $serverFile)) { exit 0 }
$serverInfo = Get-Content -LiteralPath $serverFile -Raw | ConvertFrom-Json
try {
    Invoke-RestMethod -Method Post -Uri ('http://127.0.0.1:' + $serverInfo.port + '/api/shutdown') -Headers @{'X-Studio-Token'=$serverInfo.token} | Out-Null
    Write-Host 'Subtitle Studio is stopping. Completed chunks are saved.'
} catch { Write-Host 'Subtitle Studio is already stopped.' }
