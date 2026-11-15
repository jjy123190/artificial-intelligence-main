$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($PSScriptRoot)
$statePath = Join-Path $root 'outputs\runtime\server.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output '没有本项目的网页运行记录。'; exit 0 }
$state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([IO.Path]::GetFullPath($state.root) -ne $root) { throw '进程记录不属于当前项目，拒绝操作。' }
$process = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$state.pid)
if ($null -ne $process) {
    $entry = Join-Path $root 'main.py'
    if ($process.CommandLine.IndexOf($entry,[StringComparison]::OrdinalIgnoreCase) -lt 0 -or $process.CommandLine -notmatch '\bweb\s+--port\b') { throw 'PID已不属于本项目网页服务，拒绝停止。' }
    $children = @(Get-CimInstance Win32_Process -Filter ('ParentProcessId = ' + [int]$state.pid) | Where-Object { $_.Name -match '^python(w|3)?\.exe$' -and $_.CommandLine -match 'multiprocessing' })
    foreach ($child in $children) { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
    Stop-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
}
$state.running=$false
$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
Write-Output '本项目网页服务已停止，案例文件未改动。'
