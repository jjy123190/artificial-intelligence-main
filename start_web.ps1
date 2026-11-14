param([int]$Port = 8765, [switch]$NoBrowser, [string]$PythonPath, [ValidateSet("home", "lab01", "lab02", "lab03")][string]$Entry = "home")
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
if ($Port -lt 1 -or $Port -gt 65535) { throw '端口必须在1至65535之间。' }
$root = [IO.Path]::GetFullPath($PSScriptRoot)
$mainPath = Join-Path $root 'main.py'
$runtime = Join-Path $root 'outputs\runtime'
$fragment = if ($Entry -eq 'home') { '#/home' } else { '#/labs/' + $Entry }
$statePath = Join-Path $runtime 'server.json'
. (Join-Path $root 'scripts\web_python.ps1')
$hash = [Security.Cryptography.SHA256]::Create()
$workspaceId = ([BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant())))).Replace('-','').ToLowerInvariant().Substring(0,16)
$hash.Dispose()

function Open-Website([string]$Address) {
    if ($NoBrowser) { return }
    try {
        Start-Process -FilePath $Address -ErrorAction Stop
    } catch {
        # 默认URL处理程序暂时不可用时，让资源管理器按系统关联打开。
        Start-Process -FilePath (Join-Path $env:WINDIR 'explorer.exe') -ArgumentList $Address -ErrorAction Stop
    }
}

New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$launchLog = Join-Path $runtime ('startup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + $PID + '.log')
$startupMutex = [Threading.Mutex]::new($false, ('Local\EquityWebStart-' + $workspaceId))
$mutexOwned = $false
$transcribing = $false
try {
    try { $mutexOwned = $startupMutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $mutexOwned = $true }
    if (-not $mutexOwned) {
        Write-Host '另一个启动窗口正在准备网站，正在等待它完成。'
        try { $mutexOwned = $startupMutex.WaitOne(600000) }
        catch [Threading.AbandonedMutexException] { $mutexOwned = $true }
        if (-not $mutexOwned) { throw '等待启动超时，请查看之前启动窗口的错误信息。' }
    }
    Start-Transcript -LiteralPath $launchLog -Append | Out-Null
    $transcribing = $true

# Check the recorded service first, before resolving/installing Python or rebuilding assets.
if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    try {
        $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($state.root -eq $root -and $state.running) {
            $health = Invoke-RestMethod -Uri ($state.url + '/api/health') -TimeoutSec 2
            if ($health.app -eq 'equity-reasoning-web' -and $health.workspace_id -eq $workspaceId) {
                Write-Output ('网站已运行：' + $state.url)
                Write-Output ('目标页面：' + $state.url + '/' + $fragment)
                Open-Website ($state.url + "/" + $fragment)
                exit 0
            }
        }
    } catch {}
}

$selectedPort = $null
$lastPort = [Math]::Min(65535, $Port + 9)
for ($candidate = $Port; $candidate -le $lastPort; $candidate++) {
    $candidateUrl = 'http://127.0.0.1:' + $candidate
    try {
        $health = Invoke-RestMethod -Uri ($candidateUrl + '/api/health') -TimeoutSec 1
        if ($health.app -eq 'equity-reasoning-web' -and $health.workspace_id -eq $workspaceId) {
            Write-Output ('网站已运行：' + $candidateUrl)
            Write-Output ('目标页面：' + $candidateUrl + '/' + $fragment)
            Open-Website ($candidateUrl + "/" + $fragment)
            exit 0
        }
    } catch {}
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $candidate)
    try { $listener.Start(); $selectedPort = $candidate; break }
    catch {} finally { $listener.Stop() }
}
if ($null -eq $selectedPort) { throw ('端口' + $Port + '至' + $lastPort + '均被占用，请关闭不需要的程序再启动。') }
$Port = $selectedPort
$python = Resolve-WebPython -Root $root -PythonPath $PythonPath
if (-not (Test-Path -LiteralPath (Join-Path $root 'web\dist\index.html'))) {
    Write-Output '首次启动需要构建网页，正在自动准备前端。'
    $npm = Get-Command npm.cmd -CommandType Application -ErrorAction SilentlyContinue
    if (-not $npm) { throw '首次构建需要Node.js。请安装Node.js后再次双击；已构建的网页不需要Node.js。' }
    Push-Location -LiteralPath (Join-Path $root 'web')
    try {
        if (Get-Command rtk -ErrorAction SilentlyContinue) { & rtk npm ci } else { & $npm.Source ci }
        if ($LASTEXITCODE -ne 0) { throw '网页依赖安装失败，请检查网络后再次双击启动。' }
        if (Get-Command rtk -ErrorAction SilentlyContinue) { & rtk npm run build } else { & $npm.Source run build }
        if ($LASTEXITCODE -ne 0) { throw '网页构建失败，请查看上方构建信息。' }
    } finally { Pop-Location }
}
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $runtime 'python-path.txt'), $python, (New-Object Text.UTF8Encoding($false)))
$url = 'http://127.0.0.1:' + $Port
$arguments = @('-B', '-X', 'utf8', ('"' + $mainPath + '"'), 'web', '--port', [string]$Port)
$previousLogMode = $env:EQUITY_WEB_FILE_LOG
$env:EQUITY_WEB_FILE_LOG = '1'
try {
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
} finally { $env:EQUITY_WEB_FILE_LOG = $previousLogMode }
@{pid=$process.Id;port=$Port;url=$url;root=$root;python=$python;workspace_id=$workspaceId;running=$true;started_at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
$ready = $false
for ($i = 0; $i -lt 100; $i++) {
    Start-Sleep -Milliseconds 200
    try {
        $health = Invoke-RestMethod -Uri ($url + '/api/health') -TimeoutSec 1
        if ($health.app -eq 'equity-reasoning-web' -and $health.workspace_id -eq $workspaceId) { $ready = $true; break }
    } catch {}
    if ($process.HasExited) { break }
}
if (-not $ready) {
    if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
    $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $state.running = $false
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    throw ('网页服务未就绪，请查看 ' + (Join-Path $runtime 'server-error.log'))
}
Write-Output ('网站已就绪：' + $url)
Write-Output ('目标页面：' + $url + '/' + $fragment)
# 服务已能复用，后续完整实验不再阻塞其他启动入口。
$startupMutex.ReleaseMutex()
$mutexOwned = $false
Open-Website ($url + "/" + $fragment)

# 源码模式先开放基础算法，不隐式生成报告/录屏或完整实验结果。
Write-Host '源码模式已启动；缺少历史统计时页面显示未生成，不会伪造结果。'
Write-Host '初始化实验二：python scripts/prepare_lab02.py'
Write-Host '初始化实验三：python scripts/prepare_lab03.py（真实LLM需另加 --llm）'
Write-Host '初始化后刷新页面。构建与实验命令会生成outputs/web/dist，纯源码检查前应另行归档。'
} catch {
    Write-Host ('启动失败：' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('错误日志：' + $launchLog)
    exit 1
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    if ($mutexOwned) { $startupMutex.ReleaseMutex() }
    $startupMutex.Dispose()
}
