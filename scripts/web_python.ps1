# Resolve an interpreter independently of Explorer/Conda PATH ordering.
function Test-WebPython {
    param([string]$Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    if ($Path -match '\\Microsoft\\WindowsApps\\') { return $null }
    $probe = 'import sys,importlib.util,json; ok=sys.version_info>=(3,11); names=(\"fastapi\",\"uvicorn\",\"pydantic\",\"httpx\",\"networkx\"); modules=all(importlib.util.find_spec(m) for m in names); ready=False; exec(\"if ok and modules:\n import fastapi,uvicorn,pydantic,httpx,networkx\n ready=int(pydantic.__version__.split(chr(46))[0])>=2\"); print(json.dumps(dict(executable=sys.executable,supported=ok,ready=ready)))'
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Path
    $info.Arguments = '-B -X utf8 -c "' + $probe + '"'
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(15000)) {
            $process.Kill()
            return $null
        }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { return $null }
        return ($stdout.Result.Trim() | ConvertFrom-Json)
    } catch { return $null } finally { $process.Dispose() }
}

function Get-WebPythonCandidates {
    param([string]$Root, [string]$PythonPath)
    if ($PythonPath) {
        $specified = [IO.Path]::GetFullPath($PythonPath)
        if (-not (Test-Path -LiteralPath $specified -PathType Leaf)) { throw "指定Python不存在：$specified" }
        return ,$specified
    }
    $candidates = @(
        (Join-Path $Root '.venv\Scripts\python.exe')
    )
    $cache = Join-Path $Root 'outputs\runtime\python-path.txt'
    if (Test-Path -LiteralPath $cache -PathType Leaf) {
        $candidates += ([IO.File]::ReadAllText($cache, [Text.Encoding]::UTF8)).Trim()
    }
    $candidates += @(Get-Command python.exe -All -CommandType Application -ErrorAction SilentlyContinue | ForEach-Object { $_.Source })
    # Both user and machine registrations matter: Explorer may see only the user PATH.
    foreach ($registry in @('HKCU:\Software\Python', 'HKLM:\SOFTWARE\Python', 'HKLM:\SOFTWARE\WOW6432Node\Python')) {
        if (-not (Test-Path -LiteralPath $registry)) { continue }
        foreach ($key in @(Get-ChildItem -LiteralPath $registry -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -eq 'InstallPath' })) {
            $executable = $key.GetValue('ExecutablePath')
            $install = $key.GetValue('')
            if ($executable) { $candidates += [string]$executable }
            elseif ($install) { $candidates += Join-Path ([string]$install) 'python.exe' }
        }
    }
    $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $listing = @(& $launcher.Source -0p 2>$null)
            foreach ($line in $listing) {
                if ([string]$line -match '([A-Za-z]:\\.*python(?:w)?\.exe)\s*$') { $candidates += $Matches[1] }
            }
        } catch {}
    }
    return @($candidates | Where-Object { $_ } | Select-Object -Unique)
}

function Resolve-WebPython {
    param([string]$Root, [string]$PythonPath, [switch]$NoInstall)
    $Root = [IO.Path]::GetFullPath($Root)
    $supported = @()
    foreach ($candidate in @(Get-WebPythonCandidates -Root $Root -PythonPath $PythonPath)) {
        $result = Test-WebPython -Path $candidate
        if ($null -eq $result -or -not $result.supported) { continue }
        $supported += [string]$result.executable
        if ($result.ready) {
            Write-Host ('网页运行环境：' + $result.executable)
            return [string]$result.executable
        }
    }
    if ($supported.Count -eq 0) { throw '未找到可运行的Python 3.11或更高版本。请先安装Python，再双击启动。' }
    if ($NoInstall) { throw '已找到Python，但网页依赖不完整。' }
    $environment = Join-Path $Root '.venv'
    $python = Join-Path $environment 'Scripts\python.exe'
    Write-Host '首次使用：正在准备本项目的独立Python环境，请稍候。'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        & $supported[0] -B -X utf8 -m venv $environment | Out-Host
        if ($LASTEXITCODE -ne 0) { throw '无法创建独立Python环境。请检查Python的venv组件及项目目录写入权限。' }
    }
    & $python -B -X utf8 -m pip --version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        & $python -B -X utf8 -m ensurepip --upgrade | Out-Host
        if ($LASTEXITCODE -ne 0) { throw '无法准备pip。请检查Python安装。' }
    }
    Write-Host '正在自动安装网页依赖（仅首次需要联网）……'
    & $python -B -X utf8 -m pip install --disable-pip-version-check --timeout 30 --retries 2 -r (Join-Path $Root 'requirements-web.txt') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw '网页依赖自动安装失败。请检查网络连接，再双击启动；已有源码和实验结果未改动。' }
    $result = Test-WebPython -Path $python
    if ($null -eq $result -or -not $result.ready) { throw '网页依赖安装后仍不完整。请检查上方安装信息。' }
    return [string]$result.executable
}
