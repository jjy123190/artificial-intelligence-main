param([string]$OutputDir, [switch]$Force, [string]$PythonPath, [switch]$Interactive)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$root = [IO.Path]::GetFullPath($PSScriptRoot)
. (Join-Path $root 'scripts\web_python.ps1')
$python = Resolve-WebPython -Root $root -PythonPath $PythonPath
$resultDirectory = if ($OutputDir) { [IO.Path]::GetFullPath($OutputDir) } else { Join-Path $root 'outputs\lab02' }

# 双击入口明确区分查看和重算；网站自动初始化仍使用原来的非交互流程。
if ($Interactive) {
    & $python -B -X utf8 (Join-Path $root 'scripts\prepare_lab02.py') --check --output-dir $resultDirectory
    $hasResults = $LASTEXITCODE -eq 0
    Write-Host ''
    Write-Host '实验二：实验运行与结果查看' -ForegroundColor Cyan
    Write-Host ('结果目录：' + $resultDirectory)
    if ($hasResults) { Write-Host '已检测到完整实验结果，可以直接查看。' }
    else { Write-Host '尚未检测到完整结果，请选择运行完整实验。' }
    Write-Host ''
    Write-Host '[1] 重新运行完整实验（更新模型、数据和图表，需数分钟）'
    Write-Host '[2] 查看已有结果（打开结果文件夹）'
    Write-Host '[0] 退出'
    $defaultChoice = if ($hasResults) { '2' } else { '1' }
    while ($true) {
        $choice = Read-Host ('请输入选项，直接回车选择 ' + $defaultChoice)
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = $defaultChoice }
        if ($choice -eq '0') { exit 0 }
        if ($choice -eq '1') {
            $Force = $true
            Write-Host '开始重新计算，请保持此窗口打开。完成后会显示结果位置。'
            break
        }
        if ($choice -eq '2') {
            if (-not (Test-Path -LiteralPath $resultDirectory -PathType Container)) {
                Write-Host '结果目录尚不存在，请输入1生成实验结果。' -ForegroundColor Yellow
                continue
            }
            Start-Process -FilePath $resultDirectory
            Write-Host '已打开结果文件夹。图表在figures目录，模型与数值结果在各实验子目录。'
            [void](Read-Host '按回车关闭此窗口')
            exit 0
        }
        Write-Host '请输入0、1或2。' -ForegroundColor Yellow
    }
}

# 绘图是本实验的正式要求，不能因为缺少依赖而静默跳过。
& $python -B -X utf8 -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' matplotlib
if ($LASTEXITCODE -ne 0) {
    Write-Host '首次生成图表，正在安装绘图依赖。'
    & $python -B -X utf8 -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements-plots.txt')
    if ($LASTEXITCODE -ne 0) { throw '绘图依赖安装失败，请检查网络后重试。' }
}
$arguments = @('-B', '-X', 'utf8', (Join-Path $root 'scripts\prepare_lab02.py'))
if ($OutputDir) { $arguments += @('--output-dir', [IO.Path]::GetFullPath($OutputDir)) }
if ($Force) { $arguments += '--force' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw '实验未完整运行，请检查上方具体错误。' }
if ($Interactive) {
    Write-Host ''
    Write-Host '完整实验已完成。' -ForegroundColor Green
    Write-Host ('结果已保存到：' + $resultDirectory)
    Start-Process -FilePath $resultDirectory
    [void](Read-Host '按回车关闭此窗口')
}
