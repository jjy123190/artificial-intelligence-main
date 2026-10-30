param([string]$OutputDir, [switch]$Force, [string]$PythonPath)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$root = [IO.Path]::GetFullPath($PSScriptRoot)
. (Join-Path $root 'scripts\web_python.ps1')
$python = Resolve-WebPython -Root $root -PythonPath $PythonPath

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
