param([switch]$Quick, [switch]$NoPlots)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Push-Location -LiteralPath $root
try {
    $arguments=@('-B','-X','utf8','main.py','lab02','all')
    if($Quick){$arguments+=@('--quick','--output-dir',(Join-Path $root 'outputs\qa\smoke'))}
    if($NoPlots){$arguments+='--no-plots'}
    if (Get-Command rtk -ErrorAction SilentlyContinue) { rtk proxy python @arguments } else { & python @arguments }
    if($LASTEXITCODE -ne 0){throw 'Lab 02 execution failed'}
} finally {Pop-Location}
