$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
Set-Location $repo

$pattern = 'v3_[a-zA-Z0-9_]+\.(csv|json|npz|png)'
$matches = Select-String -Path "notebooks\*.ipynb" -Pattern $pattern

if ($matches) {
    Write-Host "Found version-prefixed artifact filenames:" -ForegroundColor Yellow
    foreach ($m in $matches) {
        Write-Host ("{0}:{1}: {2}" -f $m.Path, $m.LineNumber, $m.Line.Trim())
    }
    exit 1
}

Write-Host "Notebook artifact naming check passed (no v3_ filename prefixes)." -ForegroundColor Green
exit 0
