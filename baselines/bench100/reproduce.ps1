param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Python,
    [string]$Output,
    [string]$Cache,
    [int]$Workers = 6
)
$ErrorActionPreference = 'Stop'
$finalDir = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
if (!$Python) { $Python = Join-Path $finalDir 'envs/watermark/python.exe' }
if (!$Output) { $Output = Join-Path $finalDir 'rerun/results' }
if (!$Cache) { $Cache = Join-Path $finalDir 'rerun/cache' }
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:TEMP = Join-Path $finalDir 'cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

function Invoke-BenchmarkPython {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Benchmark step failed (exit $LASTEXITCODE): $args" }
}
$smokeOut = Join-Path $Cache 'smoke-report'
$smokeCache = Join-Path $Cache 'smoke-checkpoints'
$fullCache = Join-Path $Cache 'full'
Invoke-BenchmarkPython "$PSScriptRoot/run.py" --manifest $Manifest --limit 1 --workers 1 --output $smokeOut --cache $smokeCache
Invoke-BenchmarkPython "$PSScriptRoot/verify_smoke.py" --manifest $Manifest --output $smokeOut --cache $smokeCache
Invoke-BenchmarkPython "$PSScriptRoot/run.py" --manifest $Manifest --workers $Workers --output $Output --cache $fullCache
Invoke-BenchmarkPython "$PSScriptRoot/analyze.py" --output $Output --cache $fullCache
Copy-Item -LiteralPath "$smokeOut/smoke_verification.json" -Destination $Output -Force
