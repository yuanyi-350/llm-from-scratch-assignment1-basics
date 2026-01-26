$ErrorActionPreference = "Stop"


$OutputFile = "cs336-spring2025-assignment-1-submission.zip"
$OutputFilePath = Join-Path (Get-Location) $OutputFile

if (Test-Path $OutputFilePath) {
    Remove-Item -Path $OutputFilePath -Force
    Write-Host "Removed old submission file." -ForegroundColor Gray
}

$ExclusionPatterns = @(
    '.venv', 'data', 'egg-info', 'mypy_cache', 'pytest_cache', 'build', 
    'ipynb_checkpoints', '__pycache__', 
    '^\.git', '\.venv', 
    'tests[\\/]fixtures', 'tests[\\/]_snapshots',
    '\.pkl$', '\.pickle$', '\.txt$', '\.log$', '\.json$', 
    '\.out$', '\.err$', '\.pt$', '\.pth$', '\.npy$', '\.npz$',
    '^\.'
)
$RegexPattern = ($ExclusionPatterns -join '|')

$BasePath = (Get-Location).Path

Write-Host "Scanning files..." -ForegroundColor Cyan
$FilesToZip = Get-ChildItem -Path $BasePath -Recurse -File | Where-Object {
    $relPath = $_.FullName.Substring($BasePath.Length + 1)
    
    $_.Name -ne $OutputFile -and 
    ($relPath -notmatch $RegexPattern)
}

Write-Host "Compressing $($FilesToZip.Count) files into $OutputFile..." -ForegroundColor Cyan

Add-Type -AssemblyName System.IO.Compression.FileSystem

$ZipArchiveMode = [System.IO.Compression.ZipArchiveMode]::Create
$ZipFile = [System.IO.Compression.ZipFile]::Open($OutputFilePath, $ZipArchiveMode)

try {
    foreach ($File in $FilesToZip) {
        $RelativePath = $File.FullName.Substring($BasePath.Length + 1)
        
        $EntryName = $RelativePath -replace '\\', '/'
        
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($ZipFile, $File.FullName, $EntryName)
    }
}
catch {
    Write-Error "Failed to compress files: $_"
    $ZipFile.Dispose()
    exit 1
}

$ZipFile.Dispose()

Write-Host "Success! All files compressed into $OutputFile" -ForegroundColor Green