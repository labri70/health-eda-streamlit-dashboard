param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $AppDir "..\.venv\Scripts\python.exe"

Set-Location $AppDir
& $Python -m streamlit run service_app.py --server.port $Port --server.headless true
