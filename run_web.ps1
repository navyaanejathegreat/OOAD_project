# Starts the JalMarg API (port 8000) and the React app (port 5173), then opens the browser.
# Usage:  .\run_web.ps1      Stop with Ctrl+C.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path "$root\web\node_modules")) {
    Write-Host "Installing web dependencies (first run only)..."
    Push-Location "$root\web"; npm install --no-audit --no-fund; Pop-Location
}

$api = Start-Process python -ArgumentList "-m", "uvicorn", "jalmarg.api:app", "--port", "8000" `
    -WorkingDirectory $root -PassThru -WindowStyle Hidden
try {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:5173"
    Push-Location "$root\web"
    npm run dev
}
finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}
