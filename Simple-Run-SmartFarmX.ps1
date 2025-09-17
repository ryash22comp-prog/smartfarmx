# Simple SmartFarmX Runner
param()

Write-Host "SmartFarmX Setup and Run" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Check for disease_description.json in backend/data/
$targetPath = "backend\data\disease_description.json"
$sourcePaths = @("disease_description.json", "data\disease_description.json", "..\disease_description.json")

# Create backend/data directory if needed
$targetDir = Split-Path $targetPath -Parent
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    Write-Host "✓ Created directory: $targetDir" -ForegroundColor Green
}

# Handle disease_description.json
if (Test-Path $targetPath) {
    Write-Host "✓ disease_description.json found at $targetPath" -ForegroundColor Green
} else {
    $sourceFound = $false
    foreach ($source in $sourcePaths) {
        if (Test-Path $source) {
            Copy-Item $source $targetPath -Force
            Write-Host "✓ Copied disease_description.json from $source" -ForegroundColor Green
            $sourceFound = $true
            break
        }
    }
    
    if (-not $sourceFound) {
        Write-Host "⚠ Creating placeholder disease_description.json" -ForegroundColor Yellow
        '{"Healthy":{"Description":"Plant appears healthy"}}' | Out-File $targetPath -Encoding UTF8
    }
}

# Copy to application expected location
if (-not (Test-Path "..\disease_description.json")) {
    Copy-Item $targetPath "..\disease_description.json" -Force
    Write-Host "✓ Copied to application expected location" -ForegroundColor Green
}

# Install dependencies and run
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install fastapi uvicorn tensorflow pandas numpy scikit-learn pillow python-multipart

Write-Host "Starting application..." -ForegroundColor Green
Set-Location smartfarmx_backend
python main.py
