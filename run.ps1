# Quick Start Script for Credit Card Fraud Detection

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 79 -ForegroundColor Cyan
Write-Host "Credit Card Fraud Detection - Quick Start" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 79 -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python found" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "[2/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Create output directory
Write-Host "[3/4] Creating output directory..." -ForegroundColor Yellow
if (!(Test-Path -Path "output")) {
    New-Item -ItemType Directory -Path "output" | Out-Null
}
Write-Host "✓ Output directory ready" -ForegroundColor Green
Write-Host ""

# Run main script
Write-Host "[4/4] Running fraud detection pipeline..." -ForegroundColor Yellow
Write-Host ""
python main.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 79 -ForegroundColor Cyan
    Write-Host "✓ Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "Check the 'output' folder for results and visualizations." -ForegroundColor Cyan
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 79 -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "ERROR: Pipeline failed!" -ForegroundColor Red
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
}
