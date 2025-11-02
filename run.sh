#!/bin/bash

# Quick Start Script for Credit Card Fraud Detection (Linux/Mac)

echo "================================================================================"
echo "Credit Card Fraud Detection - Quick Start"
echo "================================================================================"
echo ""

# Check Python
echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python3 not found! Please install Python 3.8+"
    exit 1
fi
python3 --version
echo "✓ Python found"
echo ""

# Install dependencies
echo "[2/4] Installing dependencies..."
pip3 install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies!"
    exit 1
fi
echo "✓ Dependencies installed"
echo ""

# Create output directory
echo "[3/4] Creating output directory..."
mkdir -p output
echo "✓ Output directory ready"
echo ""

# Run main script
echo "[4/4] Running fraud detection pipeline..."
echo ""
python3 main.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ Pipeline completed successfully!"
    echo "Check the 'output' folder for results and visualizations."
    echo "================================================================================"
else
    echo ""
    echo "ERROR: Pipeline failed!"
    echo "Please check the error messages above."
    exit 1
fi