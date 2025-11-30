#!/bin/bash
# 🌙 Moon Dev Marketplace Quick Start Script

echo "🌙 Moon Dev Strategy Marketplace Launcher"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found! Please install Anaconda/Miniconda first."
    exit 1
fi

# Activate environment
echo "📦 Activating tflow environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tflow

# Check Python version
echo "🐍 Python version: $(python --version)"

# Install Flask if needed
echo "📦 Checking Flask installation..."
python -c "import flask" 2>/dev/null || pip install flask

# Start the dashboard
echo ""
echo "🚀 Starting Strategy Marketplace Dashboard..."
echo ""
echo "📌 Access the dashboard at one of these URLs:"
echo "   - http://172.18.154.77:8002 (WSL2 direct)"
echo "   - http://localhost:8002 (if port forwarding is set up)"
echo ""
echo "💡 Tip: If you can't connect, see the QUICKSTART.md for WSL2 instructions"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Run the dashboard
python src/scripts/marketplace_dashboard.py