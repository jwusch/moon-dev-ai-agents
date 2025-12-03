#!/bin/bash

# 🌙 TradingView Authenticated Setup Script

echo "🌙 Moon Dev TradingView Authenticated API Setup"
echo "=============================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Install server dependencies
echo ""
echo "📦 Installing TradingView server dependencies..."
cd tradingview-server

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found in tradingview-server/"
    exit 1
fi

# Install dependencies
npm install

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit tradingview-server/.env with your TradingView credentials"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit tradingview-server/.env with your TradingView credentials:"
echo "   TV_USERNAME=your_username"
echo "   TV_PASSWORD=your_password"
echo ""
echo "2. Start the TradingView server:"
echo "   cd tradingview-server"
echo "   npm start"
echo ""
echo "3. In another terminal, test the Python client:"
echo "   python -m src.agents.tradingview_auth_client"
echo ""
echo "🚀 The authenticated API provides:"
echo "   - No rate limiting issues"
echo "   - Access to private indicators"
echo "   - Real-time streaming data"
echo "   - Batch operations"
echo ""