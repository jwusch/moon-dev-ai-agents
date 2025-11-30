# Aster Exchange Setup Guide

This guide explains how to set up the Aster exchange integration for the Moon Dev AI Trading System.

## Overview

Aster is a futures DEX that supports both long and short positions. The integration is optional - the system will work without it, but you won't be able to trade on Aster exchange.

## Setup Options

### Option 1: Set Environment Variable (Recommended)

Add the following to your `.env` file:

```bash
# Path to Aster-Dex-Trading-Bots repository
ASTER_BOTS_PATH=/path/to/your/Aster-Dex-Trading-Bots

# Aster API credentials
ASTER_API_KEY=your_api_key_here
ASTER_API_SECRET=your_api_secret_here
```

### Option 2: Clone to Standard Location

Clone the Aster-Dex-Trading-Bots repository to one of these locations:

```bash
# Option A: Home directory
cd ~
git clone [aster-repo-url] Aster-Dex-Trading-Bots

# Option B: Parent directory
cd ..
git clone [aster-repo-url] Aster-Dex-Trading-Bots

# Option C: GitHub folder in home
mkdir -p ~/github
cd ~/github
git clone [aster-repo-url] Aster-Dex-Trading-Bots
```

## Verifying Installation

Run this command to check if Aster is properly configured:

```bash
python -c "import src.nice_funcs_aster; print('Aster setup complete!')"
```

You should see one of these messages:
- ✅ "Aster modules imported successfully!" - Everything is working
- ⚠️ "Continuing without Aster support..." - Aster not found but system will work

## Using Aster in Trading Agent

To use Aster exchange in the trading agent, edit `src/agents/trading_agent.py`:

```python
# Line 84
EXCHANGE = "ASTER"  # Enable Aster exchange
```

## Troubleshooting

### "Failed to import Aster modules"
- Check that Aster-Dex-Trading-Bots exists at the specified path
- Verify the ASTER_BOTS_PATH environment variable is set correctly
- Ensure you have the required dependencies installed in Aster-Dex-Trading-Bots

### "ASTER API keys not found"
- Add ASTER_API_KEY and ASTER_API_SECRET to your .env file
- Get these credentials from your Aster exchange account

### Trading Agent Still Using Wrong Exchange
- Make sure to restart the trading agent after changing EXCHANGE setting
- Check that EXCHANGE = "ASTER" in trading_agent.py (line 84)

## Running Without Aster

The system is designed to work without Aster. If you don't need Aster exchange:
1. Simply ignore the warning messages about Aster modules
2. Use HYPERLIQUID or SOLANA as your exchange instead
3. The system will continue to function normally