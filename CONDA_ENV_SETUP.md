# Conda Environment Setup for Moon Dev AI Agents

## New Environment Created

A new conda environment `moondev312` has been created with Python 3.12 to resolve the pandas_ta installation issue.

### Key Features
- **Python 3.12** - Required for pandas_ta 0.4.71b0
- **pandas_ta** - Technical analysis indicators (successfully installed)
- **All core dependencies** - backtesting, ccxt, anthropic, openai, web3, etc.

### How to Use

1. **Activate the new environment:**
   ```bash
   conda activate moondev312
   ```

2. **Run the trading system:**
   ```bash
   python src/main.py
   ```

### Installed Packages
- pandas_ta==0.4.71b0 ✅
- pandas==2.3.3
- numpy==2.2.6
- backtesting==0.6.5
- ccxt==4.5.22
- anthropic==0.75.0
- openai==2.8.1
- groq==0.36.0
- google-generativeai==0.8.5
- web3==7.14.0
- hyperliquid-python-sdk==0.21.0
- scikit-learn==1.7.2
- termcolor, Flask, fastapi, ta, and more...

### Switching Between Environments

```bash
# To use the new environment (recommended)
conda activate moondev312

# To go back to the old environment
conda activate tflow

# To see all environments
conda env list
```

### Testing
Run the test script to verify everything works:
```bash
conda activate moondev312
python test_env_simple.py
```

### Notes
- The old `tflow` environment is still available if needed
- The new `moondev312` environment has all the same packages plus pandas_ta
- This resolves the pandas_ta installation issue you encountered

## Summary
✨ Your new conda environment `moondev312` is ready! It includes Python 3.12 with pandas_ta and all other required packages for the AI trading system.