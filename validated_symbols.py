"""
Enhanced symbol validation for AEGS
Generated: 2025-12-02 02:12:47
"""

from enhanced_data_fetcher import EnhancedDataFetcher

# Known good symbols after validation
ACTIVE_SYMBOLS = [
  "REVG",
  "AXP",
  "RIDE-USD",
  "GNUS-USD",
  "PGR",
  "VIEWF",
  "BBWI",
  "BBBY",
  "9N1.F",
  "DJT",
  "RUM",
  "GREE",
  "OPAD",
  "TMC",
  "CEIGALL.BO",
  "XELA"
]

# Delisted symbols to avoid
DELISTED_SYMBOLS = [
  "MULN",
  "FFIE",
  "APRN",
  "RDBX",
  "IRNT"
]

# Ticker mappings
TICKER_MAPPINGS = {
  "REV": "REVG",
  "EXPR": "AXP",
  "RIDE": "RIDE-USD",
  "GNUS": "GNUS-USD",
  "PROG": "PGR",
  "VIEW": "VIEWF",
  "BODY": "BBWI",
  "WEBR": "9N1.F",
  "DWAC": "DJT",
  "CFVI": "RUM",
  "SPRT": "GREE",
  "CEI": "CEIGALL.BO"
}

def get_valid_symbol(symbol):
    """Get valid trading symbol"""
    if symbol in TICKER_MAPPINGS:
        return TICKER_MAPPINGS[symbol]
    if symbol in DELISTED_SYMBOLS:
        return None
    return symbol

def filter_valid_symbols(symbols):
    """Filter to only valid symbols"""
    valid = []
    for symbol in symbols:
        valid_symbol = get_valid_symbol(symbol)
        if valid_symbol:
            valid.append(valid_symbol)
    return valid
