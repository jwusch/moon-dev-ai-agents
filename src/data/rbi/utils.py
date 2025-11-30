"""
🌙 Moon Dev's Data Path Utilities
Handles dynamic path resolution for backtest data files
"""
import os
import sys

def get_data_file_path(filename='BTC-USD-15m.csv'):
    """
    Find the correct path to a data file by searching in multiple locations.
    
    Args:
        filename: Name of the data file (default: BTC-USD-15m.csv)
    
    Returns:
        Full path to the data file
        
    Raises:
        FileNotFoundError if the file cannot be found
    """
    # Get the directory of the calling script
    if hasattr(sys.modules['__main__'], '__file__'):
        script_path = os.path.abspath(sys.modules['__main__'].__file__)
        script_dir = os.path.dirname(script_path)
    else:
        script_dir = os.getcwd()
    
    # List of potential paths to check
    potential_paths = [
        # Try relative to script location
        os.path.join(script_dir, filename),
        os.path.join(script_dir, '..', filename),
        os.path.join(script_dir, '..', '..', filename),
        os.path.join(script_dir, '..', '..', 'rbi', filename),
        
        # Try from project root
        os.path.join(script_dir, '..', '..', '..', '..', '..', 'src', 'data', 'rbi', filename),
        os.path.join(script_dir, '..', '..', '..', '..', '..', 'src', 'data', filename),
        
        # Try common locations
        os.path.join('/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data/rbi', filename),
        os.path.join('/mnt/c/Users/jwusc/moon-dev-ai-agents/src/data', filename),
    ]
    
    # Check each potential path
    for path in potential_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            print(f"✅ Found data file at: {normalized_path}")
            return normalized_path
    
    # If not found, list directories we checked
    checked_dirs = set()
    for path in potential_paths:
        dir_path = os.path.dirname(os.path.normpath(path))
        if os.path.exists(dir_path):
            checked_dirs.add(dir_path)
    
    error_msg = f"❌ Could not find {filename} in any of the expected locations.\n"
    error_msg += "Checked directories:\n"
    for dir_path in checked_dirs:
        error_msg += f"  - {dir_path}\n"
        # List files in directory to help debug
        try:
            files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            if files:
                error_msg += f"    Found CSV files: {', '.join(files[:3])}\n"
        except:
            pass
    
    raise FileNotFoundError(error_msg)


def prepare_backtest_data(data):
    """
    Prepare data for backtesting by standardizing column names.
    
    Args:
        data: pandas DataFrame with OHLCV data
        
    Returns:
        Prepared DataFrame with standardized columns
    """
    # Clean and prepare columns
    data.columns = data.columns.str.strip()
    
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Standardize column names
    column_mapping = {
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'datetime': 'Datetime',
        'date': 'Date'
    }
    
    # Apply case-insensitive mapping
    for col in data.columns:
        col_lower = col.lower()
        if col_lower in column_mapping:
            data.rename(columns={col: column_mapping[col_lower]}, inplace=True)
    
    # Set datetime index if available
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
    elif 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
    return data