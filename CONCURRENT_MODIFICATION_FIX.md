# Concurrent Modification Error Fix

## Issue
When AEGS tried to add NEM to invalid symbols, it encountered:
```
🚫 Added NEM to invalid symbols: Backtest error: Extra data: line 311 column 2 (char 8972)
⚠️ Error saving invalid symbols: dictionary changed size during iteration
```

## Root Cause
The `save()` method in `InvalidSymbolTracker` was iterating over the `self.invalid_symbols` dictionary while Python's `json.dump()` was serializing it. If another thread/process modified the dictionary during serialization, it would raise a RuntimeError.

## Solution Applied
Modified `src/agents/invalid_symbol_tracker.py` to:

1. **Create a copy** of the dictionary before serialization
2. **Handle RuntimeError** specifically for concurrent modifications
3. **Retry automatically** if concurrent modification is detected

### Code Changes
```python
def save(self):
    """Save invalid symbols to file"""
    # Create a copy to avoid concurrent modification errors
    invalid_symbols_copy = dict(self.invalid_symbols)
    
    data = {
        'invalid_symbols': invalid_symbols_copy,
        'metadata': {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_invalid': len(invalid_symbols_copy)
        }
    }
    
    try:
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            print(f"⚠️ Concurrent modification detected, retrying...")
            # Retry with another copy
            self.save()
        else:
            print(f"⚠️ Error saving invalid symbols: {e}")
    except Exception as e:
        print(f"⚠️ Error saving invalid symbols: {e}")
```

## Additional Notes

### About the JSON Error
- NEM (like many other symbols) is still experiencing the "Extra data: line 311" JSON parsing error
- This is the same cached error affecting 79+ symbols
- The fix applied earlier (`fix_aegs_json_error.py`) can clear these symbols

### Thread-Safe Version
A fully thread-safe version was also created at:
`src/agents/invalid_symbol_tracker_threadsafe.py`

This version includes:
- Threading locks for all operations
- Atomic file writes using temporary files
- Safe concurrent access from multiple threads/processes

## Status
✅ The concurrent modification error should now be resolved. The tracker will automatically retry if it detects concurrent access during save operations.