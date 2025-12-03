"""
🛡️ Enhanced Symbol Validator - Multi-Source Symbol Validation 🛡️

Eliminates single point of failure by validating symbols through multiple data sources:
1. Primary: yfinance validation
2. Secondary: Alpha Vantage API  
3. Tertiary: SwarmAgent web research consensus
4. Intelligent retry: Temporary vs permanent invalid classification

Created as part of AEGS Enhancement Phase 2
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.swarm_agent import SwarmAgent
from agents.invalid_symbol_tracker import InvalidSymbolTracker

class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    TEMPORARY_FAILURE = "temporary_failure"
    NEEDS_RETRY = "needs_retry"
    UNKNOWN = "unknown"

@dataclass
class ValidationResult:
    symbol: str
    status: ValidationStatus
    confidence: float  # 0-100
    sources: List[str]  # Which sources validated it
    errors: List[str]   # Any errors encountered
    last_checked: datetime
    retry_count: int = 0
    
class EnhancedSymbolValidator:
    """
    Multi-source symbol validator with intelligent retry logic
    """
    
    def __init__(self):
        self.name = "Enhanced Symbol Validator"
        self.validation_cache = {}
        self.invalid_tracker = InvalidSymbolTracker()
        
        # Configuration
        self.cache_ttl_hours = 24  # Cache valid symbols for 24 hours
        self.invalid_cache_days = 7  # Cache invalid symbols for 7 days
        self.max_retries = 3
        self.timeout_seconds = 30
        
        # Initialize SwarmAgent for web research (optional)
        try:
            self.swarm_agent = SwarmAgent()
            print("✅ SwarmAgent initialized for symbol web research")
        except Exception as e:
            print(f"⚠️ SwarmAgent initialization failed: {e}")
            self.swarm_agent = None
        
        # Alpha Vantage configuration (add your key to .env)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if self.alpha_vantage_key:
            print("✅ Alpha Vantage API key found")
        else:
            print("⚠️ No Alpha Vantage API key found (add ALPHA_VANTAGE_API_KEY to .env)")
        
    def validate_symbol(self, symbol: str, force_refresh: bool = False) -> ValidationResult:
        """
        Validate a symbol using multiple data sources with consensus scoring
        """
        symbol = symbol.strip().upper()
        
        # Check cache first (unless force refresh)
        if not force_refresh and symbol in self.validation_cache:
            cached = self.validation_cache[symbol]
            if self._is_cache_valid(cached):
                return cached
        
        print(f"🔍 Validating symbol: {symbol}")
        
        # Initialize result
        result = ValidationResult(
            symbol=symbol,
            status=ValidationStatus.UNKNOWN,
            confidence=0.0,
            sources=[],
            errors=[],
            last_checked=datetime.now()
        )
        
        # Source 1: yfinance validation
        yf_result = self._validate_yfinance(symbol)
        
        # Source 2: Alpha Vantage validation (if available)
        av_result = self._validate_alpha_vantage(symbol)
        
        # Source 3: SwarmAgent web research (if available)
        swarm_result = self._validate_swarm_research(symbol)
        
        # Combine results using consensus scoring
        result = self._calculate_consensus(symbol, [yf_result, av_result, swarm_result])
        
        # Cache the result
        self.validation_cache[symbol] = result
        
        # Update invalid symbol tracker if needed
        if result.status == ValidationStatus.INVALID:
            self.invalid_tracker.mark_invalid(
                symbol, 
                f"Multi-source validation failed: {'; '.join(result.errors)}"
            )
        
        return result
    
    def _validate_yfinance(self, symbol: str) -> Tuple[bool, str, List[str]]:
        """Validate using yfinance API"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try multiple methods to validate
            info = ticker.info
            hist = ticker.history(period="5d")
            
            # Check if we got valid data
            if len(hist) > 0 and info.get('symbol'):
                return True, "yfinance", []
            else:
                return False, "yfinance", ["No price data available"]
                
        except Exception as e:
            error_msg = str(e)
            
            # Classify error type
            if "delisted" in error_msg.lower():
                return False, "yfinance", [f"Delisted: {error_msg}"]
            elif "not found" in error_msg.lower():
                return False, "yfinance", [f"Not found: {error_msg}"]
            else:
                # Might be temporary network error
                return False, "yfinance", [f"Temporary error: {error_msg}"]
    
    def _validate_alpha_vantage(self, symbol: str) -> Tuple[bool, str, List[str]]:
        """Validate using Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return False, "alpha_vantage", ["API key not configured"]
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            data = response.json()
            
            # Check for valid response
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                if quote.get('01. symbol') == symbol:
                    return True, "alpha_vantage", []
            
            # Check for error messages
            if 'Error Message' in data:
                return False, "alpha_vantage", [data['Error Message']]
            elif 'Note' in data:
                return False, "alpha_vantage", ["API limit reached"]
            else:
                return False, "alpha_vantage", ["No data returned"]
                
        except Exception as e:
            return False, "alpha_vantage", [f"API error: {str(e)}"]
    
    def _validate_swarm_research(self, symbol: str) -> Tuple[bool, str, List[str]]:
        """Validate using SwarmAgent web research"""
        if not self.swarm_agent:
            return False, "swarm_research", ["SwarmAgent not available"]
        
        try:
            prompt = f"""Research the stock symbol {symbol} and determine if it's a valid, actively traded ticker.

Please check:
1. Is {symbol} a legitimate stock ticker?
2. Is it currently listed on a major exchange (NYSE, NASDAQ, etc.)?
3. Is it actively trading (not delisted or suspended)?
4. What company does it represent?

Return a JSON response:
{{"valid": true/false, "company": "company name", "exchange": "exchange name", "reason": "explanation"}}"""

            swarm_result = self.swarm_agent.query(prompt)
            
            if swarm_result and swarm_result.get('consensus_summary'):
                # Parse consensus for validity
                consensus = swarm_result['consensus_summary'].lower()
                
                # Look for positive indicators
                valid_indicators = ['valid', 'legitimate', 'actively trading', 'listed', 'real company']
                invalid_indicators = ['invalid', 'delisted', 'not found', 'fake', 'does not exist']
                
                valid_score = sum(1 for indicator in valid_indicators if indicator in consensus)
                invalid_score = sum(1 for indicator in invalid_indicators if indicator in consensus)
                
                if valid_score > invalid_score:
                    return True, "swarm_research", []
                else:
                    return False, "swarm_research", [f"Consensus suggests invalid: {consensus[:100]}"]
            
            return False, "swarm_research", ["No consensus reached"]
            
        except Exception as e:
            return False, "swarm_research", [f"Research failed: {str(e)}"]
    
    def _calculate_consensus(self, symbol: str, results: List[Tuple[bool, str, List[str]]]) -> ValidationResult:
        """Calculate consensus from multiple validation sources"""
        
        valid_sources = []
        all_errors = []
        
        for is_valid, source, errors in results:
            if is_valid:
                valid_sources.append(source)
            else:
                all_errors.extend([f"{source}: {err}" for err in errors])
        
        # Calculate confidence based on number of sources that agree
        total_sources = len([r for r in results if r[1] != ""])  # Exclude empty sources
        valid_count = len(valid_sources)
        
        if total_sources == 0:
            confidence = 0.0
            status = ValidationStatus.UNKNOWN
        else:
            confidence = (valid_count / total_sources) * 100
            
            # Determine status based on consensus
            if valid_count >= 2:  # At least 2 sources agree it's valid
                status = ValidationStatus.VALID
            elif valid_count == 0:  # All sources say it's invalid
                status = ValidationStatus.INVALID
            else:  # Mixed results, needs more investigation
                status = ValidationStatus.NEEDS_RETRY
        
        return ValidationResult(
            symbol=symbol,
            status=status,
            confidence=confidence,
            sources=valid_sources,
            errors=all_errors,
            last_checked=datetime.now()
        )
    
    def _is_cache_valid(self, cached_result: ValidationResult) -> bool:
        """Check if cached result is still valid"""
        age = datetime.now() - cached_result.last_checked
        
        if cached_result.status == ValidationStatus.VALID:
            return age.total_seconds() < (self.cache_ttl_hours * 3600)
        elif cached_result.status == ValidationStatus.INVALID:
            return age.total_seconds() < (self.invalid_cache_days * 24 * 3600)
        else:
            # Always re-check uncertain results
            return False
    
    def validate_batch(self, symbols: List[str], max_concurrent: int = 5) -> Dict[str, ValidationResult]:
        """Validate multiple symbols with rate limiting"""
        results = {}
        
        print(f"🔍 Batch validating {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if i > 0 and i % max_concurrent == 0:
                time.sleep(1)  # Rate limiting
            
            result = self.validate_symbol(symbol)
            results[symbol] = result
            
            status_emoji = "✅" if result.status == ValidationStatus.VALID else "❌"
            print(f"   {status_emoji} {symbol}: {result.status.value} ({result.confidence:.1f}% confidence)")
        
        return results
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        stats = {
            'total_cached': len(self.validation_cache),
            'valid': 0,
            'invalid': 0,
            'uncertain': 0
        }
        
        for result in self.validation_cache.values():
            if result.status == ValidationStatus.VALID:
                stats['valid'] += 1
            elif result.status == ValidationStatus.INVALID:
                stats['invalid'] += 1
            else:
                stats['uncertain'] += 1
        
        return stats

# Standalone testing
if __name__ == "__main__":
    validator = EnhancedSymbolValidator()
    
    # Test with known symbols
    test_symbols = ['AAPL', 'GOOGL', 'INVALID123', 'GME', 'BBIG', 'TSLA']
    
    print("🧪 Testing Enhanced Symbol Validator...")
    results = validator.validate_batch(test_symbols)
    
    print("\n📊 Validation Results:")
    for symbol, result in results.items():
        print(f"{symbol}: {result.status.value} ({result.confidence:.1f}%)")
        if result.sources:
            print(f"   Sources: {', '.join(result.sources)}")
        if result.errors:
            print(f"   Errors: {'; '.join(result.errors[:2])}")
    
    print(f"\n📈 Stats: {validator.get_validation_stats()}")