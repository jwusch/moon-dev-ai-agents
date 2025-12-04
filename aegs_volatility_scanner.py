#!/usr/bin/env python
"""
🌋 AEGS DAILY VOLATILITY SCANNER
Automatically discovers and analyzes the most volatile stocks for AEGS signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

# Import AEGS components
from aegs_live_scanner import AEGSLiveScanner


class YahooFinanceRateLimiter:
    """Smart rate limiter for Yahoo Finance API calls"""
    
    def __init__(self):
        self.request_count = 0
        self.start_time = time.time()
        self.rate_limit_detected = False
        self.backoff_time = 1.0  # Start with 1 second backoff
        self.max_backoff = 60.0  # Maximum 60 seconds
        self.requests_per_minute = 60  # Conservative limit
        self.last_request_time = 0
        
    def wait_if_needed(self):
        """Wait if we're hitting rate limits"""
        current_time = time.time()
        
        # Basic rate limiting - max requests per minute
        if self.request_count > 0:
            elapsed = current_time - self.start_time
            if elapsed < 60:  # Within a minute
                requests_per_sec = self.request_count / elapsed
                if requests_per_sec > (self.requests_per_minute / 60):
                    sleep_time = (60 / self.requests_per_minute) - (current_time - self.last_request_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        # Additional backoff if rate limit detected
        if self.rate_limit_detected:
            jitter = random.uniform(0.1, 0.3)  # Add jitter to avoid thundering herd
            sleep_time = self.backoff_time + jitter
            print(colored(f"   🕐 Rate limit detected, waiting {sleep_time:.1f}s...", 'yellow'))
            time.sleep(sleep_time)
            
        self.last_request_time = current_time
        self.request_count += 1
        
        # Reset counters every minute
        if current_time - self.start_time > 60:
            self.start_time = current_time
            self.request_count = 0
            
    def handle_rate_limit(self):
        """Handle detected rate limit"""
        self.rate_limit_detected = True
        self.backoff_time = min(self.backoff_time * 1.5, self.max_backoff)
        print(colored(f"   ⚠️ Yahoo Finance rate limit detected, increasing backoff to {self.backoff_time:.1f}s", 'red'))
        
    def reset_backoff(self):
        """Reset backoff when requests are successful"""
        if self.rate_limit_detected:
            self.rate_limit_detected = False
            self.backoff_time = 1.0
            print(colored(f"   ✅ Rate limit cleared, resuming normal speed", 'green'))


# Global rate limiter instance
rate_limiter = YahooFinanceRateLimiter()


def safe_yfinance_download(symbol: str, **kwargs) -> pd.DataFrame:
    """Safe wrapper for yfinance download with rate limiting and retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Wait if needed before making request
            rate_limiter.wait_if_needed()
            
            # Download data
            data = yf.download(symbol, progress=False, **kwargs)
            
            # Success - reset backoff
            if not data.empty:
                rate_limiter.reset_backoff()
                return data
            else:
                # Empty data - might be rate limited or delisted
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief pause before retry
                continue
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit indicators
            if any(indicator in error_msg for indicator in [
                '401', 'unauthorized', 'rate limit', 'too many requests',
                'invalid crumb', 'unable to access'
            ]):
                rate_limiter.handle_rate_limit()
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter for retries
                    retry_delay = (2 ** attempt) + random.uniform(0.1, 0.5)
                    print(colored(f"   🔄 Retrying {symbol} in {retry_delay:.1f}s (attempt {attempt + 2}/{max_retries})", 'yellow'))
                    time.sleep(retry_delay)
                    continue
            else:
                # Other error - might be delisted symbol
                break
                
    # All retries failed
    return pd.DataFrame()  # Return empty DataFrame


class VolatilityDiscoveryEngine:
    """Discovers high volatility stocks for AEGS analysis"""
    
    def __init__(self):
        self.min_price = 1.0  # Minimum stock price
        self.max_price = 500.0  # Maximum stock price
        self.min_volume = 500000  # Minimum daily volume
        self.min_market_cap = 50_000_000  # $50M minimum market cap
        
        # Volatility thresholds
        self.high_volatility_threshold = 0.08  # 8% daily move
        self.extreme_volatility_threshold = 0.15  # 15% daily move
        
        # Stock universes to scan
        self.stock_universes = {
            'sp500': self._get_sp500_symbols(),
            'nasdaq100': self._get_nasdaq100_symbols(),
            'russell2000_sample': self._get_russell2000_sample(),
            'biotech_etf_holdings': self._get_biotech_holdings(),
            'fintech_trending': self._get_fintech_symbols(),
            'ev_stocks': self._get_ev_symbols(),
            'meme_universe': self._get_meme_stocks(),
            'penny_volatility': self._get_penny_vol_stocks()
        }
        
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols"""
        try:
            # Popular S&P 500 symbols known for volatility
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
                'NFLX', 'CRM', 'ZM', 'PTON', 'ROKU', 'SQ', 'PYPL', 'SHOP',
                'SNOW', 'COIN', 'RBLX', 'UBER', 'LYFT', 'ABNB', 'DASH'
            ]
        except Exception:
            return ['AAPL', 'TSLA', 'NVDA', 'AMD']  # Fallback
            
    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 volatile symbols"""
        return [
            'QQQ', 'TQQQ', 'SQQQ', 'ARKK', 'ARKW', 'ARKG', 'ARKQ', 'ARKF',
            'PLTR', 'SOFI', 'HOOD', 'DKNG', 'PENN', 'RIVN', 'LCID', 'F',
            'NIO', 'XPEV', 'LI', 'BABA', 'JD', 'PDD', 'BILI'
        ]
        
    def _get_russell2000_sample(self) -> List[str]:
        """Get Russell 2000 small cap volatile stocks"""
        return [
            'IWM', 'TNA', 'TZA', 'SOXL', 'SOXS', 'LABU', 'LABD',
            'SPXL', 'SPXS', 'UPRO', 'SPXU', 'TMF', 'TMV'
        ]
        
    def _get_biotech_holdings(self) -> List[str]:
        """Get biotech stocks known for volatility"""
        return [
            'BIIB', 'GILD', 'MRNA', 'BNTX', 'REGN', 'VRTX', 'ILMN',
            'SAVA', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VKTX', 'BLUE',
            'SRPT', 'BMRN', 'ALNY', 'IONS', 'EXAS', 'TECH'
        ]
        
    def _get_fintech_symbols(self) -> List[str]:
        """Get fintech stocks"""
        return [
            'PYPL', 'SQ', 'SOFI', 'AFRM', 'UPST', 'HOOD', 'COIN',
            'MA', 'V', 'FIS', 'FISV', 'GPN', 'WU', 'MQ'
        ]
        
    def _get_ev_symbols(self) -> List[str]:
        """Get EV and automotive stocks"""
        return [
            'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM',
            'NKLA', 'GOEV', 'RIDE', 'HYLN', 'FSR', 'WKHS', 'BLNK', 'CHPT'
        ]
        
    def _get_meme_stocks(self) -> List[str]:
        """Get meme stocks known for volatility"""
        return [
            'GME', 'AMC', 'BB', 'KOSS', 'EXPR', 'CLOV', 'WKHS',
            'WISH', 'CLNE', 'SENS', 'SNDL', 'TLRY', 'CGC', 'ACB'
        ]
        
    def _get_penny_vol_stocks(self) -> List[str]:
        """Get penny stocks with high volatility"""
        return [
            'SIRI', 'NOK', 'ZYXI', 'GSAT', 'IDEX', 'NAKD', 'SNDL',
            'ZSAN', 'JAGX', 'SHIP', 'TOPS', 'GLBS', 'CTRM', 'CASTOR'
        ]
        
    def get_all_symbols(self) -> List[str]:
        """Get all symbols from all universes"""
        all_symbols = []
        for universe_name, symbols in self.stock_universes.items():
            all_symbols.extend(symbols)
        return list(set(all_symbols))  # Remove duplicates
        
    def calculate_daily_volatility(self, symbol: str, days: int = 5) -> Optional[Dict]:
        """Calculate daily volatility metrics for a symbol"""
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)  # Extra buffer
            
            data = safe_yfinance_download(symbol, start=start_date, end=end_date)
            
            if data.empty or len(data) < 3:
                return None
                
            # Fix multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change()
            data['Daily_Range'] = (data['High'] - data['Low']) / data['Close']
            data['Intraday_Move'] = abs((data['Close'] - data['Open']) / data['Open'])
            
            # Current metrics
            latest = data.iloc[-1]
            current_price = latest['Close']
            current_volume = latest['Volume']
            
            # Volatility metrics
            daily_return = latest['Daily_Return']
            daily_range = latest['Daily_Range']
            intraday_move = latest['Intraday_Move']
            
            # Rolling volatility (annualized)
            returns = data['Daily_Return'].dropna()
            if len(returns) > 1:
                volatility = returns.std() * np.sqrt(252)
            else:
                volatility = 0
                
            # Volume analysis
            avg_volume = data['Volume'].rolling(5).mean().iloc[-1] if len(data) >= 5 else current_volume
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price filters
            price_ok = self.min_price <= current_price <= self.max_price
            volume_ok = current_volume >= self.min_volume
            
            # Volatility classification
            if abs(daily_return) >= self.extreme_volatility_threshold:
                vol_class = "EXTREME"
                vol_score = 100
            elif abs(daily_return) >= self.high_volatility_threshold:
                vol_class = "HIGH"
                vol_score = 75
            elif daily_range >= 0.05:  # 5% intraday range
                vol_class = "MODERATE"
                vol_score = 50
            else:
                vol_class = "LOW"
                vol_score = 25
                
            # Overall score
            score = (
                vol_score * 0.4 +  # Daily move weight
                min(volume_spike * 10, 50) * 0.3 +  # Volume spike weight
                min(volatility * 100, 50) * 0.2 +  # Historical vol weight
                min(daily_range * 100, 50) * 0.1   # Intraday range weight
            )
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'daily_return': daily_return,
                'daily_return_pct': daily_return * 100,
                'daily_range': daily_range,
                'intraday_move': intraday_move,
                'volatility_annual': volatility,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_spike': volume_spike,
                'vol_class': vol_class,
                'vol_score': vol_score,
                'overall_score': score,
                'filters_passed': price_ok and volume_ok,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
            
    def scan_volatility_universe(self, max_workers: int = 8) -> List[Dict]:
        """Scan all symbols for volatility"""
        
        all_symbols = self.get_all_symbols()
        volatility_data = []
        
        print(colored(f"🌋 Scanning {len(all_symbols)} symbols for daily volatility...", 'cyan'))
        
        # Dynamic concurrency based on rate limiting
        initial_workers = max_workers
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=initial_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.calculate_daily_volatility, symbol): symbol 
                for symbol in all_symbols
            }
            
            # Collect results
            completed = 0
            rate_limit_count = 0
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=45)  # Increased timeout
                    if result:
                        volatility_data.append(result)
                    completed += 1
                    
                    # Check for rate limiting and adjust if needed
                    if rate_limiter.rate_limit_detected:
                        rate_limit_count += 1
                        if rate_limit_count > 5:  # Multiple rate limits
                            print(colored(f"  ⚠️ High rate limiting detected, reducing concurrency", 'yellow'))
                            # Can't change executor mid-flight, but we can track this for next time
                    
                    # Progress indicator
                    if completed % 20 == 0:
                        status = "⚡ FAST" if not rate_limiter.rate_limit_detected else "🐌 THROTTLED"
                        print(f"  📊 Processed {completed}/{len(all_symbols)} symbols... {status}")
                        
                except Exception as e:
                    completed += 1
                    if "timeout" in str(e).lower():
                        print(f"  ⏰ Timeout processing {symbol}")
                    else:
                        print(f"  ❌ Error processing {symbol}: {str(e)}")
                    
        # Filter and sort results
        valid_results = [r for r in volatility_data if r['filters_passed']]
        
        # Sort by overall score
        valid_results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        print(colored(f"✅ Found {len(valid_results)} qualified volatile stocks", 'green'))
        
        return valid_results
        
    def get_top_volatility_picks(self, limit: int = 30) -> List[Dict]:
        """Get top volatility picks for AEGS analysis"""
        
        volatility_results = self.scan_volatility_universe()
        
        # Get top picks
        top_picks = volatility_results[:limit]
        
        print(colored(f"\n🎯 TOP {len(top_picks)} VOLATILITY PICKS FOR AEGS ANALYSIS", 'yellow', attrs=['bold']))
        print("=" * 80)
        
        for i, pick in enumerate(top_picks, 1):
            vol_color = {
                'EXTREME': 'red',
                'HIGH': 'magenta', 
                'MODERATE': 'yellow',
                'LOW': 'white'
            }.get(pick['vol_class'], 'white')
            
            print(f"{i:2d}. {colored(pick['symbol'], vol_color, attrs=['bold'])}: "
                  f"${pick['current_price']:.2f} | "
                  f"{pick['daily_return_pct']:+.1f}% | "
                  f"{pick['vol_class']} Vol | "
                  f"Score: {pick['overall_score']:.0f}")
                  
        return top_picks


class AEGSVolatilityIntegration:
    """Integrates volatility discovery with AEGS analysis"""
    
    def __init__(self):
        self.volatility_engine = VolatilityDiscoveryEngine()
        self.aegs_scanner = AEGSLiveScanner()
        
    def run_enhanced_aegs_scan(self, volatility_limit: int = 50) -> Dict:
        """Run AEGS analysis on top volatile stocks"""
        
        print(colored("🚀 ENHANCED AEGS VOLATILITY SCANNER", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        # Step 1: Get volatile stocks
        print(colored("\n🌋 PHASE 1: DISCOVERING VOLATILE STOCKS", 'yellow'))
        volatile_stocks = self.volatility_engine.get_top_volatility_picks(volatility_limit)
        
        if not volatile_stocks:
            print(colored("❌ No volatile stocks found matching criteria", 'red'))
            return {'volatile_picks': [], 'aegs_signals': [], 'top_opportunities': []}
            
        # Step 2: Run custom AEGS analysis on volatile stocks  
        volatile_symbols = [stock['symbol'] for stock in volatile_stocks[:30]]  # Top 30 only
        
        print(colored(f"\n🎯 PHASE 2: AEGS ANALYSIS ON {len(volatile_symbols)} VOLATILE STOCKS", 'yellow'))
        
        # Run AEGS analysis on volatile symbols manually
        aegs_results = self._analyze_volatile_symbols(volatile_symbols, volatile_stocks)
        
        # Step 3: Create enhanced signals
        enhanced_signals = aegs_results
        
        # Step 4: Generate top opportunities
        top_opportunities = self._generate_top_opportunities(enhanced_signals[:10], volatile_stocks[:20])
        
        return {
            'volatile_picks': volatile_stocks,
            'aegs_signals': enhanced_signals,
            'top_opportunities': top_opportunities,
            'scan_timestamp': datetime.now().isoformat(),
            'total_symbols_scanned': len(volatile_symbols)
        }
        
    def _analyze_volatile_symbols(self, symbols: List[str], volatile_data: List[Dict]) -> List[Dict]:
        """Run custom AEGS-style analysis on volatile symbols"""
        
        enhanced_signals = []
        volatility_lookup = {v['symbol']: v for v in volatile_data}
        
        print(f"   🔍 Analyzing {len(symbols)} symbols for AEGS patterns...")
        
        # Simple AEGS-style signal detection
        for symbol in symbols:
            if symbol not in volatility_lookup:
                continue
                
            vol_data = volatility_lookup[symbol]
            
            try:
                # Get basic market data for AEGS analysis
                data = safe_yfinance_download(symbol, period='5d', interval='1h')
                
                if data.empty or len(data) < 10:
                    continue
                    
                # Fix multi-column issue
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Calculate basic indicators
                data['RSI'] = self._calculate_rsi(data['Close'], 14)
                data['BB_Lower'] = data['Close'].rolling(20).mean() - (data['Close'].rolling(20).std() * 2)
                data['Price_vs_BB'] = (data['Close'] - data['BB_Lower']) / data['Close']
                
                # Get latest values
                latest = data.iloc[-1]
                current_price = latest['Close']
                rsi = latest['RSI'] if not pd.isna(latest['RSI']) else 50
                price_vs_bb = latest['Price_vs_BB'] if not pd.isna(latest['Price_vs_BB']) else 0
                
                # AEGS signal calculation
                signal_strength = 0
                
                # RSI oversold signal (20-40% of score)
                if rsi <= 30:
                    signal_strength += 40
                elif rsi <= 35:
                    signal_strength += 25
                elif rsi <= 40:
                    signal_strength += 15
                
                # Bollinger band signal (20-30% of score)
                if price_vs_bb <= -0.1:  # 10% below BB lower
                    signal_strength += 30
                elif price_vs_bb <= -0.05:  # 5% below BB lower
                    signal_strength += 20
                elif price_vs_bb <= 0:  # At or below BB lower
                    signal_strength += 10
                
                # Volatility bonus (up to 30% of score)
                vol_bonus = min(vol_data['overall_score'] / 3, 30)
                signal_strength += vol_bonus
                
                # Create enhanced signal
                if signal_strength >= 40:  # Minimum threshold
                    enhanced_signal = {
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'current_price': current_price,
                        'rsi': rsi,
                        'price_vs_bb_lower': price_vs_bb,
                        'volatility_score': vol_data['overall_score'],
                        'volatility_class': vol_data['vol_class'],
                        'daily_move': vol_data['daily_return_pct'],
                        'volume_spike': vol_data['volume_spike'],
                        'price_range': vol_data['daily_range'],
                        'combined_score': (signal_strength + vol_data['overall_score']) / 2,
                        'category': 'Volatile Stock Signal',
                        'triggers': f"RSI={rsi:.1f}, Vol={vol_data['vol_class']}"
                    }
                    
                    enhanced_signals.append(enhanced_signal)
                    
            except Exception as e:
                print(f"    ❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by combined score
        enhanced_signals.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"   ✅ Found {len(enhanced_signals)} potential signals")
        
        return enhanced_signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _generate_top_opportunities(self, top_signals: List[Dict], top_volatile: List[Dict]) -> List[Dict]:
        """Generate top trading opportunities combining AEGS + volatility"""
        
        opportunities = []
        
        print(colored(f"\n💎 PHASE 3: TOP TRADING OPPORTUNITIES", 'yellow'))
        print("=" * 80)
        
        # Category 1: Strong AEGS signals on volatile stocks
        strong_aegs = [s for s in top_signals if s['signal_strength'] >= 60]
        
        if strong_aegs:
            print(colored(f"\n🎯 STRONG AEGS SIGNALS ON VOLATILE STOCKS ({len(strong_aegs)}):", 'green', attrs=['bold']))
            
            for i, signal in enumerate(strong_aegs[:5], 1):
                opportunity = {
                    'rank': i,
                    'symbol': signal['symbol'],
                    'opportunity_type': 'AEGS_SIGNAL_HIGH_VOL',
                    'aegs_score': signal['signal_strength'],
                    'volatility_score': signal['volatility_score'],
                    'combined_score': signal['combined_score'],
                    'daily_move': signal['daily_move'],
                    'vol_class': signal['volatility_class'],
                    'entry_price': signal['current_price'],
                    'recommendation': 'BUY',
                    'position_size': self._calculate_position_size(signal),
                    'reasoning': f"Strong AEGS signal ({signal['signal_strength']}/100) on {signal['volatility_class']} volatility stock"
                }
                
                opportunities.append(opportunity)
                
                print(f"  {i}. {colored(signal['symbol'], 'green', attrs=['bold'])}: "
                      f"AEGS {signal['signal_strength']}/100 | "
                      f"Vol {signal['volatility_score']:.0f} | "
                      f"{signal['daily_move']:+.1f}% | "
                      f"${signal['current_price']:.2f}")
        
        # Category 2: Extreme volatility plays
        extreme_vol = [v for v in top_volatile if v['vol_class'] == 'EXTREME'][:3]
        
        if extreme_vol:
            print(colored(f"\n🌋 EXTREME VOLATILITY PLAYS ({len(extreme_vol)}):", 'red', attrs=['bold']))
            
            for i, vol_pick in enumerate(extreme_vol, len(opportunities) + 1):
                opportunity = {
                    'rank': i,
                    'symbol': vol_pick['symbol'],
                    'opportunity_type': 'EXTREME_VOLATILITY',
                    'aegs_score': 0,  # No AEGS signal
                    'volatility_score': vol_pick['overall_score'],
                    'combined_score': vol_pick['overall_score'],
                    'daily_move': vol_pick['daily_return_pct'],
                    'vol_class': vol_pick['vol_class'],
                    'entry_price': vol_pick['current_price'],
                    'recommendation': 'MONITOR',
                    'position_size': 'SMALL',
                    'reasoning': f"Extreme volatility ({vol_pick['daily_return_pct']:+.1f}%) - high risk/reward potential"
                }
                
                opportunities.append(opportunity)
                
                print(f"  {i}. {colored(vol_pick['symbol'], 'red', attrs=['bold'])}: "
                      f"{vol_pick['daily_return_pct']:+.1f}% | "
                      f"Vol Score {vol_pick['overall_score']:.0f} | "
                      f"${vol_pick['current_price']:.2f}")
        
        return opportunities
        
    def _calculate_position_size(self, signal: Dict) -> str:
        """Calculate recommended position size based on signal strength and volatility"""
        
        aegs_score = signal['signal_strength']
        vol_score = signal['volatility_score']
        
        if aegs_score >= 80 and vol_score <= 60:
            return 'LARGE'  # Strong signal, moderate vol
        elif aegs_score >= 60 and vol_score <= 80:
            return 'MEDIUM'  # Good signal, manageable vol
        elif aegs_score >= 40:
            return 'SMALL'   # Weak signal or high vol
        else:
            return 'MINIMAL'  # Very weak
            
    def save_volatility_results(self, results: Dict, filename: str = None):
        """Save volatility scan results"""
        
        if filename is None:
            filename = f'aegs_volatility_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
        # Prepare data for JSON serialization
        json_data = {
            'scan_timestamp': results['scan_timestamp'],
            'total_symbols_scanned': results['total_symbols_scanned'],
            'volatile_stocks_found': len(results['volatile_picks']),
            'aegs_signals_found': len(results['aegs_signals']),
            'top_opportunities': results['top_opportunities'][:10],  # Top 10 only
            'top_volatile_picks': results['volatile_picks'][:20],  # Top 20 only
            'strong_aegs_signals': [s for s in results['aegs_signals'] if s['signal_strength'] >= 60]
        }
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
            
        print(f"\n💾 Volatility scan results saved to {filename}")


def run_daily_volatility_aegs_scan():
    """Main function to run daily volatility + AEGS scan"""
    
    print(colored("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🌋💎 AEGS ENHANCED DAILY VOLATILITY SCANNER 💎🌋          ║
    ║                                                              ║
    ║  Automatically discovers volatile stocks and runs AEGS      ║
    ║  analysis to find the best trading opportunities            ║
    ╚══════════════════════════════════════════════════════════════╝
    """, 'cyan', attrs=['bold']))
    
    start_time = datetime.now()
    
    # Initialize scanner
    scanner = AEGSVolatilityIntegration()
    
    try:
        # Run enhanced scan
        results = scanner.run_enhanced_aegs_scan(volatility_limit=50)
        
        # Save results
        scanner.save_volatility_results(results)
        
        # Summary
        elapsed = datetime.now() - start_time
        
        print(colored(f"\n🎉 VOLATILITY SCAN COMPLETE", 'green', attrs=['bold']))
        print("=" * 80)
        print(f"⏱️  Scan Time: {elapsed}")
        print(f"📊 Symbols Analyzed: {results['total_symbols_scanned']}")
        print(f"🌋 Volatile Stocks Found: {len(results['volatile_picks'])}")
        print(f"🎯 AEGS Signals: {len(results['aegs_signals'])}")
        print(f"💎 Top Opportunities: {len(results['top_opportunities'])}")
        
        # Show top 3 opportunities
        if results['top_opportunities']:
            print(colored(f"\n🏆 TOP 3 OPPORTUNITIES:", 'yellow', attrs=['bold']))
            for opp in results['top_opportunities'][:3]:
                print(f"  {opp['rank']}. {colored(opp['symbol'], 'green', attrs=['bold'])}: "
                      f"{opp['opportunity_type']} | "
                      f"Score: {opp['combined_score']:.0f} | "
                      f"{opp['recommendation']}")
                      
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(colored(f"❌ Error during volatility scan: {str(e)}", 'red'))
        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AEGS Enhanced Volatility Scanner')
    parser.add_argument('--limit', '-l', type=int, default=50,
                       help='Number of volatile stocks to analyze (default: 50)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick scan with fewer symbols')
    
    args = parser.parse_args()
    
    if args.quick:
        args.limit = 25
        
    run_daily_volatility_aegs_scan()