"""
🔥💎 AEGS DISCOVERY AGENT 💎🔥
AI-powered symbol discovery for finding the next goldmine

This agent uses multiple strategies to discover potential AEGS candidates:
1. Volatility scanners
2. Volume anomaly detection
3. Sector rotation analysis
4. Pattern matching with existing goldmines
5. Market crash recovery plays
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import ModelFactory
from termcolor import colored
from agents.backtest_history import BacktestHistory
from agents.invalid_symbol_tracker import InvalidSymbolTracker

class AEGSDiscoveryAgent:
    """
    AI-powered discovery agent for finding AEGS goldmine candidates
    """
    
    def __init__(self):
        self.name = "AEGS Discovery Agent"
        self.discovered_candidates = []
        self.discovery_reasons = {}
        
        # Initialize backtest history and enhanced validation
        self.history = BacktestHistory()
        self.invalid_tracker = InvalidSymbolTracker()
        
        # Initialize enhanced symbol validator for multi-source validation
        try:
            from agents.enhanced_symbol_validator import EnhancedSymbolValidator
            self.symbol_validator = EnhancedSymbolValidator()
            print("✅ Enhanced Symbol Validator initialized (multi-source validation)")
        except Exception as e:
            print(f"⚠️ Enhanced validator initialization failed: {e}")
            print("   Falling back to basic invalid symbol tracker")
            self.symbol_validator = None
        
        # Initialize AI models for analysis (SwarmAgent + fallback)
        self.swarm_agent = None
        self.ai_model = None
        
        try:
            # Try to initialize SwarmAgent for multi-model consensus
            from agents.swarm_agent import SwarmAgent
            self.swarm_agent = SwarmAgent()
            print("✅ SwarmAgent initialized for multi-model discovery consensus")
        except Exception as e:
            print(f"⚠️ SwarmAgent initialization failed: {e}")
            
            # Fallback to single model
            try:
                factory = ModelFactory()
                self.ai_model = factory.get_model('claude')
                print("✅ Fallback to single Claude model for discovery")
            except Exception as e2:
                print(f"⚠️ AI model initialization failed: {e2}")
                print("   AI pattern discovery will be disabled")
                self.ai_model = None
        
        # Load existing goldmine characteristics
        self.goldmine_patterns = self._load_goldmine_patterns()
        
        # Discovery parameters
        self.min_volume = 100000  # Minimum daily volume
        self.min_volatility = 0.02  # 2% daily volatility
        self.max_price = 500  # Focus on lower-priced stocks
        
    def run(self):
        """Main discovery process"""
        print(colored(f"🔍 {self.name} Starting Discovery Process...", 'cyan', attrs=['bold']))
        print("=" * 80)
        
        candidates = []
        
        # Run multiple discovery strategies
        print("\n📊 Running discovery strategies...")
        
        # Strategy 1: Volatility Explosion Scanner
        volatility_picks = self._scan_volatility_explosions()
        candidates.extend(volatility_picks)
        
        # Strategy 2: Volume Anomaly Detection
        volume_picks = self._scan_volume_anomalies()
        candidates.extend(volume_picks)
        
        # Strategy 3: Beaten Down Recovery Plays
        recovery_picks = self._scan_recovery_candidates()
        candidates.extend(recovery_picks)
        
        # Strategy 4: Sector Rotation Opportunities
        sector_picks = self._scan_sector_rotation()
        candidates.extend(sector_picks)
        
        # Strategy 5: AI Pattern Analysis
        ai_picks = self._ai_pattern_discovery()
        candidates.extend(ai_picks)
        
        # Remove duplicates and filter with enhanced validation
        unique_candidates = list(set(candidates))
        validated_candidates = self._validate_and_filter_candidates(unique_candidates)
        
        # Save results
        self._save_discoveries(validated_candidates)
        
        print(colored(f"\n✅ Discovery Complete! Found {len(validated_candidates)} validated candidates", 'green'))
        
        return validated_candidates
    
    def _scan_volatility_explosions(self) -> List[str]:
        """Find symbols with explosive volatility increases"""
        print("\n🌋 Scanning for volatility explosions...")
        
        candidates = []
        
        # List of symbols to scan (this would be expanded)
        scan_universe = self._get_scan_universe()
        
        for symbol in scan_universe[:100]:  # Limit for testing
            try:
                # Get recent data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="3mo")
                
                if len(df) < 60:
                    continue
                
                # Calculate volatility metrics
                recent_vol = df['Close'].pct_change().tail(20).std()
                historical_vol = df['Close'].pct_change().head(40).std()
                
                # Check for volatility explosion
                if recent_vol > historical_vol * 2 and recent_vol > self.min_volatility:
                    # Additional checks
                    avg_volume = df['Volume'].tail(20).mean()
                    current_price = df['Close'].iloc[-1]
                    
                    if avg_volume > self.min_volume and current_price < self.max_price:
                        candidates.append(symbol)
                        self.discovery_reasons[symbol] = f"Volatility explosion: {recent_vol/historical_vol:.1f}x increase"
                        print(f"   ✅ {symbol}: Volatility {recent_vol/historical_vol:.1f}x normal")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a delisting/not found error
                if 'delisted' in error_msg or 'no data found' in error_msg or 'symbol may be delisted' in error_msg:
                    # Use enhanced validator to check if truly invalid
                    if self.symbol_validator:
                        validation = self.symbol_validator.validate_symbol(symbol)
                        if validation.status.value == "invalid":
                            self.invalid_tracker.mark_invalid(symbol, f"Confirmed invalid: {error_msg}")
                        elif validation.status.value == "needs_retry":
                            # Could be a temporary issue or wrong symbol format
                            print(f"   ⚠️ {symbol}: Needs investigation - {validation.errors[0] if validation.errors else 'Unknown'}")
                    else:
                        # No validator, just mark as invalid
                        self.invalid_tracker.mark_invalid(symbol, error_msg)
                continue
        
        return candidates
    
    def _scan_volume_anomalies(self) -> List[str]:
        """Find symbols with unusual volume patterns"""
        print("\n📊 Scanning for volume anomalies...")
        
        candidates = []
        
        # Focus on specific sectors known for volatility
        volatile_sectors = ['SPAC', 'Biotech', 'Crypto', 'EV']
        
        # Get fresh symbols with potential volume anomalies
        # Exclude already tested symbols
        excluded = self._get_all_excluded_symbols()
        
        volume_watchlist = [
            # New De-SPACs and high volatility (removed delisted: FFIE, MULN, CENN, REV)
            'IDEX', 'SES', 'MVST',
            # Recent IPOs with volume
            'KVUE', 'CART', 'SOLV', 'VNT', 'CAVA', 'FIGS',
            # Squeeze candidates  
            'BYND', 'CVNA', 'W', 'DASH', 'UPST', 'AI', 'OPEN',
            # New meme potentials (removed delisted: APRN, GNUS, NAKD, SNDL, OCGN)
            'SENS',
        ]
        
        # Filter out already tested
        volume_watchlist = [s for s in volume_watchlist if s not in excluded]
        
        for symbol in volume_watchlist:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1mo")
                
                if len(df) < 20:
                    continue
                
                # Volume analysis
                recent_vol = df['Volume'].tail(5).mean()
                avg_vol = df['Volume'].mean()
                
                if recent_vol > avg_vol * 3:
                    # Check price action
                    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
                    
                    # Look for volume spike with price decline (panic selling)
                    if price_change < -0.15:
                        candidates.append(symbol)
                        self.discovery_reasons[symbol] = f"Volume spike {recent_vol/avg_vol:.1f}x with {price_change*100:.1f}% decline"
                        print(f"   ✅ {symbol}: Volume {recent_vol/avg_vol:.1f}x, Price {price_change*100:.1f}%")
                
            except Exception as e:
                continue
        
        return candidates
    
    def _scan_recovery_candidates(self) -> List[str]:
        """Find beaten-down stocks ready for recovery"""
        print("\n💎 Scanning for recovery candidates...")
        
        candidates = []
        
        # Categories prone to boom/bust cycles - UPDATE WITH FRESH SYMBOLS
        excluded = self._get_all_excluded_symbols()
        
        recovery_categories = {
            'Crypto': ['BTBT', 'CAN', 'SOS', 'EBON', 'LGHL', 'NCTY', 'BTCM', 'XNET'],
            'Cannabis': ['OGI'],  # Most cannabis stocks delisted
            'CleanEnergy': ['CHPT', 'BLNK', 'EVGO', 'WATT'],  # Removed delisted: VLTA, DCFC, HYZN
            'Biotech': ['ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL', 'BRDS', 'DRUG'],
            'Space': ['RKLB', 'ASTR', 'SPCE', 'MNTS', 'BKSY', 'PL', 'LUNR'],
            'EV': ['REE', 'FSR']  # Removed delisted: FFIE, MULN, CENN, ARVL, PTRA
        }
        
        # Filter out tested symbols from each category
        for category in recovery_categories:
            recovery_categories[category] = [s for s in recovery_categories[category] if s not in excluded]
        
        for category, symbols in recovery_categories.items():
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period="1y")
                    
                    if len(df) < 200:
                        continue
                    
                    # Calculate drawdown from peak
                    peak = df['High'].max()
                    current = df['Close'].iloc[-1]
                    drawdown = (current - peak) / peak
                    
                    # Look for 50%+ drawdowns with recent stabilization
                    if drawdown < -0.5:
                        # Check for stabilization (lower volatility recently)
                        recent_vol = df['Close'].pct_change().tail(20).std()
                        
                        if recent_vol < 0.05:  # Less volatile recently
                            candidates.append(symbol)
                            self.discovery_reasons[symbol] = f"{category} recovery play: {drawdown*100:.1f}% from peak"
                            print(f"   ✅ {symbol}: Down {abs(drawdown)*100:.1f}% from peak")
                
                except Exception as e:
                    continue
        
        return candidates
    
    def _scan_sector_rotation(self) -> List[str]:
        """Find sector rotation opportunities"""
        print("\n🔄 Scanning for sector rotation plays...")
        
        candidates = []
        
        # Sector ETFs to analyze
        sectors = {
            'XLE': 'Energy',
            'XLF': 'Financials', 
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'ConsumerStaples',
            'XLY': 'ConsumerDiscretionary',
            'XLU': 'Utilities',
            'XLRE': 'RealEstate'
        }
        
        # Find underperforming sectors
        sector_performance = {}
        
        for etf, sector_name in sectors.items():
            try:
                ticker = yf.Ticker(etf)
                df = ticker.history(period="3mo")
                
                if len(df) > 0:
                    performance = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                    sector_performance[sector_name] = performance
            except:
                continue
        
        # Find worst performing sectors
        if sector_performance:
            worst_sectors = sorted(sector_performance.items(), key=lambda x: x[1])[:3]
            
            for sector, perf in worst_sectors:
                print(f"   📉 {sector} sector down {abs(perf)*100:.1f}% - searching for rebounds...")
                
                # Add sector-specific stocks (abbreviated list)
                if sector == 'Energy':
                    candidates.extend(['XOM', 'CVX', 'OXY', 'DVN'])
                elif sector == 'Financials':
                    candidates.extend(['BAC', 'JPM', 'WFC', 'C'])
                # Add more sectors...
        
        return candidates
    
    def _ai_pattern_discovery(self) -> List[str]:
        """Use AI swarm consensus to discover patterns similar to existing goldmines"""
        print("\n🤖 Running AI Swarm Consensus Pattern Analysis...")
        
        # Check if SwarmAgent or fallback AI is available
        if not self.swarm_agent and not self.ai_model:
            print("   ⚠️ No AI models available, skipping pattern analysis")
            return []
        
        # Prepare goldmine characteristics for AI
        goldmine_data = {
            'WULF': 'Crypto mining, high volatility, boom/bust cycles',
            'SOL-USD': 'Cryptocurrency, extreme volatility, strong trends with sharp reversals',
            'NOK': 'Meme stock potential, telecom, high retail interest',
            'MARA': 'Bitcoin proxy, mining stock, follows crypto cycles',
            'EQT': 'Natural gas, commodity cycles, weather-driven volatility'
        }
        
        # Create prompt for AI swarm
        prompt = f"""Based on these successful AEGS goldmine characteristics:
        
{json.dumps(goldmine_data, indent=2)}

Analyze market patterns and suggest 8-10 ticker symbols with similar characteristics for mean reversion trading.

Focus on symbols with:
- High volatility and boom/bust cycles
- Sector rotation opportunities  
- Meme stock potential
- Commodity or crypto exposure
- Recent price weakness that could reverse

Return your analysis as a JSON list with ticker symbols and detailed reasoning:
{{"symbols": [{{"ticker": "XYZ", "reason": "Detailed explanation of why this matches goldmine patterns", "confidence": "high/medium/low"}}]}}

Be specific about which goldmine patterns each symbol matches and why it could be a good mean reversion candidate."""

        try:
            candidates = []
            
            # Use SwarmAgent for multi-model consensus if available
            if self.swarm_agent:
                print("   🐝 Querying AI swarm for consensus...")
                
                swarm_result = self.swarm_agent.query(prompt)
                
                if swarm_result and swarm_result.get('consensus_summary'):
                    print(f"   💬 Consensus: {swarm_result['consensus_summary'][:200]}...")
                    
                    # Parse individual model responses for symbols
                    all_suggestions = []
                    for provider, data in swarm_result.get('responses', {}).items():
                        if data.get('success'):
                            response_text = data.get('response', '')
                            suggestions = self._parse_ai_symbols(response_text, provider)
                            all_suggestions.extend(suggestions)
                    
                    # Create consensus from all model suggestions
                    symbol_votes = {}
                    for symbol, reason, provider in all_suggestions:
                        if symbol in symbol_votes:
                            symbol_votes[symbol]['votes'] += 1
                            symbol_votes[symbol]['reasons'].append(f"{provider}: {reason}")
                        else:
                            symbol_votes[symbol] = {
                                'votes': 1,
                                'reasons': [f"{provider}: {reason}"]
                            }
                    
                    # Select symbols with multiple votes (consensus)
                    for symbol, data in symbol_votes.items():
                        if data['votes'] >= 2:  # At least 2 models agree
                            candidates.append(symbol)
                            combined_reason = f"Swarm Consensus ({data['votes']} votes): " + "; ".join(data['reasons'][:2])
                            self.discovery_reasons[symbol] = combined_reason
                            print(f"   ✅ {symbol}: {data['votes']} votes - {data['reasons'][0][:100]}...")
                    
                    print(f"   🎯 SwarmAgent consensus found {len(candidates)} high-confidence symbols")
                    
            # Fallback to single model if SwarmAgent failed or unavailable
            elif self.ai_model:
                print("   🤖 Using single Claude model (fallback)...")
                
                response = self.ai_model.generate_response(
                    system_prompt="You are a financial analyst specializing in finding volatile stocks for mean reversion strategies.",
                    user_content=prompt,
                    temperature=0.7
                )
                
                # Handle ModelResponse object
                response_text = response.content if hasattr(response, 'content') else str(response)
                suggestions = self._parse_ai_symbols(response_text, "Claude")
                for symbol, reason, provider in suggestions:
                    candidates.append(symbol)
                    self.discovery_reasons[symbol] = f"AI: {reason}"
                    print(f"   ✅ {symbol}: {reason}")
            
            return candidates
                
        except Exception as e:
            print(f"   ❌ AI swarm analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return []
    
    def _parse_ai_symbols(self, response_text: str, provider: str) -> List[Tuple[str, str, str]]:
        """Parse AI response for symbol suggestions"""
        suggestions = []
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ai_data = json.loads(json_match.group())
                
                for item in ai_data.get('symbols', []):
                    if isinstance(item, dict):
                        symbol = item.get('ticker', '').strip().upper()
                        reason = item.get('reason', '')
                        if symbol and len(symbol) <= 5:  # Basic symbol validation
                            suggestions.append((symbol, reason, provider))
                            
        except Exception as e:
            # Try simple text parsing if JSON fails
            import re
            # Look for patterns like "SYMBOL - reason" or "SYMBOL: reason"  
            pattern = r'\b([A-Z]{1,5})\s*[-:]\s*([^.\n]+)'
            matches = re.findall(pattern, response_text)
            
            for symbol, reason in matches[:10]:  # Limit to 10 matches
                suggestions.append((symbol.strip(), reason.strip(), provider))
        
        return suggestions
    
    def _get_scan_universe(self) -> List[str]:
        """Get universe of symbols to scan - EXCLUDING already tested ones"""
        
        # Start with empty universe
        universe = []
        
        # Get all excluded symbols
        excluded = self._get_all_excluded_symbols()
        
        # Add known delisted symbols to exclusions (as of Dec 2025)
        known_delisted = {
            'PROG', 'VLTA', 'DCFC', 'APRN', 'GNUS', 'PTRA', 'EXPR', 
            'HYZN', 'CEI', 'ODDITY', 'NAKD', 'ARVL', 'REV', 'FFIE',
            'MULN', 'CENN', 'HEXO', 'HUGE', 'FIRE', 'WMD', 'TGOD',
            'VLNS', 'SNDL', 'OCGN', 'BBBY', 'ATER'
        }
        excluded.update(known_delisted)
        
        # 1. S&P 600 Small Cap (more volatile than S&P 500)
        try:
            small_cap_url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
            tables = pd.read_html(small_cap_url)
            sp600 = tables[0]['Symbol'].tolist()
            fresh = [s for s in sp600 if s not in excluded]
            universe.extend(fresh)
            print(f"   📊 Added {len(fresh)} fresh S&P 600 symbols")
        except:
            pass
        
        # 2. Recent IPOs and De-SPACs (high volatility)
        recent_volatile = [
            # 2024-2025 IPOs (removed delisted: ODDITY)
            'ARM', 'KVUE', 'CART', 'SOLV', 'VNT', 'CAVA',
            # EV/Battery (removed delisted: ARVL, PTRA)
            'QS', 'LCID', 'FSR', 'GOEV', 'REE',
            # Space Tech
            'RKLB', 'ASTR', 'SPCE', 'MNTS', 'BKSY', 'PL', 'LUNR',
            # Biotech SPACs
            'ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL', 'BRDS',
            # Quantum Computing
            'IONQ', 'ARQQ', 'RGTI', 'QBTS',
            # Green Energy (removed delisted: VLTA, DCFC)
            'CHPT', 'BLNK', 'EVGO', 'WATT',
            # Meme Stocks Wave 2 (removed delisted: BBBY, EXPR, REV, APRN)
            'BYND', 'OPEN',
            # Crypto/Blockchain
            'BTBT', 'CAN', 'SOS', 'EBON', 'LGHL', 'NCTY',
            # Reddit Favorites (removed delisted: ATER, PROG, CEI, GNUS, NAKD)
            'BBIG'
        ]
        
        # Only add if not already tested
        fresh = [s for s in recent_volatile if s not in excluded]
        universe.extend(fresh)
        
        # 3. High Short Interest stocks (squeeze potential) - removed delisted: HYZN
        high_short = [
            'BYND', 'CVNA', 'W', 'DASH', 'AFRM', 'UPST', 'HOOD',
            'AI', 'OPEN', 'COMP', 'LOVE', 'PSNY', 'BIRD'
        ]
        fresh = [s for s in high_short if s not in excluded]
        universe.extend(fresh)
        
        # Remove duplicates and shuffle for variety
        universe = list(set(universe))
        import random
        random.shuffle(universe)
        
        print(f"   🌍 Total scan universe: {len(universe)} fresh symbols (excluded {len(excluded)})")
        
        return universe
    
    def _get_all_excluded_symbols(self) -> set:
        """Get all symbols to exclude from discovery"""
        excluded = set()
        
        # 1. From backtest history
        tested = set(self.history.history['backtest_history'].keys())
        excluded.update(tested)
        
        # 2. From invalid symbol tracker
        invalid_symbols = self.invalid_tracker.get_all_invalid()
        excluded.update(invalid_symbols)
        
        # 3. From goldmine registry
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            for tier in registry['goldmine_symbols'].values():
                for symbol in tier.get('symbols', {}).keys():
                    excluded.add(symbol)
        except:
            pass
        
        # 4. Common symbols we've tested many times
        common_tested = {
            'WULF', 'MARA', 'NOK', 'BB', 'RIOT', 'CLSK', 'EQT', 'DVN',
            'TLRY', 'COIN', 'MSTR', 'SAVA', 'LABU', 'TQQQ', 'MRNA',
            'RIVN', 'SH', 'TNA', 'USO', 'TLT', 'SPY', 'QQQ', 'VXX'
        }
        excluded.update(common_tested)
        
        return excluded
    
    def _validate_and_filter_candidates(self, candidates: List[str]) -> List[str]:
        """Enhanced validation and filtering using multi-source validation"""
        print(f"\n🛡️ Enhanced validation of {len(candidates)} candidates...")
        
        if not candidates:
            return []
        
        validated_candidates = []
        
        # Use enhanced validator if available, otherwise fallback to basic filtering
        if self.symbol_validator:
            print("   🔍 Using multi-source validation (yfinance + Alpha Vantage + AI research)")
            
            # Batch validate symbols for efficiency
            validation_results = self.symbol_validator.validate_batch(candidates, max_concurrent=3)
            
            for symbol, result in validation_results.items():
                if result.status.value == "valid" and result.confidence >= 66.0:  # At least 2/3 confidence
                    validated_candidates.append(symbol)
                    print(f"   ✅ {symbol}: Valid ({result.confidence:.1f}% confidence, sources: {', '.join(result.sources)})")
                elif result.status.value == "invalid":
                    print(f"   ❌ {symbol}: Invalid ({'; '.join(result.errors[:1])})")
                else:
                    print(f"   ⚠️ {symbol}: Uncertain ({result.confidence:.1f}% confidence) - skipping")
            
            print(f"   🎯 Multi-source validation: {len(validated_candidates)}/{len(candidates)} symbols passed")
        else:
            print("   📊 Using basic filtering (fallback)")
            validated_candidates = self._filter_candidates(candidates)
        
        return validated_candidates
    
    def _filter_candidates(self, candidates: List[str]) -> List[str]:
        """Filter and validate candidates"""
        print("\n🔍 Filtering candidates...")
        
        # First, remove recently tested symbols
        untested_candidates = self.history.get_untested_symbols(candidates)
        print(f"   📊 After history check: {len(untested_candidates)} untested symbols")
        
        # Also check registry to avoid retesting registered symbols
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            registered_symbols = set()
            for tier in registry['goldmine_symbols'].values():
                for symbol in tier.get('symbols', {}).keys():
                    registered_symbols.add(symbol)
            
            # Remove registered symbols
            candidates_to_test = [s for s in untested_candidates if s not in registered_symbols]
            
            if len(untested_candidates) != len(candidates_to_test):
                print(f"   ℹ️ Excluding {len(untested_candidates) - len(candidates_to_test)} already registered symbols")
        except:
            candidates_to_test = untested_candidates
        
        filtered = []
        
        for symbol in candidates_to_test:
            try:
                # Quick validation
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Check if it's a valid tradeable symbol
                if info.get('regularMarketPrice'):
                    filtered.append(symbol)
                    
            except Exception as e:
                # Track failed symbols
                if "404" in str(e) or "not found" in str(e).lower():
                    self.invalid_tracker.add_invalid_symbol(symbol, "Symbol not found on Yahoo Finance", "not_found")
                continue
        
        print(f"   ✅ Final candidates: {len(filtered)} symbols ready for testing")
        
        return filtered
    
    def _save_discoveries(self, candidates: List[str]):
        """Save discovered candidates"""
        
        discovery_data = {
            'discovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'agent': self.name,
            'candidates_found': len(candidates),
            'candidates': []
        }
        
        for symbol in candidates:
            discovery_data['candidates'].append({
                'symbol': symbol,
                'reason': self.discovery_reasons.get(symbol, 'Multiple signals')
            })
        
        # Save to file
        filename = f"aegs_discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(discovery_data, f, indent=2)
        
        print(f"\n💾 Discoveries saved to {filename}")
    
    def _load_goldmine_patterns(self) -> Dict:
        """Load characteristics of existing goldmines"""
        # This would load from the registry in production
        return {
            'volatility_range': (0.02, 0.10),
            'volume_profile': 'spiky',
            'sector_preference': ['crypto', 'biotech', 'energy', 'meme'],
            'price_range': (0.1, 100),
            'market_cap': 'small to mid'
        }


def main():
    """Run the discovery agent"""
    
    print(colored("🔥💎 AEGS DISCOVERY AGENT STARTING 💎🔥", 'cyan', attrs=['bold']))
    print("=" * 80)
    
    agent = AEGSDiscoveryAgent()
    candidates = agent.run()
    
    if candidates:
        print(f"\n🎯 Ready to test {len(candidates)} candidates:")
        for symbol in candidates[:10]:  # Show first 10
            reason = agent.discovery_reasons.get(symbol, 'Unknown')
            print(f"   {symbol}: {reason}")
        
        print(f"\n✅ Run 'python aegs_batch_backtest.py' to test these candidates!")
    else:
        print("\n❌ No candidates found in this run")


if __name__ == "__main__":
    main()