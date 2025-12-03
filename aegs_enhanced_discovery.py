"""
🔥💎 ENHANCED AEGS DISCOVERY AGENT 💎🔥
Smarter discovery that avoids previously tested symbols
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.backtest_history import BacktestHistory
from src.agents.invalid_symbol_tracker import InvalidSymbolTracker
from termcolor import colored

class EnhancedDiscoveryAgent:
    """
    Enhanced discovery that learns from history
    """
    
    def __init__(self):
        self.history = BacktestHistory()
        self.invalid_tracker = InvalidSymbolTracker()
        self.excluded_symbols = set()
        self.load_all_exclusions()
        
    def load_all_exclusions(self):
        """Load all symbols we should exclude"""
        
        # 1. Load backtest history
        tested_symbols = set(self.history.history['backtest_history'].keys())
        self.excluded_symbols.update(tested_symbols)
        print(f"📚 Loaded {len(tested_symbols)} tested symbols from history")
        
        # 2. Load invalid symbols
        invalid_symbols = self.invalid_tracker.get_all_invalid()
        self.excluded_symbols.update(invalid_symbols)
        print(f"🚫 Loaded {len(invalid_symbols)} invalid/failed symbols")
        
        # 3. Load goldmine registry
        try:
            with open('aegs_goldmine_registry.json', 'r') as f:
                registry = json.load(f)
            
            goldmine_count = 0
            for tier in registry['goldmine_symbols'].values():
                for symbol in tier.get('symbols', {}).keys():
                    if symbol not in tested_symbols:
                        goldmine_count += 1
                    self.excluded_symbols.add(symbol)
            
            print(f"💎 Loaded {goldmine_count} registered goldmines")
        except:
            pass
        
        # 4. Add commonly tested symbols
        common_symbols = {
            'WULF', 'MARA', 'NOK', 'BB', 'RIOT', 'CLSK', 'EQT', 'DVN',
            'TLRY', 'COIN', 'MSTR', 'SAVA', 'LABU', 'TQQQ', 'MRNA',
            'RIVN', 'SH', 'TNA', 'USO', 'TLT', 'JPM', 'WFC', 'BAC', 'C',
            'CGC', 'INO', 'LCID', 'NKLA', 'RIDE', 'DNA', 'GINKGO', 'ME',
            'HOOD', 'SOFI', 'AFRM', 'RBLX', 'U', 'DKNG', 'PLUG', 'FCEL',
            'BLDP', 'BE', 'RUN', 'BNTX', 'NVAX', 'VXRT', 'CRON', 'ACB', 'SNDL'
        }
        self.excluded_symbols.update(common_symbols)
        
        print(f"🚫 Total excluded symbols: {len(self.excluded_symbols)}")
    
    def get_fresh_universe(self):
        """Get fresh symbols not in exclusion list"""
        
        print("\n🔍 Building fresh discovery universe...")
        
        fresh_symbols = []
        
        # 1. Get S&P 600 Small Cap Index (more volatility)
        try:
            small_cap_url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
            tables = pd.read_html(small_cap_url)
            sp600 = tables[0]['Symbol'].tolist()
            fresh = [s for s in sp600 if s not in self.excluded_symbols]
            fresh_symbols.extend(fresh)
            print(f"   📊 Added {len(fresh)} fresh S&P 600 symbols")
        except:
            pass
        
        # 2. Get Russell 2000 components
        russell_2000_samples = [
            'XELA', 'ATER', 'PROG', 'BBIG', 'CEI', 'GNUS', 'NAKD', 'SNDL',
            'OCGN', 'SENS', 'TRCH', 'MMAT', 'NEGG', 'CARV', 'OPAD', 'IRNT',
            'TMC', 'OPAL', 'GOEV', 'NKLA', 'HYLN', 'WKHS', 'RIDE', 'ARVL',
            'FFIE', 'MULN', 'CENN', 'REV', 'IDEX', 'SES', 'MVST', 'PTRA',
            'CHPT', 'BLNK', 'EVGO', 'VLTA', 'FREY', 'LEV', 'PROTERRA', 'ELMS',
            'ATAI', 'CMPS', 'MNMD', 'CYBN', 'SEEL', 'BRDS', 'DRUG', 'EYEN'
        ]
        fresh = [s for s in russell_2000_samples if s not in self.excluded_symbols]
        fresh_symbols.extend(fresh)
        
        # 3. Recent IPOs and SPACs
        recent_ipos = [
            'ARM', 'KVUE', 'CART', 'SOLV', 'ODDITY', 'VNT', 'CAVA', 'FIGS',
            'INST', 'KTTA', 'LNTH', 'MGIC', 'NUVL', 'PRMB', 'RKLB', 'SGHC',
            'SMCI', 'TMDX', 'VERV', 'VITL', 'WEAV', 'XMTR', 'ZETA'
        ]
        fresh = [s for s in recent_ipos if s not in self.excluded_symbols]
        fresh_symbols.extend(fresh)
        
        # 4. Sector-specific volatile stocks
        volatile_sectors = {
            'Biotech': ['SAVA', 'AVXL', 'BNGO', 'JAGX', 'OCUP', 'PROG', 'SESN'],
            'EV_Charging': ['CHPT', 'BLNK', 'EVGO', 'VLTA', 'DCFC', 'WATT'],
            'Space': ['RKLB', 'ASTR', 'SPCE', 'MNTS', 'BKSY', 'PL', 'LUNR'],
            'Quantum': ['IONQ', 'ARQQ', 'RGTI', 'QBTS'],
            'Hydrogen': ['PLUG', 'FCEL', 'BLDP', 'BE', 'HTOO', 'HYZN'],
            'Mining': ['LAC', 'LTHM', 'MP', 'UUUU', 'DNN', 'CCJ', 'NXE']
        }
        
        for sector, symbols in volatile_sectors.items():
            fresh = [s for s in symbols if s not in self.excluded_symbols]
            fresh_symbols.extend(fresh)
        
        # 5. Get some random screening
        if len(fresh_symbols) < 50:
            # Use finviz screener criteria
            screener_ideas = [
                'APPS', 'WISH', 'CLOV', 'WKHS', 'RIDE', 'GOEV', 'CANOO',
                'ARVL', 'REE', 'FSR', 'LCID', 'RIVN', 'FFIE', 'MULN',
                'NKLA', 'HYLN', 'XL', 'AYRO', 'SOLO', 'FUV', 'KNDI'
            ]
            fresh = [s for s in screener_ideas if s not in self.excluded_symbols]
            fresh_symbols.extend(fresh)
        
        # Remove duplicates
        fresh_symbols = list(set(fresh_symbols))
        
        print(f"✅ Built universe of {len(fresh_symbols)} fresh symbols to explore")
        
        # Shuffle for variety
        random.shuffle(fresh_symbols)
        
        return fresh_symbols
    
    def get_validated_symbols(self):
        """Get symbols from the validated penny stocks list"""
        
        print("\n💎 Loading validated penny stocks...")
        
        # Try to load the most recent validated symbols
        import glob
        validated_files = glob.glob('validated_pennies_*.json')
        
        if not validated_files:
            print("❌ No validated symbols file found")
            return []
        
        # Get most recent file
        validated_files.sort()
        latest_file = validated_files[-1]
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            symbols = []
            for candidate in data['candidates']:
                symbol = candidate['symbol']
                if symbol not in self.excluded_symbols:
                    symbols.append(symbol)
            
            print(f"✅ Loaded {len(symbols)} validated symbols from {latest_file}")
            return symbols
            
        except Exception as e:
            print(f"❌ Error loading validated symbols: {e}")
            return []
    
    def discover_new_candidates(self, max_candidates=20, use_validated=True):
        """Discover genuinely new candidates"""
        
        print(colored("\n🔥💎 ENHANCED DISCOVERY STARTING 💎🔥", 'cyan', attrs=['bold']))
        print("=" * 60)
        
        # First try validated symbols if requested
        if use_validated:
            validated_symbols = self.get_validated_symbols()
            if validated_symbols:
                fresh_universe = validated_symbols
                print(f"🎯 Using {len(fresh_universe)} pre-validated active symbols")
            else:
                fresh_universe = self.get_fresh_universe()
        else:
            fresh_universe = self.get_fresh_universe()
        
        if not fresh_universe:
            print("❌ No fresh symbols available!")
            return []
        
        candidates = []
        tested = 0
        
        print(f"\n🔬 Analyzing {min(50, len(fresh_universe))} fresh symbols...")
        
        for symbol in fresh_universe[:50]:  # Test up to 50 symbols
            if len(candidates) >= max_candidates:
                break
                
            tested += 1
            
            try:
                # Quick volatility check
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="3mo")
                
                if len(df) < 60:
                    continue
                
                # Calculate metrics
                volatility = df['Close'].pct_change().std()
                avg_volume = df['Volume'].mean()
                current_price = df['Close'].iloc[-1]
                
                # High volatility check
                if volatility > 0.03 and avg_volume > 100000:
                    # Calculate 3-month performance
                    three_month_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                    
                    # Look for beaten down or highly volatile
                    if abs(three_month_return) > 0.3 or volatility > 0.05:
                        candidates.append({
                            'symbol': symbol,
                            'volatility': volatility,
                            'volume': avg_volume,
                            'price': current_price,
                            '3m_return': three_month_return,
                            'reason': self._generate_reason(volatility, three_month_return)
                        })
                        
                        print(f"   ✅ {symbol}: Vol={volatility:.3f}, 3M={three_month_return:+.1%}")
                
            except Exception as e:
                # Track failed symbols
                if "possibly delisted" in str(e) or "No data found" in str(e):
                    self.invalid_tracker.add_invalid_symbol(symbol, "No data found - possibly delisted", "no_data")
                else:
                    self.invalid_tracker.add_invalid_symbol(symbol, f"Error: {str(e)}", "data_error")
                continue
        
        print(f"\n📊 Tested {tested} symbols, found {len(candidates)} candidates")
        
        # Sort by volatility
        candidates.sort(key=lambda x: x['volatility'], reverse=True)
        
        return candidates[:max_candidates]
    
    def _generate_reason(self, volatility, three_month_return):
        """Generate discovery reason"""
        if three_month_return < -0.5:
            return f"Extreme selloff: {three_month_return:.1%} in 3M"
        elif three_month_return > 0.5:
            return f"Momentum surge: {three_month_return:+.1%} in 3M"
        elif volatility > 0.08:
            return f"Ultra-high volatility: {volatility:.1%} daily"
        else:
            return f"High volatility play: {volatility:.1%} daily"
    
    def save_discoveries(self, candidates):
        """Save discoveries in AEGS format"""
        
        discovery_data = {
            'discovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'agent': 'Enhanced Discovery Agent',
            'candidates_found': len(candidates),
            'candidates': []
        }
        
        for c in candidates:
            discovery_data['candidates'].append({
                'symbol': c['symbol'],
                'reason': c['reason']
            })
        
        filename = f"aegs_discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(discovery_data, f, indent=2)
        
        print(f"\n💾 Saved {len(candidates)} fresh discoveries to {filename}")
        
        # Also show what we excluded
        print(f"\n📊 Discovery Summary:")
        print(f"   Excluded symbols: {len(self.excluded_symbols)}")
        print(f"   Fresh candidates: {len(candidates)}")
        
        return filename


def demo_enhanced_discovery():
    """Demo the enhanced discovery"""
    
    agent = EnhancedDiscoveryAgent()
    
    print("\n🔍 Current Exclusions:")
    print(f"   Total excluded: {len(agent.excluded_symbols)}")
    
    # Show some excluded symbols
    excluded_list = sorted(list(agent.excluded_symbols))[:20]
    print(f"   Examples: {', '.join(excluded_list)}")
    
    # Discover new candidates
    candidates = agent.discover_new_candidates(max_candidates=10)
    
    if candidates:
        print(colored("\n🎯 Fresh Candidates Found:", 'green'))
        for i, c in enumerate(candidates, 1):
            print(f"\n#{i}. {c['symbol']}")
            print(f"   Reason: {c['reason']}")
            print(f"   Volatility: {c['volatility']:.1%} daily")
            print(f"   Avg Volume: {c['volume']:,.0f}")
            print(f"   Price: ${c['price']:.2f}")
        
        # Save them
        filename = agent.save_discoveries(candidates)
        print(f"\n✅ Ready for backtesting: python src/agents/aegs_backtest_agent.py {filename}")
    else:
        print("\n❌ No new candidates found in this run")


if __name__ == "__main__":
    demo_enhanced_discovery()