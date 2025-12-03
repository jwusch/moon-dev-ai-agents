"""
🔄💎 AEGS SYMBOL VALIDATION INTEGRATION 💎🔄
Integrates enhanced data fetcher with AEGS system
"""

import json
import pandas as pd
from datetime import datetime
from termcolor import colored
from enhanced_data_fetcher import EnhancedDataFetcher
from smart_symbol_validator import SmartSymbolValidator
from aegs_enhanced_discovery import EnhancedDiscoveryAgent

class AEGSSymbolIntegration:
    """
    Integrate symbol validation with AEGS
    """
    
    def __init__(self):
        self.fetcher = EnhancedDataFetcher()
        self.validator = SmartSymbolValidator()
        self.discovery = EnhancedDiscoveryAgent()
        
    def update_swarm_discovery_lists(self):
        """
        Update all swarm discovery lists with validated symbols
        """
        
        print(colored("🔄💎 UPDATING AEGS SWARM DISCOVERY LISTS 💎🔄", 'cyan', attrs=['bold']))
        print("=" * 60)
        
        # Load current backtest history to exclude tested symbols
        tested_symbols = set(self.discovery.history.history['backtest_history'].keys())
        print(f"\n📋 Already tested symbols: {len(tested_symbols)}")
        
        # Load validated symbols from latest report
        try:
            with open('symbol_validation_report.json', 'r') as f:
                validation_data = json.load(f)
            
            active_symbols = validation_data['active_symbols']
            symbol_mappings = validation_data['symbol_mappings']
            
            print(f"✅ Found {len(active_symbols)} active symbols")
            print(f"🔄 Found {len(symbol_mappings)} symbol mappings")
            
        except:
            print("❌ No validation report found. Run smart_symbol_validator.py first")
            return
        
        # Create updated discovery lists for different strategies
        discovery_lists = {
            'volatile_pennies': [],
            'high_momentum': [],
            'oversold_bounce': [],
            'crypto_volatility': [],
            'validated_active': []
        }
        
        # Analyze each active symbol
        print("\n🔬 Analyzing symbols for strategy categorization...")
        
        for symbol in active_symbols:
            if symbol in tested_symbols:
                continue  # Skip already tested
                
            try:
                # Get recent data
                data, found_symbol = self.fetcher.fetch_data(symbol, period='1mo')
                
                if data is None or len(data) < 20:
                    continue
                
                # Calculate metrics
                current_price = data['Close'].iloc[-1]
                volatility = data['Close'].pct_change().std()
                volume = data['Volume'].mean()
                momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
                rsi = self._calculate_rsi(data['Close'], 14)
                
                # Categorize symbol
                if current_price < 10 and volatility > 0.03:
                    discovery_lists['volatile_pennies'].append({
                        'symbol': found_symbol,
                        'reason': f"${current_price:.2f} with {volatility:.1%} daily volatility",
                        'metrics': {
                            'price': current_price,
                            'volatility': volatility,
                            'volume': volume
                        }
                    })
                
                if momentum > 0.2:  # 20% gain in a month
                    discovery_lists['high_momentum'].append({
                        'symbol': found_symbol,
                        'reason': f"{momentum:.1%} gain in last month",
                        'metrics': {
                            'momentum': momentum,
                            'price': current_price,
                            'volume': volume
                        }
                    })
                
                if rsi < 35:  # Oversold
                    discovery_lists['oversold_bounce'].append({
                        'symbol': found_symbol,
                        'reason': f"RSI at {rsi:.1f} (oversold)",
                        'metrics': {
                            'rsi': rsi,
                            'price': current_price,
                            'volume': volume
                        }
                    })
                
                # All validated symbols go here
                discovery_lists['validated_active'].append({
                    'symbol': found_symbol,
                    'reason': f"Validated active symbol, ${current_price:.2f}",
                    'metrics': {
                        'price': current_price,
                        'volatility': volatility,
                        'volume': volume,
                        'momentum': momentum,
                        'rsi': rsi
                    }
                })
                
            except Exception as e:
                continue
        
        # Add crypto symbols
        crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'SHIB-USD']
        for crypto in crypto_symbols:
            if crypto not in tested_symbols:
                discovery_lists['crypto_volatility'].append({
                    'symbol': crypto,
                    'reason': 'Crypto volatility play'
                })
        
        # Save discovery lists
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for strategy, candidates in discovery_lists.items():
            if candidates:
                filename = f"aegs_validated_{strategy}_{timestamp}.json"
                
                discovery_data = {
                    'discovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'agent': f'Symbol Integration - {strategy}',
                    'candidates_found': len(candidates),
                    'candidates': candidates
                }
                
                with open(filename, 'w') as f:
                    json.dump(discovery_data, f, indent=2)
                
                print(f"\n💾 Saved {len(candidates)} {strategy} candidates to {filename}")
        
        # Create master discovery file
        self._create_master_discovery_file(discovery_lists, timestamp)
        
        print(colored("\n✅ AEGS discovery lists updated with validated symbols!", 'green'))
        
        return discovery_lists
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _create_master_discovery_file(self, discovery_lists, timestamp):
        """Create a master file with all discoveries"""
        
        all_candidates = []
        
        for strategy, candidates in discovery_lists.items():
            for candidate in candidates:
                candidate['strategy'] = strategy
                all_candidates.append(candidate)
        
        # Remove duplicates by symbol
        seen_symbols = set()
        unique_candidates = []
        
        for candidate in all_candidates:
            if candidate['symbol'] not in seen_symbols:
                seen_symbols.add(candidate['symbol'])
                unique_candidates.append(candidate)
        
        master_data = {
            'discovery_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'agent': 'AEGS Symbol Integration Master',
            'candidates_found': len(unique_candidates),
            'candidates': unique_candidates,
            'summary': {
                'total_validated': len(unique_candidates),
                'by_strategy': {
                    strategy: len(candidates) 
                    for strategy, candidates in discovery_lists.items()
                }
            }
        }
        
        filename = f"aegs_master_validated_discoveries_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(master_data, f, indent=2)
        
        print(f"\n🏆 Created master discovery file: {filename}")
        print(f"   Total unique candidates: {len(unique_candidates)}")
        
        # Show top candidates by volatility
        if unique_candidates:
            print("\n🎯 Top 10 candidates by metrics:")
            
            # Sort by volatility if available
            sorted_candidates = []
            for c in unique_candidates:
                if 'metrics' in c and 'volatility' in c['metrics']:
                    sorted_candidates.append(c)
            
            sorted_candidates.sort(key=lambda x: x['metrics']['volatility'], reverse=True)
            
            for i, candidate in enumerate(sorted_candidates[:10], 1):
                metrics = candidate['metrics']
                print(f"\n{i}. {candidate['symbol']}")
                print(f"   Price: ${metrics['price']:.2f}")
                print(f"   Volatility: {metrics['volatility']:.1%}")
                print(f"   Volume: {metrics['volume']:,.0f}")
                if 'momentum' in metrics:
                    print(f"   Momentum: {metrics['momentum']:.1%}")


def main():
    """
    Run symbol validation integration
    """
    
    integrator = AEGSSymbolIntegration()
    discovery_lists = integrator.update_swarm_discovery_lists()
    
    print("\n" + "="*60)
    print(colored("📋 NEXT STEPS:", 'yellow'))
    print("1. Run AEGS swarm with validated symbols:")
    print("   python run_aegs_swarm.py --use-validated")
    print("\n2. Test specific strategies:")
    print("   python src/agents/aegs_backtest_agent.py aegs_validated_volatile_pennies_*.json")
    print("\n3. Monitor for new goldmines:")
    print("   The swarm will now focus on validated, active symbols only")


if __name__ == "__main__":
    main()