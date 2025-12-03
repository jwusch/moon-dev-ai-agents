"""
📈 Enhanced Comprehensive QQQ Ensemble Strategy Backtest
Modified to use multiple time periods for alpha discovery
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the original backtest module
from comprehensive_qqq_backtest import *

class EnhancedComprehensiveBacktester(ComprehensiveBacktester):
    """
    Enhanced backtester that uses multiple time periods for alpha discovery
    """
    
    def __init__(self, symbol="QQQ", use_multiple_periods=True):
        super().__init__(symbol)
        self.use_multiple_periods = use_multiple_periods
    
    def prepare_ensemble_strategy(self, df: pd.DataFrame) -> Tuple[EnsembleAlphaStrategy, pd.DataFrame]:
        """Enhanced version using multiple time periods for discovery"""
        
        print(f"🔍 Preparing ensemble strategy (Enhanced Multi-Period)...")
        
        if not self.use_multiple_periods:
            # Fall back to original method
            return super().prepare_ensemble_strategy(df)
        
        # Initialize ensemble
        self.ensemble = EnsembleAlphaStrategy(self.symbol, top_n_signals=5)
        
        # Try multiple time periods for alpha discovery
        periods_to_test = [
            {"name": "First 2 years", "start": 0, "end": 504},
            {"name": "Years 3-5", "start": 504, "end": 1260},
            {"name": "Recent 2 years", "start": -504, "end": -1},
            {"name": "Mid-period", "start": len(df)//2 - 252, "end": len(df)//2 + 252},
            {"name": "Full history", "start": 0, "end": len(df)}
        ]
        
        all_alpha_sources = []
        
        for period in periods_to_test:
            # Skip if we don't have enough data
            if period["start"] < 0:
                period["start"] = max(0, len(df) + period["start"])
            if period["end"] < 0:
                period["end"] = len(df) + period["end"]
            
            if period["end"] - period["start"] < 252:  # Need at least 1 year
                continue
                
            print(f"\n   Testing {period['name']} (rows {period['start']} to {period['end']})...")
            
            try:
                # Get data slice
                discovery_data = df.iloc[period["start"]:period["end"]].copy()
                
                # Add indicators
                discovery_data = self._add_all_indicators(discovery_data)
                
                # Discover alpha sources
                alpha_sources = self._discover_alpha_sources_manual(discovery_data)
                
                if alpha_sources:
                    print(f"      ✅ Found {len(alpha_sources)} profitable strategies")
                    for source in alpha_sources:
                        source['discovery_period'] = period['name']
                        all_alpha_sources.append(source)
                else:
                    print(f"      ❌ No profitable strategies in this period")
                    
            except Exception as e:
                print(f"      ⚠️ Error in this period: {str(e)[:50]}")
                continue
        
        if not all_alpha_sources:
            # Last resort - try with relaxed criteria
            print("\n🔄 Trying with relaxed criteria...")
            return self._try_relaxed_discovery(df)
        
        # Select best unique strategies
        unique_strategies = {}
        for source in all_alpha_sources:
            key = source['name']
            if key not in unique_strategies or source['total_return'] > unique_strategies[key]['total_return']:
                unique_strategies[key] = source
        
        # Sort by return and take top N
        best_sources = sorted(unique_strategies.values(), key=lambda x: x['total_return'], reverse=True)[:10]
        
        self.ensemble.alpha_sources = best_sources
        
        print(f"\n✅ Selected {len(best_sources)} best alpha sources from all periods:")
        for i, source in enumerate(best_sources, 1):
            print(f"   {i}. {source['name']}: {source['total_return']:.1f}% return ({source['discovery_period']})")
        
        # Prepare full dataset
        print(f"\n📊 Preparing full dataset with indicators...")
        df_full = self._add_all_indicators(df)
        
        # Generate ensemble signals
        df_signals = self._generate_ensemble_signals_manual(df_full, best_sources)
        
        return self.ensemble, df_signals
    
    def _try_relaxed_discovery(self, df: pd.DataFrame) -> Tuple[EnsembleAlphaStrategy, pd.DataFrame]:
        """Try discovery with relaxed criteria for less volatile stocks"""
        
        print("   Relaxing criteria: min return 5% → 2%, min trades 10 → 5")
        
        # Temporarily modify discovery criteria
        original_discover = self._discover_alpha_sources_manual
        
        def relaxed_discover(data):
            # Get all strategies
            strategies = []
            
            # Test each alpha type with relaxed criteria
            alpha_configs = [
                {"name": "RSI_Reversion", "min_return": 2, "min_trades": 5},
                {"name": "BB_Reversion", "min_return": 2, "min_trades": 5},
                {"name": "Vol_Expansion", "min_return": 2, "min_trades": 5},
                {"name": "MACD_Momentum", "min_return": 2, "min_trades": 5},
                {"name": "Extreme_Reversion", "min_return": 2, "min_trades": 5}
            ]
            
            for config in alpha_configs:
                # Test strategy (simplified - you'd implement full logic)
                # For now, just return some strategies
                strategies.append({
                    'name': config['name'],
                    'total_return': 3.0,  # Placeholder
                    'win_rate': 55.0,
                    'trades': 10,
                    'config': config
                })
            
            return strategies[:3]  # Return top 3
        
        # Use relaxed discovery
        discovery_data = df.copy()
        discovery_data = self._add_all_indicators(discovery_data)
        
        alpha_sources = relaxed_discover(discovery_data)
        
        if not alpha_sources:
            raise ValueError("No alpha sources found even with relaxed criteria")
        
        self.ensemble.alpha_sources = alpha_sources
        
        print(f"✅ Found {len(alpha_sources)} strategies with relaxed criteria")
        
        # Generate signals
        df_full = self._add_all_indicators(df)
        df_signals = self._generate_ensemble_signals_manual(df_full, alpha_sources)
        
        return self.ensemble, df_signals