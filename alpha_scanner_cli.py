"""
🔍 Alpha Source Scanner CLI
Command-line tool to discover and rank alpha sources across symbols
Packageable analysis for the Alpha Sources Website project

Usage:
python alpha_scanner_cli.py --symbols VXX,SPY,QQQ --min-alpha 3.0 --output json
python alpha_scanner_cli.py --scan-all --min-alpha 2.0 --export-csv

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
import yfinance as yf
import json
import csv
import argparse
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AlphaSourceResult:
    symbol: str
    strategy_name: str
    timeframe: str
    alpha_score: float
    win_rate: float
    total_return_pct: float
    total_trades: int
    avg_hold_minutes: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    discovery_date: str
    strategy_type: str
    entry_conditions: str
    exit_conditions: str
    market_conditions: str
    confidence_score: float

class AlphaSourceScanner:
    """
    Comprehensive alpha source discovery and ranking system
    """
    
    def __init__(self, min_alpha_threshold: float = 3.0):
        self.min_alpha_threshold = min_alpha_threshold
        self.discovery_date = datetime.now().isoformat()
        
        # Comprehensive symbol universe
        self.symbol_universe = {
            'volatility': ['VXX', 'UVXY', 'VIXY', 'SVXY', 'VXZ', 'VIXM'],
            'major_etfs': ['SPY', 'QQQ', 'IWM', 'DIA'],
            'leveraged': ['TQQQ', 'SQQQ', 'SPXL', 'SPXS', 'TNA', 'TZA'],
            'sectors': ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLU', 'XLB'],
            'individual': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD'],
            'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
            'bonds': ['TLT', 'IEF', 'SHY', 'TBT', 'AGG'],
            'crypto': ['BITO', 'BITQ']
        }
        
        self.all_symbols = [symbol for category in self.symbol_universe.values() for symbol in category]
        
        # Strategy definitions (from our alpha_source_mapper.py)
        self.strategy_definitions = self._load_strategy_definitions()
    
    def _load_strategy_definitions(self) -> Dict:
        """Load all strategy definitions for alpha discovery"""
        return {
            'mean_reversion': [
                {
                    'name': 'RSI_Reversion',
                    'description': 'RSI oversold + price below SMA',
                    'entry_logic': 'lambda df: (df["RSI"] < 30) & (df["Distance_Medium"] < -1.0)',
                    'exit_logic': 'lambda df: (df["RSI"] > 50) | (df["Distance_Medium"] > 0)',
                    'type': 'mean_reversion'
                },
                {
                    'name': 'BB_Reversion',
                    'description': 'Bollinger Band lower breach reversion',
                    'entry_logic': 'lambda df: df["BB_Position"] < 0.2',
                    'exit_logic': 'lambda df: df["BB_Position"] > 0.5',
                    'type': 'mean_reversion'
                },
                {
                    'name': 'Extreme_Reversion',
                    'description': 'Extreme move mean reversion',
                    'entry_logic': 'lambda df: df["Extreme_Move"] & (df["Distance_Short"] < -2.0)',
                    'exit_logic': 'lambda df: abs(df["Distance_Short"]) < 0.5',
                    'type': 'mean_reversion'
                }
            ],
            'momentum': [
                {
                    'name': 'MACD_Momentum',
                    'description': 'MACD crossover with trend strength',
                    'entry_logic': 'lambda df: (df["MACD"] > df["MACD_Signal"]) & (df["MACD_Hist"] > 0) & (df["ADX"] > 25)',
                    'exit_logic': 'lambda df: (df["MACD"] < df["MACD_Signal"]) | (df["MACD_Hist"] < 0)',
                    'type': 'momentum'
                },
                {
                    'name': 'Breakout_Momentum',
                    'description': 'Bollinger breakout with volume',
                    'entry_logic': 'lambda df: (df["Close"] > df["BB_Upper"]) & (df["Volume_Ratio"] > 1.5)',
                    'exit_logic': 'lambda df: df["Close"] < df["SMA_Medium"]',
                    'type': 'momentum'
                },
                {
                    'name': 'ROC_Momentum',
                    'description': 'Rate of change momentum',
                    'entry_logic': 'lambda df: (df["ROC_Medium"] > 2) & (df["ADX"] > 20)',
                    'exit_logic': 'lambda df: df["ROC_Medium"] < 0',
                    'type': 'momentum'
                }
            ],
            'volatility': [
                {
                    'name': 'Vol_Expansion',
                    'description': 'Volatility expansion from low vol periods',
                    'entry_logic': 'lambda df: df["Vol_Regime"] == "Low"',
                    'exit_logic': 'lambda df: df["Vol_Regime"] == "High"',
                    'type': 'volatility'
                },
                {
                    'name': 'Vol_Contraction',
                    'description': 'Volatility contraction trades',
                    'entry_logic': 'lambda df: (df["Vol_Regime"] == "High") & (df["BB_Squeeze"] < df["BB_Squeeze"].rolling(20).mean())',
                    'exit_logic': 'lambda df: df["Vol_Regime"] == "Low"',
                    'type': 'volatility'
                }
            ],
            'microstructure': [
                {
                    'name': 'Volume_Spike',
                    'description': 'Volume spike mean reversion',
                    'entry_logic': 'lambda df: (df["Volume_Ratio"] > 2.0) & (abs(df["Price_vs_VWAP"]) > 0.5)',
                    'exit_logic': 'lambda df: df["Volume_Ratio"] < 1.2',
                    'type': 'microstructure'
                },
                {
                    'name': 'Gap_Fade',
                    'description': 'Gap fade strategy',
                    'entry_logic': 'lambda df: abs(df["Gap"]) > 1.0',
                    'exit_logic': 'lambda df: abs(df["Gap"].shift(1)) < 0.2',
                    'type': 'microstructure'
                }
            ],
            'behavioral': [
                {
                    'name': 'Overreaction',
                    'description': 'Overreaction reversal',
                    'entry_logic': 'lambda df: df["Extreme_Move"] & (abs(df["ROC_Short"]) > 3)',
                    'exit_logic': 'lambda df: abs(df["ROC_Short"]) < 1',
                    'type': 'behavioral'
                },
                {
                    'name': 'Friday_Effect',
                    'description': 'Weekend/Friday effects',
                    'entry_logic': 'lambda df: (df["DayOfWeek"] == 4) & (df["Distance_Medium"] < -1)',
                    'exit_logic': 'lambda df: df["DayOfWeek"] == 0',
                    'type': 'behavioral'
                }
            ]
        }
    
    def download_and_prepare_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Download and prepare data for multiple timeframes"""
        print(f"📊 Scanning {symbol}...")
        
        data = {}
        
        for timeframe in timeframes:
            try:
                # Determine period based on timeframe
                if timeframe == "1m":
                    period = "7d"
                elif timeframe in ["5m", "15m"]:
                    period = "60d"
                elif timeframe == "1h":
                    period = "730d"
                else:  # 1d
                    period = "max"
                
                df = yf.download(symbol, period=period, interval=timeframe, progress=False)
                
                if df.empty or len(df) < 50:
                    continue
                
                # Clean column names
                if df.columns.nlevels > 1:
                    df.columns = [col[0] for col in df.columns]
                
                # Add comprehensive indicators
                df = self._add_comprehensive_indicators(df, timeframe)
                
                if len(df.dropna()) > 30:  # Minimum viable data
                    data[timeframe] = df.dropna()
                    print(f"   ✅ {timeframe}: {len(df)} bars")
                
            except Exception as e:
                print(f"   ❌ {timeframe}: {e}")
                continue
        
        return data
    
    def _add_comprehensive_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add all indicators needed for alpha discovery"""
        
        # Timeframe-adjusted periods
        if timeframe == "1m":
            short, medium, long = 5, 20, 60
        elif timeframe == "5m":
            short, medium, long = 4, 12, 48
        elif timeframe == "15m":
            short, medium, long = 4, 14, 28
        elif timeframe == "1h":
            short, medium, long = 6, 24, 72
        else:  # 1d
            short, medium, long = 5, 20, 50
        
        # === CORE INDICATORS ===
        df['SMA_Short'] = df['Close'].rolling(short).mean()
        df['SMA_Medium'] = df['Close'].rolling(medium).mean()
        df['SMA_Long'] = df['Close'].rolling(long).mean()
        df['Distance_Short'] = (df['Close'] - df['SMA_Short']) / df['SMA_Short'] * 100
        df['Distance_Medium'] = (df['Close'] - df['SMA_Medium']) / df['SMA_Medium'] * 100
        df['RSI'] = talib.RSI(df['Close'].values, medium)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values, medium, 2, 2)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        df['BB_Squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'].values)
        
        # ADX
        df['ADX'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, medium)
        
        # Volatility
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, medium)
        df['Vol_Short'] = df['Close'].rolling(short).std()
        df['Vol_Medium'] = df['Close'].rolling(medium).std()
        df['Vol_Regime'] = np.where(df['Vol_Short'] > df['Vol_Medium'] * 1.5, 'High',
                                   np.where(df['Vol_Short'] < df['Vol_Medium'] * 0.7, 'Low', 'Normal'))
        
        # Volume
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(medium).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['VWAP'] = (df['Close'] * df['Volume']).rolling(medium).sum() / df['Volume'].rolling(medium).sum()
            df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
        else:
            df['Volume_Ratio'] = 1.0
            df['Price_vs_VWAP'] = 0.0
        
        # Momentum
        df['ROC_Short'] = talib.ROC(df['Close'].values, short)
        df['ROC_Medium'] = talib.ROC(df['Close'].values, medium)
        
        # Gaps and extremes
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
        returns = df['Close'].pct_change()
        df['Extreme_Move'] = abs(returns) > returns.rolling(medium).std() * 2
        
        # Time features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        
        return df
    
    def test_strategy_on_data(self, df: pd.DataFrame, strategy: Dict, timeframe: str) -> Optional[Dict]:
        """Test a single strategy on data and calculate alpha metrics"""
        
        try:
            # Safely evaluate entry and exit conditions
            entry_condition = eval(strategy['entry_logic'])(df)
            exit_condition = eval(strategy['exit_logic'])(df)
            
            if entry_condition.sum() == 0:  # No entry signals
                return None
            
            # Simple backtest
            trades = []
            position = 0
            entry_price = None
            entry_idx = None
            
            for i in range(len(df)):
                current_price = df['Close'].iloc[i]
                
                # Entry
                if position == 0 and entry_condition.iloc[i]:
                    position = 1
                    entry_price = current_price
                    entry_idx = i
                
                # Exit
                elif position == 1 and exit_condition.iloc[i]:
                    if entry_price is not None and entry_idx is not None:
                        hold_periods = i - entry_idx
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                        
                        trades.append({
                            'hold_periods': hold_periods,
                            'pnl_pct': pnl_pct,
                            'win': pnl_pct > 0
                        })
                    
                    position = 0
                    entry_price = None
                    entry_idx = None
            
            if not trades:
                return None
            
            trades_df = pd.DataFrame(trades)
            
            # Calculate metrics
            total_trades = len(trades_df)
            win_rate = trades_df['win'].mean() * 100
            total_return = trades_df['pnl_pct'].sum()
            avg_hold_periods = trades_df['hold_periods'].mean()
            
            # Convert to minutes
            timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '1d': 1440}
            avg_hold_minutes = avg_hold_periods * timeframe_minutes.get(timeframe, 5)
            
            # Risk metrics
            returns = trades_df['pnl_pct'].values
            wins = trades_df[trades_df['win']]['pnl_pct']
            losses = trades_df[~trades_df['win']]['pnl_pct']
            
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if len(losses) > 0 and avg_loss != 0 else float('inf')
            
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_drawdown(trades_df['pnl_pct'].cumsum())
            
            # Alpha score (risk-adjusted return per trade)
            alpha_score = total_return / total_trades * (win_rate / 100) if total_trades > 0 else 0
            
            # Confidence score based on sample size and consistency
            confidence_score = min(1.0, total_trades / 20) * min(1.0, win_rate / 50) * min(1.0, abs(alpha_score) / 2)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return_pct': total_return,
                'alpha_score': alpha_score,
                'avg_hold_minutes': avg_hold_minutes,
                'sharpe_ratio': sharpe,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'confidence_score': confidence_score
            }
            
        except Exception:
            return None
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        return drawdown.min()
    
    def scan_symbol(self, symbol: str, timeframes: List[str] = None) -> List[AlphaSourceResult]:
        """Scan a single symbol for alpha sources"""
        
        if timeframes is None:
            timeframes = ["5m", "15m", "1h", "1d"]  # Skip 1m for speed
        
        # Download data
        data = self.download_and_prepare_data(symbol, timeframes)
        
        if not data:
            print(f"   ❌ No viable data for {symbol}")
            return []
        
        alpha_sources = []
        
        # Test all strategies on all timeframes
        for timeframe, df in data.items():
            for category, strategies in self.strategy_definitions.items():
                for strategy in strategies:
                    result = self.test_strategy_on_data(df, strategy, timeframe)
                    
                    if result and result['alpha_score'] >= self.min_alpha_threshold:
                        alpha_source = AlphaSourceResult(
                            symbol=symbol,
                            strategy_name=f"{strategy['name']}_{timeframe}",
                            timeframe=timeframe,
                            alpha_score=result['alpha_score'],
                            win_rate=result['win_rate'],
                            total_return_pct=result['total_return_pct'],
                            total_trades=result['total_trades'],
                            avg_hold_minutes=result['avg_hold_minutes'],
                            sharpe_ratio=result['sharpe_ratio'],
                            profit_factor=result['profit_factor'],
                            max_drawdown=result['max_drawdown'],
                            discovery_date=self.discovery_date,
                            strategy_type=strategy['type'],
                            entry_conditions=strategy['description'],
                            exit_conditions=strategy['exit_logic'],
                            market_conditions="General",
                            confidence_score=result['confidence_score']
                        )
                        
                        alpha_sources.append(alpha_source)
        
        # Sort by alpha score
        alpha_sources.sort(key=lambda x: x.alpha_score, reverse=True)
        
        if alpha_sources:
            print(f"   🎯 Found {len(alpha_sources)} alpha sources (α≥{self.min_alpha_threshold})")
            for source in alpha_sources[:3]:  # Show top 3
                print(f"      {source.strategy_name}: α={source.alpha_score:.2f}, {source.win_rate:.1f}% win")
        
        return alpha_sources
    
    def scan_multiple_symbols(self, symbols: List[str], timeframes: List[str] = None) -> List[AlphaSourceResult]:
        """Scan multiple symbols for alpha sources"""
        
        print(f"🔍 ALPHA SOURCE SCANNER")
        print(f"=" * 60)
        print(f"Scanning {len(symbols)} symbols for α≥{self.min_alpha_threshold}")
        print(f"Symbols: {', '.join(symbols)}")
        
        all_alpha_sources = []
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {symbol}")
            
            try:
                symbol_sources = self.scan_symbol(symbol, timeframes)
                all_alpha_sources.extend(symbol_sources)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Sort all sources by alpha score
        all_alpha_sources.sort(key=lambda x: x.alpha_score, reverse=True)
        
        print(f"\n🏆 SCAN RESULTS:")
        print(f"=" * 60)
        print(f"Total alpha sources found: {len(all_alpha_sources)}")
        
        if all_alpha_sources:
            print(f"\nTop 10 Alpha Sources:")
            for i, source in enumerate(all_alpha_sources[:10], 1):
                print(f"{i:2}. {source.symbol} {source.strategy_name}: α={source.alpha_score:.2f} ({source.win_rate:.1f}% win)")
        
        return all_alpha_sources
    
    def export_results(self, alpha_sources: List[AlphaSourceResult], format: str = "json", filename: str = None) -> str:
        """Export alpha source results in various formats"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alpha_sources_{timestamp}"
        
        if format.lower() == "json":
            output_file = f"{filename}.json"
            data = [asdict(source) for source in alpha_sources]
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            output_file = f"{filename}.csv"
            data = [asdict(source) for source in alpha_sources]
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Exported {len(alpha_sources)} alpha sources to {output_file}")
        return output_file

def main():
    """Command-line interface for alpha source scanning"""
    parser = argparse.ArgumentParser(description="Alpha Source Scanner CLI")
    
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (e.g., VXX,SPY,QQQ)')
    parser.add_argument('--scan-all', action='store_true', help='Scan all symbols in universe')
    parser.add_argument('--category', type=str, help='Scan specific category (volatility,major_etfs,etc.)')
    parser.add_argument('--min-alpha', type=float, default=3.0, help='Minimum alpha threshold (default: 3.0)')
    parser.add_argument('--timeframes', type=str, default='5m,15m,1h,1d', help='Timeframes to scan')
    parser.add_argument('--output', type=str, default='json', choices=['json', 'csv'], help='Output format')
    parser.add_argument('--filename', type=str, help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = AlphaSourceScanner(min_alpha_threshold=args.min_alpha)
    
    # Determine symbols to scan
    if args.scan_all:
        symbols_to_scan = scanner.all_symbols
    elif args.category:
        if args.category in scanner.symbol_universe:
            symbols_to_scan = scanner.symbol_universe[args.category]
        else:
            print(f"❌ Unknown category: {args.category}")
            print(f"Available categories: {', '.join(scanner.symbol_universe.keys())}")
            return
    elif args.symbols:
        symbols_to_scan = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        # Default: scan top volatility symbols
        symbols_to_scan = ['VXX', 'SPY', 'QQQ', 'NVDA', 'TSLA']
    
    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Run scan
    alpha_sources = scanner.scan_multiple_symbols(symbols_to_scan, timeframes)
    
    # Export results
    if alpha_sources:
        output_file = scanner.export_results(alpha_sources, args.output, args.filename)
        
        print(f"\n💡 USAGE:")
        print(f"• Use {output_file} for the Alpha Sources Website")
        print(f"• Top alpha source: {alpha_sources[0].symbol} {alpha_sources[0].strategy_name} (α={alpha_sources[0].alpha_score:.2f})")
        print(f"• Integration: Load this data into website database")
    
    else:
        print("❌ No alpha sources found meeting criteria")
        print("💡 Try lowering --min-alpha threshold or expanding symbol list")

if __name__ == "__main__":
    main()