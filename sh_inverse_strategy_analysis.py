"""
📉💎 SH INVERSE S&P 500 STRATEGY ANALYSIS 💎📉
Comprehensive backtest and analysis of SH (ProShares Short S&P 500)

SH moves inversely to the S&P 500:
- When SPY drops 1%, SH gains ~1%
- Perfect for market crashes and corrections
- AEGS showed 31.8% excess return potential
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('dark_background')
sns.set_palette("husl")

class SHInverseStrategy:
    """
    Comprehensive analysis of SH inverse ETF strategy
    """
    
    def __init__(self):
        self.symbol = 'SH'
        self.benchmark = 'SPY'
        self.vix_threshold = 20  # VIX above 20 suggests volatility
        
    def download_comprehensive_data(self):
        """Download SH, SPY, and VIX data"""
        
        print("📊 Downloading comprehensive market data...")
        
        # Download 10 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3650)
        
        # Download SH (inverse S&P)
        sh = yf.download('SH', start=start_date, end=end_date)
        sh.columns = ['SH_' + col if isinstance(col, str) else 'SH_' + col[0] for col in sh.columns]
        
        # Download SPY (S&P 500)
        spy = yf.download('SPY', start=start_date, end=end_date)
        spy.columns = ['SPY_' + col if isinstance(col, str) else 'SPY_' + col[0] for col in spy.columns]
        
        # Download VIX (volatility index)
        vix = yf.download('^VIX', start=start_date, end=end_date)
        vix.columns = ['VIX_' + col if isinstance(col, str) else 'VIX_' + col[0] for col in vix.columns]
        
        # Combine all data
        data = pd.concat([sh, spy, vix], axis=1)
        data.dropna(inplace=True)
        
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
    
    def analyze_correlation(self, data):
        """Analyze SH vs SPY correlation"""
        
        print("\n📈 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Calculate daily returns
        data['SH_Return'] = data['SH_Close'].pct_change()
        data['SPY_Return'] = data['SPY_Close'].pct_change()
        
        # Correlation
        correlation = data['SH_Return'].corr(data['SPY_Return'])
        print(f"SH vs SPY Correlation: {correlation:.3f}")
        print("(Should be close to -1.0 for perfect inverse)")
        
        # Beta calculation
        covariance = data['SH_Return'].cov(data['SPY_Return'])
        spy_variance = data['SPY_Return'].var()
        beta = covariance / spy_variance
        print(f"SH Beta vs SPY: {beta:.3f}")
        
        return data
    
    def backtest_strategies(self, data):
        """Backtest multiple SH trading strategies"""
        
        print("\n🔬 BACKTESTING SH STRATEGIES")
        print("=" * 50)
        
        results = {}
        
        # Strategy 1: Buy and Hold SH (baseline)
        print("\n1️⃣ Buy & Hold SH Strategy:")
        initial_price = data['SH_Close'].iloc[0]
        final_price = data['SH_Close'].iloc[-1]
        total_return = (final_price - initial_price) / initial_price * 100
        annual_return = total_return / (len(data) / 252)
        
        results['buy_hold_sh'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': self._calculate_sharpe(data['SH_Return'].dropna())
        }
        
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Annual Return: {annual_return:.2f}%")
        
        # Strategy 2: VIX-Based Entry (Buy SH when VIX > 20)
        print("\n2️⃣ VIX-Based Strategy (VIX > 20):")
        data['VIX_Signal'] = (data['VIX_Close'] > self.vix_threshold).astype(int)
        data['VIX_Strategy_Return'] = data['SH_Return'] * data['VIX_Signal'].shift(1)
        
        vix_total_return = (1 + data['VIX_Strategy_Return']).cumprod().iloc[-1] - 1
        vix_annual_return = vix_total_return * 252 / len(data)
        
        results['vix_strategy'] = {
            'total_return': vix_total_return * 100,
            'annual_return': vix_annual_return * 100,
            'sharpe': self._calculate_sharpe(data['VIX_Strategy_Return'].dropna()),
            'days_in_market': data['VIX_Signal'].sum() / len(data) * 100
        }
        
        print(f"   Total Return: {vix_total_return * 100:.2f}%")
        print(f"   Annual Return: {vix_annual_return * 100:.2f}%")
        print(f"   Days in Market: {results['vix_strategy']['days_in_market']:.1f}%")
        
        # Strategy 3: SPY Technical (Buy SH when SPY < 50 SMA)
        print("\n3️⃣ SPY Technical Strategy (SPY < 50 SMA):")
        data['SPY_SMA50'] = data['SPY_Close'].rolling(50).mean()
        data['Tech_Signal'] = (data['SPY_Close'] < data['SPY_SMA50']).astype(int)
        data['Tech_Strategy_Return'] = data['SH_Return'] * data['Tech_Signal'].shift(1)
        
        tech_total_return = (1 + data['Tech_Strategy_Return']).cumprod().iloc[-1] - 1
        tech_annual_return = tech_total_return * 252 / len(data)
        
        results['tech_strategy'] = {
            'total_return': tech_total_return * 100,
            'annual_return': tech_annual_return * 100,
            'sharpe': self._calculate_sharpe(data['Tech_Strategy_Return'].dropna()),
            'days_in_market': data['Tech_Signal'].sum() / len(data) * 100
        }
        
        print(f"   Total Return: {tech_total_return * 100:.2f}%")
        print(f"   Annual Return: {tech_annual_return * 100:.2f}%")
        print(f"   Days in Market: {results['tech_strategy']['days_in_market']:.1f}%")
        
        # Strategy 4: Combined VIX + Technical
        print("\n4️⃣ Combined Strategy (VIX > 20 OR SPY < 50 SMA):")
        data['Combined_Signal'] = ((data['VIX_Signal'] == 1) | (data['Tech_Signal'] == 1)).astype(int)
        data['Combined_Strategy_Return'] = data['SH_Return'] * data['Combined_Signal'].shift(1)
        
        combined_total_return = (1 + data['Combined_Strategy_Return']).cumprod().iloc[-1] - 1
        combined_annual_return = combined_total_return * 252 / len(data)
        
        results['combined_strategy'] = {
            'total_return': combined_total_return * 100,
            'annual_return': combined_annual_return * 100,
            'sharpe': self._calculate_sharpe(data['Combined_Strategy_Return'].dropna()),
            'days_in_market': data['Combined_Signal'].sum() / len(data) * 100
        }
        
        print(f"   Total Return: {combined_total_return * 100:.2f}%")
        print(f"   Annual Return: {combined_annual_return * 100:.2f}%")
        print(f"   Days in Market: {results['combined_strategy']['days_in_market']:.1f}%")
        
        # Strategy 5: Crisis-Only Strategy (VIX > 30)
        print("\n5️⃣ Crisis-Only Strategy (VIX > 30):")
        data['Crisis_Signal'] = (data['VIX_Close'] > 30).astype(int)
        data['Crisis_Strategy_Return'] = data['SH_Return'] * data['Crisis_Signal'].shift(1)
        
        crisis_total_return = (1 + data['Crisis_Strategy_Return']).cumprod().iloc[-1] - 1
        crisis_annual_return = crisis_total_return * 252 / len(data)
        
        results['crisis_strategy'] = {
            'total_return': crisis_total_return * 100,
            'annual_return': crisis_annual_return * 100,
            'sharpe': self._calculate_sharpe(data['Crisis_Strategy_Return'].dropna()),
            'days_in_market': data['Crisis_Signal'].sum() / len(data) * 100
        }
        
        print(f"   Total Return: {crisis_total_return * 100:.2f}%")
        print(f"   Annual Return: {crisis_annual_return * 100:.2f}%")
        print(f"   Days in Market: {results['crisis_strategy']['days_in_market']:.1f}%")
        
        return data, results
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def analyze_crisis_performance(self, data):
        """Analyze SH performance during market crises"""
        
        print("\n🔥 CRISIS PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Define major crisis periods
        crises = {
            'COVID-19 Crash': ('2020-02-19', '2020-03-23'),
            '2022 Bear Market': ('2022-01-03', '2022-10-12'),
            '2023 Banking Crisis': ('2023-03-01', '2023-03-31'),
        }
        
        for crisis_name, (start, end) in crises.items():
            try:
                crisis_data = data[start:end]
                if len(crisis_data) == 0:
                    continue
                
                # Calculate performance
                spy_return = (crisis_data['SPY_Close'].iloc[-1] / crisis_data['SPY_Close'].iloc[0] - 1) * 100
                sh_return = (crisis_data['SH_Close'].iloc[-1] / crisis_data['SH_Close'].iloc[0] - 1) * 100
                vix_max = crisis_data['VIX_Close'].max()
                
                print(f"\n{crisis_name}:")
                print(f"   SPY Return: {spy_return:.1f}%")
                print(f"   SH Return: {sh_return:.1f}% ✨")
                print(f"   VIX Peak: {vix_max:.1f}")
                print(f"   SH Outperformance: {sh_return - spy_return:.1f}%")
                
            except:
                continue
    
    def generate_report(self, data, results):
        """Generate comprehensive analysis report"""
        
        print("\n📊 COMPREHENSIVE SH STRATEGY REPORT")
        print("=" * 60)
        
        # Rank strategies by annual return
        ranked = sorted(results.items(), key=lambda x: x[1]['annual_return'], reverse=True)
        
        print("\n🏆 STRATEGY RANKINGS (by Annual Return):")
        for i, (strategy, metrics) in enumerate(ranked, 1):
            print(f"\n{i}. {strategy.upper()}:")
            print(f"   Annual Return: {metrics['annual_return']:.2f}%")
            print(f"   Total Return: {metrics['total_return']:.2f}%")
            print(f"   Sharpe Ratio: {metrics['sharpe']:.2f}")
            if 'days_in_market' in metrics:
                print(f"   Time in Market: {metrics['days_in_market']:.1f}%")
        
        # Best performing strategy
        best_strategy = ranked[0][0]
        best_metrics = ranked[0][1]
        
        print(f"\n✨ BEST STRATEGY: {best_strategy.upper()}")
        print(f"   Would turn $10,000 into ${10000 * (1 + best_metrics['total_return']/100):,.0f}")
        
        # Risk analysis
        print("\n⚠️  RISK ANALYSIS:")
        print("   - SH loses value in bull markets (most of the time)")
        print("   - Best used as hedge or during high VIX periods")
        print("   - Consider allocation of only 5-20% of portfolio")
        
        # Current market conditions
        current_vix = data['VIX_Close'].iloc[-1]
        current_signal = "BUY SH" if current_vix > 20 else "AVOID SH"
        
        print(f"\n📍 CURRENT CONDITIONS:")
        print(f"   VIX: {current_vix:.1f}")
        print(f"   Signal: {current_signal}")
        
        # Save detailed results
        report = {
            'analysis_date': datetime.now().isoformat(),
            'symbol': 'SH',
            'data_period': f"{data.index[0].date()} to {data.index[-1].date()}",
            'strategies': results,
            'best_strategy': best_strategy,
            'current_vix': current_vix,
            'current_signal': current_signal
        }
        
        import json
        filename = f"sh_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {filename}")
        
        return report
    
    def plot_performance(self, data):
        """Create performance visualization"""
        
        print("\n📈 Generating performance charts...")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: SH vs SPY Performance
        ax1 = axes[0]
        (data['SH_Close'] / data['SH_Close'].iloc[0] * 100).plot(ax=ax1, label='SH', color='red', linewidth=2)
        (data['SPY_Close'] / data['SPY_Close'].iloc[0] * 100).plot(ax=ax1, label='SPY', color='blue', linewidth=2)
        ax1.set_title('SH vs SPY Performance (Indexed to 100)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: VIX with threshold
        ax2 = axes[1]
        data['VIX_Close'].plot(ax=ax2, label='VIX', color='yellow', linewidth=2)
        ax2.axhline(y=20, color='red', linestyle='--', label='VIX=20 Threshold')
        ax2.axhline(y=30, color='darkred', linestyle='--', label='VIX=30 Crisis')
        ax2.set_title('VIX (Volatility Index)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Strategy Performance
        ax3 = axes[2]
        (1 + data['VIX_Strategy_Return']).cumprod().plot(ax=ax3, label='VIX Strategy', linewidth=2)
        (1 + data['Tech_Strategy_Return']).cumprod().plot(ax=ax3, label='Technical Strategy', linewidth=2)
        (1 + data['Combined_Strategy_Return']).cumprod().plot(ax=ax3, label='Combined Strategy', linewidth=2)
        ax3.set_title('Strategy Performance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"sh_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Chart saved to: {filename}")
        
        plt.close()


def main():
    """Run comprehensive SH analysis"""
    
    print("📉💎 SH INVERSE S&P 500 STRATEGY ANALYSIS 💎📉")
    print("=" * 60)
    
    analyzer = SHInverseStrategy()
    
    # Download data
    data = analyzer.download_comprehensive_data()
    
    # Analyze correlation
    data = analyzer.analyze_correlation(data)
    
    # Backtest strategies
    data, results = analyzer.backtest_strategies(data)
    
    # Analyze crisis performance
    analyzer.analyze_crisis_performance(data)
    
    # Generate report
    report = analyzer.generate_report(data, results)
    
    # Create visualizations
    analyzer.plot_performance(data)
    
    print("\n✅ SH Analysis Complete!")


if __name__ == "__main__":
    main()