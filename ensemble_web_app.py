"""
🎯 Ensemble Strategy Builder Web App
Interactive Streamlit web app for building ensemble trading strategies

Run with: streamlit run ensemble_web_app.py

Author: Claude (Anthropic)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ensemble_alpha_strategy import EnsembleAlphaStrategy

# Configure Streamlit
st.set_page_config(
    page_title="Ensemble Strategy Builder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main web app"""
    
    st.title("🎯 Ensemble Alpha Strategy Builder")
    st.markdown("Build advanced trading strategies by combining top alpha sources")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Strategy Configuration")
        
        # Symbol input
        symbol = st.text_input(
            "Symbol",
            value="SPY",
            help="Enter a stock symbol (e.g., NVDA, AAPL, SPY, QQQ)"
        ).upper().strip()
        
        # Number of signals
        top_n = st.slider(
            "Top N Alpha Sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of top alpha sources to combine"
        )
        
        # Timeframes
        available_timeframes = ["1m", "5m", "15m", "1h", "1d"]
        selected_timeframes = st.multiselect(
            "Timeframes to Analyze",
            available_timeframes,
            default=["5m", "15m", "1h", "1d"],
            help="Select timeframes for alpha discovery"
        )
        
        # Data period
        period = st.selectbox(
            "Data Period",
            ["7d", "30d", "60d", "1y", "2y"],
            index=2,
            help="Historical data period for analysis"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            min_alpha_threshold = st.number_input(
                "Minimum Alpha Threshold",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Minimum alpha score for signal inclusion"
            )
            
            correlation_penalty = st.checkbox(
                "Apply Correlation Penalty",
                value=True,
                help="Reduce weights for highly correlated signals"
            )
            
            dynamic_sizing = st.checkbox(
                "Dynamic Position Sizing",
                value=True,
                help="Adjust position size based on signal confidence"
            )
        
        # Build button
        build_button = st.button(
            "🚀 Build Strategy",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if build_button:
        if not symbol:
            st.error("❌ Please enter a symbol")
            return
        
        if not selected_timeframes:
            st.error("❌ Please select at least one timeframe")
            return
        
        # Build strategy
        build_ensemble_strategy_ui(symbol, top_n, selected_timeframes, period, min_alpha_threshold)
    
    else:
        # Show welcome screen
        show_welcome_screen()
    
    # Footer with saved strategies
    show_saved_strategies()

def show_welcome_screen():
    """Show welcome screen with instructions"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Alpha Discovery
        - Scan multiple timeframes
        - Identify top performing strategies
        - Calculate risk-adjusted returns
        """)
    
    with col2:
        st.markdown("""
        ### ⚖️ Smart Combination
        - Dynamic signal weighting
        - Correlation analysis
        - Diversification optimization
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Backtesting
        - Complete performance metrics
        - Risk management rules
        - Interactive visualizations
        """)
    
    st.markdown("---")
    
    # Sample results
    st.subheader("📊 Sample Results")
    
    sample_data = {
        'Symbol': ['SPY', 'QQQ', 'NVDA'],
        'Sources Combined': [4, 4, 3],
        'Total Return': ['+5.4%', '+4.5%', '+8.2%'],
        'Win Rate': ['77.8%', '75.0%', '82.1%'],
        'Alpha Score': [0.47, 0.43, 0.89],
        'Sharpe Ratio': [0.87, 0.52, 1.23]
    }
    
    df_sample = pd.DataFrame(sample_data)
    st.dataframe(df_sample, use_container_width=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def build_ensemble_strategy_ui(symbol: str, top_n: int, timeframes: list, period: str, min_alpha_threshold: float):
    """Build ensemble strategy with caching"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize
        status_text.text("🔍 Initializing ensemble builder...")
        progress_bar.progress(10)
        
        ensemble = EnsembleAlphaStrategy(symbol, top_n_signals=top_n)
        ensemble.scanner.min_alpha_threshold = min_alpha_threshold
        
        # Step 2: Discover alpha sources
        status_text.text("🔍 Discovering alpha sources...")
        progress_bar.progress(30)
        
        alpha_sources = ensemble.discover_alpha_sources(timeframes)
        
        if not alpha_sources:
            st.error(f"❌ No alpha sources found for {symbol}")
            st.info("💡 Try lowering the minimum alpha threshold or using different timeframes")
            return
        
        # Step 3: Prepare data
        status_text.text("📊 Preparing ensemble data...")
        progress_bar.progress(50)
        
        df = ensemble.prepare_ensemble_data(period=period)
        
        if len(df) < 100:
            st.error(f"❌ Insufficient data for {symbol} (got {len(df)} bars, need 100+)")
            return
        
        # Step 4: Generate signals
        status_text.text("🎯 Generating ensemble signals...")
        progress_bar.progress(70)
        
        df = ensemble.generate_ensemble_signals(df)
        
        # Step 5: Backtest
        status_text.text("📈 Backtesting strategy...")
        progress_bar.progress(90)
        
        result = ensemble.backtest_ensemble_strategy(df)
        
        if result is None:
            st.error("❌ No trades generated - strategy may need adjustment")
            return
        
        progress_bar.progress(100)
        status_text.text("✅ Strategy built successfully!")
        
        # Display results
        display_strategy_results(symbol, ensemble, df, result, alpha_sources)
        
    except Exception as e:
        st.error(f"❌ Error building strategy: {e}")
        st.exception(e)

def display_strategy_results(symbol: str, ensemble: EnsembleAlphaStrategy, df: pd.DataFrame, 
                           result, alpha_sources):
    """Display comprehensive strategy results"""
    
    st.success(f"✅ Successfully built ensemble strategy for {symbol}")
    
    # Performance metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{result.total_return_pct:+.1f}%")
        st.metric("Win Rate", f"{result.win_rate:.1f}%")
    
    with col2:
        st.metric("Alpha Score", f"{result.alpha_score:.2f}")
        st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")
        st.metric("Total Trades", f"{result.total_trades}")
    
    with col4:
        st.metric("Profit Factor", f"{result.profit_factor:.2f}")
        st.metric("Alpha Sources", f"{len(alpha_sources)}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Signals", 
        "🎯 Alpha Sources", 
        "⚖️ Signal Analysis", 
        "📊 Performance", 
        "💾 Export"
    ])
    
    with tab1:
        show_price_and_signals(df, symbol)
    
    with tab2:
        show_alpha_sources(alpha_sources, ensemble.weights)
    
    with tab3:
        show_signal_analysis(df, ensemble)
    
    with tab4:
        show_performance_analysis(result, df)
    
    with tab5:
        show_export_options(symbol, ensemble, result, alpha_sources)

def show_price_and_signals(df: pd.DataFrame, symbol: str):
    """Show price chart with trading signals"""
    
    st.subheader(f"📈 {symbol} - Price & Trading Signals")
    
    # Create subplot with price and signals
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'{symbol} Price', 'Ensemble Signal', 'Confidence'],
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'], 
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Mark buy/sell signals
    buy_signals = df[df['trade_signal'] == 1]
    sell_signals = df[df['trade_signal'] == -1]
    
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Buy Signal'
            ),
            row=1, col=1
        )
    
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Sell Signal'
            ),
            row=1, col=1
        )
    
    # Ensemble signal
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ensemble_signal'],
            mode='lines',
            line=dict(color='cyan', width=2),
            name='Ensemble Signal'
        ),
        row=2, col=1
    )
    
    # Add thresholds
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-0.3, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    
    # Confidence
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ensemble_confidence'],
            mode='lines',
            fill='tonexty',
            line=dict(color='purple', width=1),
            name='Confidence'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_alpha_sources(alpha_sources, weights):
    """Show alpha sources details"""
    
    st.subheader("🎯 Alpha Sources Breakdown")
    
    # Create dataframe for display
    sources_data = []
    for source in alpha_sources:
        weight = weights.get(f"signal_{source.strategy_name.replace(f'_{source.timeframe}', '')}", 0)
        sources_data.append({
            'Strategy': source.strategy_name,
            'Timeframe': source.timeframe,
            'Alpha Score': round(source.alpha_score, 2),
            'Win Rate %': round(source.win_rate, 1),
            'Weight': round(weight, 3),
            'Confidence': round(source.confidence_score, 2),
            'Type': source.strategy_type.title()
        })
    
    df_sources = pd.DataFrame(sources_data)
    
    # Display table
    st.dataframe(df_sources, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Weight distribution pie chart
        fig_pie = px.pie(
            df_sources, 
            values='Weight', 
            names='Strategy',
            title="Signal Weight Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Alpha vs Win Rate scatter
        fig_scatter = px.scatter(
            df_sources,
            x='Win Rate %',
            y='Alpha Score',
            size='Weight',
            color='Type',
            hover_data=['Strategy'],
            title="Alpha Score vs Win Rate"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

def show_signal_analysis(df: pd.DataFrame, ensemble):
    """Show signal correlation and analysis"""
    
    st.subheader("⚖️ Signal Analysis")
    
    signal_columns = [col for col in df.columns if col.startswith('signal_')]
    
    if len(signal_columns) > 1:
        # Correlation matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Signal Correlation Matrix**")
            if ensemble.correlation_matrix is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                mask = np.triu(np.ones_like(ensemble.correlation_matrix, dtype=bool))
                sns.heatmap(
                    ensemble.correlation_matrix, 
                    mask=mask, 
                    annot=True, 
                    fmt='.2f',
                    cmap='RdYlGn', 
                    center=0, 
                    ax=ax
                )
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            st.markdown("**Signal Activity Timeline**")
            
            # Signal activity heatmap
            signal_activity = df[signal_columns].abs()  # Get signal strength
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(signal_activity.T.values, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(signal_columns)))
            ax.set_yticklabels([col.replace('signal_', '') for col in signal_columns])
            ax.set_xlabel('Time')
            ax.set_title('Signal Activity Over Time')
            plt.colorbar(im, ax=ax, label='Signal Strength')
            plt.tight_layout()
            st.pyplot(fig)

def show_performance_analysis(result, df):
    """Show detailed performance analysis"""
    
    st.subheader("📊 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Individual Component Performance**")
        if result.individual_performance:
            individual_data = []
            for signal_name, perf in result.individual_performance.items():
                individual_data.append({
                    'Signal': signal_name.replace('signal_', ''),
                    'Return %': round(perf['return'], 1),
                    'Win Rate %': round(perf['win_rate'], 1),
                    'Trades': perf['trades']
                })
            
            df_individual = pd.DataFrame(individual_data)
            st.dataframe(df_individual, use_container_width=True)
    
    with col2:
        st.markdown("**Strategy Quality Assessment**")
        
        # Quality score calculation
        quality_factors = {
            'Return Quality': min(100, max(0, result.total_return_pct * 10)),
            'Win Rate Quality': result.win_rate,
            'Risk Quality': max(0, 100 + result.max_drawdown),  # Less drawdown = higher quality
            'Consistency Quality': min(100, result.sharpe_ratio * 50)
        }
        
        overall_quality = sum(quality_factors.values()) / len(quality_factors)
        
        for factor, score in quality_factors.items():
            st.metric(factor, f"{score:.0f}/100")
        
        st.metric("**Overall Quality**", f"{overall_quality:.0f}/100")
    
    # Performance over time (if we had trade timestamps)
    st.markdown("**Signal Frequency Analysis**")
    
    signal_freq_data = {
        'Signal Type': ['Buy Signals', 'Sell Signals', 'No Signal'],
        'Count': [
            (df['trade_signal'] == 1).sum(),
            (df['trade_signal'] == -1).sum(),
            (df['trade_signal'] == 0).sum()
        ]
    }
    
    fig_freq = px.bar(
        x=signal_freq_data['Signal Type'],
        y=signal_freq_data['Count'],
        title="Trading Signal Frequency"
    )
    st.plotly_chart(fig_freq, use_container_width=True)

def show_export_options(symbol: str, ensemble, result, alpha_sources):
    """Show export and save options"""
    
    st.subheader("💾 Export & Save Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Strategy**")
        
        # Prepare export data
        export_data = {
            'symbol': symbol,
            'alpha_sources': [
                {
                    'name': source.strategy_name,
                    'timeframe': source.timeframe,
                    'alpha_score': source.alpha_score,
                    'win_rate': source.win_rate,
                    'confidence': source.confidence_score
                }
                for source in alpha_sources
            ],
            'weights': ensemble.weights,
            'performance': {
                'total_return_pct': result.total_return_pct,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'alpha_score': result.alpha_score,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'profit_factor': result.profit_factor
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Download button for JSON
        st.download_button(
            label="📥 Download Strategy (JSON)",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"{symbol}_ensemble_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Save locally button
        if st.button("💾 Save Strategy Locally"):
            filename = f"{symbol}_ensemble_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            st.success(f"✅ Strategy saved as {filename}")
    
    with col2:
        st.markdown("**Implementation Code**")
        
        # Generate implementation code
        code = f"""
# Implementation code for {symbol} ensemble strategy
from ensemble_alpha_strategy import EnsembleAlphaStrategy

# Initialize ensemble
ensemble = EnsembleAlphaStrategy('{symbol}', top_n_signals={len(alpha_sources)})

# Set weights (from your optimization)
ensemble.weights = {ensemble.weights}

# Use in trading system
def get_trading_signal(current_data):
    df = ensemble.generate_ensemble_signals(current_data)
    latest_signal = df['trade_signal'].iloc[-1]
    latest_confidence = df['ensemble_confidence'].iloc[-1]
    
    return {{
        'signal': latest_signal,
        'confidence': latest_confidence,
        'position_size': latest_confidence if latest_signal != 0 else 0
    }}
"""
        
        st.code(code, language='python')

def show_saved_strategies():
    """Show previously saved strategies"""
    
    st.markdown("---")
    st.subheader("📂 Previously Saved Strategies")
    
    # Find strategy files
    strategy_files = [f for f in os.listdir('.') if f.endswith('_ensemble_strategy.json')]
    
    if strategy_files:
        # Limit to most recent 5
        strategy_files = sorted(strategy_files, reverse=True)[:5]
        
        cols = st.columns(len(strategy_files))
        
        for i, filename in enumerate(strategy_files):
            with cols[i]:
                try:
                    with open(filename, 'r') as f:
                        strategy_data = json.load(f)
                    
                    symbol = strategy_data['symbol']
                    performance = strategy_data['performance']
                    
                    with st.container():
                        st.markdown(f"**{symbol}**")
                        st.metric("Return", f"{performance['total_return_pct']:+.1f}%")
                        st.metric("Win Rate", f"{performance['win_rate']:.1f}%")
                        
                        if st.button(f"Load {symbol}", key=f"load_{i}"):
                            st.session_state['loaded_strategy'] = strategy_data
                            st.rerun()
                
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
    else:
        st.info("No saved strategies found. Build your first strategy above!")

if __name__ == "__main__":
    main()