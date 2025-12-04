"""
🔍 DATA QUALITY MONITORING SYSTEM
Real-time monitoring and alerting for data quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
import warnings
from collections import deque
import threading
import time


class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    STALE_DATA = "stale_data"
    INVALID_OHLC = "invalid_ohlc"
    SUSPICIOUS_VOLUME = "suspicious_volume"
    GAP_DETECTED = "gap_detected"
    DUPLICATE_DATA = "duplicate_data"
    NEGATIVE_PRICE = "negative_price"
    EXTREME_SPREAD = "extreme_spread"
    TIMESTAMP_ERROR = "timestamp_error"


@dataclass
class QualityAlert:
    """Data quality alert"""
    timestamp: datetime
    issue_type: DataQualityIssue
    severity: str  # 'low', 'medium', 'high', 'critical'
    symbol: str
    description: str
    data_sample: Optional[Dict] = None
    suggested_action: Optional[str] = None


@dataclass
class DataQualityMetrics:
    """Overall data quality metrics"""
    timestamp: datetime
    total_points: int
    missing_points: int
    outlier_count: int
    invalid_ohlc_count: int
    gap_count: int
    quality_score: float  # 0-100
    latency_ms: Optional[float] = None
    issues_by_type: Optional[Dict[str, int]] = None


class DataQualityMonitor:
    """Main data quality monitoring system"""
    
    def __init__(self, 
                 outlier_threshold: float = 5.0,  # Standard deviations
                 max_gap_seconds: int = 60,
                 min_volume: float = 0,
                 max_spread_pct: float = 0.05,
                 alert_callback: Optional[callable] = None):
        
        self.outlier_threshold = outlier_threshold
        self.max_gap_seconds = max_gap_seconds
        self.min_volume = min_volume
        self.max_spread_pct = max_spread_pct
        self.alert_callback = alert_callback
        
        # Tracking
        self.alerts = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=1000)
        self.rolling_stats = {}
        
        # Logging
        self.logger = logging.getLogger('DataQualityMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Statistics cache
        self._stats_cache = {}
        self._cache_timestamp = None
        
    def check_data_quality(self, data: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """Comprehensive data quality check"""
        
        start_time = time.time()
        issues = []
        
        # Basic validation
        if data.empty:
            self._create_alert(
                DataQualityIssue.MISSING_DATA,
                'critical',
                symbol,
                "No data available"
            )
            return DataQualityMetrics(
                timestamp=datetime.now(),
                total_points=0,
                missing_points=0,
                outlier_count=0,
                invalid_ohlc_count=0,
                gap_count=0,
                quality_score=0
            )
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns 
                         if col not in data.columns and col.lower() not in data.columns]
        
        if missing_columns:
            self._create_alert(
                DataQualityIssue.MISSING_DATA,
                'critical',
                symbol,
                f"Missing columns: {missing_columns}"
            )
            return DataQualityMetrics(
                timestamp=datetime.now(),
                total_points=len(data),
                missing_points=len(data),
                outlier_count=0,
                invalid_ohlc_count=0,
                gap_count=0,
                quality_score=0
            )
        
        # Standardize column names
        data = self._standardize_columns(data)
        
        # Run quality checks
        missing_count = self._check_missing_data(data, symbol)
        outlier_count = self._check_outliers(data, symbol)
        invalid_ohlc_count = self._check_ohlc_validity(data, symbol)
        gap_count = self._check_gaps(data, symbol)
        volume_issues = self._check_volume(data, symbol)
        price_issues = self._check_prices(data, symbol)
        timestamp_issues = self._check_timestamps(data, symbol)
        
        # Calculate quality score
        total_issues = (missing_count + outlier_count + invalid_ohlc_count + 
                       gap_count + volume_issues + price_issues + timestamp_issues)
        
        quality_score = max(0, 100 - (total_issues / len(data) * 100))
        
        # Compile metrics
        latency = (time.time() - start_time) * 1000
        
        metrics = DataQualityMetrics(
            timestamp=datetime.now(),
            total_points=len(data),
            missing_points=missing_count,
            outlier_count=outlier_count,
            invalid_ohlc_count=invalid_ohlc_count,
            gap_count=gap_count,
            quality_score=quality_score,
            latency_ms=latency,
            issues_by_type={
                'missing': missing_count,
                'outliers': outlier_count,
                'invalid_ohlc': invalid_ohlc_count,
                'gaps': gap_count,
                'volume': volume_issues,
                'price': price_issues,
                'timestamp': timestamp_issues
            }
        )
        
        self.metrics_history.append(metrics)
        self._update_rolling_stats(symbol, metrics)
        
        # Alert on low quality
        if quality_score < 80:
            severity = 'critical' if quality_score < 50 else 'high' if quality_score < 70 else 'medium'
            self._create_alert(
                DataQualityIssue.MISSING_DATA,
                severity,
                symbol,
                f"Low data quality score: {quality_score:.1f}%",
                suggested_action="Review data source and consider fallback"
            )
        
        return metrics
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        column_map = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }
        
        return data.rename(columns=column_map)
    
    def _check_missing_data(self, data: pd.DataFrame, symbol: str) -> int:
        """Check for missing data points"""
        missing_count = data[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum().sum()
        
        if missing_count > 0:
            missing_pct = missing_count / (len(data) * 5) * 100
            self._create_alert(
                DataQualityIssue.MISSING_DATA,
                'high' if missing_pct > 5 else 'medium',
                symbol,
                f"{missing_count} missing values ({missing_pct:.1f}%)",
                data_sample={'missing_by_column': data.isna().sum().to_dict()},
                suggested_action="Use forward fill or interpolation"
            )
            
        return missing_count
    
    def _check_outliers(self, data: pd.DataFrame, symbol: str) -> int:
        """Check for statistical outliers"""
        outlier_count = 0
        
        # Price outliers using rolling statistics
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                # Calculate rolling mean and std
                rolling_mean = data[col].rolling(20, min_periods=10).mean()
                rolling_std = data[col].rolling(20, min_periods=10).std()
                
                # Identify outliers
                z_scores = np.abs((data[col] - rolling_mean) / rolling_std)
                outliers = z_scores > self.outlier_threshold
                
                if outliers.any():
                    outlier_indices = data.index[outliers].tolist()
                    outlier_count += len(outlier_indices)
                    
                    self._create_alert(
                        DataQualityIssue.OUTLIER,
                        'medium',
                        symbol,
                        f"{len(outlier_indices)} outliers in {col}",
                        data_sample={
                            'column': col,
                            'outlier_values': data.loc[outliers, col].head(5).to_dict()
                        },
                        suggested_action="Verify with alternative data source"
                    )
        
        return outlier_count
    
    def _check_ohlc_validity(self, data: pd.DataFrame, symbol: str) -> int:
        """Check OHLC relationship validity"""
        
        # High should be >= Low
        invalid_hl = data['High'] < data['Low']
        
        # Close should be between High and Low
        invalid_close = (data['Close'] > data['High']) | (data['Close'] < data['Low'])
        
        # Open should be between High and Low
        invalid_open = (data['Open'] > data['High']) | (data['Open'] < data['Low'])
        
        invalid_count = invalid_hl.sum() + invalid_close.sum() + invalid_open.sum()
        
        if invalid_count > 0:
            self._create_alert(
                DataQualityIssue.INVALID_OHLC,
                'high',
                symbol,
                f"{invalid_count} invalid OHLC relationships",
                data_sample={
                    'invalid_high_low': invalid_hl.sum(),
                    'invalid_close': invalid_close.sum(),
                    'invalid_open': invalid_open.sum(),
                    'sample_indices': data.index[invalid_hl | invalid_close | invalid_open].tolist()[:5]
                },
                suggested_action="Reject invalid bars or use previous valid values"
            )
            
        return invalid_count
    
    def _check_gaps(self, data: pd.DataFrame, symbol: str) -> int:
        """Check for time gaps in data"""
        gap_count = 0
        
        if isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff()
            
            # Expected frequency
            median_diff = time_diffs.median()
            
            # Gaps are > 2x expected frequency
            gaps = time_diffs > (median_diff * 2)
            
            if gaps.any():
                gap_count = gaps.sum()
                gap_locations = data.index[gaps].tolist()
                
                self._create_alert(
                    DataQualityIssue.GAP_DETECTED,
                    'medium',
                    symbol,
                    f"{gap_count} time gaps detected",
                    data_sample={
                        'gap_locations': gap_locations[:5],
                        'expected_frequency': str(median_diff),
                        'max_gap': str(time_diffs.max())
                    },
                    suggested_action="Check data source connectivity"
                )
                
        return gap_count
    
    def _check_volume(self, data: pd.DataFrame, symbol: str) -> int:
        """Check volume data quality"""
        issues = 0
        
        # Zero volume
        zero_volume = data['Volume'] == 0
        if zero_volume.any():
            issues += zero_volume.sum()
            self._create_alert(
                DataQualityIssue.SUSPICIOUS_VOLUME,
                'low',
                symbol,
                f"{zero_volume.sum()} bars with zero volume",
                suggested_action="Normal for some instruments during off-hours"
            )
        
        # Negative volume
        negative_volume = data['Volume'] < 0
        if negative_volume.any():
            issues += negative_volume.sum()
            self._create_alert(
                DataQualityIssue.SUSPICIOUS_VOLUME,
                'high',
                symbol,
                f"{negative_volume.sum()} bars with negative volume",
                suggested_action="Data error - reject these bars"
            )
            
        # Volume spikes (>10x average)
        avg_volume = data['Volume'].rolling(20).mean()
        volume_spikes = data['Volume'] > (avg_volume * 10)
        
        if volume_spikes.any():
            spike_count = volume_spikes.sum()
            self._create_alert(
                DataQualityIssue.SUSPICIOUS_VOLUME,
                'medium',
                symbol,
                f"{spike_count} extreme volume spikes",
                data_sample={
                    'spike_ratios': (data['Volume'] / avg_volume)[volume_spikes].head(5).to_dict()
                },
                suggested_action="Verify against news events or splits"
            )
            
        return issues
    
    def _check_prices(self, data: pd.DataFrame, symbol: str) -> int:
        """Check price data quality"""
        issues = 0
        
        # Negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            negative = data[col] <= 0
            if negative.any():
                issues += negative.sum()
                self._create_alert(
                    DataQualityIssue.NEGATIVE_PRICE,
                    'critical',
                    symbol,
                    f"{negative.sum()} negative/zero prices in {col}",
                    suggested_action="Critical data error - do not trade"
                )
        
        # Extreme spreads
        spread_pct = (data['High'] - data['Low']) / data['Close']
        extreme_spreads = spread_pct > self.max_spread_pct
        
        if extreme_spreads.any():
            issues += extreme_spreads.sum()
            self._create_alert(
                DataQualityIssue.EXTREME_SPREAD,
                'medium',
                symbol,
                f"{extreme_spreads.sum()} bars with extreme spreads",
                data_sample={
                    'spread_pcts': spread_pct[extreme_spreads].head(5).to_dict()
                },
                suggested_action="Possible data error or extreme volatility"
            )
            
        return issues
    
    def _check_timestamps(self, data: pd.DataFrame, symbol: str) -> int:
        """Check timestamp integrity"""
        issues = 0
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for duplicates
            duplicates = data.index.duplicated()
            if duplicates.any():
                issues += duplicates.sum()
                self._create_alert(
                    DataQualityIssue.DUPLICATE_DATA,
                    'high',
                    symbol,
                    f"{duplicates.sum()} duplicate timestamps",
                    data_sample={
                        'duplicate_times': data.index[duplicates].tolist()[:5]
                    },
                    suggested_action="Remove duplicates, keep last"
                )
            
            # Check for future timestamps (handle timezone aware data)
            try:
                now = datetime.now()
                # If data index is timezone aware, make comparison timezone aware
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    import pytz
                    now = now.replace(tzinfo=pytz.UTC)
                
                future = data.index > now
                if future.any():
                    issues += future.sum()
                    self._create_alert(
                        DataQualityIssue.TIMESTAMP_ERROR,
                        'critical',
                        symbol,
                        f"{future.sum()} future timestamps",
                        suggested_action="Critical error - check system time"
                    )
            except Exception:
                # Skip timestamp comparison if timezone issues
                pass
                
            # Check for very old data
            if len(data) > 0:
                try:
                    latest_time = data.index[-1]
                    now = datetime.now()
                    
                    # Handle timezone aware timestamps
                    if hasattr(latest_time, 'tz') and latest_time.tz is not None:
                        # Convert to naive for comparison
                        latest_time = latest_time.tz_localize(None) if hasattr(latest_time, 'tz_localize') else latest_time.replace(tzinfo=None)
                        
                    age = now - latest_time
                    if age > timedelta(minutes=5):
                        self._create_alert(
                            DataQualityIssue.STALE_DATA,
                            'medium',
                            symbol,
                            f"Latest data is {age} old",
                            suggested_action="Check data feed connection"
                        )
                except Exception:
                    # Skip age check if timestamp issues
                    pass
                    
        return issues
    
    def _create_alert(self, issue_type: DataQualityIssue, severity: str,
                     symbol: str, description: str, 
                     data_sample: Optional[Dict] = None,
                     suggested_action: Optional[str] = None):
        """Create and dispatch alert"""
        
        alert = QualityAlert(
            timestamp=datetime.now(),
            issue_type=issue_type,
            severity=severity,
            symbol=symbol,
            description=description,
            data_sample=data_sample,
            suggested_action=suggested_action
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, f"{symbol} - {issue_type.value}: {description}")
        
        # Callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
                
    def _update_rolling_stats(self, symbol: str, metrics: DataQualityMetrics):
        """Update rolling statistics for symbol"""
        
        if symbol not in self.rolling_stats:
            self.rolling_stats[symbol] = deque(maxlen=100)
            
        self.rolling_stats[symbol].append(metrics.quality_score)
    
    def get_quality_report(self, symbol: Optional[str] = None) -> Dict:
        """Get comprehensive quality report"""
        
        if symbol and symbol in self.rolling_stats:
            scores = list(self.rolling_stats[symbol])
            avg_quality = np.mean(scores) if scores else 0
            min_quality = min(scores) if scores else 0
            trend = "improving" if len(scores) > 2 and scores[-1] > scores[-2] else "declining"
        else:
            # Overall stats
            all_scores = []
            for scores in self.rolling_stats.values():
                all_scores.extend(scores)
            avg_quality = np.mean(all_scores) if all_scores else 0
            min_quality = min(all_scores) if all_scores else 0
            trend = "stable"
            
        # Recent alerts summary
        recent_alerts = list(self.alerts)[-50:]
        alert_summary = {}
        for alert in recent_alerts:
            key = f"{alert.issue_type.value}_{alert.severity}"
            alert_summary[key] = alert_summary.get(key, 0) + 1
            
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol or 'all',
            'average_quality_score': avg_quality,
            'min_quality_score': min_quality,
            'quality_trend': trend,
            'total_alerts': len(recent_alerts),
            'alert_summary': alert_summary,
            'recent_issues': [
                {
                    'time': alert.timestamp.isoformat(),
                    'type': alert.issue_type.value,
                    'severity': alert.severity,
                    'description': alert.description
                }
                for alert in recent_alerts[:10]
            ]
        }
    
    def monitor_live_feed(self, data_generator, symbol: str, 
                         check_interval: int = 60):
        """Monitor live data feed continuously"""
        
        def monitor_loop():
            while True:
                try:
                    # Get latest data
                    data = data_generator()
                    
                    # Check quality
                    metrics = self.check_data_quality(data, symbol)
                    
                    # Log metrics
                    self.logger.info(
                        f"{symbol} quality: {metrics.quality_score:.1f}% "
                        f"(issues: {sum(metrics.issues_by_type.values())})"
                    )
                    
                    # Sleep
                    time.sleep(check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitor error: {e}")
                    time.sleep(check_interval)
                    
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info(f"Started monitoring {symbol} every {check_interval}s")
        
        return monitor_thread


class DataQualityDashboard:
    """Web dashboard for data quality monitoring"""
    
    def __init__(self, monitor: DataQualityMonitor):
        self.monitor = monitor
        
    def generate_html_report(self) -> str:
        """Generate HTML dashboard"""
        
        report = self.monitor.get_quality_report()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Monitor</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 20px;
                    background: #f0f0f0;
                    border-radius: 8px;
                }}
                .good {{ background: #d4edda; }}
                .warning {{ background: #fff3cd; }}
                .bad {{ background: #f8d7da; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #4CAF50; color: white; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .alert-critical {{ background: #f44336; color: white; }}
                .alert-high {{ background: #ff9800; color: white; }}
                .alert-medium {{ background: #ffeb3b; color: black; }}
                .alert-low {{ background: #2196f3; color: white; }}
            </style>
            <script>
                function refreshData() {{
                    location.reload();
                }}
                setInterval(refreshData, 30000); // Refresh every 30s
            </script>
        </head>
        <body>
            <h1>📊 Data Quality Monitor</h1>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metrics">
                <div class="metric {'good' if report['average_quality_score'] > 80 else 'warning' if report['average_quality_score'] > 60 else 'bad'}">
                    <h3>Average Quality</h3>
                    <h1>{report['average_quality_score']:.1f}%</h1>
                </div>
                
                <div class="metric">
                    <h3>Total Alerts</h3>
                    <h1>{report['total_alerts']}</h1>
                </div>
                
                <div class="metric">
                    <h3>Quality Trend</h3>
                    <h1>{report['quality_trend'].upper()}</h1>
                </div>
            </div>
            
            <h2>Recent Issues</h2>
            <div class="alerts">
        """
        
        for issue in report['recent_issues']:
            severity_class = f"alert-{issue['severity']}"
            html += f"""
                <div class="alert {severity_class}">
                    <strong>{issue['time']}</strong> - 
                    {issue['type'].replace('_', ' ').title()}: 
                    {issue['description']}
                </div>
            """
            
        html += """
            </div>
            
            <h2>Issue Summary</h2>
            <table>
                <tr>
                    <th>Issue Type</th>
                    <th>Severity</th>
                    <th>Count</th>
                </tr>
        """
        
        for issue_key, count in report['alert_summary'].items():
            issue_type, severity = issue_key.split('_', 1)
            html += f"""
                <tr>
                    <td>{issue_type.replace('_', ' ').title()}</td>
                    <td>{severity.upper()}</td>
                    <td>{count}</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


def demo_quality_monitoring():
    """Test data quality monitoring with real market data"""
    
    print("🔍 DATA QUALITY MONITORING - REAL DATA")
    print("=" * 60)
    
    # Create monitor
    def alert_handler(alert: QualityAlert):
        print(f"\n⚠️ ALERT: [{alert.severity.upper()}] {alert.symbol}")
        print(f"   {alert.description}")
        if alert.suggested_action:
            print(f"   💡 {alert.suggested_action}")
            
    monitor = DataQualityMonitor(alert_callback=alert_handler)
    
    # Test with real market data
    test_symbols = ['SPY', 'QQQ', 'AAPL']
    
    try:
        import yfinance as yf
        
        for symbol in test_symbols:
            print(f"\n📊 Checking data quality for {symbol}...")
            
            # Get recent data
            data = yf.download(symbol, period='5d', interval='1m', progress=False)
            
            if not data.empty:
                # Fix multi-column issue if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Check quality
                metrics = monitor.check_data_quality(data, symbol)
                print(f"   Quality Score: {metrics.quality_score:.1f}%")
                print(f"   Data Points: {metrics.total_points}")
                print(f"   Issues: {sum(metrics.issues_by_type.values()) if metrics.issues_by_type else 0}")
            else:
                print(f"   ❌ No data available for {symbol}")
                
    except ImportError:
        print("❌ yfinance not available, using static test data")
        
        # Fallback to basic test
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        test_data = pd.DataFrame({
            'Open': 100,
            'High': 101,
            'Low': 99,
            'Close': 100,
            'Volume': 1000
        }, index=dates)
        
        metrics = monitor.check_data_quality(test_data, "TEST_DATA")
        print(f"Quality Score: {metrics.quality_score:.1f}%")
    
    # Get overall quality report
    print("\n📊 OVERALL QUALITY REPORT")
    print("=" * 60)
    report = monitor.get_quality_report()
    
    print(f"Average Quality Score: {report['average_quality_score']:.1f}%")
    print(f"Total Alerts: {report['total_alerts']}")
    print(f"Quality Trend: {report['quality_trend']}")
    
    # Create dashboard
    dashboard = DataQualityDashboard(monitor)
    html_report = dashboard.generate_html_report()
    
    with open('data_quality_report.html', 'w') as f:
        f.write(html_report)
        
    print("\n✅ Dashboard saved to data_quality_report.html")
    print("✅ Data quality check complete")


def monitor_positions_data_quality():
    """Monitor data quality for current open positions"""
    
    print("🔍 MONITORING DATA QUALITY FOR OPEN POSITIONS")
    print("=" * 60)
    
    try:
        from src.data.position_tracker import PositionTracker
        import yfinance as yf
        
        tracker = PositionTracker()
        positions = tracker.get_open_positions()
        
        if positions.empty:
            print("❌ No open positions to monitor")
            return
            
        # Create monitor
        def alert_handler(alert: QualityAlert):
            print(f"⚠️ ALERT: [{alert.severity.upper()}] {alert.symbol} - {alert.description}")
            
        monitor = DataQualityMonitor(alert_callback=alert_handler)
        
        print(f"Checking data quality for {len(positions)} open positions...")
        
        overall_quality = []
        
        for _, position in positions.iterrows():
            symbol = position['symbol']
            print(f"\n📊 {symbol}:")
            
            # Get recent data
            data = yf.download(symbol, period='2d', interval='1m', progress=False)
            
            if not data.empty:
                # Fix multi-column issue if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Check quality
                metrics = monitor.check_data_quality(data, symbol)
                overall_quality.append(metrics.quality_score)
                
                print(f"   Quality Score: {metrics.quality_score:.1f}%")
                print(f"   Data Points: {metrics.total_points}")
                if metrics.issues_by_type:
                    issues = sum(metrics.issues_by_type.values())
                    if issues > 0:
                        print(f"   Issues Found: {issues}")
                        for issue_type, count in metrics.issues_by_type.items():
                            if count > 0:
                                print(f"     {issue_type}: {count}")
                    else:
                        print("   ✅ No issues detected")
            else:
                print(f"   ❌ No data available")
                
        # Summary
        if overall_quality:
            avg_quality = sum(overall_quality) / len(overall_quality)
            print(f"\n📈 SUMMARY:")
            print(f"   Average Quality Score: {avg_quality:.1f}%")
            
            if avg_quality > 90:
                print(colored("   ✅ Excellent data quality", 'green'))
            elif avg_quality > 80:
                print(colored("   👍 Good data quality", 'yellow'))
            else:
                print(colored("   ⚠️ Data quality issues detected", 'red'))
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--positions':
        monitor_positions_data_quality()
    else:
        demo_quality_monitoring()