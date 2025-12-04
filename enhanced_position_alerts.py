#!/usr/bin/env python
"""
🚨 ENHANCED POSITION ALERTS SYSTEM
Advanced alert and notification system for position monitoring
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
try:
    import smtplib
    from email.mime.text import MimeText
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
import subprocess
import platform
from termcolor import colored


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss" 
    REGIME_CHANGE = "regime_change"
    TECHNICAL_SIGNAL = "technical_signal"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_MOVEMENT = "price_movement"
    TIME_BASED = "time_based"


@dataclass
class TradingAlert:
    """Trading alert data structure"""
    timestamp: datetime
    symbol: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    current_price: float
    pnl_pct: float
    pnl_dollar: float
    recommended_action: str
    metadata: Optional[Dict] = None


class AlertManager:
    """Manages trading alerts and notifications"""
    
    def __init__(self):
        self.alerts_history = []
        self.sent_alerts = set()  # Avoid duplicate alerts
        self.alert_config = self._load_alert_config()
        
    def _load_alert_config(self) -> Dict:
        """Load alert configuration"""
        default_config = {
            "notifications": {
                "desktop": True,
                "email": False,
                "sound": True,
                "console": True
            },
            "thresholds": {
                "profit_target_pct": 15.0,
                "stop_loss_pct": -8.0,
                "large_move_pct": 5.0,
                "volume_spike_multiplier": 4.0
            },
            "email": {
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "to_address": ""
            },
            "cooldown_minutes": 15  # Minimum time between similar alerts
        }
        
        try:
            with open('alert_config.json', 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except:
            # Save default config
            with open('alert_config.json', 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
            
    def create_alert(self, symbol: str, alert_type: AlertType, priority: AlertPriority,
                    title: str, message: str, current_price: float, 
                    pnl_pct: float, pnl_dollar: float, 
                    recommended_action: str, metadata: Optional[Dict] = None) -> TradingAlert:
        """Create a new trading alert"""
        
        alert = TradingAlert(
            timestamp=datetime.now(),
            symbol=symbol,
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            current_price=current_price,
            pnl_pct=pnl_pct,
            pnl_dollar=pnl_dollar,
            recommended_action=recommended_action,
            metadata=metadata or {}
        )
        
        # Check if we should send this alert (avoid spam)
        if self._should_send_alert(alert):
            self._send_alert(alert)
            self.alerts_history.append(alert)
            
        return alert
        
    def _should_send_alert(self, alert: TradingAlert) -> bool:
        """Check if alert should be sent based on cooldown and duplicates"""
        
        # Create unique key for this type of alert
        alert_key = f"{alert.symbol}_{alert.alert_type.value}_{alert.priority.value}"
        
        # Check cooldown
        cooldown = timedelta(minutes=self.alert_config['cooldown_minutes'])
        cutoff_time = datetime.now() - cooldown
        
        # Look for recent similar alerts
        for recent_alert in self.alerts_history:
            if recent_alert.timestamp > cutoff_time:
                recent_key = f"{recent_alert.symbol}_{recent_alert.alert_type.value}_{recent_alert.priority.value}"
                if recent_key == alert_key:
                    return False  # Skip due to cooldown
                    
        return True
        
    def _send_alert(self, alert: TradingAlert):
        """Send alert via configured channels"""
        
        config = self.alert_config['notifications']
        
        # Console notification
        if config.get('console', True):
            self._send_console_alert(alert)
            
        # Desktop notification
        if config.get('desktop', False):
            self._send_desktop_notification(alert)
            
        # Sound alert
        if config.get('sound', False):
            self._play_alert_sound(alert.priority)
            
        # Email notification
        if config.get('email', False):
            self._send_email_alert(alert)
            
        # Log to file
        self._log_alert_to_file(alert)
        
    def _send_console_alert(self, alert: TradingAlert):
        """Send alert to console"""
        priority_colors = {
            AlertPriority.LOW: 'blue',
            AlertPriority.MEDIUM: 'yellow', 
            AlertPriority.HIGH: 'magenta',
            AlertPriority.CRITICAL: 'red'
        }
        
        color = priority_colors.get(alert.priority, 'white')
        attrs = ['bold'] if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL] else []
        
        print("\n" + "="*60)
        print(colored(f"🚨 TRADING ALERT - {alert.priority.value.upper()}", color, attrs=attrs))
        print(colored(f"Symbol: {alert.symbol}", color))
        print(colored(f"Title: {alert.title}", color))
        print(colored(f"Message: {alert.message}", color))
        print(colored(f"Price: ${alert.current_price:.2f}", color))
        print(colored(f"P&L: {alert.pnl_pct:+.1%} (${alert.pnl_dollar:+.2f})", color))
        print(colored(f"Action: {alert.recommended_action}", color, attrs=['bold']))
        print(colored(f"Time: {alert.timestamp.strftime('%H:%M:%S')}", color))
        print("="*60)
        
        # Add blinking for critical alerts
        if alert.priority == AlertPriority.CRITICAL:
            for _ in range(3):
                print(colored("🚨🚨🚨 CRITICAL ALERT 🚨🚨🚨", 'red', attrs=['bold', 'blink']))
                time.sleep(0.5)
                
    def _send_desktop_notification(self, alert: TradingAlert):
        """Send desktop notification"""
        title = f"{alert.symbol} - {alert.title}"
        message = f"{alert.message}\nP&L: {alert.pnl_pct:+.1%} (${alert.pnl_dollar:+.2f})\nAction: {alert.recommended_action}"
        
        try:
            if platform.system() == "Windows":
                # Windows toast notification
                import plyer
                plyer.notification.notify(
                    title=title,
                    message=message,
                    app_name="Trading Monitor",
                    timeout=10
                )
            elif platform.system() == "Darwin":  # macOS
                subprocess.run([
                    'osascript', '-e',
                    f'display notification "{message}" with title "{title}"'
                ])
            else:  # Linux
                subprocess.run(['notify-send', title, message])
        except Exception as e:
            print(f"Desktop notification failed: {e}")
            
    def _play_alert_sound(self, priority: AlertPriority):
        """Play alert sound based on priority"""
        try:
            if platform.system() == "Windows":
                import winsound
                if priority == AlertPriority.CRITICAL:
                    winsound.Beep(1000, 1000)  # High pitch, long beep
                elif priority == AlertPriority.HIGH:
                    winsound.Beep(800, 500)   # Medium pitch, medium beep
                else:
                    winsound.Beep(600, 200)   # Low pitch, short beep
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(['afplay', '/System/Library/Sounds/Glass.aiff'])
            else:  # Linux
                subprocess.run(['aplay', '/usr/share/sounds/alsa/Front_Left.wav'], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            print("🔔")  # Fallback visual alert
            
    def _send_email_alert(self, alert: TradingAlert):
        """Send email alert"""
        if not EMAIL_AVAILABLE:
            return
            
        email_config = self.alert_config.get('email', {})
        
        if not all([email_config.get('smtp_server'), email_config.get('username'), 
                   email_config.get('password'), email_config.get('to_address')]):
            return  # Email not configured
            
        try:
            subject = f"Trading Alert: {alert.symbol} - {alert.title}"
            body = f"""
Trading Alert Details:

Symbol: {alert.symbol}
Alert Type: {alert.alert_type.value}
Priority: {alert.priority.value.upper()}
Title: {alert.title}
Message: {alert.message}

Position Details:
Current Price: ${alert.current_price:.2f}
P&L: {alert.pnl_pct:+.1%} (${alert.pnl_dollar:+.2f})

Recommended Action: {alert.recommended_action}

Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from your trading monitor.
"""
            
            msg = MimeText(body)
            msg['Subject'] = subject
            msg['From'] = email_config['username']
            msg['To'] = email_config['to_address']
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            print(f"Email alert failed: {e}")
            
    def _log_alert_to_file(self, alert: TradingAlert):
        """Log alert to file"""
        log_entry = {
            'timestamp': alert.timestamp.isoformat(),
            'symbol': alert.symbol,
            'alert_type': alert.alert_type.value,
            'priority': alert.priority.value,
            'title': alert.title,
            'message': alert.message,
            'current_price': alert.current_price,
            'pnl_pct': alert.pnl_pct,
            'pnl_dollar': alert.pnl_dollar,
            'recommended_action': alert.recommended_action,
            'metadata': alert.metadata
        }
        
        filename = f'trading_alerts_{datetime.now().strftime("%Y%m%d")}.jsonl'
        with open(filename, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def get_recent_alerts(self, hours: int = 24) -> List[TradingAlert]:
        """Get recent alerts within specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history if alert.timestamp > cutoff]
        
    def get_alerts_by_symbol(self, symbol: str) -> List[TradingAlert]:
        """Get all alerts for a specific symbol"""
        return [alert for alert in self.alerts_history if alert.symbol == symbol]
        
    def clear_old_alerts(self, days: int = 7):
        """Remove old alerts from memory"""
        cutoff = datetime.now() - timedelta(days=days)
        self.alerts_history = [alert for alert in self.alerts_history if alert.timestamp > cutoff]


def create_smart_alerts_from_exit_signals(symbol: str, exit_signals: List[Dict], 
                                        metrics: Dict, alert_manager: AlertManager) -> List[TradingAlert]:
    """Convert exit signals to smart alerts"""
    alerts = []
    
    for signal in exit_signals:
        # Determine alert type and priority
        alert_type = AlertType.TECHNICAL_SIGNAL
        priority = AlertPriority.MEDIUM
        
        if signal['type'] == 'PROFIT_TARGET':
            alert_type = AlertType.PROFIT_TARGET
            priority = AlertPriority.HIGH
        elif signal['type'] == 'STOP_LOSS':
            alert_type = AlertType.STOP_LOSS  
            priority = AlertPriority.CRITICAL
        elif signal['type'] == 'REGIME_CHANGE':
            alert_type = AlertType.REGIME_CHANGE
            priority = AlertPriority.HIGH
        elif 'VOLUME' in signal['type']:
            alert_type = AlertType.VOLUME_ANOMALY
            priority = AlertPriority.MEDIUM
            
        # Create alert
        alert = alert_manager.create_alert(
            symbol=symbol,
            alert_type=alert_type,
            priority=priority,
            title=signal['type'].replace('_', ' ').title(),
            message=signal['reason'],
            current_price=metrics['current_price'],
            pnl_pct=metrics['pnl_pct'],
            pnl_dollar=metrics['pnl_dollar'],
            recommended_action=signal['action'],
            metadata={
                'urgency': signal['urgency'],
                'days_held': metrics['days_held'],
                'hurst_analysis': metrics.get('hurst_analysis')
            }
        )
        
        alerts.append(alert)
        
    return alerts


def demo_alert_system():
    """Demo the enhanced alert system"""
    print(colored("🚨 ENHANCED ALERT SYSTEM DEMO", 'cyan', attrs=['bold']))
    print("="*60)
    
    alert_manager = AlertManager()
    
    # Demo alerts of different priorities
    test_alerts = [
        {
            'symbol': 'DEMO',
            'alert_type': AlertType.PROFIT_TARGET,
            'priority': AlertPriority.HIGH,
            'title': 'Profit Target Reached',
            'message': 'Position up 18% - consider taking profits',
            'current_price': 118.50,
            'pnl_pct': 0.18,
            'pnl_dollar': 1850.00,
            'recommended_action': 'SELL_75%'
        },
        {
            'symbol': 'TEST',
            'alert_type': AlertType.REGIME_CHANGE,
            'priority': AlertPriority.MEDIUM,
            'title': 'Market Regime Change',
            'message': 'Hurst exponent indicates shift to mean-reverting regime',
            'current_price': 45.30,
            'pnl_pct': 0.06,
            'pnl_dollar': 120.00,
            'recommended_action': 'MONITOR_CLOSELY'
        }
    ]
    
    for alert_data in test_alerts:
        alert = alert_manager.create_alert(**alert_data)
        time.sleep(2)  # Space out alerts
        
    # Show recent alerts
    recent = alert_manager.get_recent_alerts(1)
    print(f"\n📋 Recent alerts: {len(recent)}")
    
    print("\n✅ Alert system demo complete")


if __name__ == "__main__":
    demo_alert_system()