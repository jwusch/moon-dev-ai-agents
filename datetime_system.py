"""
🕐 DateTime Awareness System for Claude Code AI
Foolproof system for accurate date/time handling

Author: Claude (Anthropic)
"""

import os
import sys
from datetime import datetime, timezone, timedelta
import pytz
import platform
import subprocess
from typing import Dict, Tuple, Optional
import json

class DateTimeAwareness:
    """
    Provides foolproof datetime awareness for AI systems.
    
    Key Features:
    - Auto-detects user's timezone
    - Provides current time in multiple formats
    - Calculates market hours globally
    - Handles timezone conversions
    - Validates datetime assumptions
    """
    
    def __init__(self):
        self.user_timezone = self._detect_timezone()
        self.market_timezones = {
            'US/Eastern': 'NYSE/NASDAQ',
            'Europe/London': 'LSE',
            'Europe/Berlin': 'XETRA',  # Frankfurt uses Berlin timezone
            'Asia/Tokyo': 'TSE',
            'Asia/Hong_Kong': 'HKSE',
            'Asia/Shanghai': 'SSE',
            'Australia/Sydney': 'ASX'
        }
        
    def _detect_timezone(self) -> pytz.timezone:
        """Auto-detect user's timezone"""
        # Method 1: Environment variable
        tz_env = os.environ.get('TZ')
        if tz_env:
            try:
                return pytz.timezone(tz_env)
            except:
                pass
        
        # Method 2: System timezone (Linux/Mac)
        if platform.system() != 'Windows':
            try:
                with open('/etc/timezone', 'r') as f:
                    tz_str = f.read().strip()
                    return pytz.timezone(tz_str)
            except:
                pass
        
        # Method 3: Windows timezone
        if platform.system() == 'Windows':
            try:
                # Use PowerShell to get timezone
                result = subprocess.run(
                    ['powershell', '-Command', 'Get-TimeZone | Select-Object -ExpandProperty Id'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    tz_str = result.stdout.strip()
                    # Map Windows timezone to pytz
                    windows_to_pytz = {
                        'Eastern Standard Time': 'US/Eastern',
                        'Central Standard Time': 'US/Central',
                        'Mountain Standard Time': 'US/Mountain',
                        'Pacific Standard Time': 'US/Pacific',
                    }
                    if tz_str in windows_to_pytz:
                        return pytz.timezone(windows_to_pytz[tz_str])
            except:
                pass
        
        # Default to US/Eastern (most common for finance)
        return pytz.timezone('US/Eastern')
    
    def get_current_time_report(self) -> Dict[str, str]:
        """
        Get comprehensive current time report.
        
        Returns dict with:
        - utc_time: Current UTC time
        - local_time: User's local time
        - local_timezone: User's timezone name
        - epoch: Unix timestamp
        - iso_format: ISO 8601 format
        - market_status: Status of major markets
        """
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(self.user_timezone)
        
        report = {
            'utc_time': now_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'local_time': now_local.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'local_timezone': str(self.user_timezone),
            'epoch': int(now_utc.timestamp()),
            'iso_format': now_utc.isoformat(),
            'day_of_week': now_local.strftime('%A'),
            'market_status': self._get_market_status(now_utc)
        }
        
        return report
    
    def _get_market_status(self, utc_time: datetime) -> Dict[str, str]:
        """Check which markets are currently open"""
        status = {}
        
        for tz_name, market_name in self.market_timezones.items():
            market_tz = pytz.timezone(tz_name)
            market_time = utc_time.astimezone(market_tz)
            
            # Check if weekend
            if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                status[market_name] = "Closed (Weekend)"
                continue
            
            hour = market_time.hour
            
            # Simplified market hours (actual hours vary)
            if market_name in ['NYSE/NASDAQ', 'US/Eastern']:
                is_open = 9.5 <= hour < 16  # 9:30 AM - 4:00 PM
            elif market_name == 'LSE':
                is_open = 8 <= hour < 16.5  # 8:00 AM - 4:30 PM
            elif market_name in ['XETRA', 'Europe/Berlin']:
                is_open = 9 <= hour < 17.5  # 9:00 AM - 5:30 PM
            elif market_name == 'TSE':
                is_open = (9 <= hour < 11.5) or (12.5 <= hour < 15)  # Morning and afternoon sessions
            elif market_name in ['HKSE', 'SSE']:
                is_open = (9.5 <= hour < 12) or (13 <= hour < 16)
            elif market_name == 'ASX':
                is_open = 10 <= hour < 16  # 10:00 AM - 4:00 PM
            else:
                is_open = False
            
            if is_open:
                status[market_name] = f"Open ({market_time.strftime('%H:%M')} local)"
            else:
                status[market_name] = f"Closed ({market_time.strftime('%H:%M')} local)"
        
        # Crypto markets
        status['Crypto'] = "Open 24/7"
        
        return status
    
    def convert_time(self, time_str: str, from_tz: str, to_tz: str) -> str:
        """Convert time between timezones"""
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)
        
        # Parse the time string
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        
        # Localize to source timezone
        dt = from_timezone.localize(dt)
        
        # Convert to target timezone
        dt = dt.astimezone(to_timezone)
        
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    def get_market_open_time(self, market: str, date: Optional[datetime] = None) -> Dict[str, str]:
        """Get market open/close times for a specific date"""
        if date is None:
            date = datetime.now(timezone.utc)
        
        # Define market hours
        market_hours = {
            'NYSE': {'tz': 'US/Eastern', 'open': '09:30', 'close': '16:00'},
            'NASDAQ': {'tz': 'US/Eastern', 'open': '09:30', 'close': '16:00'},
            'LSE': {'tz': 'Europe/London', 'open': '08:00', 'close': '16:30'},
            'TSE': {'tz': 'Asia/Tokyo', 'open': '09:00', 'close': '15:00'},
            'CRYPTO': {'tz': 'UTC', 'open': '00:00', 'close': '23:59'}
        }
        
        if market.upper() not in market_hours:
            return {'error': f'Unknown market: {market}'}
        
        info = market_hours[market.upper()]
        market_tz = pytz.timezone(info['tz'])
        
        # Get market date
        market_date = date.astimezone(market_tz).date()
        
        # Create open/close times
        open_time = market_tz.localize(datetime.strptime(f"{market_date} {info['open']}", '%Y-%m-%d %H:%M'))
        close_time = market_tz.localize(datetime.strptime(f"{market_date} {info['close']}", '%Y-%m-%d %H:%M'))
        
        # Convert to user's timezone
        open_local = open_time.astimezone(self.user_timezone)
        close_local = close_time.astimezone(self.user_timezone)
        
        return {
            'market': market.upper(),
            'date': str(market_date),
            'open_market_tz': open_time.strftime('%H:%M %Z'),
            'close_market_tz': close_time.strftime('%H:%M %Z'),
            'open_user_tz': open_local.strftime('%H:%M %Z'),
            'close_user_tz': close_local.strftime('%H:%M %Z'),
            'is_open_now': open_time <= date.astimezone(market_tz) < close_time
        }
    
    def create_context_file(self) -> str:
        """Create a context file for AI to read"""
        report = self.get_current_time_report()
        
        context = f"""# Current DateTime Context

Generated at: {datetime.now().isoformat()}

## Current Time
- UTC: {report['utc_time']}
- Local: {report['local_time']} 
- Timezone: {report['local_timezone']}
- Day: {report['day_of_week']}

## Market Status
"""
        for market, status in report['market_status'].items():
            context += f"- {market}: {status}\n"
        
        context += f"""
## Key Information
- User is in {report['local_timezone']} timezone
- When user says "now" or "current time", they mean {report['local_time']}
- Today's date in user's timezone: {datetime.now(self.user_timezone).date()}

## Time Conversion Examples
- 9:30 AM EST = 14:30 UTC
- 4:00 PM EST = 21:00 UTC
- Market hours are in local exchange time
"""
        
        # Save to file
        with open('datetime_context.txt', 'w') as f:
            f.write(context)
        
        return context


def main():
    """Demo the datetime system"""
    dt_system = DateTimeAwareness()
    
    print("=" * 80)
    print("🕐 DATETIME AWARENESS SYSTEM")
    print("=" * 80)
    
    # Get current time report
    report = dt_system.get_current_time_report()
    
    print("\n📅 CURRENT TIME REPORT:")
    print("-" * 40)
    for key, value in report.items():
        if key != 'market_status':
            print(f"{key:15}: {value}")
    
    print("\n🏛️ MARKET STATUS:")
    print("-" * 40)
    for market, status in report['market_status'].items():
        status_color = 'green' if 'Open' in status and 'Closed' not in status else 'red'
        print(f"{market:15}: {status}")
    
    # Demo timezone conversion
    print("\n🌍 TIMEZONE CONVERSION DEMO:")
    print("-" * 40)
    
    if 'Eastern' in str(dt_system.user_timezone):
        demo_time = "2025-12-02 09:30:00"
        print(f"NYSE Open (EST): {demo_time}")
        utc_time = dt_system.convert_time(demo_time, 'US/Eastern', 'UTC')
        print(f"In UTC: {utc_time}")
        tokyo_time = dt_system.convert_time(demo_time, 'US/Eastern', 'Asia/Tokyo')
        print(f"In Tokyo: {tokyo_time}")
    
    # Market hours
    print("\n📈 MARKET HOURS:")
    print("-" * 40)
    for market in ['NYSE', 'LSE', 'TSE']:
        info = dt_system.get_market_open_time(market)
        if 'error' not in info:
            print(f"\n{market}:")
            print(f"  Open: {info['open_user_tz']} (your time)")
            print(f"  Close: {info['close_user_tz']} (your time)")
            print(f"  Status: {'OPEN' if info['is_open_now'] else 'CLOSED'}")
    
    # Create context file
    print("\n📄 Creating datetime context file...")
    context = dt_system.create_context_file()
    print("Context file created: datetime_context.txt")
    
    print("\n" + "=" * 80)
    print("✅ System ready for foolproof datetime handling!")


if __name__ == "__main__":
    main()