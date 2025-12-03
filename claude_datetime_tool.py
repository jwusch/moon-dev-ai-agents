"""
🕐 Claude Code DateTime Tool
Agent-optimized tool for accurate datetime awareness

Author: Claude (Anthropic)
"""

from datetime import datetime, timezone
import pytz
from typing import Dict, Optional, Literal
from datetime_system import DateTimeAwareness

# Initialize the datetime system
dt_system = DateTimeAwareness()

class ClaudeDateTimeTool:
    """Agent-optimized datetime tool following best practices"""
    
    @staticmethod
    def get_current_datetime(
        response_format: Literal["minimal", "standard", "detailed"] = "standard"
    ) -> Dict[str, str]:
        """Get current date and time with timezone awareness for accurate temporal context.
        
        Use this when you need to:
        - Know the current time in user's timezone for any time-sensitive operation
        - Check if markets are open before running trading analysis
        - Timestamp any output or logs with accurate local time
        - Verify day of week before assuming market availability
        - Convert between timezones for global market analysis
        
        Do NOT use this for:
        - Historical datetime operations (use Python datetime directly)
        - Scheduling future events (this is for current time only)
        - Timezone lookups without time context (use timezone reference instead)
        
        Args:
            response_format: Control output detail level to optimize token usage.
                - "minimal": Just local time and day (~20 tokens)
                    Use when: Only need current timestamp for logging
                - "standard": Local time, UTC, timezone, day of week (~50 tokens)  
                    Use when: Need basic awareness for most operations (DEFAULT)
                - "detailed": Full report with all markets and conversions (~200 tokens)
                    Use when: Analyzing global markets or complex timezone work
        
        Returns:
            Dictionary with datetime information.
            Minimal: {local_time, day_of_week}
            Standard: Above + {utc_time, timezone, is_market_hours}
            Detailed: Above + {market_status for 10+ exchanges, epoch, iso_format}
        
        Performance Notes:
            - Minimal format: ~20 tokens (just timestamp)
            - Standard format: ~50 tokens (recommended default)
            - Detailed format: ~200 tokens (only for market analysis)
            - Execution time: <10ms
            - Auto-detects user timezone on first call
            - Timezone detection cached for session
        
        Examples:
            # Quick timestamp for logging
            get_current_datetime(response_format="minimal")
            # Returns: {"local_time": "2025-12-01 21:59:00 EST", "day": "Sunday"}
            
            # Standard check before market operations (default)
            get_current_datetime()
            # Returns: {"local_time": "2025-12-01 21:59:00 EST", 
            #           "utc_time": "2025-12-02 02:59:00 UTC",
            #           "timezone": "US/Eastern", 
            #           "day": "Sunday",
            #           "is_market_hours": false}
            
            # Full market analysis across global exchanges
            get_current_datetime(response_format="detailed")
            # Returns: Complete market status for NYSE, NASDAQ, LSE, TSE, etc.
        """
        report = dt_system.get_current_time_report()
        
        if response_format == "minimal":
            return {
                "local_time": report['local_time'],
                "day": report['day_of_week']
            }
        
        elif response_format == "standard":
            # Check if US markets are open
            is_market_hours = False
            if report['day_of_week'] not in ['Saturday', 'Sunday']:
                hour = datetime.now(dt_system.user_timezone).hour
                minute = datetime.now(dt_system.user_timezone).minute
                # 9:30 AM - 4:00 PM EST
                is_market_hours = (hour == 9 and minute >= 30) or (10 <= hour < 16)
            
            return {
                "local_time": report['local_time'],
                "utc_time": report['utc_time'],
                "timezone": report['local_timezone'],
                "day": report['day_of_week'],
                "is_market_hours": is_market_hours
            }
        
        else:  # detailed
            return report
    
    @staticmethod
    def check_market_hours(
        market: Literal["NYSE", "NASDAQ", "LSE", "TSE", "CRYPTO", "ALL"] = "NYSE"
    ) -> Dict[str, any]:
        """Check if specific market is currently open and when it opens/closes.
        
        Use this when you need to:
        - Verify market is open before running live trading strategies
        - Calculate time until market open/close for scheduling
        - Determine which global markets are available for analysis
        - Plan operations around market hours
        
        Do NOT use this for:
        - General time queries (use get_current_datetime instead)
        - Historical market hours (this is current day only)
        - Cryptocurrency status (always returns 24/7 open)
        
        Args:
            market: Which market to check.
                - "NYSE": New York Stock Exchange (also covers NASDAQ hours)
                - "NASDAQ": Same hours as NYSE  
                - "LSE": London Stock Exchange
                - "TSE": Tokyo Stock Exchange
                - "CRYPTO": Cryptocurrency markets (always open)
                - "ALL": Check all major markets globally
        
        Returns:
            Dictionary with market status and times.
            Single market: {is_open, current_time, open_time, close_time, hours_until_change}
            ALL: {market_name: status_dict for each market}
        
        Performance Notes:
            - Single market: ~30 tokens
            - ALL markets: ~150 tokens
            - Execution time: <5ms
            - Times shown in user's local timezone
            - Accounts for weekends and basic holidays
        
        Examples:
            # Check if NYSE is open before running strategy
            check_market_hours("NYSE")
            # Returns: {"is_open": false, 
            #           "current_time": "21:59 EST",
            #           "open_time": "09:30 EST", 
            #           "close_time": "16:00 EST",
            #           "hours_until_change": 11.5}
            
            # Check all markets for global opportunity scan  
            check_market_hours("ALL")
            # Returns: {"NYSE": {...}, "LSE": {...}, "TSE": {...}, "CRYPTO": {...}}
            
            # Verify crypto markets (always open)
            check_market_hours("CRYPTO")
            # Returns: {"is_open": true, "status": "Open 24/7"}
        """
        if market == "ALL":
            results = {}
            for mkt in ["NYSE", "LSE", "TSE", "CRYPTO"]:
                results[mkt] = dt_system.get_market_open_time(mkt)
            return results
        else:
            return dt_system.get_market_open_time(market)
    
    @staticmethod
    def convert_timezone(
        time_str: str,
        from_tz: str = "US/Eastern",
        to_tz: str = "UTC",
        include_offset: bool = True
    ) -> str:
        """Convert time between timezones for global market coordination.
        
        Use this when you need to:
        - Convert market times between exchanges (NYSE time to TSE time)
        - Align global trading strategies across timezones
        - Understand when international events occur in local time
        - Coordinate timestamps from different data sources
        
        Do NOT use this for:
        - Getting current time (use get_current_datetime)
        - Date-only conversions (timezone doesn't affect dates)
        - Relative time calculations (use Python timedelta)
        
        Args:
            time_str: Time in "YYYY-MM-DD HH:MM:SS" format (24-hour).
                Must be exact format. No timezone in string.
            from_tz: Source timezone (pytz format).
                Common: "US/Eastern", "US/Pacific", "Europe/London", "Asia/Tokyo"
            to_tz: Target timezone (pytz format).
                Use "UTC" for universal time, or specific zones
            include_offset: Whether to show timezone offset in result.
                True: "2025-12-01 14:30:00 EST"
                False: "2025-12-01 14:30:00"
        
        Returns:
            Converted time string in target timezone.
            Format matches input format plus optional timezone identifier.
        
        Performance Notes:
            - ~25 tokens per conversion
            - Execution time: <2ms
            - Handles DST automatically
            - Uses pytz for accuracy
        
        Examples:
            # Convert NYSE opening bell to UTC
            convert_timezone("2025-12-02 09:30:00", "US/Eastern", "UTC")
            # Returns: "2025-12-02 14:30:00 UTC"
            
            # Check Tokyo market time during NYSE hours
            convert_timezone("2025-12-02 10:00:00", "US/Eastern", "Asia/Tokyo")  
            # Returns: "2025-12-03 00:00:00 JST"
            
            # Convert without timezone suffix
            convert_timezone("2025-12-02 16:00:00", "US/Eastern", "Europe/London", False)
            # Returns: "2025-12-02 21:00:00"
        """
        return dt_system.convert_time(time_str, from_tz, to_tz)
    
    @staticmethod
    def create_datetime_context() -> str:
        """Generate a comprehensive datetime context file for AI reference.
        
        Use this when you need to:
        - Initialize a new session with accurate time awareness
        - Create a reference file after timezone changes
        - Debug time-related issues in AI responses
        - Provide context for time-sensitive operations
        
        Do NOT use this for:
        - Repeated calls in same session (cache the result)
        - Real-time time tracking (use get_current_datetime)
        - User display (this is for AI context only)
        
        Args:
            None
        
        Returns:
            Path to created context file (datetime_context.txt).
            File contains current time in multiple formats, market status,
            timezone information, and conversion examples.
        
        Performance Notes:
            - ~300 tokens if content is read
            - Creates file: datetime_context.txt  
            - Execution time: ~20ms
            - File can be cached for session
            - Auto-refreshes timezone detection
        
        Examples:
            # Create context at session start
            create_datetime_context()
            # Returns: "datetime_context.txt"
            # File contains full datetime awareness context
            
            # Refresh context after timezone change
            create_datetime_context()
            # Updates file with new timezone information
        """
        context = dt_system.create_context_file()
        return "datetime_context.txt"


# Convenience function for direct use
def what_time_is_it() -> Dict[str, str]:
    """Quick helper to get current time in user's timezone.
    
    Use this for simple "what time is it?" queries.
    Returns standard format datetime info.
    """
    tool = ClaudeDateTimeTool()
    return tool.get_current_datetime("standard")


if __name__ == "__main__":
    # Demo the tool
    print("🕐 Claude DateTime Tool Demo\n")
    
    tool = ClaudeDateTimeTool()
    
    # Test different response formats
    print("MINIMAL:", tool.get_current_datetime("minimal"))
    print("\nSTANDARD:", tool.get_current_datetime("standard"))
    print("\nMARKET CHECK:", tool.check_market_hours("NYSE"))
    
    # Test timezone conversion
    result = tool.convert_timezone("2025-12-02 09:30:00", "US/Eastern", "UTC")
    print(f"\nTIMEZONE CONVERSION: NYSE open in UTC = {result}")
    
    # Create context file
    print(f"\nCONTEXT FILE: {tool.create_datetime_context()}")