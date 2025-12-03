"""
🕐 Setup DateTime Awareness for Claude Code
One-time setup script to ensure foolproof datetime handling

Author: Claude (Anthropic)
"""

import os
import json
from datetime import datetime
from claude_datetime_tool import ClaudeDateTimeTool
from datetime_system import DateTimeAwareness

def setup_datetime_awareness():
    """Setup datetime awareness system"""
    
    print("🕐 SETTING UP DATETIME AWARENESS SYSTEM")
    print("=" * 60)
    
    # 1. Initialize system
    dt_system = DateTimeAwareness()
    tool = ClaudeDateTimeTool()
    
    # 2. Create initial context file
    print("\n1. Creating datetime context file...")
    context_file = tool.create_datetime_context()
    print(f"   ✓ Created: {context_file}")
    
    # 3. Test timezone detection
    print("\n2. Testing timezone detection...")
    current = tool.get_current_datetime("detailed")
    print(f"   ✓ Detected timezone: {current['local_timezone']}")
    print(f"   ✓ Current local time: {current['local_time']}")
    
    # 4. Create CLAUDE.md update
    claude_md_addition = f"""
## DateTime Awareness System

This project includes a foolproof datetime awareness system to prevent timezone confusion.

### Quick Usage

```python
from claude_datetime_tool import what_time_is_it

# Get current time in user's timezone
time_info = what_time_is_it()
print(time_info['local_time'])  # "2025-12-01 22:15:00 EST"
```

### Key Commands

```python
# Get current time (always accurate to user's timezone)
from claude_datetime_tool import ClaudeDateTimeTool
tool = ClaudeDateTimeTool()

# Simple time check
current = tool.get_current_datetime("standard")
print(f"User's time: {{current['local_time']}}")
print(f"Market status: Check with tool.check_market_hours('NYSE')")

# Check specific market
nyse = tool.check_market_hours("NYSE")
print(f'NYSE is {{"open" if nyse["is_open"] else "closed"}}')

# Convert between timezones
utc_time = tool.convert_timezone("2025-12-02 09:30:00", "US/Eastern", "UTC")
```

### Important Notes

- **User timezone detected as: {current['local_timezone']}**
- When user says "now" or "current time", they mean their LOCAL time
- Always use the datetime tools instead of assuming times
- Run `python setup_datetime_awareness.py` to refresh timezone detection
"""
    
    # 5. Save configuration
    config = {
        "user_timezone": str(dt_system.user_timezone),
        "detected_at": datetime.now().isoformat(),
        "local_time_example": current['local_time'],
        "setup_complete": True
    }
    
    print("\n3. Saving datetime configuration...")
    with open('.datetime_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("   ✓ Saved: .datetime_config.json")
    
    # 6. Create startup check script
    startup_script = '''#!/usr/bin/env python3
"""Auto-load datetime awareness"""
try:
    from claude_datetime_tool import what_time_is_it
    time_info = what_time_is_it()
    print(f"⏰ Current time: {time_info['local_time']}")
except:
    pass
'''
    
    with open('datetime_check.py', 'w') as f:
        f.write(startup_script)
    print("   ✓ Created: datetime_check.py")
    
    # 7. Show summary
    print("\n" + "=" * 60)
    print("✅ DATETIME AWARENESS SETUP COMPLETE")
    print("=" * 60)
    
    print(f"\n📍 User timezone: {current['local_timezone']}")
    print(f"📅 Current time: {current['local_time']}")
    
    # Check if it's a weekday and market hours
    markets_status = "CLOSED"
    if current['day_of_week'] not in ['Saturday', 'Sunday']:
        # Simple check for US market hours
        hour = int(current['local_time'].split()[1].split(':')[0])
        if 9 <= hour < 16:
            markets_status = "OPEN (or near open)"
    print(f"📈 Markets: {markets_status}")
    
    print("\n📋 CLAUDE.md Addition:")
    print("-" * 40)
    print(claude_md_addition)
    
    print("\n🎯 Next Steps:")
    print("1. Add the above section to CLAUDE.md")
    print("2. Run 'from claude_datetime_tool import what_time_is_it' at start of sessions")
    print("3. Use ClaudeDateTimeTool for all time-related operations")
    
    return config

if __name__ == "__main__":
    config = setup_datetime_awareness()