#!/usr/bin/env python3
"""Auto-load datetime awareness"""
try:
    from claude_datetime_tool import what_time_is_it
    time_info = what_time_is_it()
    print(f"⏰ Current time: {time_info['local_time']}")
except:
    pass
