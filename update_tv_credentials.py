"""
🌙 Update TradingView Credentials
Helper to update TV_USERNAME and TV_PASSWORD in .env file
"""

import os
import getpass
from pathlib import Path

print("🌙 Update TradingView Credentials")
print("=" * 60)

# Check current .env
env_file = Path('.env')
env_content = ""
tv_username_line = -1
tv_password_line = -1

if env_file.exists():
    lines = env_file.read_text().split('\n')
    
    for i, line in enumerate(lines):
        if line.startswith('TV_USERNAME='):
            current_username = line.split('=', 1)[1].strip()
            print(f"Current username: {current_username}")
            tv_username_line = i
        elif line.startswith('TV_PASSWORD='):
            tv_password_line = i
            print(f"Current password: {'*' * 8} (hidden)")
            
    env_content = '\n'.join(lines)
else:
    print("❌ .env file not found!")
    print("Creating a new .env file...")
    
# Get new credentials
print("\n📝 Enter new credentials (or press Enter to keep current)")

new_username = input("TradingView email: ").strip()
new_password = getpass.getpass("TradingView password: ").strip()

if not new_username and tv_username_line == -1:
    print("❌ Username is required!")
    exit(1)

# Update or add credentials
if env_file.exists():
    lines = env_content.split('\n')
    
    # Update username
    if new_username:
        if tv_username_line >= 0:
            lines[tv_username_line] = f'TV_USERNAME={new_username}'
        else:
            # Add username
            lines.append(f'TV_USERNAME={new_username}')
    
    # Update password
    if new_password:
        if tv_password_line >= 0:
            lines[tv_password_line] = f'TV_PASSWORD={new_password}'
        else:
            # Add password
            lines.append(f'TV_PASSWORD={new_password}')
    
    # Write back
    env_file.write_text('\n'.join(lines))
    
else:
    # Create new .env
    content = f"""# TradingView Credentials
TV_USERNAME={new_username}
TV_PASSWORD={new_password}
TV_SERVER_URL=http://localhost:8888
"""
    env_file.write_text(content)

print("\n✅ Credentials updated in .env file")
print("\n⚠️  Important:")
print("1. Restart the TradingView server for changes to take effect")
print("2. Never commit the .env file to git")
print("3. Make sure your password doesn't have special characters that need escaping")

# Test the credentials
print("\n🔍 Quick validation...")
if new_username:
    if '@' not in new_username:
        print("⚠️  Username doesn't look like an email address")
    else:
        print("✅ Username format looks good")
        
if new_password:
    if len(new_password) < 6:
        print("⚠️  Password seems short")
    else:
        print("✅ Password length looks good")