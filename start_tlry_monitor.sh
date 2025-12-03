#!/bin/bash
# Start TLRY Exit Monitor

echo "🚨 TLRY EXIT MONITOR STARTER 🚨"
echo "==============================="
echo ""
echo "Choose your monitoring option:"
echo ""
echo "1) One-time check (no entry price):"
echo "   python tlry_exit_tracker.py"
echo ""
echo "2) One-time check with P&L (replace 7.00 with your entry price):"
echo "   python tlry_exit_tracker.py --entry 7.00"
echo ""
echo "3) Continuous monitoring every 15 minutes:"
echo "   python tlry_auto_monitor.py --entry 7.00 --interval 15"
echo ""
echo "4) Continuous monitoring every 5 minutes (more frequent):"
echo "   python tlry_auto_monitor.py --entry 7.00 --interval 5"
echo ""
echo "5) Continuous monitoring every 30 minutes (less frequent):"
echo "   python tlry_auto_monitor.py --entry 7.00 --interval 30"
echo ""
echo "==============================="
echo "Press Ctrl+C to stop continuous monitoring"
echo ""

# Default to continuous monitoring with 15 min intervals
echo "Starting default continuous monitor (15 min intervals)..."
echo "Replace 7.00 with your actual entry price:"
echo ""
python tlry_auto_monitor.py --entry 7.00 --interval 15