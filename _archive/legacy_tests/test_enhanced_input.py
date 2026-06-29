#!/usr/bin/env python3
"""
Test script for the enhanced input components
"""

import sys
import os
import shutil

# Add project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix agents.py before importing
agents_fixed_path = os.path.join('cosmic_cli', 'agents_fixed.py')
agents_path = os.path.join('cosmic_cli', 'agents.py')

# Make a backup of the original file
agents_backup_path = os.path.join('cosmic_cli', 'agents.py.bak')
shutil.copy2(agents_path, agents_backup_path)

# Replace the agents.py with the fixed version
shutil.copy2(agents_fixed_path, agents_path)
print("✅ Fixed agents.py file")

# Import our components
from cosmic_cli.enhanced_input import demo_enhanced_input
from cosmic_cli.ui_enhanced import run_enhanced_ui

if __name__ == "__main__":
    print("Testing Enhanced Input Components for Cosmic CLI")
    print("1. Test the enhanced input component only")
    print("2. Test the full enhanced UI")
    print("3. Restore original agents.py")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        demo_enhanced_input()
    elif choice == "2":
        run_enhanced_ui()
    elif choice == "3":
        # Restore original file
        shutil.copy2(agents_backup_path, agents_path)
        print("✅ Restored original agents.py file")
    else:
        print("Invalid choice!")
