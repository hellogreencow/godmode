#!/usr/bin/env python3
"""
GodMode Installation Script
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def install_godmode():
    """Install GodMode CLI"""
    print("🚀 Installing GodMode CLI...")
    
    # Install main package
    if not run_command("pip install -e .", "Installing GodMode core"):
        return False
    
    # Install CLI
    if not run_command("pip install -e . --force-reinstall", "Installing CLI"):
        return False
    
    # Install additional dependencies
    if not run_command("pip install click rich", "Installing CLI dependencies"):
        return False
    
    print("✅ GodMode CLI installed successfully!")
    print("\n🎯 Try these commands:")
    print("  godmode --help")
    print("  godmode test")
    print("  godmode start")
    
    return True

if __name__ == "__main__":
    install_godmode()
