#!/usr/bin/env python3
"""
Interactive wrapper for moviewatching.py
Usage: python run_movie.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Path to the original script
    script_path = Path.home() / "Documents/moviewatching/experiment/moviewatching.py"
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Prompt for subject ID
    subject_id = input("Enter Subject ID: ").strip()
    
    if not subject_id:
        print("Error: Subject ID cannot be empty")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(script_path), "--subject", subject_id]
    
    print(f"\nStarting experiment for subject: {subject_id}\n")
    
    # Run the script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script exited with code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()