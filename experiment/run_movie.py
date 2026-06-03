#!/usr/bin/env python3
"""
Interactive wrapper for moviewatching.py with FFmpeg warning filtering
Usage: 
    python run_movie.py                    # Interactive mode
    python run_movie.py --mock             # With mock tracker
    python run_movie.py --debug --mock     # Debug mode with mock tracker
"""

import subprocess
import sys
import threading
from pathlib import Path
import argparse

def filter_output(pipe, output_stream, filter_terms):
    """Filter lines containing certain terms from a pipe"""
    for line in iter(pipe.readline, b''):
        line_str = line.decode('utf-8', errors='replace')
        # Check if line should be filtered
        should_filter = any(term in line_str for term in filter_terms)
        if not should_filter:
            output_stream.write(line_str)
            output_stream.flush()
    pipe.close()

def main():
    # Path to the original script (same directory as this wrapper)
    script_dir = Path(__file__).parent.resolve()
    script_path = script_dir / "moviewatching.py"
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        print(f"Looking in: {script_dir}")
        sys.exit(1)
    
    # Parse command line arguments for pass-through
    parser = argparse.ArgumentParser(description="Wrapper for moviewatching.py with FFmpeg filtering")
    parser.add_argument('--subject', type=str, help='Subject ID (will prompt if not provided)')
    parser.add_argument('--mock', action='store_true', help='Use mock eye tracker')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true', help='Disable randomization of calibration/validation point order')
    parser.add_argument('--baby', action='store_true', help='Enable baby-specific experiment mode')
    args = parser.parse_args()
    
    # Get subject ID
    if args.subject:
        subject_id = args.subject
    else:
        subject_id = input("Enter Subject ID: ").strip()
    
    if not subject_id:
        print("Error: Subject ID cannot be empty")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(script_path), "--subject", subject_id]
    
    # Add optional flags
    if args.mock:
        cmd.append("--mock")
    if args.debug:
        cmd.append("--debug")
    if args.no_shuffle:
        cmd.append("--no-shuffle")
    if args.baby:
        cmd.append("--baby")
    
    flags_str = " ".join(["--mock" if args.mock else "", "--debug" if args.debug else ""]).strip()
    print(f"\nStarting experiment for subject: {subject_id} {flags_str}\n")
    
    # Terms to filter out
    filter_terms = [
        '[swscaler @',
        'No accelerated colorspace conversion',
        '[mov,mp4,m4a,3gp,3g2,mj2 @',
        'Missing key frame while searching',
        'Cannot find an index entry',
    ]
    
    # Run the script with filtered stderr only (stdout remains interactive)
    try:
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,  # Only pipe stderr
            # stdout and stdin remain connected to terminal for interactivity
            bufsize=0
        )
        
        # Only filter stderr (where FFmpeg warnings go)
        stderr_thread = threading.Thread(
            target=filter_output,
            args=(process.stderr, sys.stderr, filter_terms),
            daemon=True
        )
        
        stderr_thread.start()
        
        # Wait for completion
        returncode = process.wait()
        stderr_thread.join()
        
        if returncode != 0:
            print(f"\nError: Script exited with code {returncode}")
            sys.exit(returncode)
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        process.terminate()
        sys.exit(130)

if __name__ == "__main__":
    main()