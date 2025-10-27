#!/usr/bin/env python3
"""
Script to check all encoded JSON files for validity.
Prints the filename of any invalid JSON files found.
"""

import json
import os
import glob
import sys
from pathlib import Path


def check_json_file(file_path):
    """Check if a JSON file is valid."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"File error: {str(e)}"


def main():
    """Check all encoded JSON files for validity."""
    # Find all encoded JSON files
    pattern = '/home/u5as/as1748.u5as/frodobots/projects_link/frodobots_captions/**/*_encoded.json'
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} encoded JSON files to check...")
    print("=" * 60)
    
    invalid_files = []
    valid_count = 0
    
    for i, file_path in enumerate(files, 1):
        if i % 100 == 0:
            print(f"Checked {i}/{len(files)} files...", end='\r')
        
        is_valid, error = check_json_file(file_path)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_files.append((file_path, error))
            print(f"INVALID: {os.path.basename(file_path)}")
            print(f"  Path: {file_path}")
            print(f"  Error: {error}")
            print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total files checked: {len(files)}")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print(f"\nInvalid files found:")
        for file_path, error in invalid_files:
            print(f"  {file_path}")
    else:
        print("\nAll files are valid!")


if __name__ == "__main__":
    main()
