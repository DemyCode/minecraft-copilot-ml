#!/usr/bin/env python3
"""
Script to validate a sample of schematic files in the minecraft-schematics-raw directory.
This checks that each file can be loaded without errors using our schematic_loader function.
"""

import os
import sys
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from schematic_loader import load_schematic_to_numpy

# Number of files to sample
SAMPLE_SIZE = 500

def validate_schematic(file_path):
    """
    Attempt to load a schematic file and return success/failure info.
    
    Args:
        file_path (str): Path to the schematic file
        
    Returns:
        tuple: (file_path, success, error_message)
    """
    try:
        blocks, dimensions = load_schematic_to_numpy(file_path)
        return (file_path, True, dimensions)
    except Exception as e:
        error_message = traceback.format_exc()
        return (file_path, False, error_message)

def main():
    # Check if the directory exists
    raw_dir = 'minecraft-schematics-raw'
    if not os.path.exists(raw_dir):
        print(f"Error: Directory '{raw_dir}' not found.")
        sys.exit(1)
    
    # Get all .schematic files
    schematic_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.schematic'):
                schematic_files.append(os.path.join(root, file))
    
    total_files = len(schematic_files)
    if total_files == 0:
        print(f"No .schematic files found in '{raw_dir}'.")
        sys.exit(0)
    
    # Sample a subset of files
    sample_size = min(SAMPLE_SIZE, total_files)
    sampled_files = random.sample(schematic_files, sample_size)
    
    print(f"Found {total_files} schematic files. Validating a sample of {sample_size} files...")
    
    # Install tqdm if not already installed
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress display...")
        os.system('pip install tqdm')
        from tqdm import tqdm
    
    # Process files with a progress bar
    start_time = time.time()
    successful = 0
    failed = 0
    failed_files = []
    
    # Use multiprocessing to speed up validation
    max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 workers max
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(validate_schematic, file): file for file in sampled_files}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=sample_size, desc="Validating"):
            file_path, success, result = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                failed_files.append((file_path, result))
    
    # Calculate statistics
    success_rate = (successful / sample_size) * 100
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n--- Validation Results ---")
    print(f"Sample size: {sample_size} out of {total_files} total files")
    print(f"Successfully loaded: {successful} ({success_rate:.2f}%)")
    print(f"Failed to load: {failed} ({100 - success_rate:.2f}%)")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Write detailed error report if there were failures
    if failed:
        error_report = "error_report.txt"
        with open(error_report, 'w') as f:
            f.write(f"Schematic Validation Error Report\n")
            f.write(f"Sample size: {sample_size} out of {total_files} total files\n")
            f.write(f"Failed files: {failed}\n\n")
            
            for file_path, error in failed_files:
                f.write(f"\n--- {file_path} ---\n")
                f.write(f"{error}\n")
                f.write("-" * 80 + "\n")
        
        print(f"\nDetailed error report written to {error_report}")
    
    # Return success code based on validation results
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())