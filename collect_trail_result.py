#!/usr/bin/env python3
"""
Script to collect and organize trial results from all trial_X_info.txt files
in subdirectories of cm16 folder. Extracts all parameters dynamically.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

def parse_trial_info(file_path: Path) -> Optional[Dict[str, any]]:
    """Parse a trial_X_info.txt file and extract all parameters and results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {}
        
        # Extract trial number from filename
        trial_match = re.search(r'trial_(\d+)_info\.txt', file_path.name)
        if trial_match:
            result['trial_number'] = int(trial_match.group(1))
        
        # Extract basic trial information
        val_acc_match = re.search(r'Best Validation Accuracy:\s*([\d.]+)', content)
        if val_acc_match:
            result['best_val_acc'] = float(val_acc_match.group(1))
        
        best_epoch_match = re.search(r'Best Epoch:\s*(\d+)', content)
        if best_epoch_match:
            result['best_epoch'] = int(best_epoch_match.group(1))
        
        best_threshold_match = re.search(r'Best Threshold:\s*([\d.]+)', content)
        if best_threshold_match:
            result['best_threshold'] = float(best_threshold_match.group(1))
        
        status_match = re.search(r'Status:\s*(\w+)', content)
        if status_match:
            result['status'] = status_match.group(1)
        
        # Extract Parameters section - dynamically extract all parameters
        params_section = re.search(r'Parameters:\s*\n=+\s*\n(.*?)(?:\n\n|\nLoss Weights:)', content, re.DOTALL)
        if params_section:
            params_text = params_section.group(1)
            # Extract all parameters using regex: param_name: value
            param_pattern = r'(\w+):\s*([\d.eE+-]+)'
            for match in re.finditer(param_pattern, params_text):
                param_name = match.group(1)
                param_value_str = match.group(2)
                # Try to convert to appropriate type
                try:
                    # Check if it's an integer
                    if '.' not in param_value_str and 'e' not in param_value_str.lower():
                        result[param_name] = int(param_value_str)
                    else:
                        result[param_name] = float(param_value_str)
                except ValueError:
                    result[param_name] = param_value_str
        
        # Extract Loss Weights
        w_bag_match = re.search(r'w_bag.*?:\s*([\d.]+)', content)
        if w_bag_match:
            result['w_bag'] = float(w_bag_match.group(1))
        
        w_ft_match = re.search(r'w_ft.*?:\s*([\d.]+)', content)
        if w_ft_match:
            result['w_ft'] = float(w_ft_match.group(1))
        
        w_max_match = re.search(r'w_max.*?:\s*([\d.]+)', content)
        if w_max_match:
            result['w_max'] = float(w_max_match.group(1))
        
        # Extract Test Results
        test_section = re.search(r'Test Results:\s*\n=+\s*\n(.*?)(?:\n\n|\nFinal Loss Weights Used:)', content, re.DOTALL)
        if test_section:
            test_text = test_section.group(1)
            # Extract Test Accuracy
            acc_match = re.search(r'Test Accuracy:\s*([\d.]+)', test_text)
            if acc_match:
                result['test_acc'] = float(acc_match.group(1))
            
            # Extract Test AUC
            auc_match = re.search(r'Test AUC:\s*([\d.]+)', test_text)
            if auc_match:
                result['test_auc'] = float(auc_match.group(1))
            
            # Extract Final Threshold
            final_threshold_match = re.search(r'Final Threshold:\s*([\d.]+)', test_text)
            if final_threshold_match:
                result['final_threshold'] = float(final_threshold_match.group(1))
        
        # Extract Timestamp
        timestamp_match = re.search(r'Timestamp:\s*([\d\s\-:]+)', content)
        if timestamp_match:
            result['timestamp'] = timestamp_match.group(1).strip()
        
        # Only return if we have essential fields (at least trial number and some results)
        if 'trial_number' in result:
            return result
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return None

def collect_all_trials(base_dir: Path) -> List[Dict[str, any]]:
    """Collect all trial results from subdirectories."""
    all_results = []
    
    # Find all trial_X_info.txt files
    for trial_file in base_dir.rglob('trial_*_info.txt'):
        result = parse_trial_info(trial_file)
        if result:
            all_results.append(result)
    
    return all_results

def format_float(value: float, decimals: int = 3) -> str:
    """Format float to avoid precision issues."""
    if value is None:
        return ''
    return f"{value:.{decimals}f}".rstrip('0').rstrip('.')

def get_all_fieldnames(results: List[Dict[str, any]]) -> List[str]:
    """Get all unique field names from all results."""
    all_fields: Set[str] = set()
    for result in results:
        all_fields.update(result.keys())
    
    # Define preferred order for common fields
    preferred_order = [
        'trial_number',
        'best_val_acc',
        'best_epoch',
        'best_threshold',
        'status',
        'sparsity_ratio',
        'eps_explore',
        'r1',
        'k_base',
        'n_heads',
        'pvm_d_state',
        'pvm_d_conv',
        'pvm_expand',
        'w_bag',
        'w_ft',
        'w_max',
        'test_acc',
        'test_auc',
        'final_threshold',
        'timestamp'
    ]
    
    # Start with preferred fields in order
    fieldnames = []
    for field in preferred_order:
        if field in all_fields:
            fieldnames.append(field)
            all_fields.remove(field)
    
    # Add remaining fields in sorted order
    fieldnames.extend(sorted(all_fields))
    
    return fieldnames

def write_results_csv(results: List[Dict[str, any]], output_file: Path):
    """Write results to CSV file."""
    if not results:
        print("No results to write.")
        return
    
    # Get all fieldnames dynamically
    fieldnames = get_all_fieldnames(results)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {}
            for field in fieldnames:
                value = result.get(field)
                if value is None:
                    row[field] = ''
                elif isinstance(value, float):
                    # Use appropriate decimal places based on field type
                    if field in ['test_acc', 'test_auc', 'best_val_acc', 'best_threshold', 'final_threshold']:
                        row[field] = format_float(value, 4)
                    elif field in ['w_bag', 'w_ft', 'w_max']:
                        row[field] = format_float(value, 4)
                    elif field in ['sparsity_ratio', 'eps_explore', 'r1']:
                        row[field] = format_float(value, 3)
                    else:
                        row[field] = format_float(value, 3)
                elif isinstance(value, int):
                    row[field] = str(value)
                else:
                    row[field] = str(value)
            writer.writerow(row)
    
    print(f"Results written to {output_file}")
    print(f"Total trials collected: {len(results)}")
    print(f"Total fields: {len(fieldnames)}")

def main():
    parser = argparse.ArgumentParser(
        description='Collect and organize trial results from trial_X_info.txt files'
    )
    parser.add_argument(
        'scan_path',
        type=str,
        nargs='?',
        default='/data/ljc/source/outputs/TopoMIL/cm16',
        help='Path to directory to scan for trial_X_info.txt files (default: /data/ljc/source/outputs/TopoMIL/cm16)'
    )
    parser.add_argument(
        'output_csv',
        type=str,
        nargs='?',
        default='/data/ljc/source/outputs/TopoMIL/cm16/trial_results_summary.csv',
        help='Path to output CSV file (default: /data/ljc/source/outputs/TopoMIL/cm16/trial_results_summary.csv)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    base_dir = Path(args.scan_path)
    output_file = Path(args.output_csv)
    
    # Validate scan path exists
    if not base_dir.exists():
        print(f"Error: Scan path does not exist: {base_dir}")
        return
    
    if not base_dir.is_dir():
        print(f"Error: Scan path is not a directory: {base_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {base_dir}")
    results = collect_all_trials(base_dir)
    
    # Sort results by trial number
    results.sort(key=lambda x: x.get('trial_number', 0))
    
    write_results_csv(results, output_file)

if __name__ == '__main__':
    main()

