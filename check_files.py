"""
File Series Checker

This script checks for missing and misaligned files in a directory based
on a specified file prefix, upgrade number, and step length. It
processes two file series with prefixes, `com_meta` and
`com_eulp_hvac_elec_MWh` (or `res_...`), and groups results by upgrade.

Features:
- Identifies missing files in the series
- Detects misaligned files where sequence overlaps are incorrect.
- Handles multiple upgrades and dynamic step lengths.
- Outputs results to both a text file and the terminal.

Usage:
- Update `directory` to the target folder path.
- Customize the `file_prefixes`, `upgrades` and `step_length` as needed.
- Run the script to generate and view the results.

Output:
- Results are printed to the terminal and saved to `check_results.txt`
  in the specified directory.

TODO (optional):
- Identify missing files before the first file.
- Remove "Misaligned files" processes as they are not necessary (?).
- Ensure that there are no files that should come after the last file
  (using meta_master?).
- Write main function for better modularity and reusability.
"""

import os
import re

def check_missing_files(directory, upgrade, file_prefix, step=15):
    """
    Checks for missing files and incorrect overlaps in a series based on
    the provided file prefix and upgrade number.

    Args:
        directory (str): Path to the directory containing the files.
        upgrade (int): Upgrade number to filter files.
        file_prefix (str): Prefix of the file series to check.
        step (int): Step size for the file ranges (default is 15).

    Returns:
        str: Results of the check for the given prefix and upgrade.
    """
    # Build regex pattern for the specific file group
    pattern = rf'{file_prefix}_upgrade{upgrade}_\d{{4}}-\d{{4}}\.csv'
    
    # List and filter files matching the pattern
    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    
    if not files:
        return f"No matching files found for prefix '{file_prefix}' and upgrade '{upgrade}'.\n"
    
    # Extract start and end indices
    indices = []
    for file in files:
        match = re.search(r'_(\d{4})-(\d{4})\.csv', file)
        if match:
            start, end = map(int, match.groups())
            indices.append((start, end))
    
    # Sort indices based on the start value
    indices.sort()
    
    # Check for missing files or incorrect overlap
    missing_ranges = []
    misaligned_files = []
    for i in range(len(indices) - 1):
        current_start, current_end = indices[i]
        next_start, next_end = indices[i + 1]
        
        # Check for missing files
        if current_end < next_start:
            # Add missing ranges between current_end and next_start
            for missing_start in range(current_end, next_start, step):
                missing_end = missing_start + step
                missing_ranges.append((missing_start, missing_end))
        
        # Check for misalignment (overlap must exist)
        if current_end != next_start:
            misaligned_files.append((current_start, current_end, next_start, next_end))
    
    # Build results summary
    results = []
    results.append(f"Results for {file_prefix} (upgrade {upgrade}):")
    results.append(f"Total files found: {len(files)}")
    results.append(f"First file: {file_prefix}_upgrade{upgrade}_{indices[0][0]:04d}-{indices[0][1]:04d}.csv")
    results.append(f"Last file: {file_prefix}_upgrade{upgrade}_{indices[-1][0]:04d}-{indices[-1][1]:04d}.csv")
    
    if missing_ranges:
        results.append("\nMissing file ranges:")
        for start, end in missing_ranges:
            results.append(f"{file_prefix}_upgrade{upgrade}_{start:04d}-{end:04d}.csv")
    else:
        results.append("\nNo missing files detected.")
    
    if misaligned_files:
        results.append("\nMisaligned files (missing overlap):")
        for current_start, current_end, next_start, next_end in misaligned_files:
            results.append(f"{file_prefix}_upgrade{upgrade}_{current_start:04d}-{current_end:04d}.csv -> "
                           f"{file_prefix}_upgrade{upgrade}_{next_start:04d}-{next_end:04d}.csv")
    else:
        results.append("\nNo misaligned files detected.")
    
    return "\n".join(results) + "\n"

if __name__ == "__main__":
    # Replace with your target directory
    directory = "/projects/geohc/jm_ReEDS-2.0/postprocessing/reValue/parallel_agg/outputs/outputs_2025-01-20-15-17-29"
    
    # File prefixes begin with either com_ or res_
    file_prefixes = ["com_meta", "com_eulp_hvac_elec_MWh"]
    
    # List of upgrades
    upgrades = [0, 1, 18]
    
    # Step length
    step_length = 15
    
    # Output file
    output_file = os.path.join(directory, "check_results.txt")
    
    # Clear previous results
    with open(output_file, "w") as f:
        f.write("File Check Results\n")
        f.write("=" * 40 + "\n")
    
    # Perform checks and group outputs by upgrade
    grouped_results = {}
    for upgrade in upgrades:
        grouped_results[upgrade] = []
        for prefix in file_prefixes:
            result = check_missing_files(directory, upgrade, prefix, step=step_length)
            grouped_results[upgrade].append(result)
    
    # Write results to file and print to terminal
    with open(output_file, "a") as f:
        for upgrade, results in grouped_results.items():
            upgrade_header = f"\nUpgrade {upgrade} Results\n" + "=" * 40 + "\n"
            f.write(upgrade_header)
            print(upgrade_header)  # Print to terminal
            
            for result in results:
                f.write(result + "\n")
                print(result)  # Print to terminal
    
    print(f"\nResults written to {output_file}")
