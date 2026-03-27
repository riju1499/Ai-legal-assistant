import json
import os
from pathlib import Path

# Directory containing the JSON files
cases_dir = "/home/salon/Wakalat Sewa/processed_llm/cases"

# Counter for tracking
total_files = 0
deleted_files = 0
error_files = []

# Iterate through all JSON files in the directory
for json_file in Path(cases_dir).glob("*.json"):
    total_files += 1
    
    try:
        # Read the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it has the error key with the specific value
        if data.get("error") is not None and data.get("error") != "" and type(data.get("error")) == str:
            # Delete the file
            os.remove(json_file)
            deleted_files += 1
            print(f"Deleted: {json_file.name}")
    
    except Exception as e:
        error_files.append((json_file.name, str(e)))
        print(f"Error processing {json_file.name}: {e}")

# Print summary
print("\n" + "="*60)
print(f"Summary:")
print(f"Total JSON files processed: {total_files}")
print(f"Files deleted: {deleted_files}")
print(f"Files with errors: {len(error_files)}")

if error_files:
    print("\nFiles with errors:")
    for filename, error in error_files:
        print(f"  - {filename}: {error}")