#!/usr/bin/env python3
"""
Fix subject_id di semua JSON metadata files.
Ubah dari "835" (tangan kiri) menjadi "836" (tangan kanan).
"""

import json
from pathlib import Path
import sys


def fix_json_metadata(json_path: Path, old_subject_id: str, new_subject_id: str) -> bool:
    """
    Update subject_id di JSON file.
    
    Returns:
        True if updated, False if skipped
    """
    try:
        # Read JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if subject_id exists and needs update
        if 'subject_id' not in data:
            return False
        
        if data['subject_id'] != old_subject_id:
            return False
        
        # Update subject_id
        data['subject_id'] = new_subject_id
        
        # Write back
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def main():
    dataset_root = Path("dataset_multi_distance/836")
    
    if not dataset_root.exists():
        print(f"Error: {dataset_root} tidak ditemukan")
        sys.exit(1)
    
    print(f"Fixing subject_id metadata in {dataset_root}")
    print("Changing: 835 (tangan kiri) → 836 (tangan kanan)")
    print("=" * 80)
    
    total_updated = 0
    total_skipped = 0
    
    # Find all JSON files
    json_files = list(dataset_root.rglob("*_preprocess.json"))
    
    print(f"Found {len(json_files)} JSON metadata files\n")
    
    for json_path in sorted(json_files):
        updated = fix_json_metadata(json_path, "835", "836")
        
        if updated:
            total_updated += 1
            print(f"✓ Updated: {json_path.relative_to(dataset_root.parent)}")
        else:
            total_skipped += 1
    
    print("\n" + "=" * 80)
    print(f"DONE")
    print(f"  Updated: {total_updated} files")
    print(f"  Skipped: {total_skipped} files")


if __name__ == "__main__":
    main()
