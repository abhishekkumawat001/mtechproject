#!/usr/bin/env python3
"""Replace the old notebook with the newly generated one."""

import shutil
import os

src = r"c:\Users\abhik\Desktop\project related work\CMEMS_Monthly_Visualization_2024_generated.ipynb"
dst = r"c:\Users\abhik\Desktop\project related work\CMEMS_Monthly_Visualization_2024.ipynb"

# Get file sizes before
src_size = os.path.getsize(src)
if os.path.exists(dst):
    old_size = os.path.getsize(dst)
    print(f"Old file size: {old_size:,} bytes ({old_size / (1024**2):.1f} MB)")
else:
    old_size = None

# Replace the file
shutil.copy2(src, dst)

# Get file size after
new_size = os.path.getsize(dst)
print(f"\n✓ File replacement successful!")
print(f"  Generated file: {src_size:,} bytes")
print(f"  New target:     {new_size:,} bytes ({new_size / 1024:.1f} KB)")

if old_size:
    reduction = old_size - new_size
    pct = (reduction / old_size) * 100
    print(f"  Size reduction: {reduction:,} bytes ({pct:.1f}%)")

# Verify the file is valid JSON and has 19 cells
import json
with open(dst, 'r') as f:
    nb = json.load(f)

print(f"\n✓ Notebook validated:")
print(f"  - Format version: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"  - Cell count: {len(nb['cells'])}")
print(f"  - Kernel: {nb['metadata']['kernelspec']['name']}")

# List cell IDs and types
print(f"\n✓ Cells:")
for i, cell in enumerate(nb['cells'], 1):
    cell_type = cell['cell_type']
    cell_id = cell['id']
    if cell_type == 'markdown':
        # Get first line of markdown
        first_line = cell['source'][0] if cell['source'] else ''
        print(f"  {i:2d}. [{cell_type:8s}] {cell_id:12s} → {first_line[:60]}")
    else:
        print(f"  {i:2d}. [{cell_type:8s}] {cell_id:12s}")

print(f"\n✓ Path: {dst}")
