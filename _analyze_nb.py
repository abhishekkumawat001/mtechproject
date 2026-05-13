import json, sys

nb = json.load(open(r'Cyclone_Scale_Decomposition_Analysis_copy.ipynb', 'r', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
print(f"Kernel: {nb.get('metadata', {}).get('kernelspec', {})}")
print()

# Write full source to a text file for analysis
with open('_nb_full_source.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        f.write(f"{'='*80}\n")
        f.write(f"CELL {i} ({c['cell_type']}, {len(c.get('source',[]))} source lines)\n")
        f.write(f"{'='*80}\n")
        f.write(src)
        f.write('\n\n')
    
print(f"Written to _nb_full_source.txt")

# Summary
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = src.count('\n') + 1
    first_line = src.split('\n')[0][:120] if src else '(empty)'
    print(f"Cell {i:2d} | {c['cell_type']:8s} | {lines:4d} lines | {first_line}")
