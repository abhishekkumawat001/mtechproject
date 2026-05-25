import json

nb_path = r'C:\Users\abhik\Desktop\project related work\Cyclone_Scale_Decomposition_Analysis_for_multiple_cyclone.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'phase=phase' in source:
            print(f"Found 'phase=phase' in cell {i} (execution_count: {cell.get('execution_count')})")
            print("--- SOURCE ---")
            print(source)
            print("--- END SOURCE ---")
