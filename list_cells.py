import json
nb = json.load(open(r'c:\Users\abhik\Desktop\project related work\CMEMS_Monthly_Visualization_2024.ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f'Cell {i} ({c["cell_type"]}): {repr(src[:100])}')
