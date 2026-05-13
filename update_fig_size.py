import json

file_path = r'c:\Users\abhik\Desktop\project related work\Cyclone_Scale_Decomposition_Analysis_copy.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_block = """        panel_w = 3.0
        fig_w   = max(26, n_cols * panel_w + 2.0)
        fig_h   = n_rows * 4.6 + 1.6
        fig = plt.figure(figsize=(fig_w, fig_h))

        gs = gridspec.GridSpec(
            n_rows + 1, n_cols,
            height_ratios=[0.055] + [1] * n_rows,
            hspace=0.22, wspace=0.03,
            left=0.05, right=0.90,
            top=0.93, bottom=0.05
        )"""

new_block = """        panel_w = 3.2
        panel_h = 3.2
        
        # Calculate figure size to ensure perfectly square grid cells for the map plots
        margin_left = 1.2
        margin_right = 2.5  # Room for colorbars
        margin_top = 1.2
        margin_bottom = 1.0
        
        wspace = 0.05
        hspace = 0.15
        header_h = 0.25 # Height of the header cell
        
        total_panel_width = n_cols * panel_w + (n_cols - 1) * panel_w * wspace
        total_panel_height = n_rows * panel_h + (n_rows - 1) * panel_h * hspace
        
        fig_w = margin_left + total_panel_width + margin_right
        fig_h = margin_bottom + total_panel_height + header_h + margin_top
        
        left = margin_left / fig_w
        right = 1.0 - (margin_right / fig_w)
        bottom = margin_bottom / fig_h
        top = 1.0 - (margin_top / fig_h)
        
        fig = plt.figure(figsize=(fig_w, fig_h))

        height_ratios = [header_h / panel_h] + [1] * n_rows
        
        gs = gridspec.GridSpec(
            n_rows + 1, n_cols,
            height_ratios=height_ratios,
            hspace=hspace, wspace=wspace,
            left=left, right=right,
            top=top, bottom=bottom
        )"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "panel_w = 3.0" in source:
            source = source.replace(old_block, new_block)
            
            # The colorbar alignment code also needs a slight adjustment
            # because we changed the figure size dynamics.
            # Old:
            # y0 = 0.05 + (n_rows - 1 - ri) * cbar_row_h + cbar_row_h * 0.12
            # cax = fig.add_axes([0.92, y0, 0.012, cbar_row_h * 0.70])
            
            # Since the rows are now defined properly with margins:
            # Height of one data row in figure fraction:
            # row_h_frac = panel_h / fig_h
            # The bottom of row ri (where ri=0 is the top data row, just below header)
            # wait, GridSpec indices: ri=0 is row 1 (since row 0 is header).
            # The physical bottom of row ri from the bottom of the figure:
            # bottom_margin + (n_rows - 1 - ri) * (panel_h + hspace*panel_h)
            
            # Instead of replacing colorbar placement here, we'll replace the loop
            old_cbar_block = """        cbar_row_h = 0.82 / n_rows
        cbar_axes = []
        for ri in range(n_rows):
            y0 = 0.05 + (n_rows - 1 - ri) * cbar_row_h + cbar_row_h * 0.12
            cax = fig.add_axes([0.92, y0, 0.012, cbar_row_h * 0.70])
            cbar_axes.append(cax)"""

            new_cbar_block = """        cbar_axes = []
        for ri in range(n_rows):
            # Calculate exact position for each row's colorbar
            # ri=0 is the top data row.
            row_bottom_in_inches = margin_bottom + (n_rows - 1 - ri) * (panel_h * (1 + hspace))
            y0 = row_bottom_in_inches / fig_h
            height = panel_h / fig_h
            
            # Add some vertical padding to colorbar so it doesn't touch the borders
            pad = 0.1 * height
            cax = fig.add_axes([right + 0.02, y0 + pad, 0.012, height - 2*pad])
            cbar_axes.append(cax)"""
            
            source = source.replace(old_cbar_block, new_cbar_block)
            
            # Adjust the sigma annotation position so it's not offscreen
            old_sig = "        fig.text(\n            0.50, 0.01,"
            new_sig = "        fig.text(\n            0.50, 0.02,"
            source = source.replace(old_sig, new_sig)
            
            # Adjust legend position
            old_leg = "bbox_to_anchor=(0.915, 0.055)"
            new_leg = "bbox_to_anchor=(right + 0.01, 0.02)"
            source = source.replace(old_leg, new_leg)
            
            cell['source'] = source.splitlines(True)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Figure dimensions dynamically calculated for square cells!")
