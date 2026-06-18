download the l3 data 
write script for sorting the data *sorting_one_time_run.ipynb "not pushed on git yet"
write script to print all the map of cycle 1 of ROI swoth_l3_data_plot.ipynb "not push on git yet"

How original passes were found: The AVISO website publishes a shapefile of every SWOT orbit pass (its swath footprint on Earth). Each polygon has a pass number (ID_PASS). You intersect that shapefile with your region's bounding box → the polygon IDs that intersect are the pass numbers you hard-coded.

Downloads sph_science_swath.zip from AVISO into a swot_orbit_data/ folder next to your notebook.

