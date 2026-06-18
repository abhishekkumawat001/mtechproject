import json

# Load file
with open("cyclone_info_2024.json", "r") as f:
    data = json.load(f)

total_removed = 0

# Loop through all basins
for basin_name, basin_data in data.items():
    
    # Skip if no storms key (safety check)
    if "storms" not in basin_data:
        continue
    
    storms = basin_data["storms"]
    
    seen = set()
    unique_storms = []
    
    for storm in storms:
        name = storm["name"]
        
        if name not in seen:
            seen.add(name)
            unique_storms.append(storm)
    
    removed = len(storms) - len(unique_storms)
    total_removed += removed
    
    # Replace cleaned list
    basin_data["storms"] = unique_storms
    
    print(f"{basin_name}: removed {removed} duplicates")

# Save cleaned file
with open("cyclone_info_2024_1.json", "w") as f:
    json.dump(data, f, indent=4)

print(f"\nTotal duplicates removed: {total_removed}")