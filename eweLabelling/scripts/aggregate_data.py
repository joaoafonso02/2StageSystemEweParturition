import os
import pandas as pd
from pathlib import Path

# Define the source base directory and the destination directory
source_base_dir = Path(".")
destination_base_dir = Path("/aggregated_data")
destination_base_dir.mkdir(parents=True, exist_ok=True)

# List of days to aggregate
days_to_aggregate = ["20240520", "20240521", "20240522", "20240523", "20240524", "20240525", "20240526", "20240527", "20240528"]

# Get a list of all collars
collars = set()
for day in days_to_aggregate:
    day_dir = source_base_dir / day
    if day_dir.exists():
        for collar_dir in day_dir.iterdir():
            if collar_dir.is_dir():
                collars.add(collar_dir.name)
    else:
        print(f"Directory for day {day} does not exist")

# Debugging: Print the list of collars found
print(f"Collars found: {collars}")

# Aggregate data for each collar
for collar in collars:
    aggregated_data = pd.DataFrame()
    for day in days_to_aggregate:
        source_file = source_base_dir / day / collar / "all_data.csv"
        if source_file.exists():
            print(f"Reading data for collar {collar} on {day}")
            day_data = pd.read_csv(source_file)
            aggregated_data = pd.concat([aggregated_data, day_data], ignore_index=True)
        else:
            print(f"No data for collar {collar} on {day}")

    # Save the aggregated data to a new CSV file
    if not aggregated_data.empty:
        output_file = destination_base_dir / f"{collar}_20240520_20240528.csv"
        aggregated_data.to_csv(output_file, index=False)
        print(f"Aggregated data saved to {output_file}")
    else:
        print(f"No data to aggregate for collar {collar}")

print("Aggregation completed.")