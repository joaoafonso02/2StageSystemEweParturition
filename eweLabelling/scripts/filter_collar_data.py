import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

# Define the source directory for the large CSV files and the output directory
source_base_dir = Path("../data/finalDataSet")
output_path = Path("../data/ewes")
output_path.mkdir(parents=True, exist_ok=True)

# Load JSON data
json_file = Path("../data/ewe_data.json")
with open(json_file, 'r') as f:
    ewe_data = json.load(f)

# Function to process each chunk of the CSV file
def process_chunk(chunk, lambing_times, lambing_type, start_time, end_time):
    chunk['Time'] = pd.to_datetime(chunk['Time'], unit='ms')  # Convert Time from ms to datetime
    print(f"Filtering data between {start_time} and {end_time}")
    chunk = chunk[(chunk['Time'] >= start_time) & (chunk['Time'] <= end_time)]  # Filter data based on start and end times
    chunk['Class'] = chunk['Time'].apply(lambda x: assign_class(x, lambing_times, lambing_type))
    return chunk

# Function to assign class based on lambing times
def assign_class(time, lambing_times, lambing_type):
    if not lambing_times:
        return 13  # Default to non-partum if no lambing times are provided

    first_lambing = lambing_times[0]
    last_lambing = get_last_lambing_time(lambing_times, lambing_type)
    class_0_end = last_lambing + timedelta(minutes=5)

    for lambing_time in lambing_times:
        diff = (time - lambing_time).total_seconds() / 3600  # Difference in hours

        # Class 0: From first lambing to 5 minutes after the last lambing within 2 hours
        if first_lambing <= time <= class_0_end:
            return 0

        # Post-Partum Classes
        if 0.083 < diff <= 1:
            return -1  # 5 minutes after to 1 hour after lambing

        # Pre-Partum Classes
        if -1 <= diff < -0.083:
            return 1  # 5 minutes before to 1 hour before lambing
        elif -2 <= diff < -1:
            return 2  # 1 to 2 hours before lambing
        elif -3 <= diff < -2:
            return 3  # 2 to 3 hours before lambing
        elif -4 <= diff < -3:
            return 4  # 3 to 4 hours before lambing
        elif -5 <= diff < -4:
            return 5  # 4 to 5 hours before lambing
        elif -6 <= diff < -5:
            return 6  # 5 to 6 hours before lambing
        elif -7 <= diff < -6:
            return 7  # 6 to 7 hours before lambing
        elif -8 <= diff < -7:
            return 8  # 7 to 8 hours before lambing
        elif -9 <= diff < -8:
            return 9  # 8 to 9 hours before lambing
        elif -10 <= diff < -9:
            return 10  # 9 to 10 hours before lambing
        elif -11 <= diff < -10:
            return 11  # 10 to 12 hours before lambing
        elif -12 <= diff < -11:
            return 12  # 11 to 12 hours before lambing

        # Non-Partum Classes
        if diff > 1:
            return 13  # More than 1 hour after lambing time
        elif diff < -12:
            return 13  # More than 12 hours before lambing time

    return 13  # Default to non-partum

# Function to get the last lambing time based on lambing type
def get_last_lambing_time(lambing_times, lambing_type):
    if lambing_type == "Single":
        return lambing_times[0]
    elif lambing_type == "Double" and len(lambing_times) >= 2:
        return lambing_times[1]
    elif lambing_type == "Triple" and len(lambing_times) >= 3:
        return lambing_times[2]
    return lambing_times[0]

# Iterate over each animal ID in the JSON data
for animal_id, details in ewe_data.items():
    collar = details.get("collar")
    lambing_times_str = details.get("lambing_times", [])
    lambing_type = details.get("lambing_type", "Single")
    placement_time_str = details.get("placement")
    removal_time_str = details.get("removal")
    
    if not collar:
        print(f"Skipping {animal_id}: Missing collar information.")
        continue
    
    lambing_times = [datetime.fromisoformat(t) for t in lambing_times_str]
    placement_time = datetime.fromisoformat(placement_time_str) + timedelta(minutes=10)
    
    # Adjust removal time to the correct value
    removal_time = datetime.fromisoformat(removal_time_str) - timedelta(minutes=10)
    removal_time_corrected = removal_time - timedelta(minutes=20)  
    
    print(f"\nProcessing ewe [{animal_id}] with collar ID [{collar}]")
    print(f"Placement time: {placement_time}, Removal time: {removal_time_corrected}")

    # Load the large CSV file for the collar
    collar_file = source_base_dir / f"{collar}.csv"
    if not collar_file.exists():
        print(f"Skipping ewe [{animal_id}]: No CSV file found for collar [{collar}].")
        continue

    # Process the CSV file in chunks
    chunk_size = 10000  # Adjust chunk size based on available memory
    for chunk in pd.read_csv(collar_file, delimiter=';', chunksize=chunk_size):
        print(f"Processing chunk for ewe [{animal_id}] with collar ID [{collar}]")
        processed_chunk = process_chunk(chunk, lambing_times, lambing_type, placement_time, removal_time_corrected)
        output_file = output_path / f"{animal_id}_{collar}.csv"
        if not output_file.exists():
            processed_chunk.to_csv(output_file, index=False, mode='w', header=True, sep=';')
            print(f"Created new file for ewe [{animal_id}] with collar ID [{collar}]")
        else:
            processed_chunk.to_csv(output_file, index=False, mode='a', header=False, sep=';')
            print(f"Appended to file for ewe [{animal_id}] with collar ID [{collar}]")

print("Processing completed.")
