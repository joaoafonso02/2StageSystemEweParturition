import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

def parse_metadata(input_csv, output_json):
    ewe_data = {}
    delay = timedelta(minutes=10)  # 10 minutes delay

    date_formats = [
        "%d/%m/%y %I:%M %p",       
        "%d/%m/%Y %I:%M %p",        
        "%d/%m/%Y %H:%M:%S",        
        "%Y-%m-%dT%H.%M.%S%z",      
        "%d/%m/%Y %H:%M",           
        "%d/%m/%y %H:%M:%S",        
        "%d/%m/%Y %H:%M:%S",       
    ]

    def try_parse(date_str):
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        print(f"Unable to parse date: {date_str}")
        return None

    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2): # Skip header
            animal_id = f"Animal_{row['AnimalID']}"
            lambing_times = []
            for key in ["Lambing Time", "Lambing Time(2nd)", "Lambing Time(3rd)"]:
                if row.get(key):
                    parsed_time = try_parse(row[key].strip())
                    if parsed_time:
                        lambing_times.append(parsed_time.isoformat())
                    else:
                        print(f"Row {row_number}: Invalid lambing time format for {key} -> '{row[key]}'")

            # Remove any None values
            lambing_times = [lt for lt in lambing_times if lt is not None]

            # Parse placement and removal times
            placement = try_parse(row["Placement"].strip())
            removal = try_parse(row["Removal"].strip())

            if not placement or not removal:
                print(f"Row {row_number}: Skipping {animal_id} due to invalid placement/removal dates.")
                continue

            placement += delay
            removal -= delay

            ewe_data[animal_id] = {
                "collar": row["Collar"].strip(),
                "placement": placement.isoformat(),
                "removal": removal.isoformat(),
                "lambing_times": lambing_times,
                "lambing_type": row["Lambing Type"].strip(),
                "assistance": row["Assistance"].strip(),
                "comments": row["Comments"].strip(),
                "problems": row["Problems"].strip()
            }

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(ewe_data, json_file, indent=4)

# Update paths to match the new structure
input_csv_path = Path("../data/ewe_metadata.csv")
output_json_path = Path("../data/ewe_data.json")

parse_metadata(input_csv_path, output_json_path)