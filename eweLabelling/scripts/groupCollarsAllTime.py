import os
import pandas as pd
from pathlib import Path

# Define the source directories and the destination directory
pos_data_path = Path("../data/pos_data/lamb_data")
nibio_data_path = Path("../data/collar_data/NIBIO")
aggregated_data_path = Path("../data/collar_data/daysData/aggregated_data")
final_data_path = Path("../data/finalDataSet")
final_data_path.mkdir(parents=True, exist_ok=True)

# Define the date ranges
pos_date_range = ("2024-04-23", "2024-05-14")
nibio_date_range = ("2024-05-14", "2024-05-20")
aggregated_date_range = ("2024-05-19", "2024-05-28")

# Function to read and filter CSV files by date range
def read_and_filter_csv(file_path, start_date, end_date):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    mask = (df['Time'] >= start_date) & (df['Time'] <= end_date)
    return df.loc[mask]

# Get a list of all collars
collars = set()
for file in pos_data_path.glob("POS_*.csv"):
    collar = file.stem.split('_')[1]
    collars.add(collar)
for file in nibio_data_path.glob("*.csv"):
    collar = file.stem
    collars.add(collar)
for file in aggregated_data_path.glob("*.csv"):
    collar = file.stem.split('_')[0]
    collars.add(collar)

# Aggregate data for each collar
for collar in collars:
    aggregated_data = pd.DataFrame()

    # Process POS data
    pos_files = pos_data_path.glob(f"POS_*{collar}*.csv")
    for pos_file in pos_files:
        pos_data = read_and_filter_csv(pos_file, pos_date_range[0], pos_date_range[1])
        aggregated_data = pd.concat([aggregated_data, pos_data], ignore_index=True)

    # Process NIBIO data
    nibio_file = nibio_data_path / f"{collar}.csv"
    if nibio_file.exists():
        nibio_data = read_and_filter_csv(nibio_file, nibio_date_range[0], "2024-05-19 23:59:59")
        aggregated_data = pd.concat([aggregated_data, nibio_data], ignore_index=True)

    # Process aggregated data
    aggregated_file = aggregated_data_path / f"{collar}_20240520_20240528.csv"
    if aggregated_file.exists():
        aggregated_data_part = read_and_filter_csv(aggregated_file, "2024-05-20", aggregated_date_range[1])
        aggregated_data = pd.concat([aggregated_data, aggregated_data_part], ignore_index=True)

    # Sort the aggregated data by time
    if not aggregated_data.empty:
        aggregated_data.sort_values(by='Time', inplace=True)
        output_file = final_data_path / f"{collar}.csv"
        aggregated_data.to_csv(output_file, index=False)
        print(f"Aggregated data saved to {output_file}")
    else:
        print(f"No data to aggregate for collar {collar}")

print("Aggregation completed.")