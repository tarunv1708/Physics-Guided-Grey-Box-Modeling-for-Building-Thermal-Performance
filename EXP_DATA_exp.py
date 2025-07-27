import pandas as pd
import os
from datetime import datetime

# Define input directory
data_directory = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\RC_Modelling\8"

# List all CSV files - SORT BY FILENAME DESCENDING (prioritize newer files)
all_files = sorted(
    [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".csv")],
    reverse=True  # Newest files first
)

# Initialize empty DataFrame
merged_data = pd.DataFrame()

# Load and Merge all CSVs with duplicate prevention
for file in all_files:
    df = pd.read_csv(file)
    print(f"Processing: {os.path.basename(file)}")

    # 🛠️ Repair MST3 files
    if "MST_3" in file:
        df = df.dropna(subset=["Date"])

    # ⚙️ Standardize date parsing
    if df['Date'].dtype == 'object':
        # Remove timezone strings if present
        df['Date'] = df['Date'].str.replace(r'(\s+\+\d{4})|(\.\d+)$', '', regex=True)

    # 📅 Parse dates consistently
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce', utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["Date"])

    # ⏰ Filter specific files to date range
    if "MST_4" in file or "54_MST" in file or "04_16" in file:
        df = df[df["Date"] > pd.Timestamp("2025-02-26 11:00:00")]

    # ❗ Remove duplicates BEFORE merging (prioritize current file)
    # Keep only new dates not in merged_data
    existing_dates = set(merged_data["Date"]) if not merged_data.empty else set()
    df = df[~df["Date"].isin(existing_dates)]

    merged_data = pd.concat([merged_data, df], ignore_index=True)

# 🔄 Final cleanup
merged_data = merged_data.sort_values("Date")
merged_data = merged_data.drop_duplicates(subset=["Date"], keep="first")  # Final safeguard
merged_data.reset_index(drop=True, inplace=True)

print(f"[✅] Total cleaned rows: {len(merged_data)}")
print(f"[📅] Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")

# 📈 Sensor Mappings
sensor_mappings = {
    "Middle_Bay_Ceiling": [
        "Temperature (RXW-TMB 22082383:22074964-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22074965-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22074967-1), *F, ASU EnKoat MicroRx"
    ],
    "Middle_Bay_Roof": [
        "Temperature (RXW-TMB 22082383:22087281-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22087280-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22085407-1), *F, ASU EnKoat MicroRx"
    ],
    "Middle_Bay_MidLevel": [
        "Temperature (RXW-THC 22082383:22074930-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-THC 22082383:22074947-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-THC 22082383:22074945-1), *F, ASU EnKoat MicroRx"
    ],
    "South_Bay_Ceiling": [
        "Temperature (RXW-TMB 22082383:22074966-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22074968-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22074969-1), *F, ASU EnKoat MicroRx"
    ],
    "South_Bay_Roof": [
        "Temperature (RXW-TMB 22082383:22085408-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22087278-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-TMB 22082383:22087279-1), *F, ASU EnKoat MicroRx"
    ],
    "South_Bay_MidLevel": [
        "Temperature (RXW-THC 22082383:22074946-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-THC 22082383:22074931-1), *F, ASU EnKoat MicroRx",
        "Temperature (RXW-THC 22082383:22074929-1), *F, ASU EnKoat MicroRx"
    ]
}


# 💾 Save outputs
def process_and_save(df, sensor_list, output_name):
    selected = ["Date"] + sensor_list
    try:
        subset = df[selected].copy()
        for col in sensor_list:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
        subset[f"Average_{output_name}"] = subset[sensor_list].mean(axis=1)
        subset.to_excel(os.path.join(data_directory, f"{output_name}.xlsx"), index=False)
        print(f"[✅] Saved: {output_name}")
    except Exception as e:
        print(f"[⚠️] Skipped {output_name} due to: {e}")


for zone, sensors in sensor_mappings.items():
    process_and_save(merged_data, sensors, zone)

print("✅ All zone outputs saved.")
