import pandas as pd

# File paths for 2024 and 2025 AZMET raw data
file_path_2024 = r"E:\Research_Assistant\Tarun\AZmet\2024_hourly_data.txt"
file_path_2025 = r"E:\Research_Assistant\Tarun\AZmet\2025_hourly_data2.txt"

# Define column names based on AZMET format
columns = ["Year", "DOY", "Hour", "DBT", "RH", "VPD", "SolarRad", "Precip",
           "SoilTemp4in", "SoilTemp20in", "WS", "WindVecMag", "WD", "WindDirSD",
           "MaxWS", "ETo", "ActualVaporPressure", "DewPoint"]

# Function to process each year's data separately
def process_azmet_data(file_path, year):
    # Load data
    df = pd.read_csv(file_path, header=None, names=columns, dtype=str)  # Load as strings

    # ✅ Convert Year, DOY, and Hour to integers to avoid float formatting issues
    df["Year"] = df["Year"].astype(int)
    df["DOY"] = df["DOY"].astype(int)
    df["Hour"] = df["Hour"].astype(int)

    # ✅ Fix "Hour = 24" issue by rolling over to the next day
    df.loc[df["Hour"] == 24, "Hour"] = 0  # Change 24:00 to 00:00
    df.loc[df["Hour"] == 0, "DOY"] += 1  # Increment DOY for those rows

    # ✅ Ensure DOY doesn't exceed the year's valid days
    def adjust_year_doy(row):
        is_leap = (row["Year"] % 4 == 0 and (row["Year"] % 100 != 0 or row["Year"] % 400 == 0))
        max_days = 366 if is_leap else 365

        if row["DOY"] > max_days:
            row["DOY"] = 1
            row["Year"] += 1  # Increment the year

        return row

    df = df.apply(adjust_year_doy, axis=1)

    # ✅ Ensure DOY and Hour are properly formatted before datetime conversion
    df["DOY"] = df["DOY"].astype(str).str.zfill(3)  # Ensures DOY is always 3 digits (001, 002, ..., 365)
    df["Hour"] = df["Hour"].astype(str).str.zfill(2)  # Ensures Hour is always 2 digits (01, 02, ..., 23)

    # ✅ Convert Year, DOY, Hour to DateTime format
    df["Datetime"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["DOY"] + " " + df["Hour"] + ":00",
        format="%Y-%j %H:%M"
    )

    # ✅ Convert required columns with proper units
    df["DBT"] = df["DBT"].astype(float)  # °C (No conversion needed)
    df["RH"] = df["RH"].astype(float)    # % (No conversion needed)

    # ✅ Convert Solar Radiation (MJ/m² → W/m²)
    df["SolarRad"] = df["SolarRad"].astype(float) * 277.78  # Correct conversion

    df["WD"] = df["WD"].astype(float)  # Degrees (No conversion needed)

    # ✅ Wind Speed is already in m/s (DO NOT convert again)
    df["WS"] = df["WS"].astype(float)  # Keep original m/s value

    # ✅ Extract only necessary columns
    df_cleaned = df[["Datetime", "DBT", "RH", "SolarRad", "WD", "WS"]]

    # ✅ Rename columns for final output format
    df_cleaned.columns = ["Date", "DryBulb (°C)", "Relative Humidity (%)", "Global Horizontal Radiation (Wh/m²)", "Wind Direction (°)", "Wind Speed (m/s)"]

    return df_cleaned

# ✅ Process 2024 and 2025 data separately
df_2024 = process_azmet_data(file_path_2024, year=2024)
df_2025 = process_azmet_data(file_path_2025, year=2025)

# ✅ Define output file paths
output_file_2024 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2024v3.csv"
output_file_2025 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2025v3.csv"

# ✅ Save cleaned data to separate CSV files
df_2024.to_csv(output_file_2024, index=False, encoding="utf-8")
df_2025.to_csv(output_file_2025, index=False, encoding="utf-8")

print(f"Processed 2024 data saved to {output_file_2024}")
print(f"Processed 2025 data saved to {output_file_2025}")
