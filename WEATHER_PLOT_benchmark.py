import os
import pandas as pd
import matplotlib.pyplot as plt

# === ✅ USER CONFIGURATION ===
WEATHER_CSV_PATH = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2025v2.csv"
OUTPUT_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\WEATHER_PLOT\1"

SIMULATION_START = "2025-02-01 00:00:00"
SIMULATION_END = "2025-03-28 00:00:00"

# === ✅ Load Weather Data ===
def load_weather_data():
    df = pd.read_csv(WEATHER_CSV_PATH)
    df = df.rename(columns={
        'Date': 'DateTime',
        'DryBulb (°C)': 'T_outside',
        'Global Horizontal Radiation (Wh/m²)': 'GHI'
    })
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    df = df[(df.index >= SIMULATION_START) & (df.index <= SIMULATION_END)]
    df = df.interpolate(method='time')
    return df

# === ✅ Plot Weather Variables with Dual Axis ===
def plot_weather_variables():
    df = load_weather_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df['T_outside_F'] = df['T_outside'] * 9 / 5 + 32


    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Left Y-axis: Temperature (°C)
    ax1.plot(df.index, df['T_outside_F'], color='tab:blue', linewidth=1.5, label='T_outside (°F)')
    ax1.set_xlabel("DateTime")
    ax1.set_ylabel("T_outside (°F)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Right Y-axis: GHI
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['GHI'], color='tab:orange', linewidth=1.5, label='GHI (Wh/m²)')
    ax2.set_ylabel("GHI (Wh/m²)", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Title and formatting
    plt.title(f"AZMET Weather — Outdoor Temp (°C) and GHI (Wh/m²)\n({SIMULATION_START} to {SIMULATION_END})")
    fig.tight_layout()

    # Save and show
    out_path = os.path.join(OUTPUT_DIR, f"Weather_GHI_Toutside_Feb2025_DualAxis.png")
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")
    plt.show()

# === ✅ Execute
plot_weather_variables()
