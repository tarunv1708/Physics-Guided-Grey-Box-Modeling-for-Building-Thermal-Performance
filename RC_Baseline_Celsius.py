import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import timedelta
import numpy.fft as fft
from scipy.fft import fft, fftfreq
import time
from tqdm import tqdm

SENSOR_ACCURACY = 0.5
SIMULATION_START = "2025-05-11 00:00:00"
SIMULATION_END = "2025-05-21 00:00:00"

BAYS = ["Middle_Bay", "South_Bay"]
SENSOR_POSITIONS = ["Ceiling", "Roof", "MidLevel"]

# Optimization parameter names
OPTIMIZE_PARAMS = [
    'R_insulation',
    'C_insulation',
    'C_membrane',
    'C_metal_deck',
    'time_shift',  #  Phase alignment
    'solar_absorptance',
    'K_p',
    'K_i'
]

# User-defined simulation time step (in seconds)
DT_SECONDS = 300  # 🟢 Use 600 for detailed runs, 3600 for faster testing
# Add near the top of your script (with other config flags)
CONVERT_GHI_TO_WATTS = True  # ⚠ Ensure this is True if GHI is in Wh/m²


# File paths
EXPERIMENTAL_DATA_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\RC_Modelling\All_Data_till_June_2025"
WEATHER_FILE_2024 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2024v3.csv"
WEATHER_FILE_2025 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2025v3.csv"
OUTPUT_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\RC_baeline_timeconstant\iter4"

# ===================== THERMAL MODEL PARAMETERS =====================
# Constants
sigma = 5.67e-8
A_roof = 921.35
T_set = 22  # in Celsius
epsilon = 0.95  # Emissivity for radiative resistance

K_p = 2000
K_i = 50
solar_absorptance = 0.28


# Initialize with nominal values (will be updated during optimization)
# Material Properties (Revised for Realism)
material_props = {
    "roof_membrane": {"k": 0.16, "density": 950, "cp": 1800, "thickness": 0.003},
    "insulation": {"k": 0.035, "density": 40, "cp": 1400, "thickness": 0.127},      # Fiberglass (ρ=40 kg/m³)
    "metal_deck": {"k": 50, "density": 7850, "cp": 500, "thickness": 0.0127},    # Near-zero thermal mass
    "air": {"density": 1.225, "cp": 1005, "volume": 3.0},
}

# Global variables for optimization resistances (used inside simulate_state_space)
global_vars = {
    'R_membrane': None,
    'R_roof_insulation': None,
    'R_metal_deck': None,

}




# Compute Capacitances (Total for Roof Area)
def compute_capacitance(material, A):
    return material["density"] * material["cp"] * material["thickness"] * A  # [J/K]

# ✅ Add this function now
def initialize_resistances():
    """Calculate initial resistances and store in global_vars"""
    global_vars['R_membrane'] =1.3 * compute_resistance(material_props["roof_membrane"], A_roof)
    global_vars['R_roof_insulation'] = 1.2 * compute_resistance(material_props["insulation"], A_roof)
    global_vars['R_metal_deck'] = compute_resistance(material_props["metal_deck"], A_roof)



C_membrane = compute_capacitance(material_props["roof_membrane"], A_roof)
C_insulation = compute_capacitance(material_props["insulation"], A_roof)
C_metal_deck = compute_capacitance(material_props["metal_deck"], A_roof)

h_ceiling = 3.0  # in meters
zone_volume = h_ceiling * A_roof


# ===================== ENHANCED CORE FUNCTIONS =====================
def compute_resistance(material, A):
    return material["thickness"] / (material["k"] * A)



def load_experimental_data(bay):
    """Load experimental data for a single bay."""
    experimental_data = {}

    # Define file paths only for the selected bay
    files = {
        "Ceiling": os.path.join(EXPERIMENTAL_DATA_DIR, f"{bay}_Ceiling.xlsx"),
        "MidLevel": os.path.join(EXPERIMENTAL_DATA_DIR, f"{bay}_MidLevel.xlsx"),
        "Roof": os.path.join(EXPERIMENTAL_DATA_DIR, f"{bay}_Roof.xlsx"),
    }

    for position, file_path in files.items():
        try:
            df = pd.read_excel(file_path)

            # ✅ Fix date parsing with correct format
            df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y %H:%M:%S", errors='coerce')

            df = df[(df['Date'] >= SIMULATION_START) & (df['Date'] <= SIMULATION_END)]
            df = df.set_index('Date').sort_index()
            experimental_data[position] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            experimental_data[position] = None  # Handle missing data

    print(f"✅ Loaded experimental data for {bay}: dict_keys({list(experimental_data.keys())})")
    return experimental_data





def compute_T_sky(T_air, RH):
    e = (RH / 100.0) * 6.11 * 10 ** (7.5 * T_air / (237.3 + T_air))  # Note this!
    return T_air * (1.22 * np.sqrt(e / 10) - 0.22)



# Thermal model functions
def wind_direction_modifier(wind_direction):
    return 1.0 if 0 <= wind_direction <= 180 else 0.5



def h_conv_tarp(T_surface, T_ambient, wind_speed, wind_direction):
    delta_T = T_surface - T_ambient
    W_f = wind_direction_modifier(wind_direction)
    # Natural convection (h_n) and forced convection (h_f)
    h_n = 1.31 * abs(delta_T) ** (1/3)  # Same as before
    h_f = W_f * 3.26 * wind_speed ** 0.89  # ASHRAE outdoor convection
    return max(np.sqrt(h_n ** 2 + h_f ** 2), 1e-3)

def R_conv(T_surface, T_ambient, wind_speed, wind_direction):
    h_conv = h_conv_tarp(T_surface, T_ambient, wind_speed, wind_direction)
    return max(1 / h_conv, 1e-6)  # Per m² [K·m²/W]

def R_indoor_dynamic(T_surface, T_inside):
    delta_T = abs(T_surface - T_inside)
    h = 1.5 if delta_T < 1 else 3.5 + 0.2 * delta_T if delta_T < 5 else 8.0
    return max(1 / h, 1e-6)  # Per m² [K·m²/W]

# HVAC Control (Scaled to Total Roof Area)


def HVAC_heat_transfer_PI(T_zone, T_set, K_p, K_i, integral, dt, max_rate=35520):
    error = T_set - T_zone
    integral += error * dt
    Q = K_p * error + K_i * integral
    return np.clip(Q, -max_rate, max_rate), integral

def compute_HVAC_series(df, T_set, K_p, K_i, dt=600):
    T_zone_series = df['T_zone_air'].values
    HVAC_list = []
    integral = 0.0
    for T_zone in T_zone_series:
        Q_HVAC, integral = HVAC_heat_transfer_PI(T_zone, T_set, K_p, K_i, integral, dt)
        HVAC_list.append(Q_HVAC)
    return HVAC_list

def process_weather_data():
    # Load both weather files
    df_2024 = pd.read_csv(WEATHER_FILE_2024, encoding='utf-8')
    df_2025 = pd.read_csv(WEATHER_FILE_2025, encoding='utf-8')

    # Clean column names
    for df in [df_2024, df_2025]:
        df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip().str.upper()

    # Rename columns
    rename_map = {
        'DATE': 'Datetime',
        'DRYBULB C': 'DBT',
        'RELATIVE HUMIDITY': 'RH',
        'GLOBAL HORIZONTAL RADIATION WH M': 'GHI',
        'WIND DIRECTION': 'WD',
        'WIND SPEED M S': 'WS'
    }

    df_2024 = df_2024.rename(columns=rename_map)[['Datetime', 'DBT', 'RH', 'GHI', 'WD', 'WS']]
    df_2025 = df_2025.rename(columns=rename_map)[['Datetime', 'DBT', 'RH', 'GHI', 'WD', 'WS']]

    # Parse datetime
    df_2024['Datetime'] = pd.to_datetime(df_2024['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df_2025['Datetime'] = pd.to_datetime(df_2025['Datetime'], format='%Y-%m-%d %H:%M:%S')

    # Combine and sort
    data = pd.concat([df_2024, df_2025]).sort_values('Datetime').drop_duplicates()
    data = data.set_index('Datetime')

    # ✅ Convert GHI from Wh/m² to W/m² if needed
    if CONVERT_GHI_TO_WATTS:
        data['GHI'] = data['GHI']  # Already assumed W/m² if True
        # OR: data['GHI'] *= 277.78  # Uncomment if original was in Wh/m²

    # ✅ Add derived fields
    data['T_outside'] = data['DBT']
    data['T_sky'] = compute_T_sky(data['DBT'], data['RH'])
    data['LatentHeat'] = 2256000 * data['RH'] / 100  # J/kg

    # Time in hours since start
    data['HourIndex'] = (data.index - data.index[0]).total_seconds() / 3600
    time_steps = data['HourIndex'].values

    # ✅ Interpolators (used in state-space simulation)
    interpolators = {
        'T_outside': interp1d(time_steps, data['T_outside'], fill_value="extrapolate"),
        'GHI': interp1d(time_steps, data['GHI'], fill_value="extrapolate"),
        'WS': interp1d(time_steps, data['WS'], fill_value="extrapolate"),
        'WD': interp1d(time_steps, data['WD'], fill_value="extrapolate"),
        'RH': interp1d(time_steps, data['RH'], fill_value="extrapolate"),
        'T_sky': interp1d(time_steps, data['T_sky'], fill_value="extrapolate"),
        'LatentHeat': interp1d(time_steps, data['LatentHeat'], fill_value="extrapolate")
    }

    return {'data': data, 'interpolators': interpolators, 'time_steps': time_steps}





    # In process_weather_data()
    if CONVERT_GHI_TO_WATTS:
        data['GHI'] = data['GHI']  # ✅ Correct for W/m²

    data['LatentHeat'] = 2256000 * data['RH'] / 100  # J/kg added heat capacity

    time_steps = data['HourIndex'].values
    interpolators = {
        'T_outside': interp1d(time_steps, data['DBT'], fill_value="extrapolate"),
        'GHI': interp1d(time_steps, data['GHI'], fill_value="extrapolate"),
        'WS': interp1d(time_steps, data['WS'], fill_value="extrapolate"),
        'WD': interp1d(time_steps, data['WD'], fill_value="extrapolate"),
        'T_sky': interp1d(time_steps, data['T_sky'], fill_value="extrapolate"),
        'LatentHeat': interp1d(time_steps, data['LatentHeat'], fill_value="extrapolate"),
        'RH': interp1d(time_steps, data['RH'], fill_value="extrapolate")

    }


    return {'data': data, 'interpolators': interpolators, 'time_steps': time_steps}



# ===================== OPTIMIZATION CORE =====================
def simulate_state_space(params, weather_data, interpolators, start_time, end_time):
    # 🔍 DEBUG BLOCK — Add near the top of simulate_state_space()
    invalid_param_debug = False

    global global_vars


    # Load constants from global vars
    R_roof_insulation = global_vars.get("R_roof_insulation", 0.003938)
    R_membrane = compute_resistance(material_props["roof_membrane"], A_roof)
    R_metal_deck = compute_resistance(material_props["metal_deck"], A_roof)

    integral = 0.0
    dt = DT_SECONDS
    print(f"Running simulate_state_space with user-defined dt = {dt} sec")

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    mask = (weather_data.index >= pd.to_datetime(start_time)) & (weather_data.index <= pd.to_datetime(end_time))

    relevant_data = weather_data[mask].copy()
    time_steps = relevant_data['HourIndex'].values
    t_sim = np.arange(time_steps[0], time_steps[-1], dt / 3600)

    x_out = np.zeros((len(t_sim), 4))  # [T_membrane, T_insulation, T_metal_deck, T_zone]
    T_init = interpolators['T_outside'](time_steps[0])
    x_out[0] = [T_init] * 4



    for i in tqdm(range(1, len(t_sim)), desc="⏳ Simulating timestep", ncols=70):
        T_membrane, T_insulation, T_metal_deck, T_zone = x_out[i - 1]
        t_current = t_sim[i]

        T_outside = interpolators['T_outside'](t_current)
        RH = interpolators['RH'](t_current)
        T_sky = interpolators['T_sky'](t_current)
        wind_speed = interpolators['WS'](t_current)
        wind_dir = interpolators['WD'](t_current)
        GHI = interpolators['GHI'](t_current)

        C_air = material_props["air"]["density"] * material_props["air"]["cp"] * zone_volume
        C_air += interpolators['LatentHeat'](t_current) * zone_volume  # ← add latent capacity


        R_indoor = R_indoor_dynamic(T_metal_deck, T_zone) / A_roof

        Q_HVAC, integral = HVAC_heat_transfer_PI(T_zone, T_set, K_p, K_i, integral, dt)

        A = np.zeros((4, 4))
        A[0, 0] = - (1 / (R_membrane * C_membrane))
        A[0, 1] = 1 / (R_membrane * C_membrane)
        A[1, 0] = 1 / (R_membrane * C_insulation)
        A[1, 1] = - (1 / (R_membrane * C_insulation) + 1 / (R_roof_insulation * C_insulation))
        A[1, 2] = 1 / (R_roof_insulation * C_insulation)
        A[2, 1] = 1 / (R_roof_insulation * C_metal_deck)
        A[2, 2] = - (1 / (R_roof_insulation * C_metal_deck) + 1 / (R_indoor * C_metal_deck))
        A[2, 3] = 1 / (R_indoor * C_metal_deck)
        A[3, 2] = 1 / (R_indoor * C_air)
        A[3, 3] = -1 / (R_indoor * C_air) - K_p / C_air

        B = np.zeros((4, 3))
        #B[0, 0] = 1 / (C_membrane * R_conv_rad)
        B[3, 2] = 1 / C_air

        U = np.array([T_outside, GHI, Q_HVAC])

        T_membrane_K = T_membrane + 273.15
        T_sky_K = T_sky + 273.15
        Q_solar = GHI * A_roof * solar_absorptance
        Q_rad = epsilon * sigma * A_roof * (T_membrane_K ** 4 - T_sky_K ** 4)
        h_conv = h_conv_tarp(T_membrane, T_outside, wind_speed, wind_dir)
        Q_conv = h_conv * A_roof * (T_membrane - T_outside)

        #if i % 50 == 0:
            #print(f"[Step {i}]")
            #print(f"  Q_solar = {Q_solar:.2f} W")
            #print(f"  Q_rad   = {Q_rad:.2f} W")
            #print(f"  Q_conv  = {Q_conv:.2f} W")
            #print(f"  Net     = {(Q_solar - Q_rad - Q_conv):.2f} W")
            #print(f"  T_mem   = {T_membrane:.2f} °C | T_sky = {T_sky:.2f} °C | T_out = {T_outside:.2f} °C")

        rhs = x_out[i - 1] + dt * (B @ U)
        rhs[0] += dt * (Q_solar - Q_rad - Q_conv) / C_membrane
        lhs = np.eye(4) - dt * A



        x_out[i] = np.linalg.solve(lhs, rhs)
        x_out[i] = np.clip(x_out[i], -50, 100)

    sim_index = pd.date_range(start=start_time, periods=len(t_sim), freq=f'{int(dt)}s')

    return pd.DataFrame({
        'T_membrane': x_out[:, 0],
        'T_metal_deck': x_out[:, 2],
        'T_zone_air': x_out[:, 3]
    }, index=sim_index)

if __name__ == "__main__":
    bay = "South_Bay"  # or "Middle_Bay"

    # Define output directory
    OUTPUT_DIR = os.path.join(
        r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\RC_baeline_timeconstant",
        bay, datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load all data
    initialize_resistances()
    exp_dict = load_experimental_data(bay)
    weather_bundle = process_weather_data()

    # Step 2: Run baseline simulation (no optimization)
    sim_df = simulate_state_space(
        params=None,
        weather_data=weather_bundle['data'],
        interpolators=weather_bundle['interpolators'],
        start_time=SIMULATION_START,
        end_time=SIMULATION_END
    )

    # 🔁 Convert simulated temperatures from °C to °F
    #sim_df['T_membrane'] = sim_df['T_membrane'] * 9 / 5 + 32
    #sim_df['T_metal_deck'] = sim_df['T_metal_deck'] * 9 / 5 + 32
    #sim_df['T_zone_air'] = sim_df['T_zone_air'] * 9 / 5 + 32

    for key, exp_df in exp_dict.items():
        if exp_df is None or sim_df is None:
            continue

        exp = (exp_df.iloc[:, 0].copy() - 32) * 5 / 9  # °F → °C  # Use first column
        exp.index = pd.to_datetime(exp.index)
        sim = sim_df[f"T_{'membrane' if key == 'Roof' else 'metal_deck' if key == 'Ceiling' else 'zone_air'}"]

        # Align and resample to hourly
        exp_hourly = exp.resample('1h').mean()
        sim_hourly = sim.reindex_like(exp_hourly).interpolate(limit=6)

        # Day/Night mask
        ghi_resampled = weather_bundle['data']['GHI'].reindex(exp_hourly.index).interpolate(limit=6)
        is_day = ghi_resampled >= 10
        is_night = ~is_day

        # Error metrics
        mask = ~np.isnan(exp_hourly) & ~np.isnan(sim_hourly)
        mae = mean_absolute_error(exp_hourly[mask], sim_hourly[mask])
        rmse = mean_squared_error(exp_hourly[mask], sim_hourly[mask]) ** 0.5

        mae_day = mean_absolute_error(exp_hourly[is_day & mask], sim_hourly[is_day & mask])
        mae_night = mean_absolute_error(exp_hourly[is_night & mask], sim_hourly[is_night & mask])
        rmse_day = mean_squared_error(exp_hourly[is_day & mask], sim_hourly[is_day & mask]) ** 0.5
        rmse_night = mean_squared_error(exp_hourly[is_night & mask], sim_hourly[is_night & mask]) ** 0.5

        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(exp_hourly, label="Experimental", linewidth=1.8)
        plt.plot(sim_hourly, label="Simulation", alpha=0.7)
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Time (Datetime)")

        plt.title(
            f"{bay} - {key} Zone\n"
            f"MAE: {mae:.2f} °C | RMSE: {rmse:.2f} °C\n"
            f"MAE-Day: {mae_day:.2f} °C | MAE-Night: {mae_night:.2f} °C\n"
            f"RMSE-Day: {rmse_day:.2f} °C | RMSE-Night: {rmse_night:.2f} °C"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{key}_comparison_baseline.png"))
        plt.show()
        plt.close()

    # Plot delta T = Roof - Ceiling
    try:
        roof_exp = (exp_dict["Roof"].iloc[:, 0] - 32) * 5 / 9
        ceil_exp = (exp_dict["Ceiling"].iloc[:, 0] - 32) * 5 / 9

        roof_exp = roof_exp.resample('1h').mean()
        ceil_exp = ceil_exp.resample('1h').mean()

        delta_exp = roof_exp - ceil_exp

        delta_sim = sim_df['T_membrane'].resample('1h').mean() - sim_df['T_metal_deck'].resample('1h').mean()

        mask = ~np.isnan(delta_exp) & ~np.isnan(delta_sim)
        mae = mean_absolute_error(delta_exp[mask], delta_sim[mask])
        rmse = mean_squared_error(delta_exp[mask], delta_sim[mask]) ** 0.5

        mae_day = mean_absolute_error(delta_exp[is_day & mask], delta_sim[is_day & mask])
        mae_night = mean_absolute_error(delta_exp[is_night & mask], delta_sim[is_night & mask])
        rmse_day = mean_squared_error(delta_exp[is_day & mask], delta_sim[is_day & mask]) ** 0.5
        rmse_night = mean_squared_error(delta_exp[is_night & mask], delta_sim[is_night & mask]) ** 0.5

        plt.figure(figsize=(14, 6))
        plt.plot(delta_exp, label="Experimental ΔT", linewidth=1.8)
        plt.plot(delta_sim, label="Simulated ΔT", alpha=0.7)
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Time (Datetime)")

        plt.title(
            f"{bay} - ΔT (Roof - Ceiling)\n"
            f"MAE: {mae:.2f} °C | RMSE: {rmse:.2f} °C\n"
            f"MAE-Day: {mae_day:.2f} °C | MAE-Night: {mae_night:.2f} °C\n"
            f"RMSE-Day: {rmse_day:.2f} °C | RMSE-Night: {rmse_night:.2f} °C"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"DeltaT_comparison_baseline.png"))
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Delta T plot failed: {e}")
