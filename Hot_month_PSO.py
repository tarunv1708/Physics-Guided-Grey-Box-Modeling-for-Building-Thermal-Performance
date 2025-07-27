import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate
from pyswarm import pso  # Changed optimization library
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import timedelta
import numpy.fft as fft
from scipy.fft import fft, fftfreq
import time
from tqdm import tqdm


# ===================== CONFIGURATION =====================

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
EXPERIMENTAL_DATA_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\RC_Modelling\8"
WEATHER_FILE_2024 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2024v3.csv"
WEATHER_FILE_2025 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2025v3.csv"
OUTPUT_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\Hot_month_PSO\iter3"
# ===================== THERMAL MODEL PARAMETERS =====================
# Constants
sigma = 5.67e-8
A_roof = 921.35
T_set = 22  # in Celsius
epsilon = 0.95  # Emissivity for radiative resistance


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




def validate_params(params):
    if len(params) != 8:
        print(f"❌ Parameter count mismatch: expected 8, got {len(params)}")
        return False

    R_ins, C_ins, C_mem, C_metal, time_shift, absorp, K_p, K_i = params

    # Check ranges match your bounds
    return (
        2.55e-03 <= R_ins <= 5.32e-03 and
        4.25e+06 <= C_ins <= 8.85e+06 and
        3.30e+06 <= C_mem <= 6.14e+06 and
        3.21e+07 <= C_metal <= 5.74e+07 and
        -8.0 <= time_shift <= 8.0 and
        0.12 <= absorp <= 0.60 and
        500 <= K_p <= 4000 and
        0 <= K_i <= 300
    )




def test_sensitivity(base_params, experimental_data, weather_data, interpolators):
    exp_roof = experimental_data['Roof'].iloc[:, :3].mean(axis=1).resample('h').mean()

    print("\n🔍 Sensitivity Analysis (ΔMAE from Baseline with +20% perturbation):")

    # Run baseline simulation
    baseline_sim = simulate_state_space(base_params, weather_data, interpolators, SIMULATION_START, SIMULATION_END)
    if baseline_sim is None:
        print("⚠️ Baseline simulation failed. Cannot continue sensitivity analysis.")
        return

    baseline_vals = baseline_sim['T_membrane']
    exp_vals = (exp_roof[:len(baseline_vals)] - 32) * 5 / 9  # Convert experimental to °C
    baseline_vals = baseline_vals[:len(exp_vals)]
    baseline_mae = mean_absolute_error(exp_vals, baseline_vals)

    # Loop over each parameter
    for i, name in enumerate(OPTIMIZE_PARAMS):
        perturbed = base_params.copy()
        perturbed[i] *= 1.2  # +20% perturbation

        sim = simulate_state_space(
            perturbed, weather_data, interpolators,
            SIMULATION_START, SIMULATION_END
        )

        if sim is not None:
            sim_vals = sim['T_membrane']  # Already in °C
            sim_vals = sim_vals[:len(exp_vals)]
            dMAE = mean_absolute_error(exp_vals, sim_vals)
            delta_mae = dMAE - baseline_mae

            print(f"📌 {name:20s} → ΔMAE = {delta_mae:+.2f}°C (New MAE: {dMAE:.2f}°C)")
        else:
            print(f"⚠️ Simulation failed for {name}")


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

    if len(params) == 8:
        R_insulation, C_insulation, C_membrane, C_metal_deck, time_shift, solar_absorptance, K_p, K_i = params
    else:
        raise ValueError(f"❌ Unsupported number of parameters: {len(params)}")

    if not validate_params(params):
        print(f"❌ Invalid parameters: {params}")
        return None

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

    if not validate_params(params):
        return None

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


cost_log = []
cost_log_roof = []
cost_log_ceiling = []

def objective_function(params, experimental_data, weather_data, interpolators):
    position_mappings = {
        'Roof': 'T_membrane',
        'Ceiling': 'T_metal_deck',
        'MidLevel': 'T_zone_air'
    }

    try:
        # ⏳ Time shift extraction
        if 'time_shift' in OPTIMIZE_PARAMS:
            time_shift_index = OPTIMIZE_PARAMS.index('time_shift')
            time_shift = params[time_shift_index]
        else:
            time_shift = 0.0

        sim_df = simulate_state_space(params, weather_data, interpolators, SIMULATION_START, SIMULATION_END)
        if sim_df is None:
            print(f"❌ simulate_state_space returned None for: {params}")
            return 1e6

        total_error = 0
        metrics = {}

        for position in ['Roof', 'Ceiling']:
            if experimental_data[position] is None:
                continue

            # 🧪 Experimental and simulated series
            exp = (experimental_data[position].iloc[:, :3].mean(axis=1).resample('h').mean() - 32) * 5 / 9
            sim_celsius = sim_df[position_mappings[position]]

            sim_time_hours = (sim_celsius.index - sim_celsius.index[0]).total_seconds() / 3600
            exp_time_hours = (exp.index - sim_celsius.index[0]).total_seconds() / 3600
            shifted_time = sim_time_hours + time_shift

            sim_interp = np.interp(exp_time_hours, shifted_time, sim_celsius.values)

            exp_vals = exp.values[:len(sim_interp)]
            sim_vals = sim_interp[:len(exp_vals)]

            night_mask = (exp.index.hour >= 20) | (exp.index.hour <= 6)
            mae_night = mean_absolute_error(exp_vals[night_mask], sim_vals[night_mask])

            # 📈 Error Metrics
            mae = mean_absolute_error(exp_vals, sim_vals)

            cross_corr = correlate(exp_vals - np.mean(exp_vals),
                                   sim_vals - np.mean(sim_vals), mode='same')
            max_corr = np.max(np.abs(cross_corr))
            phase_penalty = 1 / np.sqrt(max(max_corr, 1e-3))  # ✅ Smoother phase

            dT_sim = np.diff(sim_vals)
            dT_exp = np.diff(exp_vals)
            deriv_penalty = np.mean((dT_sim - dT_exp) ** 2)    # ✅ No /100

            sim_amp = np.max(sim_vals) - np.min(sim_vals)
            exp_amp = np.max(exp_vals) - np.min(exp_vals)
            amplitude_penalty = abs(sim_amp - exp_amp) / exp_amp  # ✅ Relative

            mean_bias = np.mean(sim_vals) - np.mean(exp_vals)
            bias_penalty = abs(mean_bias)

            # 🧮 Zone-specific weights
            if position == "Ceiling":
                w_mae = 2.0
                w_phase = 0.4
                w_deriv = 0.4
                w_amp = 0.7
                w_night = 1.2
                w_bias = 0.3
                w_shift = 0.2
            else:
                w_mae = 2.5
                w_phase = 0.8
                w_deriv = 0.4
                w_amp = 0.7
                w_night = 1.2
                w_bias = 0.3
                w_shift = 0.2

            # ✅ Apply time shift penalty directly
            time_shift_penalty = w_shift * abs(time_shift)

            zone_cost = (
                w_mae * mae +
                w_phase * phase_penalty +
                w_deriv * deriv_penalty +
                w_amp * amplitude_penalty +
                w_night * mae_night +
                w_bias * bias_penalty +
                time_shift_penalty
            )

            total_error += zone_cost

            if position == "Roof":
                cost_log_roof.append(zone_cost)
            elif position == "Ceiling":
                cost_log_ceiling.append(zone_cost)

            # 📊 Store metrics for diagnostics
            metrics[position] = {
                'MAE': mae,
                'PhasePenalty': phase_penalty,
                'DerivPenalty': deriv_penalty,
                'AmplitudePenalty': amplitude_penalty,
                'MAE_Night': mae_night,
                'BiasPenalty': bias_penalty,
                'TimeShift': time_shift,
                'ZoneCost': zone_cost
            }

            # 🔍 Print debug info
            print(f"[🔍 DEBUG] MAE={mae:.2f}, Phase={phase_penalty:.4f}, Deriv={deriv_penalty:.2f}, Amp={amplitude_penalty:.2f}, Shift={time_shift:.2f}")
            print(f"[📊 Contribution] {position} → Cost: {zone_cost:.2f}")

        avg_cost = total_error / len(metrics) if metrics else np.inf
        cost_log.append(avg_cost)

        if len(cost_log) <= 5 or len(cost_log) % 10 == 0:
            print(f"[🌀 Iteration {len(cost_log)}] Total Cost: {avg_cost:.2f}")

        return avg_cost

    except Exception as e:
        print(f"[❌ Objective Error] {e}")
        return np.inf





def check_data_alignment(exp_data, weather_data):
        """Enhanced temporal alignment verification"""
        print("\n🔍 Data Alignment Diagnostics:")

        # Weather data timeframe
        weather_start = weather_data['data'].index.min()
        weather_end = weather_data['data'].index.max()

        for position in ['Roof', 'Ceiling']:


            if exp_data[position] is None:
                continue

            exp_start = exp_data[position].index.min()
            exp_end = exp_data[position].index.max()

            # Calculate overlap percentage
            overlap_start = max(exp_start, weather_start)
            overlap_end = min(exp_end, weather_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600
            total_duration = (weather_end - weather_start).total_seconds() / 3600
            coverage = overlap_duration / total_duration * 100

            print(f"\n{position} Sensor:")
            print(f"  Experimental: {exp_start} to {exp_end}")
            print(f"  Weather:      {weather_start} to {weather_end}")
            print(f"  Overlap:      {overlap_start} to {overlap_end}")
            print(f"  Coverage:     {coverage:.1f}%")

            # Critical warning for low coverage
            if coverage < 95:
                print(f"⚠️ WARNING: Only {coverage:.1f}% temporal overlap!")




def run_optimization(bay, experimental_data, weather_data, interpolators):
    """Run parallel PSO optimization for a single bay."""
    print(f"\nStarting PSO optimization for {bay}...")

    R_ins_nom = 3.938320e-03
    C_ins_nom = 6.5526415e+06
    C_mem_nom = 4.7265255e+06
    C_metal_nom = 4.592699e+07


    bounds = [
        [R_ins_nom * 0.7, R_ins_nom * 1.3],  # R_insulation
        [C_ins_nom * 0.7, C_ins_nom * 1.3],  # C_insulation
        [C_mem_nom * 0.7, C_mem_nom * 1.3],  # C_membrane
        [C_metal_nom*0.7, C_metal_nom*1.25],
        [-6.0, 6.0],  # time_shift (widen this too)
        [0.12, 0.60],  # solar_absorptance (expanded on both sides)
        [500, 4000],  # K_p (allow more gain range)
        [0, 300]  # K_i (let it explore more integration)
    ]

    # Create shared data wrapper for parallel processing
    class SharedData:
        def __init__(self, exp, weather, interp):
            self.exp = exp
            self.weather = weather
            self.interp = interp

    shared = SharedData(experimental_data, weather_data['data'], weather_data['interpolators'])



    # Parallel objective function wrapper
    def parallel_objective(params):
        try:
            return objective_function(params, shared.exp, shared.weather, shared.interp)
        except Exception as e:
            print(f"❌ PSO objective crashed: {e} with params {params}")
            return 1e6

    # Configure parallel PSO
    # ✅ Just call PSO directly
    optimized_params, _ = pso(
        parallel_objective,  # Your wrapper objective
        [b[0] for b in bounds],  # Lower bounds
        [b[1] for b in bounds],  # Upper bounds
        swarmsize=60,  # More particles
        maxiter=100,  # Longer search
        phig=0.7,  # Global weight
        phip=0.7,  # Cognitive weight
        omega=0.6,  # Inertia
        debug=True
    )

    bay_plot_dir = os.path.join(OUTPUT_DIR, bay)
    os.makedirs(bay_plot_dir, exist_ok=True)

    # 🟣 Plot total cost convergence
    plt.figure(figsize=(8, 4))
    plt.plot(cost_log, label='Total Objective Cost', color='black', marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"{bay.replace('_', ' ')} Total Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(bay_plot_dir, "convergence_total_cost.png"), dpi=300)
    plt.close()

    # 🟧 Plot Roof zone cost convergence
    if cost_log_roof:
        plt.figure(figsize=(8, 4))
        plt.plot(cost_log_roof, label='Roof Cost', color='orange', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Zone Cost")
        plt.title(f"{bay.replace('_', ' ')} Roof Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(bay_plot_dir, "convergence_roof_cost.png"), dpi=300)
        plt.close()

    # 🟦 Plot Ceiling zone cost convergence
    if cost_log_ceiling:
        plt.figure(figsize=(8, 4))
        plt.plot(cost_log_ceiling, label='Ceiling Cost', color='steelblue', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Zone Cost")
        plt.title(f"{bay.replace('_', ' ')} Ceiling Convergence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(bay_plot_dir, "convergence_ceiling_cost.png"), dpi=300)
        plt.close()

    if optimized_params is None:
        print("❌ PSO did not return valid parameters.")
        return None

    # ✅ Save optimized parameters to CSV (as done in Updated_PSWON_timeconstant.py)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_csv_path = os.path.join(OUTPUT_DIR, bay, f"optimized_parameters_{timestamp}.csv")
    os.makedirs(os.path.dirname(opt_csv_path), exist_ok=True)

    opt_param_names = OPTIMIZE_PARAMS
    opt_df = pd.DataFrame([optimized_params], columns=opt_param_names)
    opt_df.to_csv(opt_csv_path, index=False)

    print(f"✅ Optimized parameters saved to {opt_csv_path}")

    return optimized_params


    return optimized_params  # ✅ This is your actual return value


def analyze_results(bay, experimental_data, base_df, optimized_df, optimized_params, results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bay_results = {}

    # Time shift (if optimized)
    if 'time_shift' in OPTIMIZE_PARAMS:
        time_shift_index = OPTIMIZE_PARAMS.index('time_shift')
        time_shift = optimized_params[time_shift_index]
    else:
        time_shift = 0.0

    #conversion_note = "(GHI converted)" if CONVERT_GHI_TO_WATTS else "(GHI raw)"
    base_time_hours = (base_df.index - base_df.index[0]).total_seconds() / 3600
    opt_time_hours = (optimized_df.index - optimized_df.index[0]).total_seconds() / 3600

    for position in ['Roof', 'Ceiling']:  # ✅ Expanded to include Ceiling

        plt.figure(figsize=(15, 6))

        if position not in experimental_data or experimental_data[position] is None:
            print(f"⚠️ No experimental data for {bay} {position}, skipping...")
            continue

        exp = (experimental_data[position].iloc[:, :3].mean(axis=1).resample('h').mean() - 32) * 5 / 9
        if exp.empty:
            print(f"⚠️ Experimental data is empty for {bay} {position}, skipping...")
            continue

        position_mappings = {
            'Roof': 'T_membrane',
            'Ceiling': 'T_metal_deck',  # ✅ Corrected key
            'MidLevel': 'T_zone_air'
        }

        col = position_mappings[position]
        exp_time_hours = (exp.index - base_df.index[0]).total_seconds() / 3600

        # Interpolate values to experimental time axis
        base_vals = np.interp(exp_time_hours, base_time_hours, base_df[col].values)
        opt_vals = np.interp(exp_time_hours, opt_time_hours, optimized_df[col].values)
        opt_shifted_vals = np.interp(exp_time_hours, opt_time_hours + time_shift, optimized_df[col].values)

        # Align series
        base_series = pd.Series(base_vals, index=exp.index)
        opt_series = pd.Series(opt_vals, index=exp.index)
        opt_shifted_series = pd.Series(opt_shifted_vals, index=exp.index)

        common_idx = exp.index.intersection(base_series.index)
        exp = exp.loc[common_idx]
        base_series = base_series.loc[common_idx]
        opt_series = opt_series.loc[common_idx]
        opt_shifted_series = opt_shifted_series.loc[common_idx]

        # Error metrics
        def compute_metrics(sim, name):
            return {
                name: {
                    'MAE': mean_absolute_error(exp, sim),
                    'MBE': (exp - sim).mean(),
                    'RMSE': mean_squared_error(exp, sim, squared=False),
                    'StdDev': (exp - sim).std()
                }
            }

        if position not in bay_results:
            bay_results[position] = {}

        bay_results[position].update(compute_metrics(base_series, 'Original'))
        bay_results[position].update(compute_metrics(opt_series, 'Optimized'))
        bay_results[position].update(compute_metrics(opt_shifted_series, 'Optimized+Shifted'))

        # Plotting temperature comparison
        plt.plot(exp, label='Experimental', linewidth=2)
        plt.plot(exp.index, base_series, '--', label='Baseline Sim', alpha=0.7)
        plt.plot(exp.index, opt_series, '-', label='Optimized Sim', alpha=0.9)
        plt.plot(exp.index, opt_shifted_series, '-', label='Optimized + Shift', alpha=0.9)

        plt.title(
            f"{bay.replace('_', ' ')} {position} Comparison\n"
            f"MAE ↓ {bay_results[position]['Original']['MAE']:.2f}°C → {bay_results[position]['Optimized+Shifted']['MAE']:.2f}°C"
        )

        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)

        # ✅ PLACE THIS LINE HERE
        plt.show()

        plot_dir = os.path.join(OUTPUT_DIR, bay)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{position}_comparison.png"), dpi=300)
        plt.close()

        # ✅ Residual & FFT plots now apply to both Roof and Ceiling
        night_mask = (exp.index.hour >= 20) | (exp.index.hour <= 6)
        residual_night = exp[night_mask] - opt_shifted_series[night_mask]
        residual_all = exp - opt_shifted_series

        # Nighttime Residual Plot
        plt.figure(figsize=(12, 4))
        plt.plot(residual_night, label='Nighttime Residual (Exp - Sim)', color='purple')
        plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        plt.title(f"{bay.replace('_', ' ')} Nighttime Residuals ({position})")
        plt.ylabel("Residual (°C)")
        plt.xlabel("Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"{position}_nighttime_residual.png"), dpi=300)
        plt.close()

        # Full Residual Plot
        plt.figure(figsize=(12, 4))
        plt.plot(residual_all, label='Residual (Exp - Sim)', color='darkred')
        plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        plt.title(f"{bay.replace('_', ' ')} Residuals ({position}) — Full Duration")
        plt.ylabel("Residual (°C)")
        plt.xlabel("Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"{position}_residual_full.png"), dpi=300)
        plt.close()

        # FFT Residual
        fft_path = os.path.join(plot_dir, f"{position}_fft_residual.png")
        plot_fft_residual(residual_all, f"{bay.replace('_', ' ')} {position}", fft_path)

    results[bay] = bay_results

def plot_fft_residual(residual_series, title, output_path):
    n = len(residual_series)
    dt = 3600  # Sampling interval in seconds (1 hour)

    # Zero-mean for FFT clarity
    residual_zero_mean = residual_series - residual_series.mean()

    # Compute FFT
    freq = fftfreq(n, d=dt)
    amplitude = np.abs(fft(residual_zero_mean))**2

    # Only keep positive frequencies
    mask = freq > 0
    freq_positive = freq[mask]

    # Convert freq (Hz) → period (hours)
    period_seconds = 1 / freq_positive
    period_hours = period_seconds / 3600

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(period_hours, amplitude[mask])
    plt.xscale('log')
    plt.xlabel("Period (hours)")
    plt.ylabel("Power (Residual²)")
    plt.title(f"FFT Spectrum of Residuals: {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


    def plot_fft_residual(residual_series, title, output_path):
        n = len(residual_series)
        dt = 3600  # Sampling interval in seconds (1 hour)

        # Zero-mean for FFT clarity
        residual_zero_mean = residual_series - residual_series.mean()

        # Compute FFT
        freq = fftfreq(n, d=dt)
        amplitude = np.abs(fft(residual_zero_mean)) ** 2

        # Only keep positive frequencies
        mask = freq > 0
        freq_positive = freq[mask]

        # Convert freq (Hz) → period (hours)
        period_seconds = 1 / freq_positive
        period_hours = period_seconds / 3600

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(period_hours, amplitude[mask])
        plt.xscale('log')
        plt.xlabel("Period (hours)")
        plt.ylabel("Power (Residual²)")
        plt.title(f"FFT Spectrum of Residuals: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

def print_performance_comparison(results):
    print("\n📊 Performance Comparison:\n")

    for bay, bay_results in results.items():
        print(f"\n{bay.replace('_', ' ')} Performance:")
        print(f"{'Position':<12} {'Metric':<18} {'Original':<12} {'Optimized':<12} {'+ Shift':<12}")
        print("=" * 70)

        for pos in bay_results.keys():  # Only loop over available zones
            orig = bay_results[pos].get('Original', {})
            opt = bay_results[pos].get('Optimized', {})
            shifted = bay_results[pos].get('Optimized+Shifted', {})

            # Format numbers safely
            def fmt(val): return f"{val:.2f}" if isinstance(val, (float, int)) else "N/A"

            print(
                f"{pos:<12} {'MAE (°C)':<18} {fmt(orig.get('MAE')):<12} {fmt(opt.get('MAE')):<12} {fmt(shifted.get('MAE')):<12}")
            print(
                f"{'':<12} {'MBE (°C)':<18} {fmt(orig.get('MBE')):<12} {fmt(opt.get('MBE')):<12} {fmt(shifted.get('MBE')):<12}")
            print(
                f"{'':<12} {'RMSE (°C)':<18} {fmt(orig.get('RMSE')):<12} {fmt(opt.get('RMSE')):<12} {fmt(shifted.get('RMSE')):<12}")
            print(
                f"{'':<12} {'StdDev (°C)':<18} {fmt(orig.get('StdDev')):<12} {fmt(opt.get('StdDev')):<12} {fmt(shifted.get('StdDev')):<12}")
            print("-" * 70)


# ===================== MAIN WORKFLOW =====================
# ===================== MAIN WORKFLOW =====================
BAYS = ["South_Bay"]


def main():
    try:
        start_time = time.time()
        weather = process_weather_data()
        initialize_resistances()

        print(f"Weather time range: {weather['time_steps'][0]} to {weather['time_steps'][-1]}")
        print(f"Simulation window: {SIMULATION_START} to {SIMULATION_END}")

        results = {}

        for bay in BAYS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n🔧 Processing {bay}...")

            experimental_data = load_experimental_data(bay)

            base_params = [
                3.938320e-03,  # R_insulation
                6.552641e+06,  # C_insulation
                4.726526e+06,  # C_membrane
                4.592699e+07,  # C_metal_deck
                0.0,  # time_shift
                0.28,  # solar_absorptance
                2000,  # K_p
                30  # K_i
            ]

            base_df = simulate_state_space(base_params, weather['data'], weather['interpolators'],
                                           SIMULATION_START, SIMULATION_END)

            optimized_params = run_optimization(bay, experimental_data, weather, weather['interpolators'])

            if optimized_params is None:
                print("❗ Optimization failed. Skipping this bay.")
                continue

            optimized_df = simulate_state_space(optimized_params, weather['data'], weather['interpolators'],
                                                SIMULATION_START, SIMULATION_END)

            # Get time shift parameter
            time_shift = optimized_params[OPTIMIZE_PARAMS.index('time_shift')]

            # Create clean 5-min experimental grid
            base_exp_index = experimental_data['Roof'].iloc[:, :3].mean(axis=1).index
            base_exp_index = base_exp_index[(base_exp_index >= SIMULATION_START) &
                                            (base_exp_index <= SIMULATION_END)]
            base_exp_index = base_exp_index.round("5min").drop_duplicates()

            # Apply time shift (corrected sign)
            shifted_exp_index = base_exp_index - pd.Timedelta(hours=time_shift)


            # Interpolate simulation values to shifted times
            sim_time = optimized_df.index
            sim_seconds = (sim_time - sim_time[0]).total_seconds()
            shifted_seconds = (shifted_exp_index - sim_time[0]).total_seconds()

            T_mem_final = np.interp(shifted_seconds, sim_seconds, optimized_df['T_membrane'].values)
            T_deck_final = np.interp(shifted_seconds, sim_seconds, optimized_df['T_metal_deck'].values)
            T_zone_final = np.interp(shifted_seconds, sim_seconds, optimized_df['T_zone_air'].values)  # ✅ New line

            # Compute HVAC values (crucial addition)
            HVAC_series = compute_HVAC_series(
                optimized_df,
                T_set=22,
                K_p=optimized_params[6],
                K_i=optimized_params[7],
                dt=DT_SECONDS
            )
            HVAC_final = np.interp(shifted_seconds, sim_seconds, HVAC_series)

            # ✅ Create DataFrame with T_zone_air added
            pinn_input_df = pd.DataFrame({
                'T_membrane': T_mem_final,
                'T_metal_deck': T_deck_final,
                'T_zone_air': T_zone_final,  # ✅ Added
                'HVAC': HVAC_final
            }, index=base_exp_index)  # Using experimental timestamps

            # Save output
            bay_dir = os.path.join(OUTPUT_DIR, bay)
            os.makedirs(bay_dir, exist_ok=True)

            pinn_input_df.to_csv(os.path.join(bay_dir, f"PINN_input_{timestamp}_SHIFTED.csv"))

            # For MAE consistency, use the same data in analysis and export
            analyze_results(bay, experimental_data, base_df, optimized_df, optimized_params, results)

        print_performance_comparison(results)
        print(f"\n⏱️ Total runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"[❌ MAIN ERROR] {str(e)}")
        raise


if __name__ == "__main__":
    main()

