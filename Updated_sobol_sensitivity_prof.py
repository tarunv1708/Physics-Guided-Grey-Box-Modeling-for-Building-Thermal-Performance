import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze
from multiprocessing import Pool
from tqdm import tqdm
import time



# Constants

sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m².K⁴)
A_roof = 921.35  # Roof area in m²
epsilon = 0.9  # Emissivity for radiative resistance
solar_absorptance = 0.35  # Solar absorptance
T_set = 22  # Indoor temperature setpoint (°C)

variation = {
    "R_membrane": 0.4,
    "R_insulation": 0.35,
    "R_metal_deck": 0.2,

    "C_membrane": 0.3,
    "C_insulation": 0.35,
    "C_metal_deck": 0.2,
     "C_air": 0.25
}


# Material Properties (Revised for Realism)
material_props = {
    "roof_membrane": {"k": 0.16, "density": 950, "cp": 1800, "thickness": 0.003},
    "insulation": {"k": 0.035, "density": 40, "cp": 1400, "thickness": 0.127},      # Fiberglass (ρ=40 kg/m³)
    "metal_deck": {"k": 50, "density": 7850, "cp": 500, "thickness": 0.0127},    # Near-zero thermal mass
    "air": {"density": 1.225, "cp": 1005, "volume": 3.0*921.35},
}


# Helper Functions ------------------------------------------------------------
def compute_resistance(material, A):
    return material["thickness"] / (material["k"] * A)

# Compute Capacitances (Total for Roof Area)
def compute_capacitance(material, A):
    return material["density"] * material["cp"] * material["thickness"] * A  # [J/K]


def compute_sobol_bounds(material_props, A_roof, variation):
    def R(material): return material["thickness"] / (material["k"] * A_roof)
    def C(material): return material["density"] * material["cp"] * material["thickness"] * A_roof

    # Physical values
    R_values = {
        "R_membrane": R(material_props["roof_membrane"]),
        "R_insulation": R(material_props["insulation"]),
        "R_metal_deck": R(material_props["metal_deck"]),
    }

    C_values = {
        "C_membrane": C(material_props["roof_membrane"]),
        "C_insulation": C(material_props["insulation"]),
        "C_metal_deck": C(material_props["metal_deck"]),
         "C_air": material_props["air"]["density"] * material_props["air"]["cp"] * material_props["air"]["volume"] * A_roof
    }

    bounds = []
    for key in list(R_values.keys()) + list(C_values.keys()):
        nominal = R_values.get(key) or C_values.get(key)
        perc = variation[key]
        lower = nominal * (1 - perc)
        upper = nominal * (1 + perc)
        bounds.append((lower, upper))

    return bounds

problem = {
    "num_vars": 7,
    "names": [
        "R_membrane", "R_insulation", "R_metal_deck",
        "C_membrane", "C_insulation", "C_metal_deck", "C_air"],
    "bounds": compute_sobol_bounds(material_props, A_roof, variation)
}



# Weather Data Processing ------------------------------------------------------
def load_weather_data():
    # ✅ Load both 2024 and 2025 weather files
    weather_2024 = pd.read_csv(r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2024v3.csv")
    weather_2025 = pd.read_csv(r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2025v3.csv")

    data = pd.concat([weather_2024, weather_2025])

    # Clean column names
    data.columns = (
        data.columns
        .str.strip()
        .str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.upper()
    )

    column_mapping = {
        'DATE': 'Datetime',
        'DRYBULB C': 'DBT',
        'RELATIVE HUMIDITY': 'RH',
        'GLOBAL HORIZONTAL RADIATION WH M': 'GloHorzRad',
        'WIND DIRECTION': 'WD',
        'WIND SPEED M S': 'WS'
    }

    data = (
        data.rename(columns=column_mapping)
        [['Datetime', 'DBT', 'RH', 'GloHorzRad', 'WD', 'WS']]
    )

    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')

    # ✅ Filter to the time window you want
    data = data[
        (data['Datetime'] >= '2024-10-16 00:00:00') &
        (data['Datetime'] <= '2025-06-04 23:55:00')
    ].reset_index(drop=True)

    # Create hour index
    data['HourIndex'] = (data['Datetime'] - data['Datetime'].iloc[0]).dt.total_seconds() / 3600

    # Compute additional variables
    def compute_T_sky(T_air, RH):
        e = (RH / 100) * 6.11 * 10 ** ((7.5 * T_air) / (237.3 + T_air))
        return T_air * (1.22 * (e / 10) ** 0.5 - 0.22)

    data['T_sky'] = compute_T_sky(data['DBT'], data['RH'])
    data['LatentHeat'] = 2256000 * data['RH'] / 100

    # Convert GHI from Wh/m2 to W/m2
    data['GloHorzRad'] = data['GloHorzRad']  # Already converted earlier if needed

    time_steps = data['HourIndex'].values

    return (
        interp1d(time_steps, data['DBT'], fill_value="extrapolate"),
        interp1d(time_steps, data['T_sky'], fill_value="extrapolate"),
        interp1d(time_steps, data['WS'], fill_value="extrapolate"),
        interp1d(time_steps, data['WD'], fill_value="extrapolate"),
        interp1d(time_steps, data['GloHorzRad'], fill_value="extrapolate"),
        interp1d(time_steps, data['LatentHeat'], fill_value="extrapolate"),
        interp1d(time_steps, data['RH'], fill_value="extrapolate"),
        time_steps,
        data['Datetime'].iloc[0]  # ✅ Return start datetime for index building
    )


# Thermal Calculations ---------------------------------------------------------
def wind_direction_modifier(wind_direction):
    return 1.0 if 0 <= wind_direction <= 180 else 0.5

# Thermal model functions (Revised)
def R_rad(T_roof, T_sky, epsilon=0.95):  # ← ADD default again

    T_roof_K = T_roof + 273.15
    T_sky_K = T_sky + 273.15
    h_rad = epsilon * sigma * ((T_roof_K ** 2 + T_sky_K ** 2) * (T_roof_K + T_sky_K))
    return max(1 / h_rad, 1e-6)  # Per m² [K·m²/W]

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

# HVAC Control Parameters
K_p = 1500
K_i = 30

def HVAC_heat_transfer_PI(T_zone, T_set, K_p, K_i, integral, dt, max_rate=35520):
    error = T_set - T_zone
    integral += error * dt
    Q = K_p * error + K_i * integral
    return np.clip(Q, -max_rate, max_rate), integral


# Main Simulation Function (Updated for RC Time Constant Model) -------------------------
def simulate_state_space(params=None):
    global T_outside_interpolator, T_sky_interpolator, wind_speed_interpolator, \
        wind_direction_interpolator, GHI_interpolator, latent_heat_interpolator, \
        RH_interpolator, time_steps, start_time

    # Load weather data (only once)
    if 'T_outside_interpolator' not in globals():
        (T_outside_interpolator, T_sky_interpolator, wind_speed_interpolator,
         wind_direction_interpolator, GHI_interpolator, latent_heat_interpolator,
         RH_interpolator, time_steps, start_time) = load_weather_data()

    # Set parameters or use defaults
    if params is None:
        params = [
            compute_resistance(material_props["roof_membrane"], A_roof),
            compute_resistance(material_props["insulation"], A_roof),
            compute_resistance(material_props["metal_deck"], A_roof),
            compute_capacitance(material_props["roof_membrane"]),
            compute_capacitance(material_props["insulation"]),
            compute_capacitance(material_props["metal_deck"]),
            material_props["air"]["density"] * material_props["air"]["cp"] * material_props["air"]["volume"]
        ]

    # Unpack parameters
    (R_membrane, R_roof_insulation, R_metal_deck,
     C_membrane, C_insulation, C_metal_deck, C_air) = params


    dt = 600  # 10 minutes in seconds
    print(f"Running simulate_state_space with fixed dt = {dt} sec")

    # Simulation time base
    t_sim = np.arange(time_steps[0], time_steps[-1], dt / 3600)  # in hours
    x_out = np.zeros((len(t_sim), 4))
    x_out[0] = [22.0] * 4
    integral = 0.0

    for i in range(1, len(t_sim)):
        T_membrane, T_insulation, T_metal_deck, T_zone = x_out[i - 1]
        t_current = t_sim[i]

        T_outside = T_outside_interpolator(t_current)
        RH = RH_interpolator(t_current)
        T_sky = T_sky_interpolator(t_current)
        wind_speed = wind_speed_interpolator(t_current)
        wind_dir = wind_direction_interpolator(t_current)
        GHI = GHI_interpolator(t_current)

        #C_air = material_props["air"]["density"] * material_props["air"]["cp"] * volume

        R_conv_val = R_conv(T_membrane, T_outside, wind_speed, wind_dir) / A_roof
        R_rad_val = R_rad(T_membrane, T_sky) / A_roof
        R_conv_rad = 1 / (1 / R_conv_val + 1 / R_rad_val)
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

        if i % 50 == 0:
            print(f"[Step {i}]")
            print(f"  Q_solar = {Q_solar:.2f} W")
            print(f"  Q_rad   = {Q_rad:.2f} W")
            print(f"  Q_conv  = {Q_conv:.2f} W")
            print(f"  Net     = {(Q_solar - Q_rad - Q_conv):.2f} W")
            print(f"  T_mem   = {T_membrane:.2f} °C | T_sky = {T_sky:.2f} °C | T_out = {T_outside:.2f} °C")

        rhs = x_out[i - 1] + dt * (B @ U)
        rhs[0] += dt * (Q_solar - Q_rad - Q_conv) / C_membrane
        lhs = np.eye(4) - dt * A
        x_out[i] = np.linalg.solve(lhs, rhs)
        x_out[i] = np.clip(x_out[i], -50, 100)

    sim_index = pd.date_range(start=start_time, periods=len(t_sim), freq=f'{int(dt)}s')

    T_roof = x_out[-1, 0]  # T_membrane
    T_ceiling = x_out[-1, 2]  # T_metal_deck
    Delta_T = T_roof - T_ceiling
    return [T_ceiling, T_roof, Delta_T]


# Sensitivity Analysis ---------------------------------------------------------
if __name__ == "__main__":
    # Generate samples
    N = 512 # For real analysis, try 512 or 1024
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    (T_outside_interpolator, T_sky_interpolator, wind_speed_interpolator,
     wind_direction_interpolator, GHI_interpolator, latent_heat_interpolator,
     RH_interpolator, time_steps, start_time) = load_weather_data()

    import time

    # Test a single simulation for speed
    print("⏱️ Timing one simulate_state_space run...")
    start_time = time.time()
    simulate_state_space(param_values[0])
    print("✅ Done. Time taken: {:.2f} seconds".format(time.time() - start_time))

    # Run simulations with multiprocessing + progress bar
    with Pool(processes=os.cpu_count()) as pool:
        outputs = np.array(list(tqdm(pool.imap(simulate_state_space, param_values), total=len(param_values))))

    # Separate outputs
    T_ceiling_outputs = outputs[:, 0]
    T_roof_outputs = outputs[:, 1]
    DeltaT_outputs = outputs[:, 2]

    # Save outputs
    output_path = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\sobol"
    os.makedirs(output_path, exist_ok=True)

    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame({
        "T_roof": T_roof_outputs,
        "T_ceiling": T_ceiling_outputs,
        "Delta_T": DeltaT_outputs
    }).to_csv(os.path.join(output_path, "sobol_outputs_Troof_T_ceiling_DeltaT.csv"), index=False)

    # Sobol analysis (no second-order)
    Si_T_ceiling = sobol_analyze.analyze(problem, T_ceiling_outputs, calc_second_order=False)
    Si_T_roof = sobol_analyze.analyze(problem, T_roof_outputs, calc_second_order=False)
    Si_DeltaT = sobol_analyze.analyze(problem, DeltaT_outputs, calc_second_order=False)

    # Plot helper
    def plot_sobol(title, S1, ST, filename, color1, color2):
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(problem["names"])), S1, width=0.4, color=color1, label="First-Order")
        plt.bar(np.arange(len(problem["names"])), ST, width=0.4, alpha=0.7, color=color2, label="Total-Order")
        plt.title(title)
        plt.ylabel("Sensitivity Index")
        plt.xticks(np.arange(len(problem["names"])), problem["names"], rotation=90)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, filename), dpi=300)
        plt.close()

    # Save plots
    plot_sobol("Sobol Indices: Roof Temp (T_roof)", Si_T_roof["S1"], Si_T_roof["ST"], "sobol_roof_only.png", "gold", "orange")
    plot_sobol("Sobol Indices: Ceiling Temp (T_ceiling)", Si_T_ceiling["S1"], Si_T_ceiling["ST"], "sobol_ceiling_only.png", "red", "steelblue")
    plot_sobol("Sobol Indices: ΔT (Roof - Ceiling)", Si_DeltaT["S1"], Si_DeltaT["ST"], "sobol_DeltaT_colored.png", "royalblue", "mediumseagreen")

    # Print summary
    print("\nRoof Temperature Sensitivity (T_roof):")
    print("First-order indices:", dict(zip(problem["names"], Si_T_roof["S1"])))
    print("Total-order indices:", dict(zip(problem["names"], Si_T_roof["ST"])))

    print("\nCeiling Temperature Sensitivity (T_ceiling):")
    print("First-order indices:", dict(zip(problem["names"], Si_T_ceiling["S1"])))
    print("Total-order indices:", dict(zip(problem["names"], Si_T_ceiling["ST"])))

    print("\nRoof-to-Ceiling ΔT Sensitivity:")
    print("First-order indices:", dict(zip(problem["names"], Si_DeltaT["S1"])))
    print("Total-order indices:", dict(zip(problem["names"], Si_DeltaT["ST"])))

    # === Roof Temp Plot ===
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(problem["names"])), Si_T_roof["S1"], width=0.4, color="gold", label="First-Order")
    plt.bar(np.arange(len(problem["names"])), Si_T_roof["ST"], width=0.4, alpha=0.7, color="orange",
            label="Total-Order")
    plt.title("Sobol Indices: Roof Temp (T_roof)")
    plt.ylabel("Sensitivity Index")
    plt.xticks(np.arange(len(problem["names"])), problem["names"], rotation=90)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    plt.tight_layout()
    plt.savefig("updated_sobol_roof_only.png", dpi=300)
    plt.show()

    # === Ceiling Temp Plot ===
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(problem["names"])), Si_T_ceiling["S1"], width=0.4, color="red", label="First-Order")
    plt.bar(np.arange(len(problem["names"])), Si_T_ceiling["ST"], width=0.4, alpha=0.7, color="steelblue",
            label="Total-Order")
    plt.title("Sobol Indices: Ceiling Temp (T_ceiling)")
    plt.ylabel("Sensitivity Index")
    plt.xticks(np.arange(len(problem["names"])), problem["names"], rotation=90)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    plt.tight_layout()
    plt.savefig("updated_sobol_ceiling_only.png", dpi=300)
    plt.show()

    # === Separate Plot: ΔT ===
    plt.figure(figsize=(10, 5))

    plt.bar(np.arange(len(problem["names"])), Si_DeltaT["S1"], width=0.4, color="royalblue", label="First-Order")
    plt.bar(np.arange(len(problem["names"])), Si_DeltaT["ST"], width=0.4, alpha=0.7, color="mediumseagreen",
            label="Total-Order")

    plt.title("Sobol Indices: ΔT (Roof - Ceiling)")
    plt.ylabel("Sensitivity Index")
    plt.xticks(np.arange(len(problem["names"])), problem["names"], rotation=90)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("updated_sobol_DeltaT_colored.png", dpi=300)
    plt.show()


