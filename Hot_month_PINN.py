import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

EXPERIMENTAL_DATA_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\RC_Modelling\8"


class ThermalDataset(Dataset):
    def __init__(self, sequences, weather_data, rc_params, return_datetime=False):
        self.sequences = sequences
        self.weather_data = weather_data
        self.rc_params = rc_params
        self.return_datetime = return_datetime  # NEW

        # Precompute normalization
        self.feature_mean = np.mean(weather_data[['DBT', 'GHI', 'WS', 'WD', 'RH']].values, axis=0)
        self.feature_std = np.std(weather_data[['DBT', 'GHI', 'WS', 'WD', 'RH']].values, axis=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]

        # Time shift for physics alignment
        time_shift_hr = self.rc_params.get('time_shift', 0)
        t = (seq_data.index + pd.Timedelta(hours=time_shift_hr) - seq_data.index[0]).total_seconds().values

        # Match weather data
        seq_start = seq_data.index.min()
        seq_end = seq_data.index.max()
        weather_mask = (self.weather_data.index >= seq_start) & (self.weather_data.index <= seq_end)
        seq_weather = self.weather_data[weather_mask]

        features, targets, baselines, ts_list = [], [], [], []

        for i in range(min(len(seq_data), len(seq_weather))):
            try:
                hour = t[i] / 3600 % 24
                time_feat = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]

                # Get raw weather data and clamp any NaNs
                weather_raw = seq_weather.iloc[i][['DBT', 'GHI', 'WS', 'WD', 'RH']].values
                weather_raw = np.nan_to_num(weather_raw, nan=0.0, posinf=0.0, neginf=0.0)
                weather = (weather_raw - self.feature_mean) / self.feature_std

                # HVAC normalization
                hvac = seq_data['HVAC'].iloc[i]
                hvac = np.nan_to_num(hvac, nan=0.0, posinf=0.0, neginf=0.0) / 35520.0

                # 🔸 T_zone as input feature (new step)
                tzone = seq_data['T_zone_air'].iloc[i]
                tzone = np.nan_to_num(tzone, nan=0.0, posinf=0.0, neginf=0.0)

                # Required columns
                if not all(col in seq_data.columns for col in ['Roof', 'Ceiling', 'T_membrane', 'T_metal_deck']):
                    continue

                # Skip if target or baseline are NaN
                target_vals = seq_data[['Roof', 'Ceiling']].iloc[i].values
                baseline_vals = seq_data[['T_membrane', 'T_metal_deck']].iloc[i].values
                if np.isnan(target_vals).any() or np.isnan(baseline_vals).any():
                    continue

                input_vector = np.concatenate([time_feat, weather, [hvac, tzone]])  # ✅ Now 9 features
                if np.isnan(input_vector).any():
                    continue

                # All checks passed — append
                features.append(input_vector)
                targets.append(target_vals)
                baselines.append(baseline_vals)
                ts_list.append(seq_data.index[i])

            except Exception as e:
                print(f"⚠️ Skipping step {i} in seq {idx} due to error: {e}")
                continue

        if len(features) == 0:
            raise ValueError(f"❌ Sequence {idx} has no usable data after filtering.")

        return {
            'features': torch.FloatTensor(np.array(features)),
            'target': torch.FloatTensor(np.array(targets)),
            'baseline': torch.FloatTensor(np.array(baselines)),
            'timestamps': ts_list if self.return_datetime else torch.tensor(
                pd.Series(ts_list).astype(np.int64).values // 1e9, dtype=torch.float32)
        }


class ThermalPINN(nn.Module):

    @staticmethod
    def compute_T_sky(T_air, RH):
        e = (RH / 100.0) * 6.11 * 10 ** (7.5 * T_air / (237.3 + T_air))
        return T_air * (1.22 * torch.sqrt(e / 10) - 0.22)

    @staticmethod
    def h_conv_tarp(T_surface, T_ambient, wind_speed, wind_dir):
        delta_T = T_surface - T_ambient
        W_f = torch.where((wind_dir >= 0) & (wind_dir <= 180), 1.0, 0.5)
        h_n = 1.31 * torch.abs(delta_T) ** (1 / 3)
        h_f = W_f * 3.26 * wind_speed ** 0.89
        return torch.sqrt(h_n ** 2 + h_f ** 2).clamp_min(1e-3)

    def __init__(self, rc_params, feature_mean, feature_std):
        super().__init__()

        self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32)
        self.feature_std = torch.tensor(feature_std, dtype=torch.float32)

        self.sigma = 5.67e-8
        self.A_roof = 921.35
        self.epsilon = 0.85
        self.T_set = 22

        self.solar_abs = torch.tensor(rc_params['solar_absorptance'])
        self.K_p = torch.tensor(rc_params['K_p'])
        self.K_i = torch.tensor(rc_params['K_i'])

        self.residual_nn = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.delta_scale = 10.0  # or max 10.0
        # or 30.0 to allow stronger residual correction
        # Scale factor to match real-world residual range

        self.blend_layer = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

        # 🔬 For basic learning test — replace with a linear layer
        # self.residual_nn = nn.Linear(8, 2)

        # Physics parameters (fixed from PSO optimization)
        self.register_buffer('R_ins', torch.tensor(rc_params['R_insulation']))
        self.register_buffer('C_mem', torch.tensor(rc_params['C_membrane']))
        self.register_buffer('C_ins', torch.tensor(rc_params['C_insulation']))
        self.register_buffer('C_metal', torch.tensor(rc_params['C_metal_deck']))

        # 🔁 NEW: Add configurable R2
        R2_default = 1 / (2.5 * self.A_roof)  # convection to zone air
        self.register_buffer('R2', torch.tensor(rc_params.get('R2', R2_default), dtype=torch.float32))

        self.hvac_gain = nn.Parameter(torch.tensor(0.1))  # Trainable HVAC scaling factor

        def hvac_transfer(self, T_zone, integral, dt):
            error = self.T_set - T_zone
            integral += error * dt
            Q = self.K_p * error + self.K_i * integral
            return torch.clamp(Q, -35520, 35520), integral

    def forward(self, features, baseline):
        # 🔧 Handle 2D input during inference (from apply_pinn_correction)
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, seq, feat]
            baseline = baseline.unsqueeze(0)  # [1, seq, 2]

        ghi = features[:, :, 3]  # GHI index assumed correct
        is_night = (ghi < 10).float()
        is_day = 1.0 - is_night

        baseline_safe = torch.clamp(baseline, min=1.0)
        # max_delta_day = torch.minimum(baseline_safe * 0.10, torch.tensor(5.0).to(baseline.device))
        # max_delta_night = torch.minimum(baseline_safe * 0.10, torch.tensor(4.0).to(baseline.device))

        # max_delta = is_day.unsqueeze(-1) * max_delta_day + is_night.unsqueeze(-1) * max_delta_night

        delta = self.residual_nn(features) * self.delta_scale
        baseline_safe = torch.clamp(baseline, min=1.0)
        max_delta_day = torch.minimum(baseline_safe * 0.50, torch.tensor(12.0))
        max_delta_night = torch.minimum(baseline_safe * 0.60, torch.tensor(14.0))

        max_delta = is_day.unsqueeze(-1) * max_delta_day + is_night.unsqueeze(-1) * max_delta_night
        delta = torch.clamp(delta, -max_delta, max_delta)

        # delta = torch.clamp(delta, -max_delta, max_delta)

        learned_target = baseline + delta

        alpha = self.blend_layer(features)
        alpha_day = torch.sigmoid(alpha * 3.0)  # No clamp
        alpha_night = torch.sigmoid(alpha * 4.0)

        if self.training == False:
            print(f"🔍 Mean alpha_day: {alpha_day.mean().item():.2f}, alpha_night: {alpha_night.mean().item():.2f}")

        alpha = is_day.unsqueeze(-1) * alpha_day + is_night.unsqueeze(-1) * alpha_night

        corrected = (1 - alpha) * baseline + alpha * learned_target
        return corrected

    def physics_loss(self, preds, prev_state, inputs, dt):
        T_mem, T_ceil = preds[:, 0], preds[:, 1]
        T_mem_prev, T_ceil_prev = prev_state[:, 0], prev_state[:, 1]

        # ✅ Extract T_zone from input features (index 8)
        T_zone_prev = inputs[:, 8]

        # Inputs: [sin, cos, T_out, RH, GHI, WS, WD, HVAC]
        T_out = inputs[:, 2] * self.feature_std[0] + self.feature_mean[0]
        RH = inputs[:, 3]
        GHI = inputs[:, 4] * self.feature_std[1] + self.feature_mean[1]
        WS = inputs[:, 5] * self.feature_std[2] + self.feature_mean[2]
        WD = inputs[:, 6] * self.feature_std[3] + self.feature_mean[3]
        HVAC = inputs[:, 7] * 35520  # Undo normalization

        # Compute physics terms
        T_sky = self.compute_T_sky(T_out, RH)
        h_conv = self.h_conv_tarp(T_mem_prev, T_out, WS, WD)

        Q_conv = h_conv * self.A_roof * (T_mem_prev - T_out)
        Q_conv = torch.nan_to_num(Q_conv, nan=0.0, posinf=0.0, neginf=0.0)  # 🔒 clamp

        Q_rad = self.epsilon * self.sigma * self.A_roof * ((T_mem_prev + 273.15) ** 4 - (T_sky + 273.15) ** 4)
        Q_rad = torch.nan_to_num(Q_rad, nan=0.0, posinf=0.0, neginf=0.0)  # 🔒 clamp

        Q_solar = GHI * self.A_roof * self.solar_abs
        Q_solar = torch.nan_to_num(Q_solar, nan=0.0, posinf=0.0, neginf=0.0)  # 🔒 clamp

        # Membrane energy balance (same)
        dT_mem = (Q_solar - Q_conv - Q_rad) / self.C_mem
        dT_mem = torch.nan_to_num(dT_mem, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp HVAC energy to prevent instability
        Q_hvac = torch.clamp(HVAC, -35520, 35520)

        # ✅ Add HVAC to ceiling energy balance with trainable gain
        dT_ceil = (
                          (T_mem_prev - T_ceil_prev) / self.R_ins
                          - (T_ceil_prev - T_zone_prev) / self.R2
                          + self.hvac_gain * Q_hvac / self.A_roof
                  ) / self.C_ins

        dT_ceil = torch.nan_to_num(dT_ceil, nan=0.0, posinf=0.0, neginf=0.0)

        # Physics residuals
        res_mem = (T_mem - T_mem_prev) / dt - dT_mem
        res_ceil = (T_ceil - T_ceil_prev) / dt - dT_ceil

        return torch.mean(res_mem ** 2 + res_ceil ** 2)


def train_pinn(model, dataloader, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.3)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features']
            targets = batch['target']
            baseline = batch['baseline']
            time = batch['timestamps']

            features = torch.nan_to_num(features, nan=0.0)
            baseline = torch.nan_to_num(baseline, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)

            preds = model(features, baseline)

            if torch.isnan(preds).any():
                print(f"⚠️ Batch {batch_idx}: NaN in predictions. Skipping.")
                continue

            # Core losses
            delta_true = targets - baseline
            delta_pred = preds - baseline

            delta_loss = torch.mean(torch.abs(delta_pred - delta_true))
            data_loss = F.l1_loss(preds, targets)
            correction_penalty = torch.mean(torch.abs(preds - baseline))

            # Nighttime-specific loss
            ghi = features[:, :, 4]  # assuming GHI is 4th feature
            is_night = (ghi < 10).float()
            night_mask = is_night.unsqueeze(-1).expand_as(preds)
            night_loss = torch.mean(torch.abs(preds - targets) * night_mask)

            # Physics loss
            dt = time[1] - time[0] if len(time) > 1 else 600
            phys_losses = []
            for i in range(preds.shape[0]):
                try:
                    loss = model.physics_loss(preds[i], baseline[i], features[i], dt)
                    if not torch.isnan(loss):
                        phys_losses.append(loss)
                except Exception as e:
                    print(f"⚠️ Physics loss error: {e}")

            if len(phys_losses) == 0:
                continue
            phys_loss = torch.stack(phys_losses).mean()

            # Final total loss (🎯 updated weights)
            total_loss = (
                    0.05 * data_loss +
                    0.6 * delta_loss +
                    0.35 * phys_loss +
                    0.15 * night_loss
            )

            if torch.isnan(total_loss):
                print(f"❌ Epoch {epoch}: NaN loss. Skipping backward.")
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step(epoch_loss)
        print(f"📉 Epoch {epoch}: Loss = {epoch_loss:.4f}")

    return model


from collections import defaultdict


def apply_pinn_correction(sim_df, weather_df, model, rc_params):
    """
    Sequence-aware PINN correction using baseline + delta strategy with overlapping windows.
    """
    # 🔁 Split sim_df into overlapping sequences
    splitter = TemporalSequenceSplitter(sim_df, seq_length_days=2, stride_days=1)
    sequences = splitter.get_sequences()

    preds_by_timestamp = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if seq.empty:
                continue

            # Create dataset and get sample
            dataset = ThermalDataset([seq], weather_df, rc_params, return_datetime=True)
            try:
                sample = dataset[0]
            except Exception as e:
                print(f"⚠️ Skipping sequence {i}: {e}")
                continue

            features = sample['features']
            baseline = sample['baseline']
            timestamps = sample['timestamps']

            # Predict residuals (delta)
            # delta = model(features, baseline)

            # Corrected = baseline + delta
            # 🔧 Constraint: prevent overestimation
            # Strategy: Clip delta to ±30% of baseline value (or ±10°F if baseline too small)
            # max_delta = torch.clamp(torch.abs(baseline) * 0.3, min=1.0, max=10.0)  # Avoid 0 baseline

            # Clamp delta to avoid large jumps
            # delta = torch.clamp(delta, -max_delta, max_delta)

            # Apply correction
            corrected = model(features, baseline)

            if corrected.dim() == 3:
                corrected = corrected.squeeze(0)  # Ensure correct shape

            for j, ts in enumerate(timestamps):
                preds_by_timestamp[ts].append(corrected[j].cpu().numpy())

    # ✅ Average overlapping predictions
    all_ts = sorted(preds_by_timestamp.keys())
    T_membrane_corrected = []
    T_metal_deck_corrected = []

    for ts in all_ts:
        values = np.array(preds_by_timestamp[ts])
        T_membrane_corrected.append(values[:, 0].mean())
        T_metal_deck_corrected.append(values[:, 1].mean())

    corrected_df = pd.DataFrame({
        "T_membrane_corrected": T_membrane_corrected,
        "T_metal_deck_corrected": T_metal_deck_corrected
    }, index=pd.to_datetime(all_ts))

    corrected_df = corrected_df[~corrected_df.index.duplicated(keep='first')]
    corrected_df = corrected_df.sort_index()

    # ✅ Merge back with original sim_df
    sim_df = sim_df.copy()
    corrected_sim_df = sim_df.join(corrected_df, how='left')

    # Optional: interpolate short gaps if needed
    corrected_sim_df['T_membrane_corrected'] = corrected_sim_df['T_membrane_corrected'].interpolate(limit=6)
    corrected_sim_df['T_metal_deck_corrected'] = corrected_sim_df['T_metal_deck_corrected'].interpolate(limit=6)

    print("✅ Corrected Membrane (non-NaN):", corrected_sim_df['T_membrane_corrected'].notna().sum())
    print("🔍 NaNs in Corrected:", corrected_sim_df['T_membrane_corrected'].isna().sum())

    return corrected_sim_df, corrected_df


# Select which bay to optimize
bay = "Middle_Bay"  # Change to "South_Bay" as needed

# ===================== CONFIGURATION =====================
SIMULATION_START = "2025-5-11"
SIMULATION_END = "2025-05-21"

BAY_NAME = "South_Bay"  # ⚠️ Set this to your target bay ("South_Bay" or "Middle_Bay")
CONVERT_GHI_TO_WATTS = True  # Set based on your data format


# ===================== DATA LOADING =====================

def compute_T_sky_numpy(T_air, RH):
    e = (RH / 100.0) * 6.11 * 10 ** (7.5 * T_air / (237.3 + T_air))
    return T_air * (1.22 * np.sqrt(e / 10) - 0.22)


def load_weather_data():
    """Load and process weather data for PINN"""
    path_2024 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2024v3.csv"
    path_2025 = r"E:\Research_Assistant\Tarun\AZmet\AZMET_extracted_Hourly_2025v3.csv"

    # Load and clean data
    df_2024 = pd.read_csv(path_2024, encoding='utf-8')
    df_2025 = pd.read_csv(path_2025, encoding='utf-8')

    # Standardize column names
    for df in [df_2024, df_2025]:
        df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip().str.upper()

    # Column mapping
    rename_map = {
        'DATE': 'Datetime',
        'DRYBULB C': 'DBT',
        'RELATIVE HUMIDITY': 'RH',
        'GLOBAL HORIZONTAL RADIATION WH M': 'GHI',
        'WIND DIRECTION': 'WD',
        'WIND SPEED M S': 'WS'
    }

    # Process each year's data
    df_2024 = df_2024.rename(columns=rename_map)[['Datetime', 'DBT', 'RH', 'GHI', 'WD', 'WS']]
    df_2025 = df_2025.rename(columns=rename_map)[['Datetime', 'DBT', 'RH', 'GHI', 'WD', 'WS']]

    # Combine and clean
    weather = pd.concat([df_2024, df_2025])
    weather['Datetime'] = pd.to_datetime(weather['Datetime'], errors='coerce')
    weather = weather.sort_values('Datetime').drop_duplicates().set_index('Datetime')

    # Filter to simulation period
    weather = weather.loc[SIMULATION_START:SIMULATION_END]

    # Convert GHI units if needed
    if CONVERT_GHI_TO_WATTS:
        weather['GHI'] = weather['GHI']  # Already in W/m²
        # weather['GHI'] *= 277.78  # Uncomment if original was in Wh/m²

    # Calculate derived parameters
    weather['T_sky'] = compute_T_sky_numpy(weather['DBT'], weather['RH'])
    weather['LatentHeat'] = 2256000 * weather['RH'] / 100  # J/kg

    # Resample to 10-minute frequency with interpolation
    weather = weather.resample("10min").interpolate()

    print("✅ Weather data loaded with columns:", list(weather.columns))
    return weather


def load_experimental_data(bay):
    """Load experimental data for specified bay"""
    experimental_data = {}
    files = {
        "Ceiling": f"{bay}_Ceiling.xlsx",
        "MidLevel": f"{bay}_MidLevel.xlsx",
        "Roof": f"{bay}_Roof.xlsx",
    }

    for position, filename in files.items():
        file_path = os.path.join(EXPERIMENTAL_DATA_DIR, filename)
        try:
            df = pd.read_excel(file_path)

            # Clean and filter data
            df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y %H:%M:%S", errors='coerce')
            df = df[(df['Date'] >= SIMULATION_START) & (df['Date'] <= SIMULATION_END)]
            df = df.set_index('Date').sort_index()
            df = df[~df.index.duplicated(keep='first')]  # ✅ REMOVE DUPLICATE TIMESTAMPS HERE

            # Average multiple sensors if present
            experimental_data[position] = df.mean(axis=1) if len(df.columns) > 1 else df.squeeze()

        except Exception as e:
            print(f"❌ Error loading {position} data: {e}")
            experimental_data[position] = None

    # Combine and validate
    exp_df = pd.DataFrame(experimental_data)
    exp_df = exp_df[~exp_df.index.duplicated(keep='first')]  # ✅ Also clean combined DataFrame

    print(f"\n✅ Experimental data loaded for {bay}")
    print(f"Time range: {exp_df.index.min()} to {exp_df.index.max()}")
    print(f"Columns: {list(exp_df.columns)}\n")

    exp_df = exp_df.interpolate(method='time', limit_direction='both')

    for col in exp_df.columns:
        exp_df[col] = (exp_df[col] - 32) * 5 / 9

    return exp_df


class TemporalSequenceSplitter:
    def __init__(self, data, seq_length_days=1, stride_days=0.5):  # Shorter, overlapping sequences
        self.data = data
        self.seq_length = pd.Timedelta(days=seq_length_days)
        self.stride = pd.Timedelta(days=stride_days)

    def get_sequences(self):
        sequences = []
        start_date = self.data.index.min()
        end_date = self.data.index.max() - self.seq_length

        while start_date <= end_date:
            seq_end = start_date + self.seq_length
            mask = (self.data.index >= start_date) & (self.data.index < seq_end)
            chunk = self.data[mask]
            if len(chunk) > 0:
                sequences.append(chunk)
            start_date += self.stride

        return sequences


def stratified_temporal_split_by_months(sequences, train_months=[ 2,3,5], test_months=[4,6]):
    train_seqs, test_seqs = [], []
    for seq in sequences:
        if seq.empty:
            continue  # Skip empty sequences

        start_time = seq.index[0]
        month = start_time.month

        if month in train_months:
            train_seqs.append(seq)
        elif month in test_months:
            test_seqs.append(seq)
    return train_seqs, test_seqs

def load_simulation_data():
    sim = pd.read_csv(
        r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\Hot_month_PSO\iter2\South_Bay\PINN_input_20250625_004621_SHIFTED.csv",
        parse_dates=["Date"],         # ✅ Column that actually exists
        index_col="Date"              # ✅ Use that column as index
    )

    sim = sim.interpolate(method='time', limit_direction='both')

    # ✅ Ensure T_membrane, T_metal_deck, T_zone_air, HVAC are loaded
    sim = sim[['T_membrane', 'T_metal_deck', 'T_zone_air', 'HVAC']]

    return sim.loc["2025-05-11":"2025-05-21"]




if __name__ == "__main__":
    bay = "South_Bay"  # or "South_Bay"

    OUTPUT_DIR = os.path.join(r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\Hot_month_PINN\iter5", bay, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rc_params = {
        'R_insulation': 0.00422618,
        'C_insulation': 4586849.05,
        'C_membrane': 4821496.13,
        'C_metal_deck': 32148893,
        'time_shift': -1.33329142,
        'solar_absorptance': 0.25445411,
        'K_p': 500,
        'K_i': 201.6991987
    }

    sim_df = load_simulation_data()
    sim_df = sim_df.interpolate(method='time')



    exp_df = load_experimental_data(bay)
    weather_df = load_weather_data()
    print(f"✅ Weather data range: {weather_df.index.min()} to {weather_df.index.max()}")

    # ✅ STEP 1: Merge experimental columns into simulation DataFrame
    # Ensure datetime index is sorted and unique
    sim_df = sim_df[~sim_df.index.duplicated(keep='first')].sort_index()
    exp_df = exp_df[~exp_df.index.duplicated(keep='first')].sort_index()

    # Now safely concatenate
    combined_df = pd.concat([sim_df, exp_df[['Roof', 'Ceiling', 'MidLevel']]], axis=1)

    # ✅ STEP 2: Debug sanity checks
    print("✅ Sample Combined Data:")
    print(combined_df[['T_membrane', 'T_metal_deck', 'Roof', 'Ceiling']].head(5))
    print("❓ NaNs after merge:")
    print(combined_df[['T_membrane', 'T_metal_deck', 'Roof', 'Ceiling']].isna().sum())

    # ✅ STEP 3: Drop rows with missing targets
    combined_df = combined_df.dropna(subset=['Roof', 'Ceiling'])
    print("✅ Non-NaN Roof rows:", combined_df['Roof'].notna().sum())
    print("✅ After dropping NaNs, shape:", combined_df.shape)
    print("✅ Combined data time range:", combined_df.index.min(), "→", combined_df.index.max())

    # ✅ STEP 4: Sequence generation
    splitter = TemporalSequenceSplitter(combined_df, seq_length_days=2, stride_days=1)
    raw_sequences = splitter.get_sequences()
    print(f"🧩 Total sequences generated: {len(raw_sequences)}")

    # ✅ STEP 5: Filter valid sequences (no errors and fixed length)
    valid_sequences = []
    for idx, seq in enumerate(raw_sequences):
        try:
            _ = ThermalDataset([seq], weather_df, rc_params)[0]  # test sample
            valid_sequences.append(seq)
        except Exception as e:
            print(f"⚠️ Skipping sequence {idx} due to error: {e}")

    if len(valid_sequences) == 0:
        raise ValueError("❌ No valid sequences found. Check alignment or missing columns.")

    print(f"✅ Valid sequences after initial filtering: {len(valid_sequences)}")

    print("\n📏 Sequence lengths before filtering:")
    for i, seq in enumerate(valid_sequences):
        print(f"Sequence {i}: length = {len(seq)}")

    TARGET_LENGTH = len(valid_sequences[0])
    print(f"✅ Using dynamic TARGET_LENGTH = {TARGET_LENGTH}")


    # ✅ Apply train/test split by month (for now all in October)
    train_seqs, test_seqs = stratified_temporal_split_by_months(
        valid_sequences,
        train_months=[  2,3,5],  # October and November for training
        test_months=[4,6]  # December for testing
    )

    # ⏱️ Use small number of sequences for debugging
    #train_seqs = train_seqs[:2]
    #test_seqs = test_seqs[:1]

    print(f"✅ Train sequences: {len(train_seqs)}")
    print(f"✅ Test sequences: {len(test_seqs)}")

    # Build datasets
    train_dataset = ThermalDataset(train_seqs, weather_df, rc_params)
    test_dataset = ThermalDataset(test_seqs, weather_df, rc_params)

    feature_mean = train_dataset.feature_mean
    feature_std = train_dataset.feature_std

    # Build model
    pinn = ThermalPINN(rc_params, feature_mean, feature_std)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ✅ Build training dataset and model
    train_dataset = ThermalDataset(train_seqs, weather_df, rc_params)
    test_dataset = ThermalDataset(test_seqs, weather_df, rc_params)

    feature_mean = train_dataset.feature_mean
    feature_std = train_dataset.feature_std
    pinn = ThermalPINN(rc_params, feature_mean, feature_std)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train model
    trained_pinn = train_pinn(pinn, train_loader, epochs=300)

    sample_batch = next(iter(train_loader))
    features = sample_batch['features']
    baseline = sample_batch['baseline']
    targets = sample_batch['target']
    timestamps = sample_batch['timestamps']

    with torch.no_grad():
        preds = trained_pinn(features, baseline)
        delta_true = targets - baseline
        delta_pred = preds - baseline

    plt.figure(figsize=(10, 5))

    plt.plot(delta_true[0, :, 0].detach().numpy(),
             label='True Roof Δ Seq 1', linewidth=2)
    plt.plot(delta_true[1, :, 0].detach().numpy(),
             label='True Roof Δ Seq 2', linewidth=2)

    plt.plot(delta_pred[0, :, 0].detach().numpy(),
             '--', label='Pred Roof Δ Seq 1', linewidth=2)
    plt.plot(delta_pred[1, :, 0].detach().numpy(),
             '--', label='Pred Roof Δ Seq 2', linewidth=2)

    plt.legend()
    plt.title("PINN Residuals - Roof Δ (Sequence 1 & 2)")
    plt.xlabel("Timestep (Within Sequence)")
    plt.ylabel("Δ (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roof_delta_debug.png"))
    plt.show()

    # ✅ Apply PINN correction
    corrected_sim, corrected_df = apply_pinn_correction(combined_df, weather_df, trained_pinn, rc_params)

    # ✅ Define native simulation time step (same as PSO)
    DT_SECONDS = 300

    # ✅ Define base index using uniformly spaced time points
    base_index = exp_df['Roof'].resample(f"{DT_SECONDS}s").mean().index

    # ✅ Drop duplicates before reindexing to avoid ValueError
    exp_df = exp_df[~exp_df.index.duplicated(keep='first')]
    sim_df = sim_df[~sim_df.index.duplicated(keep='first')]
    corrected_sim = corrected_sim[~corrected_sim.index.duplicated(keep='first')]

    # ✅ Reindex all time series
    exp_resampled = exp_df['Roof'].reindex(base_index).interpolate(limit=6)
    sim_f = sim_df['T_membrane'].reindex(base_index).interpolate(limit=6)
    pinn_f = corrected_sim['T_membrane_corrected'].reindex(base_index).interpolate(limit=6)

    # ✅ Day/Night masks
    ghi_resampled = weather_df['GHI'].reindex(exp_resampled.index).interpolate(limit=6)
    is_day = ghi_resampled >= 10
    is_night = ~is_day

    # ✅ MAE / RMSE calculations
    mask_sim = ~np.isnan(exp_resampled) & ~np.isnan(sim_f)
    mask_pinn = ~np.isnan(exp_resampled) & ~np.isnan(pinn_f)

    mae_sim = mean_absolute_error(exp_resampled[mask_sim], sim_f[mask_sim])
    mae_pinn = mean_absolute_error(exp_resampled[mask_pinn], pinn_f[mask_pinn])
    rmse_sim = mean_squared_error(exp_resampled[mask_sim], sim_f[mask_sim]) ** 0.5
    rmse_pinn = mean_squared_error(exp_resampled[mask_pinn], pinn_f[mask_pinn]) ** 0.5

    mae_day_sim = mean_absolute_error(exp_resampled[is_day & mask_sim], sim_f[is_day & mask_sim])
    mae_night_sim = mean_absolute_error(exp_resampled[is_night & mask_sim], sim_f[is_night & mask_sim])
    rmse_day_sim = mean_squared_error(exp_resampled[is_day & mask_sim], sim_f[is_day & mask_sim]) ** 0.5
    rmse_night_sim = mean_squared_error(exp_resampled[is_night & mask_sim], sim_f[is_night & mask_sim]) ** 0.5

    mae_day = mean_absolute_error(exp_resampled[is_day & mask_pinn], pinn_f[is_day & mask_pinn])
    mae_night = mean_absolute_error(exp_resampled[is_night & mask_pinn], pinn_f[is_night & mask_pinn])
    rmse_day = mean_squared_error(exp_resampled[is_day & mask_pinn], pinn_f[is_day & mask_pinn]) ** 0.5
    rmse_night = mean_squared_error(exp_resampled[is_night & mask_pinn], pinn_f[is_night & mask_pinn]) ** 0.5

    # ✅ Plot comparison: Roof
    plt.figure(figsize=(14, 6))
    plt.plot(exp_resampled, label=f'Experimental ({bay})', linewidth=1.8)
    plt.plot(sim_f, label=f'Simulation ({bay})', alpha=0.7)
    plt.plot(pinn_f, '--', label=f'PINN Corrected ({bay})', linewidth=1.5)

    plt.title(
        f"{bay}: Sim vs Exp vs PINN (Roof)\n"
        f"MAE(Sim): {mae_sim:.2f} °C | MAE(PINN): {mae_pinn:.2f} °C\n"
        f"MAE-Day: {mae_day_sim:.2f} → {mae_day:.2f} °C | MAE-Night: {mae_night_sim:.2f} → {mae_night:.2f} °C\n"
        f"RMSE-Day: {rmse_day_sim:.2f} → {rmse_day:.2f} °C | RMSE-Night: {rmse_night_sim:.2f} → {rmse_night:.2f} °C"
    )

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'pinn_comparison_all_{bay}.png'))
    plt.show()

    # === ✅ Ceiling Evaluation ===
    ceiling_exp = exp_df['Ceiling'].resample(f"{DT_SECONDS}s").mean().interpolate(limit=6)
    ceiling_sim = sim_df['T_metal_deck'].reindex(ceiling_exp.index).interpolate(limit=6)
    ceiling_pinn = corrected_sim['T_metal_deck_corrected'].reindex(ceiling_exp.index).interpolate(limit=6)

    mask_ceil_sim = ~np.isnan(ceiling_exp) & ~np.isnan(ceiling_sim)
    mask_ceil_pinn = ~np.isnan(ceiling_exp) & ~np.isnan(ceiling_pinn)

    mae_ceil_sim = mean_absolute_error(ceiling_exp[mask_ceil_sim], ceiling_sim[mask_ceil_sim])
    mae_ceil_pinn = mean_absolute_error(ceiling_exp[mask_ceil_pinn], ceiling_pinn[mask_ceil_pinn])

    plt.figure(figsize=(12, 5))
    plt.plot(ceiling_exp, label="Exp Ceiling (°C)")
    plt.plot(ceiling_sim, label="Sim Ceiling (°C)")
    plt.plot(ceiling_pinn, "--", label="PINN Corrected Ceiling (°C)")

    plt.title(
        f"Ceiling Correction\n"
        f"MAE Sim: {mae_ceil_sim:.2f} °C | MAE PINN: {mae_ceil_pinn:.2f} °C"
    )

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ceiling_zone_{bay}.png"))
    plt.show()

    # === ✅ ΔT (ROOF - CEILING) Extended Metrics and Plot ===
    deltaT_index = exp_resampled.index
    roof_exp = exp_df['Roof'].reindex(deltaT_index).interpolate(limit=6)
    ceiling_exp_aligned = exp_df['Ceiling'].reindex(deltaT_index).interpolate(limit=6)
    roof_sim_aligned = sim_f  # already reindexed
    roof_pinn = pinn_f
    ceiling_sim_aligned = sim_df['T_metal_deck'].reindex(deltaT_index).interpolate(limit=6)
    ceiling_pinn_aligned = corrected_sim['T_metal_deck_corrected'].reindex(deltaT_index).interpolate(limit=6)

    # --- Delta T computation
    deltaT_exp = roof_exp - ceiling_exp_aligned
    deltaT_sim = roof_sim_aligned - ceiling_sim_aligned
    deltaT_pinn = roof_pinn - ceiling_pinn_aligned

    # --- Valid masks
    mask_dT_sim = ~np.isnan(deltaT_exp) & ~np.isnan(deltaT_sim)
    mask_dT_pinn = ~np.isnan(deltaT_exp) & ~np.isnan(deltaT_pinn)
    mask_day = is_day & mask_dT_sim & mask_dT_pinn
    mask_night = is_night & mask_dT_sim & mask_dT_pinn

    # --- Error metrics
    mae_dT_sim = mean_absolute_error(deltaT_exp[mask_dT_sim], deltaT_sim[mask_dT_sim])
    rmse_dT_sim = mean_squared_error(deltaT_exp[mask_dT_sim], deltaT_sim[mask_dT_sim]) ** 0.5
    mae_dT_pinn = mean_absolute_error(deltaT_exp[mask_dT_pinn], deltaT_pinn[mask_dT_pinn])
    rmse_dT_pinn = mean_squared_error(deltaT_exp[mask_dT_pinn], deltaT_pinn[mask_dT_pinn]) ** 0.5

    # Day/Night metrics
    mae_day_sim_dT = mean_absolute_error(deltaT_exp[mask_day], deltaT_sim[mask_day])
    mae_day_pinn_dT = mean_absolute_error(deltaT_exp[mask_day], deltaT_pinn[mask_day])
    rmse_day_sim_dT = mean_squared_error(deltaT_exp[mask_day], deltaT_sim[mask_day]) ** 0.5
    rmse_day_pinn_dT = mean_squared_error(deltaT_exp[mask_day], deltaT_pinn[mask_day]) ** 0.5

    mae_night_sim_dT = mean_absolute_error(deltaT_exp[mask_night], deltaT_sim[mask_night])
    mae_night_pinn_dT = mean_absolute_error(deltaT_exp[mask_night], deltaT_pinn[mask_night])
    rmse_night_sim_dT = mean_squared_error(deltaT_exp[mask_night], deltaT_sim[mask_night]) ** 0.5
    rmse_night_pinn_dT = mean_squared_error(deltaT_exp[mask_night], deltaT_pinn[mask_night]) ** 0.5

    # --- Plot
    plt.figure(figsize=(12, 5))
    plt.plot(deltaT_exp, label="ΔT Experimental (°C)", linewidth=1.8)
    plt.plot(deltaT_sim, label="ΔT Sim (°C)", alpha=0.8)
    plt.plot(deltaT_pinn, '--', label="ΔT PINN Corrected (°C)", linewidth=1.5)

    plt.title(
        f"Delta T (Roof - Ceiling)\n"
        f"MAE(Sim): {mae_dT_sim:.2f}°C | MAE(PINN): {mae_dT_pinn:.2f}°C\n"
        f"MAE-Day: {mae_day_sim_dT:.2f} → {mae_day_pinn_dT:.2f}°C | MAE-Night: {mae_night_sim_dT:.2f} → {mae_night_pinn_dT:.2f}°C\n"
        f"RMSE-Day: {rmse_day_sim_dT:.2f} → {rmse_day_pinn_dT:.2f}°C | RMSE-Night: {rmse_night_sim_dT:.2f} → {rmse_night_pinn_dT:.2f}°C"
    )

    plt.xlabel("Date")
    plt.ylabel("ΔT (°C)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"deltaT_roof_minus_ceiling_{bay}.png"))
    plt.show()

    # ✅ Export final corrected simulation
    corrected_sim.to_csv(os.path.join(OUTPUT_DIR, "corrected_simulation.csv"))
