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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXPERIMENTAL_DATA_DIR = r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\RC_Modelling\8"


# -------------------------------
# ⚙️ Black-box toggle flag
# -------------------------------
use_blackbox = True  # ➡️ Set to False for physics-based PINN


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

        # Time feature (sin/cos)
        time_shift_hr = self.rc_params.get('time_shift', 0)
        t = (seq_data.index + pd.Timedelta(hours=time_shift_hr) - seq_data.index[0]).total_seconds().values

        # Match weather data to time range
        seq_start = seq_data.index.min()
        seq_end = seq_data.index.max()
        weather_mask = (self.weather_data.index >= seq_start) & (self.weather_data.index <= seq_end)
        seq_weather = self.weather_data[weather_mask]

        features, targets, ts_list = [], [], []

        for i in range(min(len(seq_data), len(seq_weather))):
            try:
                hour = t[i] / 3600 % 24
                time_feat = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]

                # Weather inputs
                weather_raw = seq_weather.iloc[i][['DBT', 'GHI', 'WS', 'WD', 'RH']].values
                weather_raw = np.nan_to_num(weather_raw, nan=0.0, posinf=0.0, neginf=0.0)
                weather = (weather_raw - self.feature_mean) / self.feature_std

                # Target: Experimental surface temperatures
                target_vals = seq_data[['Roof', 'Ceiling']].iloc[i].values
                if np.isnan(target_vals).any():
                    continue

                input_vector = np.concatenate([time_feat, weather])
                if np.isnan(input_vector).any():
                    continue

                features.append(input_vector)
                targets.append(target_vals)
                ts_list.append(seq_data.index[i])

            except Exception as e:
                print(f"⚠️ Skipping step {i} in seq {idx} due to error: {e}")
                continue

        if len(features) == 0:
            raise ValueError(f"❌ Sequence {idx} has no usable data after filtering.")

        return {
            'features': torch.FloatTensor(np.array(features)),
            'target': torch.FloatTensor(np.array(targets)),
            'timestamps': ts_list if self.return_datetime else torch.tensor(
                pd.Series(ts_list).astype(np.int64).values // 1e9, dtype=torch.float32)
        }


# 🔳 BLACK-BOX MODEL (No physics)
class BlackBoxNN(nn.Module):
    def __init__(self, feature_mean, feature_std):
        super().__init__()
        self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32)
        self.feature_std = torch.tensor(feature_std, dtype=torch.float32)

        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, dataloader, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            features = torch.nan_to_num(features, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)

            optimizer.zero_grad()
            preds = model(features)
            loss = nn.MSELoss()(preds, targets)

            if torch.isnan(loss):
                print("❌ Skipping batch due to NaN loss.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f}")

    return model


# 🔁 Black-box training function (MSE only)
def train_blackbox(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for batch in dataloader:
        features = batch['features'].to(device)
        targets = batch['target'].to(device)

        preds = model(features)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)



from collections import defaultdict





# Select which bay to optimize
bay = "Middle_Bay"  # Change to "South_Bay" as needed

# ===================== CONFIGURATION =====================
SIMULATION_START = "2025-02-27"
SIMULATION_END = "2025-06-04"

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

    return sim.loc["2025-02-27":"2025-06-04"]


def apply_blackbox_prediction(sim_df, weather_df, model, rc_params):
    """Sequence-aware prediction for black-box model"""
    splitter = TemporalSequenceSplitter(sim_df, seq_length_days=2, stride_days=1)
    sequences = splitter.get_sequences()

    preds_by_timestamp = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            if seq.empty:
                continue

            dataset = ThermalDataset([seq], weather_df, rc_params, return_datetime=True)
            try:
                sample = dataset[0]
            except Exception as e:
                print(f"⚠️ Skipping sequence {i}: {e}")
                continue

            features = sample['features']
            timestamps = sample['timestamps']

            preds = model(features)
            if preds.dim() == 3:
                preds = preds.squeeze(0)

            for j, ts in enumerate(timestamps):
                preds_by_timestamp[ts].append(preds[j].cpu().numpy())

    # Average overlapping predictions
    all_ts = sorted(preds_by_timestamp.keys())
    T_roof_predicted = []
    T_ceiling_predicted = []

    for ts in all_ts:
        values = np.array(preds_by_timestamp[ts])
        T_roof_predicted.append(values[:, 0].mean())
        T_ceiling_predicted.append(values[:, 1].mean())

    predicted_df = pd.DataFrame({
        "T_membrane_predicted": T_roof_predicted,
        "T_metal_deck_predicted": T_ceiling_predicted
    }, index=pd.to_datetime(all_ts))

    predicted_df = predicted_df[~predicted_df.index.duplicated(keep='first')].sort_index()

    # Merge with simulation data
    sim_df = sim_df.copy()
    predicted_sim_df = sim_df.join(predicted_df, how='left')

    # Interpolate short gaps
    predicted_sim_df['T_membrane_predicted'] = predicted_sim_df['T_membrane_predicted'].interpolate(limit=6)
    predicted_sim_df['T_metal_deck_predicted'] = predicted_sim_df['T_metal_deck_predicted'].interpolate(limit=6)

    return predicted_sim_df, predicted_df


if __name__ == "__main__":
    bay = "South_Bay"  # or "South_Bay"

    OUTPUT_DIR = os.path.join(r"E:\Research_Assistant\Tarun\Analysis_Results\Tarun\Analysis_Results\Staged_workflow\Blackbox_vs_PINN\iter1", bay, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rc_params = {

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


    print(f"✅ Train sequences: {len(train_seqs)}")
    print(f"✅ Test sequences: {len(test_seqs)}")

    # Build datasets
    train_dataset = ThermalDataset(train_seqs, weather_df, rc_params)
    test_dataset = ThermalDataset(test_seqs, weather_df, rc_params)

    feature_mean = train_dataset.feature_mean
    feature_std = train_dataset.feature_std


    # Build model
    if use_blackbox:
        model = BlackBoxNN(feature_mean, feature_std)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ✅ Build training dataset and model
    train_dataset = ThermalDataset(train_seqs, weather_df, rc_params)
    test_dataset = ThermalDataset(test_seqs, weather_df, rc_params)

    feature_mean = train_dataset.feature_mean
    feature_std = train_dataset.feature_std
    # Set toggle
    use_blackbox = True  # 🔁 Change to False for PINN

    # Build model
    if use_blackbox:
        model = BlackBoxNN(feature_mean, feature_std).to(device)


    # Load data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train model
    trained_model = train_model(model, train_loader, epochs=300)

    sample_batch = next(iter(train_loader))
    features = sample_batch['features']
    baseline = sample_batch['baseline']
    targets = sample_batch['target']
    timestamps = sample_batch['timestamps']

    with torch.no_grad():
        if use_blackbox:
            preds = trained_model(features)
            delta_true = targets  # no baseline subtraction
            delta_pred = preds
        else:
            preds = trained_model(features, baseline)
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
    plt.title("Residuals - Roof Δ (Black-box or PINN, Seq 1 & 2)")
    plt.xlabel("Timestep (Within Sequence)")
    plt.ylabel("Δ (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roof_delta_debug.png"))
    plt.show()

    # ✅ Apply PINN correction
    if use_blackbox:
        corrected_sim, corrected_df = apply_blackbox_prediction(
            combined_df, weather_df, trained_model, rc_params
        )


    # ✅ Define native simulation time step (same as PSO)
    DT_SECONDS = 300

    # ✅ Define base index using uniformly spaced time points
    base_index = exp_df['Roof'].resample(f"{DT_SECONDS}s").mean().index

    # ✅ Drop duplicates before reindexing to avoid ValueError
    exp_df = exp_df[~exp_df.index.duplicated(keep='first')]
    sim_df = sim_df[~sim_df.index.duplicated(keep='first')]

    # === ✅ Generalized Plotting for PINN or Black Box ===
    # === ✅ Unified Plotting: PINN vs Black-box ===
    if corrected_sim is not None:
        corrected_sim = corrected_sim[~corrected_sim.index.duplicated(keep='first')]

        exp_resampled = exp_df['Roof'].reindex(base_index).interpolate(limit=6)
        ghi_resampled = weather_df['GHI'].reindex(exp_resampled.index).interpolate(limit=6)
        is_day = ghi_resampled >= 10
        is_night = ~is_day

        if use_blackbox:
            pred_f = corrected_sim['T_membrane_predicted'].reindex(base_index).interpolate(limit=6)
            label = 'Black-box Predicted'
        else:
            pred_f = corrected_sim['T_membrane_corrected'].reindex(base_index).interpolate(limit=6)
            label = 'PINN Corrected'

        # === MAE/RMSE Metrics ===
        mask_pred = ~np.isnan(exp_resampled) & ~np.isnan(pred_f)
        mae = mean_absolute_error(exp_resampled[mask_pred], pred_f[mask_pred])
        rmse = mean_squared_error(exp_resampled[mask_pred], pred_f[mask_pred]) ** 0.5
        mae_day = mean_absolute_error(exp_resampled[is_day & mask_pred], pred_f[is_day & mask_pred])
        mae_night = mean_absolute_error(exp_resampled[is_night & mask_pred], pred_f[is_night & mask_pred])
        rmse_day = mean_squared_error(exp_resampled[is_day & mask_pred], pred_f[is_day & mask_pred]) ** 0.5
        rmse_night = mean_squared_error(exp_resampled[is_night & mask_pred], pred_f[is_night & mask_pred]) ** 0.5

        # ✅ Plot Roof
        plt.figure(figsize=(14, 6))
        plt.plot(exp_resampled, label=f'Experimental ({bay})', linewidth=1.8)
        plt.plot(pred_f, '--', label=f'{label} ({bay})', linewidth=1.5)
        plt.title(
            f"{bay}: Roof Comparison\n"
            f"MAE: {mae:.2f} °C | MAE-Day: {mae_day:.2f} | MAE-Night: {mae_night:.2f}\n"
            f"RMSE-Day: {rmse_day:.2f} | RMSE-Night: {rmse_night:.2f}"
        )
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'roof_comparison_{bay}.png'))
        plt.show()

        # === ✅ Ceiling
        ceiling_exp = exp_df['Ceiling'].resample(f"{DT_SECONDS}s").mean().interpolate(limit=6)
        if use_blackbox:
            ceiling_pred = corrected_sim['T_metal_deck_predicted'].reindex(ceiling_exp.index).interpolate(limit=6)
        else:
            ceiling_pred = corrected_sim['T_metal_deck_corrected'].reindex(ceiling_exp.index).interpolate(limit=6)

        mask_ceil = ~np.isnan(ceiling_exp) & ~np.isnan(ceiling_pred)
        mae_ceil = mean_absolute_error(ceiling_exp[mask_ceil], ceiling_pred[mask_ceil])
        rmse_ceil = mean_squared_error(ceiling_exp[mask_ceil], ceiling_pred[mask_ceil]) ** 0.5

        plt.figure(figsize=(12, 5))
        plt.plot(ceiling_exp, label="Exp Ceiling (°C)")
        plt.plot(ceiling_pred, "--", label=f"{label} Ceiling (°C)")
        plt.title(f"Ceiling Comparison\nMAE: {mae_ceil:.2f} °C | RMSE: {rmse_ceil:.2f} °C")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"ceiling_comparison_{bay}.png"))
        plt.show()

        # === ✅ Delta T (Roof - Ceiling)
        ceiling_aligned = exp_df['Ceiling'].reindex(base_index).interpolate(limit=6)
        roof_pred = pred_f

        # ✅ Use appropriate ceiling prediction column
        if use_blackbox:
            ceiling_pred_aligned = corrected_sim['T_metal_deck_predicted'].reindex(base_index).interpolate(limit=6)
        else:
            ceiling_pred_aligned = corrected_sim['T_metal_deck_corrected'].reindex(base_index).interpolate(limit=6)

        deltaT_exp = exp_resampled - ceiling_aligned
        deltaT_pred = roof_pred - ceiling_pred_aligned

        mask_dT = ~np.isnan(deltaT_exp) & ~np.isnan(deltaT_pred)
        mask_day_dT = is_day & mask_dT
        mask_night_dT = is_night & mask_dT

        mae_dT = mean_absolute_error(deltaT_exp[mask_dT], deltaT_pred[mask_dT])
        rmse_dT = mean_squared_error(deltaT_exp[mask_dT], deltaT_pred[mask_dT]) ** 0.5
        mae_day_dT = mean_absolute_error(deltaT_exp[mask_day_dT], deltaT_pred[mask_day_dT])
        mae_night_dT = mean_absolute_error(deltaT_exp[mask_night_dT], deltaT_pred[mask_night_dT])
        rmse_day_dT = mean_squared_error(deltaT_exp[mask_day_dT], deltaT_pred[mask_day_dT]) ** 0.5
        rmse_night_dT = mean_squared_error(deltaT_exp[mask_night_dT], deltaT_pred[mask_night_dT]) ** 0.5

        plt.figure(figsize=(12, 5))
        plt.plot(deltaT_exp, label="ΔT Experimental (°C)", linewidth=1.8)
        plt.plot(deltaT_pred, '--', label=f"ΔT {label} (°C)", linewidth=1.5)
        plt.title(
            f"ΔT (Roof - Ceiling): {label}\n"
            f"MAE: {mae_dT:.2f} | Day: {mae_day_dT:.2f}, Night: {mae_night_dT:.2f} | "
            f"RMSE-Day: {rmse_day_dT:.2f}, RMSE-Night: {rmse_night_dT:.2f}"
        )
        plt.xlabel("Date")
        plt.ylabel("ΔT (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"deltaT_comparison_{bay}.png"))
        plt.show()

        # ✅ Save corrected or predicted simulation
        corrected_sim.to_csv(os.path.join(OUTPUT_DIR, "corrected_simulation.csv"))
