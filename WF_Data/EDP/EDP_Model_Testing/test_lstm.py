import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import pandas as pd
import argparse

import torch
torch.backends.mkldnn.enabled = False
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm


class WindTurbineDataset(Dataset):
    def __init__(self, X_path=None, y_path=None, timestamps_path=None, X=None, y=None, timestamps=None):
        self.X = torch.load(X_path).float() if X is None else X  # (num_samples, seq_len, num_features)
        self.y = torch.load(y_path).float() if y is None else y  # (num_samples, 1)
        self.timestamps = self._load_pickle(timestamps_path) if timestamps is None else timestamps  # (num_samples,)

        valid_samples = ~torch.isnan(self.y).squeeze()
        self.X, self.y, self.timestamps = self.X[valid_samples], self.y[valid_samples], self.timestamps[valid_samples]

        self.mask = (~torch.isnan(self.X).any(dim=2)).float()
        self.X = torch.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx], self.timestamps[idx].timestamp()

    @staticmethod
    def _load_pickle(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


class MaskedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_dropout=False, dropout_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob) if use_dropout else nn.Identity()
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()  # Batch (batch_size, seq_length, input_size)
        h = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        # Mask for which step to ignore
        mask = mask.float()  
        
        # Iterate over time steps in sequence
        for t in range(seq_len):
            # Select the feature vector for the current time step
            x_t = x[:, t, :].unsqueeze(1)
            # Extracts the mask and reshapes for elementwise operations
            mask_t = mask[:, t].view(1, -1, 1)

            # Pass through LSTM and update h, c states
            _, (h_new, c_new) = self.lstm(x_t, (h, c))
            h = mask_t * h_new + (1 - mask_t) * h
            c = mask_t * c_new + (1 - mask_t) * c

        return torch.relu(self.fc(self.dropout(h.squeeze(0))))


def train_model(model_name, model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10):
    model.to(device)
    
    best_val_loss = float("inf")
    best_model_state = None

    def get_base_dataset(loader):
        dataset = loader.dataset
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset

    features_size = get_base_dataset(train_loader).X.size(-1)
    feature_weights_log = {col: [] for col in range(features_size)}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x, mask, y, ts in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)  # Move tensors to device that are used in training

            optimizer.zero_grad()
            outputs = model(x, mask)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, mask, y, ts in val_loader:  # Add ts here
                x, mask, y = x.to(device), mask.to(device), y.to(device)  # Move tensors to device
                # ts is not used in validation, so we don't need to move it to the device
                outputs = model(x, mask)
                loss = loss_fn(outputs, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Log feature importance each epoch
        with torch.no_grad():
            lstm_weights = model.lstm.weight_ih_l0.cpu().numpy()
            for i in range(features_size):
                feature_weights_log[i].append(lstm_weights[:, i].mean())

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    plot_feature_importance(feature_weights_log)

    # will overwrite model there
    torch.save(best_model_state, f"/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/EDP_Model_Testing/models/{model_name}.pth")


def inference(model, data_loader, device, threshold=None, data_loader_name="WARNING: pass in data loader name"):
    model.eval()
    predictions, actuals, anomalies, timestamps = [], [], [], []
    
    with torch.no_grad():
        for x, mask, y, ts in tqdm(data_loader, desc=f"Running Inference of {data_loader_name}", unit="batch"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            
            outputs = model(x, mask)
            predictions.append(outputs.cpu())
            actuals.append(y.cpu())
            timestamps.append(ts)

            if threshold is not None:
                anomalies.extend(torch.abs(outputs.cpu() - y.cpu()) > threshold)
        
    return torch.cat(predictions), torch.cat(actuals), torch.cat(anomalies) if threshold is not None else anomalies, torch.cat(timestamps)
    

def plot_predictions_distribution(preds, true_values=None, bins=50):
    plt.figure(figsize=(8, 5))
    
    # Plot histogram
    sns.histplot(preds.numpy(), bins=bins, kde=True, color="blue", label="Predictions")
    
    # Optionally overlay true values
    if true_values is not None:
        sns.histplot(true_values.numpy(), bins=bins, kde=True, color="red", label="True Values", alpha=0.6)

    plt.xlabel("Predicted Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predictions")
    plt.legend()
    plt.show()


def plot_feature_importance(
    feature_weights_log, 
    column_names = [
    "Wind_speed", "Wind_speed_std", "Wind_rel_dir", "Amb_temp",
    "Gen_speed", "Gen_speed_std", "Rotor_speed", "Rotor_speed_std",
    "Blade_pitch", "Blade_pitch_std", "Gen_phase_temp_1", "Gen_phase_temp_2",
    "Gen_phase_temp_3", "Transf_temp_p1", "Transf_temp_p2", "Transf_temp_p3",
    "Gen_bearing_temp_1", "Gen_bearing_temp_2", "Hyd_oil_temp", "Gear_oil_temp",
    "Gear_bear_temp", "Nacelle_position"
    ]
):
    plt.figure(figsize=(12, 8))

    for feature_idx, weights in feature_weights_log.items():
        feature_name = column_names[feature_idx]  # Map index to feature name
        plt.plot(range(len(weights)), weights, label=feature_name)

    plt.xlabel('Index')
    plt.ylabel('Average Weight')
    plt.title('Feature Importance Over Time')
    plt.legend()
    plt.show()


def get_data_loaders(dataset, train_split=0.7, val_split=0.1, batch_size=32):
    total_size = len(dataset)
    test_size = int((1-train_split-val_split)*total_size)
    train_val_size = total_size-test_size

    test_indices = list(range(train_val_size, total_size))
    test_dataset = Subset(dataset, test_indices)

    train_val_subset = Subset(dataset, list(range(0, train_val_size)))

    train_size = int(train_split / (train_split + val_split) * train_val_size)
    val_size = train_val_size - train_size
    train_dataset, val_dataset = random_split(train_val_subset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # will retain the timestamps associated with data samples
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_data_loaders_by_timestamp(dataset, split_timestamp, val_split=0.15, batch_size=32):
    if isinstance(split_timestamp, str):
        split_timestamp = pd.Timestamp(split_timestamp)
        
    indices = list(range(len(dataset)))
    timestamps = dataset.timestamps

    train_val_indices = [i for i, t in zip(indices, timestamps) if t < split_timestamp]
    test_indices = [i for i, t in zip(indices, timestamps) if t >= split_timestamp]

    test_dataset = Subset(dataset, test_indices)
    train_val_dataset = Subset(dataset, train_val_indices)

    train_size = int((1 - val_split) * len(train_val_indices))
    val_size = len(train_val_indices) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def permutation_importance(model, test_loader, loss_fn, device, baseline_mse=None, input_size = 22):
    feature_importance = np.zeros(input_size)
    permutation_results = {}  # To store predictions and actuals for each permutation
    
    # Run inference to calculate baseline MSE if not provided
    if baseline_mse is None:
        predictions, actuals, _, timestamps = inference(model, val_loader, device, threshold=None, data_loader_name = "permuation inference with test loader")
        baseline_mse = loss_fn(predictions, actuals).item()
        permutation_results["baseline"] = {
            "predictions": predictions,
            "actuals": actuals,
            "mse": baseline_mse
        }
    
    # Loop through each feature
    for i in range(input_size):
        permuted_mse_list = []
        permuted_predictions, permuted_actuals = [], []
        
        # Permute the feature across the entire dataset 
        for x, mask, y, ts in tqdm(test_loader, desc=f"Permuting feature {i}", unit="batch"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            
            # Clone the input and permute the feature
            X_permuted = x.clone()
            idx = torch.randperm(x.shape[0])
            X_permuted[:, :, i] = x[idx, :, i]
            
            # Ensure the mask is also permuted if necessary
            mask_permuted = mask.clone()
            mask_permuted = mask_permuted[idx]
            
            # Get model predictions with permuted feature
            with torch.no_grad():
                outputs = model(X_permuted, mask_permuted).detach().cpu()
                permuted_mse = loss_fn(outputs, y.cpu()).item()
                permuted_mse_list.append(permuted_mse)
                
                # Store predictions and actuals on CPU
                permuted_predictions.append(outputs)
                permuted_actuals.append(y.cpu())
        
        # Compute the average MSE for this permuted feature
        avg_permuted_mse = np.mean(permuted_mse_list)
        feature_importance[i] = avg_permuted_mse - baseline_mse
        
        # Store results for this permutation
        permutation_results[f"feature_{i}"] = {
            "predictions": torch.cat(permuted_predictions),
            "actuals": torch.cat(permuted_actuals),
            "mse": avg_permuted_mse
        }

        torch.cuda.empty_cache()
        gc.collect()
        
    return feature_importance, permutation_results

def get_aggregate_data(data_folder_path): 
    print("Aggregating data ...")
    X_tensor = None
    y_tensor = None
    timestamps_array = []

    def load_pickle(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    for filename in sorted(os.listdir(data_folder_path)):
        print(filename)
        file_path = os.path.join(data_folder_path, filename)
        
        if filename.endswith("X.pt"):
            tensor = torch.load(file_path)
            X_tensor = tensor if X_tensor is None else torch.cat((X_tensor, tensor), dim=0)
            
        elif filename.endswith("y.pt"):
            tensor = torch.load(file_path)
            y_tensor = tensor if y_tensor is None else torch.cat((y_tensor, tensor), dim=0)
            
        elif filename.endswith("timestamp.pkl"):
            timestamps = load_pickle(file_path)
            timestamps = np.array(timestamps)  # Ensure it's a NumPy array
            timestamps_array = (
                timestamps if timestamps_array is None else np.concatenate((timestamps_array, timestamps))
            )

    # Convert timestamps to DataFrame if needed
    timestamp_df = pd.DataFrame(timestamps_array, columns=["Timestamp"])

    # Print summary
    print("X_tensor shape:", X_tensor.shape if X_tensor is not None else "None")
    print("y_tensor shape:", y_tensor.shape if y_tensor is not None else "None")
    print("Number of timestamps:", len(timestamps_array))

    return X_tensor, y_tensor, timestamps_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process wind turbine IDs.")
    parser.add_argument(
        "--WTs",
        nargs="+",  
        default=["01", "06", "07", "11", "aggregate"],  # Default to using all
        help="List of wind turbine IDs",
    )

    parser.add_argument(
        "--LOAD_MODEL",
        action="store_true",  # Sets it to True if provided, False otherwise
        help="Whether to load a pre-trained model"
    )
    
    parser.add_argument(
        "--DROPOUT",
        action="store_true", 
        help="Whether to include dropout in the model"
    )
    
    parser.add_argument(
        "--FAULTS_INCLUDED",
        action="store_true",  
        help="Whether to include faults in the model"
    )
    
    args = parser.parse_args()
    
    MODEL_FOLDER = "/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/EDP_Model_Testing/models"

    WTs = args.WTs
    LOAD_MODEL = args.LOAD_MODEL
    DROPOUT = args.DROPOUT
    FAULTS_INCLUDED = args.FAULTS_INCLUDED

    print(f"Using wind turbines: {WTs}")

    # Map Importance to Feature Names
    feature_names = [
        "Wind_speed", "Wind_speed_std", "Wind_rel_dir", "Amb_temp",
        "Gen_speed", "Gen_speed_std", "Rotor_speed", "Rotor_speed_std",
        "Blade_pitch", "Blade_pitch_std", "Gen_phase_temp_1", "Gen_phase_temp_2",
        "Gen_phase_temp_3", "Transf_temp_p1", "Transf_temp_p2", "Transf_temp_p3",
        "Gen_bearing_temp_1", "Gen_bearing_temp_2", "Hyd_oil_temp", "Gear_oil_temp",
        "Gear_bear_temp", "Nacelle_position"
    ]

    INPUT_SIZE = 22
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOSS_FN = nn.MSELoss()

    DATA_FOLDER = "/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/EDP_Model_Testing/data_prep"
    if FAULTS_INCLUDED: 
        DATA_FOLDER += "_faults_included"
        print(f"Using faults in data. DATA_FOLDER: {DATA_FOLDER}")

    for WT in WTs: 
        if LOAD_MODEL: print(f"Loading in WT model {WT}... \n")
        else: print(f"Training in WT model {WT} from scratch... \n")

        # make sure cuda is cleared
        torch.cuda.empty_cache()
        gc.collect()

        # get model name 
        MODEL_NAME = f"simple_lstm_WT_{WT}"
        CKPT = f"/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/EDP_Model_Testing/models/simple_lstm_WT_{WT}_deleting_later"

        if FAULTS_INCLUDED: 
            MODEL_NAME += "_faults_included"
            CKPT += "_faults_included"
        
        if DROPOUT: 
            MODEL_NAME += "_dropout"
            CKPT += "_dropout.pth"
        else: 
            CKPT += ".pth"
        
        model = MaskedLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, use_dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

        if WT == "aggregate": 
            X_tensor, y_tensor, timestamps_array = get_aggregate_data(DATA_FOLDER)
            full_dataset = WindTurbineDataset(
                X=X_tensor,
                y=y_tensor,
                timestamps=timestamps_array
            )         
        else: 
            full_dataset = WindTurbineDataset(
                X_path=f"{DATA_FOLDER}/EDP_WT_{WT}_X.pt",
                y_path=f"{DATA_FOLDER}/EDP_WT_{WT}_y.pt",
                timestamps_path=f"{DATA_FOLDER}/EDP_WT_{WT}_timestamp.pkl"
            )

        train_loader, val_loader, test_loader = get_data_loaders_by_timestamp(full_dataset, split_timestamp="2017-06-01")

        if LOAD_MODEL: 
            checkpoint = torch.load(CKPT, weights_only=True)
            model.load_state_dict(checkpoint)
            model.eval()
        else:
            train_model(MODEL_NAME, model, train_loader, val_loader, optimizer, LOSS_FN, DEVICE, NUM_EPOCHS)
                
        val_preds, val_true, _, val_timestamps = inference(model, val_loader, DEVICE, threshold=None, data_loader_name = "validation loader")
        threshold = np.quantile(val_preds-val_true, 0.98)
        print(threshold)
        
        preds, true, anomalies, timestamps = inference(model, test_loader, DEVICE, threshold=threshold, data_loader_name = "test loader")
        mse = LOSS_FN(preds, true)
        print(f"Final Test MSE: {mse.item():.4f}")

        # # Plot anomalies with timestamps
        # anomaly_timestamps = timestamps[anomalies]
        # plt.figure(figsize=(12, 6))
        # plt.plot(timestamps, true, label="True Values")
        # plt.plot(timestamps, preds, label="Predictions")
        # plt.scatter(anomaly_timestamps, true[anomalies], color='red', label="Anomalies")
        # plt.xlabel("Timestamp")
        # plt.ylabel("Value")
        # plt.title("Anomaly Detection with Timestamps")
        # plt.legend()
        # # plt.show()
        # plt.savefig(os.getcwd(), dpi=300, bbox_inches='tight')  # Save with high resolution

        # plot_predictions_distribution(preds, true_values=true)
        # diff = preds-true
        # plot_predictions_distribution(diff)
        
        # # Compute Permutation Importance    
        # feature_importance, permutation_results = permutation_importance(model, test_loader, LOSS_FN, DEVICE, baseline_mse=mse)

        # # Correct mapping of feature names to importance values
        # importance_dict = dict(zip(feature_names, feature_importance.tolist()))

        # print("Feature Importance:")
        # for feature, importance in sorted(importance_dict.items(), key=lambda x: -abs(x[1])):
        #     print(f"{feature}: {importance:.4f}")

        # print("-" * 50)
        # print("\n" * 5)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()