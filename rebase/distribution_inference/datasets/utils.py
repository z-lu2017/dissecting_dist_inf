from distribution_inference.utils import warning_string
import os, copy, re, pickle, warnings, requests, shutil, math, argparse, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch as ch
from xlsx2csv import Xlsx2csv
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, Subset

from distribution_inference.datasets import new_census, celeba, boneage,census, texas, arxiv, maadface, wind_turbines, lstm_wind_turbine

DATASET_INFO_MAPPING = {
    "new_census": new_census.DatasetInformation,
    "celeba": celeba.DatasetInformation,
    "boneage": boneage.DatasetInformation,
    "old_census": census.DatasetInformation,
    "texas": texas.DatasetInformation,
    "arxiv": arxiv.DatasetInformation,
    "maadface": maadface.DatasetInformation,
    "wind_turbines": wind_turbines.DatasetInformation,
    "lstm_wind_turbine": lstm_wind_turbine.DatasetInformation
}

DATASET_WRAPPER_MAPPING = {
    "new_census": new_census.CensusWrapper,
    "celeba": celeba.CelebaWrapper,
    "boneage": boneage.BoneWrapper,
    "old_census": census.CensusWrapper,
    "texas": texas.TexasWrapper,
    "arxiv": arxiv.ArxivWrapper,
    "maadface": maadface.MaadFaceWrapper,
    "wind_turbines": wind_turbines.WindTurbineWrapper,
    "lstm_wind_turbine": lstm_wind_turbine.LSTMWindTurbineWrapper,
}


def get_dataset_wrapper(dataset_name: str):
    wrapper = DATASET_WRAPPER_MAPPING.get(dataset_name, None)
    if not wrapper:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return wrapper


def get_dataset_information(dataset_name: str):
    info = DATASET_INFO_MAPPING.get(dataset_name, None)
    if not info:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return info


# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def filter(df, condition, ratio, verbose: bool = True, get_indices: bool = False):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            sampled_indices = np.concatenate((qualify, nqi))
            concat_df = pd.concat([df.iloc[qualify], df.iloc[nqi]])
            if get_indices:
                return concat_df, sampled_indices
            return concat_df
        
        if get_indices:
            return df.iloc[qualify], qualify
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            concat_df = pd.concat([df.iloc[qi], df.iloc[notqualify]])
            sampled_indices = np.concatenate((qi, notqualify))
            if get_indices:
                return concat_df, sampled_indices
            return concat_df

        if get_indices:
            return df.iloc[notqualify], notqualify
        return df.iloc[notqualify]


def heuristic(df, condition, ratio: float,
              cwise_sample: int,
              class_imbalance: float = 2.0,
              n_tries: int = 1000,
              tot_samples: int = None,
              class_col: str = "label",
              verbose: bool = True,
              get_indices: bool = False):
    if tot_samples is not None and class_imbalance is not None:
        raise ValueError("Cannot request class imbalance and total-sample based methods together")

    vals, pckds, indices = [], [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        # Binary class- simply sample (as requested)
        # From each class
        pckd_df, pckd_ids = filter(df, condition, ratio, verbose=False, get_indices=True)
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]
        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(
                    one_ids)[:cwise_sample]
            elif class_imbalance < 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(
                    one_ids)[:int(1 / class_imbalance * cwise_sample)]
            else:
                raise ValueError(f"Invalid class_imbalance value: {class_imbalance}")
            
            # Combine them together
            pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
            pckd_df = pckd_df.iloc[pckd]

        elif tot_samples is not None:
            # Combine both and randomly sample 'tot_samples' from them
            pckd = np.random.permutation(np.concatenate([zero_ids, one_ids]))[:tot_samples]
            pckd = np.sort(pckd)
            pckd_df = pckd_df.iloc[pckd]
        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)
        indices.append(pckd_ids[pckd])

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    picked_indices = indices[np.argmin(vals)]
    if get_indices:
        return picked_df.reset_index(drop=True), picked_indices
    return picked_df.reset_index(drop=True)


def multiclass_heuristic(
        df, condition, ratio: float,
        total_samples: int,
        class_ratio_maintain: bool,
        n_tries: int = 1000,
        class_col: str = "label",
        verbose: bool = True):
    """
        Heuristic for ratio-based sampling, implemented
        for the multi-class setting.
    """
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)

    class_labels, class_counts = np.unique(
        df[class_col].to_numpy(), return_counts=True)
    class_counts = class_counts / (1. * np.sum(class_counts))
    per_class_samples = class_counts * total_samples
    for _ in iterator:

        if class_ratio_maintain:
            # For each class
            inner_pckds = []
            for i, cid in enumerate(class_labels):
                # Find rows that have that specific class label
                df_i = df[df[class_col] == cid]
                pcked_df = filter(df_i, condition, ratio, verbose=False)
                # Randomly sample from this set
                # Since sampling is uniform at random, should preserve ratio
                # Either way- we pick a sample that is closest to desired ratio
                # So that aspect should be covered anyway

                if int(per_class_samples[i]) < 1:
                    raise ValueError(f"Not enough data to sample from class {cid}")
                if int(per_class_samples[i]) > len(pcked_df):
                    print(warning_string(
                        f"Requested {int(per_class_samples[i])} but only {len(pcked_df)} avaiable for class {cid}"))
                else:
                    pcked_df = pcked_df.sample(
                        int(per_class_samples[i]), replace=True)
                inner_pckds.append(pcked_df.reset_index(drop=True))
            # Concatenate all inner_pckds into one
            pckd_df = pd.concat(inner_pckds)
        else:
            pckd_df = filter(df, condition, ratio, verbose=False)

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)


def collect_data(loader, expect_extra: bool = True):
    X, Y = [], []
    for datum in loader:
        if expect_extra:
            x, y, _ = datum
        else:
            x, y = datum
        X.append(x)
        Y.append(y)
    # Concatenate both torch tensors across batches
    X = ch.cat(X, dim=0)
    Y = ch.cat(Y, dim=0)
    return X, Y


def aggregate_data(data_folder_path, WTs, data_split): 

    if WTs is None:
        print(f"Aggregating data for all available WTs at {data_folder_path}")
        WT_folders = [folder for folder in sorted(os.listdir(data_folder_path)) if data_split in folder]
    else:
        print(f"Aggregating data for {WTs} at {data_folder_path}")
        WT_folders = [folder for folder in sorted(os.listdir(data_folder_path)) if any(WT in folder for WT in WTs) and data_split in folder]

    X = []
    y = []
    masks = []
    timestamps = []

    print(WT_folders)

    for folder in WT_folders: 
        folder_path = os.path.join(data_folder_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)       
            np_array = (np.load(file_path, allow_pickle=True))
            
            if file == "X.npy": X.append(np_array)
            elif file == "y.npy": y.append(np_array)
            elif file == "mask.npy": masks.append(np_array)
            elif file == "timestamps.npy": timestamps.append(np_array)
            else: raise KeyError(f"No file found in {folder_path} with the name {file}")

    return (
        np.concatenate(X, axis=0), 
        np.concatenate(y, axis=0), 
        np.concatenate(masks, axis=0), 
        np.concatenate(timestamps, axis=0)
        )

def split_for_training(data, timestamp_split):
    for timestamp in timestamp_split:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", timestamp), "Invalid date format! Use YYYY-MM-DD"

    X_tensor, y_tensor, timestamps_array = data

    valid_samples = ~ch.isnan(y_tensor).squeeze()
    X_tensor, y_tensor, timestamps_array = X_tensor[valid_samples], y_tensor[valid_samples], timestamps_array[valid_samples]

    mask = (~ch.isnan(X_tensor).any(dim=2)).float()
    X_tensor = ch.nan_to_num(X_tensor, nan=0.0)

    timestamps = pd.to_datetime(timestamps_array)
    
    # Get the min and max timestamp from the data
    data_start, data_end = timestamps.min(), timestamps.max()

    if len(timestamp_split) == 1:  
        timestamp_split = pd.Timestamp(timestamp_split[0])  
        if timestamp_split < data_start:
            print(f"WARNING: Single timestamp {timestamp_split} is before the data range ({data_start} to {data_end}). Training data will be empty.")
        elif timestamp_split > data_end:
            print(f"WARNING: Single timestamp {timestamp_split} is after the data range ({data_start} to {data_end}). Testing data will be empty.")
        train_mask = timestamps < timestamp_split
        test_mask = timestamps >= timestamp_split
    elif len(timestamp_split) == 2:
        start_timestamp, end_timestamp = pd.Timestamp(timestamp_split[0]), pd.Timestamp(timestamp_split[1])
        if start_timestamp < data_start or end_timestamp > data_end:
            print(f"WARNING: Timestamp range ({start_timestamp} to {end_timestamp}) is partially or completely outside the data range ({data_start} to {data_end}). Some splits may be empty.")

        train_mask = (timestamps >= start_timestamp) & (timestamps <= end_timestamp)
        test_mask = (timestamps < start_timestamp) | (timestamps > end_timestamp)
    else:
        raise ValueError("timestamp_split should contain either 1 or 2 timestamps.")

    def splits(split_mask, X, y, mask, ts):
        return X[split_mask], y[split_mask], mask[split_mask], ts[split_mask]

    return splits(train_mask, X_tensor, y_tensor, mask, timestamps_array), \
           splits(test_mask, X_tensor, y_tensor, mask, timestamps_array)


def get_segmented_data(data_folder_path): 
    result = []

    def load_pickle(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    files = sorted(os.listdir(data_folder_path))
    wt_numbers = sorted({re.search(r"EDP_WT_(\d+)", f).group(1) for f in files if re.search(r"EDP_WT_(\d+)", f)})
    for WT in wt_numbers:
        X_tensor = None
        y_tensor = None
        timestamps_array = []
        
        for file in files:
            file_path = os.path.join(data_folder_path, file)
            
            if file.endswith(f"{WT}_X.pt"):
                tensor = ch.load(file_path)
                X_tensor = tensor if X_tensor is None else ch.cat((X_tensor, tensor), dim=0)
                
            elif file.endswith(f"{WT}_y.pt"):
                tensor = ch.load(file_path)
                y_tensor = tensor if y_tensor is None else ch.cat((y_tensor, tensor), dim=0)
                
            elif file.endswith(f"{WT}_timestamp.pkl"):
                timestamps = load_pickle(file_path)
                timestamps = np.array(timestamps)  # Ensure it's a NumPy array
                timestamps_array = (
                    timestamps if timestamps_array is None else np.concatenate((timestamps_array, timestamps))
                )

        # print("X_tensor shape:", X_tensor.shape if X_tensor is not None else "None")
        # print("y_tensor shape:", y_tensor.shape if y_tensor is not None else "None")
        # print("Number of timestamps:", len(timestamps_array))
        
        result.append((WT, (X_tensor, y_tensor, timestamps_array)))

    return result

def save_processed_data(data, directory):
    os.makedirs(directory, exist_ok=True) 
    
    X_tensor, y_tensor, mask, timestamps_array = data

    X_array = X_tensor.numpy()  # (num_samples, sequence_length, num_features)
    y_array = y_tensor.numpy()
    mask_array = mask.numpy()

    # Save as .npy files
    np.save(os.path.join(directory, 'y.npy'), y_array)
    np.save(os.path.join(directory, 'mask.npy'), mask_array)
    np.save(os.path.join(directory, 'X.npy'), X_array)
    np.save(os.path.join(directory, 'timestamps.npy'), timestamps_array)


class WindTurbineDataset(Dataset):
    def __init__(self, X_path=None, y_path=None, timestamps_path=None, X=None, y=None, timestamps=None):
        self.X = ch.load(X_path).float() if X is None else X  # (num_samples, seq_len, num_features)
        self.y = ch.load(y_path).float() if y is None else y  # (num_samples, 1)
        self.timestamps = self._load_pickle(timestamps_path) if timestamps is None else timestamps  # (num_samples,)

        valid_samples = ~ch.isnan(self.y).squeeze()
        self.X, self.y, self.timestamps = self.X[valid_samples], self.y[valid_samples], self.timestamps[valid_samples]

        self.mask = (~ch.isnan(self.X).any(dim=2)).float()
        self.X = ch.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx], self.timestamps[idx].timestamp()

    @staticmethod
    def _load_pickle(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


def create_dirs(base_dir):
    paths = [
        "temp", "data_normalisation", "data_prep",
        "EDP", "data_prep_faults_included", "fault_logs"
    ]
    for path in paths:
        full_path = os.path.join(base_dir, path)
        os.makedirs(full_path, exist_ok=True)


def get_and_save_data(base_dir, data_type="normal"):
    """
    Download and save either normal EDP data or fault logs to the provided base directory.

    Parameters:
        base_dir (str): The directory where the data should be saved.
        data_type (str): "normal" for EDP data or "fault" for fault logs.
    """
    # Create a temporary download directory inside the base directory
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    if data_type == "normal":
        print("\nDownloading EDP data")
        urls = {
            "EDP_2016.xlsx": "https://www.edp.com/sites/default/files/2023-04/Wind-Turbine-SCADA-signals-2016.xlsx",
            "EDP_2017.xlsx": "https://www.edp.com/sites/default/files/2023-04/Wind-Turbine-SCADA-signals-2017_0.xlsx"
        }
        save_subdir = os.path.join(base_dir, "EDP")
    elif data_type == "fault":
        print("\nDownloading EDP fault logs")
        urls = {
            "EDP_faults_2016.xlsx": "https://www.edp.com/sites/default/files/2023-04/Historical-Failure-Logbook-2016.xlsx",
            "EDP_faults_2017.xlsx": "https://www.edp.com/sites/default/files/2023-04/opendata-wind-failures-2017.xlsx"
        }
        save_subdir = os.path.join(base_dir, "fault_logs")
    else:
        raise ValueError("data_type must be either 'normal' or 'fault'")
    
    os.makedirs(save_subdir, exist_ok=True)

    # Download files
    for filename, url in urls.items():
        file_path = os.path.join(temp_dir, filename)
        print(f"Downloading {filename}...")
        response = requests.get(url, allow_redirects=True)
        with open(file_path, 'wb') as f:
            f.write(response.content)

    # Convert Excel files to CSV
    for year in ["2016", "2017"]:
        if data_type == "normal":
            xlsx_filename = f"EDP_{year}.xlsx"
            csv_filename = f"EDP_{year}.csv"
            print(f"Converting EDP {year}")
        else:
            xlsx_filename = f"EDP_faults_{year}.xlsx"
            csv_filename = f"EDP_faults_{year}.csv"
            print(f"Converting Fault Logs {year}")
            
        xlsx_path = os.path.join(temp_dir, xlsx_filename)
        csv_path = os.path.join(temp_dir, csv_filename)
        Xlsx2csv(xlsx_path, outputencoding="utf-8").convert(csv_path)

    # Read and concatenate CSVs
    if data_type == "normal":
        df_2016 = pd.read_csv(os.path.join(temp_dir, "EDP_2016.csv"), parse_dates=["Timestamp"])
        df_2017 = pd.read_csv(os.path.join(temp_dir, "EDP_2017.csv"), parse_dates=["Timestamp"])
        df = pd.concat([df_2016, df_2017])
    else:
        df_2016 = pd.read_csv(os.path.join(temp_dir, "EDP_faults_2016.csv"), parse_dates=["Timestamp"])
        df_2017 = pd.read_csv(os.path.join(temp_dir, "EDP_faults_2017.csv"), parse_dates=["Timestamp"])
        df = pd.concat([df_2016, df_2017])
        # Drop unnecessary columns for fault logs
        df.drop(columns=['Component', 'Remarks'], inplace=True)
    
    # Save data per turbine
    for t in df["Turbine_ID"].unique():
        d = copy.deepcopy(df[df["Turbine_ID"] == t])
        # Extract last two characters of the turbine ID as a turbine number
        tnum = t[-2:]
        output_path = os.path.join(save_subdir, f"WT_{tnum}.csv")
        if data_type == "normal":
            print(f"Saving EDP wind turbine WT_{tnum} at {output_path}")
        else:
            print(f"Saving Fault Logs for WT_{tnum} at {output_path}")
        d.to_csv(output_path, index=False)
    
    # Cleanup temporary directory
    shutil.rmtree(temp_dir)

class data_farm:
    def __init__(
            self, 
            name, 
            cols, 
            list_turbine_name, 
            BASE_DIR, 
            rename, 
            sensitive_properties,
            predicted_property,
            include_faults = False,
        ):
        """
            name: str, name of dataset (EDP)
            cols: List[str], original cols of dataset ... will change these to in_cols based on data processing
            list_turbine_name: List[str], names of the turbines in the dataset
            BASE_DIR: str, where the dataset is located
            rename: Dict, new column names for dataset
            sensitive_properties: List[str], proporty/property combo to use for sensitive property
            predicted_property: List[str], property to use for predictive task
            include_faults: bool, whether or not to include faulty data periods
        """
        self.name = name
        self.columns = cols
        self.in_cols = [col for col in cols if col not in predicted_property and col != "Timestamp"]
        self.out_cols = predicted_property
        self.sensitive_properties = sensitive_properties
        self.list_turbine_name = list_turbine_name
        self.BASE_DIR = BASE_DIR
        self.rename = rename
        self.CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data_prep") if not include_faults else os.path.join(BASE_DIR, "data_prep_faults_included")
        self.normalized_dict = {}
        self.time_name = "Timestamp"
        self.include_faults = include_faults

        for t in self.list_turbine_name:
            print(f"Initializing {t}")
            self.initialize_data(t)
        self.prepare_data()

        self.X, self.y, self.timestamps = {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}

    def initialize_data(self, t):
        setattr(self, t, self.load_df(f"{t}.csv", rename=self.rename, include_faults=self.include_faults))
        
        assert "Timestamp" in getattr(self, t).columns, f"Need a timestamp column"
        assert "Wind_speed" in getattr(self, t).columns, f"Need a Wind_speed column"
        
        index = pd.to_datetime(getattr(self, t)["Timestamp"]).dt.tz_convert(None) 
        getattr(self, t).set_index(index, inplace=True)
        getattr(self, t).drop(["Timestamp"], axis=1, inplace=True)

    def prepare_data(self): 
        self.feature_cut("Wind_speed", min=0, max= 25, min_value=0)
        self.feature_cut("Power", min=0,  min_value=0)
        self.feature_cut("Rotor_speed", min=0,  min_value=0)
        self.feature_cut("Gen_speed", min=0)
        self.line_cut("Wind_speed", "Power", a = 0, b= 1100, xmin = 4.2, xmax= 25)
        cols = self.out_cols + self.in_cols

        for t in self.list_turbine_name:
            setattr(self, t, getattr(self, t)[~getattr(self, t).index.duplicated()])

        self.drop_col([c for c in getattr(self, self.list_turbine_name[0]).columns if c not in cols])
        self.time_filling(interpolation_limit = 24)

        # Normalization
        self.normalize_min_max(cols=[c for c in cols if c in [
                                                                "Wind_speed", "Amb_temp", "Gen_speed", "Rotor_speed",
                                                                "Blade_pitch", "Gen_phase_temp_1", "Gen_phase_temp_2", 
                                                                "Gen_phase_temp_3", "Transf_temp_p1", "Transf_temp_p2", 
                                                                "Transf_temp_p3", "Gen_bear_temp_avg", "Hyd_oil_temp",
                                                                "Gear_oil_temp", "Gear_bear_temp", "Nacelle_position"
                                                             ]])
        self.normalize_mean_std(cols=[c for c in cols if c in [
                                                                "Wind_speed_std", 
                                                                "Gen_speed_std", 
                                                                "Rotor_speed_std", 
                                                                "Blade_pitch_std"
                                                            ]])
        self.circ_embedding(cols=[c for c in cols if c in [
                                                                "Wind_rel_dir", 
                                                                "Nacelle_position"
                                                            ]])

        self.feature_weighted_combination("Gen_bear_temp_avg", ["Gen_bearing_temp_1", "Gen_bearing_temp_2"])
        self.feature_weighted_combination("sensitive_property", self.sensitive_properties)

        self.save_data()
        
    def load_df(self, file_name, rename={}, include_faults = False):
        wt_df = pd.read_csv(os.path.join(self.BASE_DIR, "EDP", file_name), parse_dates=[self.time_name])
        
        # Data cleaning, renaming
        if len(rename) > 0:
            rename = {key: rename[key] for key in rename.keys() if key in wt_df.columns}
            wt_df.rename(columns=rename, inplace=True)

        wt_df["Timestamp"] = pd.to_datetime(wt_df["Timestamp"], utc=True)

        # Remove rows with NaN timestamps and sort
        wt_df = wt_df.dropna(subset=["Timestamp"]).sort_values(by="Timestamp").reset_index(drop=True)
        wt_df = wt_df.drop_duplicates(subset=["Timestamp"])
        full_range = pd.date_range(start=wt_df["Timestamp"].min(), 
                                end=wt_df["Timestamp"].max(), 
                                freq="10min") 
        
        # Reindexing the dataframe to include the full range of timestamps
        wt_df = wt_df.set_index("Timestamp").reindex(full_range).reset_index()
        wt_df.rename(columns={"index": "Timestamp"}, inplace=True)
        wt_df = wt_df[[c for c in self.columns if c in wt_df.columns]]
    
        # Whether or not include faulty data periods in the dataset
        if self.include_faults:
            fault_df = pd.read_csv(os.path.join(self.BASE_DIR, "fault_logs", file_name), parse_dates=[self.time_name])

            wt_df.set_index("Timestamp", inplace=True)
            for fault_time in fault_df['Timestamp']:
                start_time = fault_time - pd.Timedelta(weeks=2)
                end_time = fault_time + pd.Timedelta(days=3)
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.in_cols] = np.nan
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.out_cols] = np.nan
            
        # Removing periods where the data appears faulty (Wind > 4.0, but 0.0 value sensor readings)
        cols_of_interest = (['Gen_speed', 'Rotor_speed', 'Power']) 
        for col in cols_of_interest:
            zero_col_with_wind = (wt_df[col] == 0) & (wt_df["Wind_speed"] > 4.0)
            if zero_col_with_wind.any():
                print(f"Wind is > 4.0, but {col} is 0 for {len(wt_df.index[zero_col_with_wind])} timestamps")
                wt_df.loc[zero_col_with_wind, self.in_cols] = np.nan
                wt_df.loc[zero_col_with_wind, self.out_cols] = np.nan
            else:
                print(f"No occurrences found for {col}.")
            
        wt_df.reset_index(drop=False, inplace=True)

        return wt_df 

    def save_data(self, turbine = None):
        turbine = turbine if turbine is not None else self.list_turbine_name
        for t in turbine:
            getattr(self, t).to_csv(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}.csv")) 
    
    def feature_cut(self, feature_name, min = None, max = None, min_value = "drop", max_value = "drop", turbine_name = None, verbose = False):
        turbine = turbine_name if turbine_name is not None else self.list_turbine_name

        for t in turbine:
            assert feature_name in getattr(self, t).columns, f"Wrong feature name: {feature_name}"
            n = len(getattr(self, t)[feature_name])
            if min is not None:
                if min_value == "drop":
                    cut_offs = getattr(self, t)[getattr(self, t)[feature_name] < min].index
                    getattr(self, t).drop(cut_offs, inplace=True)
                else:
                    neg_powers = getattr(self, t)[getattr(self, t)[feature_name] < min].index
                    getattr(self, t).loc[neg_powers, feature_name] = min_value
            if max is not None: 
                if max_value == "drop":
                    cut_offs = getattr(self, t)[getattr(self, t)[feature_name] > max].index
                    getattr(self, t).drop(cut_offs, inplace=True)
                else:
                    neg_powers = getattr(self, t)[getattr(self, t)[feature_name] > max].index
                    getattr(self, t).loc[neg_powers, feature_name] = max_value 
            if verbose:
                print(f"Cutted {n-len(getattr(self, t)[feature_name])} element from {t} for {feature_name} in {self.name}")

    def line_cut(self, input_feature_name, output_feature_name, a, b, xmin, xmax, under = True, turbine_name = None):
        turbine = turbine_name if turbine_name is not None else self.list_turbine_name
        
        for t in turbine:
            assert input_feature_name in getattr(self, t).columns, f"Wrong feature name: {input_feature_name}"
            assert output_feature_name in getattr(self, t).columns, f"Wrong feature name: {output_feature_name}"
            if under:
                f = lambda x: x[input_feature_name] <= xmin or a*x[input_feature_name] + b < x[output_feature_name] or x[input_feature_name] >= xmax
            else:
                f = lambda x: x[input_feature_name] <= xmin or a*x[input_feature_name] + b > x[output_feature_name] or x[input_feature_name] >= xmax
            setattr(self, t, getattr(self, t)[getattr(self, t).apply(f, axis=1)])
            
    def feature_averaging(self, name, input_names):
        for t in self.list_turbine_name:
            assert all([name in getattr(self,t).columns for name in input_names]), f"The input names are incorrect: {[name for name in input_names if name not in getattr(self,t).columns]}"
            getattr(self, t)[name] = getattr(self, t)[input_names].mean(axis= 1)

    def feature_weighted_combination(self, name, input_names, weights=None):
        for t in self.list_turbine_name:
            # Check that all input columns exist
            assert all([col in getattr(self, t).columns for col in input_names]), (
                f"The input names are incorrect: {[col for col in input_names if col not in getattr(self, t).columns]}"
            )

            if len(input_names) == 1:
                getattr(self, t)[name] = getattr(self, t)[input_names[0]]
            else:
                if weights is None:
                    weights = [1.0/len(input_names)] * len(input_names)
                else:
                    assert len(weights) == len(input_names), "Weights length must match input_names length."
                
                norm_weights = np.array(weights) / np.sum(weights)
                
                combined = sum(norm_weights[i] * getattr(self, t)[col] for i, col in enumerate(input_names))
                getattr(self, t)[name] = combined
            
            setattr(self, t, getattr(self, t).drop(columns=input_names))
        
        for remove_col in input_names: self.in_cols.remove(remove_col)
        self.in_cols.append(name)

    def naive_feautre_binning(self, feature): 
        pass

    def normalize_min_max(self, cols):
        global_stats = {} # using min/max w/ global values to maintain variability btw WTs
        for c in cols:
            all_values = pd.concat([getattr(self, t)[c] for t in self.list_turbine_name])
            global_stats[c] = (all_values.max(), all_values.min())
        
        for t in self.list_turbine_name:
            self.normalized_dict[t] = {c: global_stats[c] for c in cols}
            for c in cols:
                col_max, col_min = global_stats[c]
                getattr(self, t)[c] = (getattr(self, t)[c] - col_min) / (col_max - col_min)
        with open(os.path.join(self.BASE_DIR, 'data_normalisation', f'min_max_normalisation_{self.name}.pkl'), 'wb') as f:
            pickle.dump(self.normalized_dict, f)

    def normalize_mean_std(self, cols):
        global_stats = {} # using min/max w/ global values to maintain variability btw WTs
        for c in cols:
            all_values = pd.concat([getattr(self, t)[c] for t in self.list_turbine_name])
            global_stats[c] = (all_values.mean(), all_values.std())
        
        for t in self.list_turbine_name:
            self.normalized_dict[t] = {c: global_stats[c] for c in cols}
            for c in cols:
                col_mean, col_std = global_stats[c]
                getattr(self, t)[c] = (getattr(self, t)[c] - col_mean) / col_std
        with open(os.path.join(self.BASE_DIR, 'data_normalisation', f'std_mean_normalisation_{self.name}.pkl'), 'wb') as f:
            pickle.dump(self.normalized_dict, f)

    def circ_embedding(self, cols):
        # Assuming that the unit is ° (0 to 360)
        for t in self.list_turbine_name:
            for c in cols:
                getattr(self, t)[f"{c}_cos"] = np.cos(getattr(self,t)[c]*2*np.pi/360)
                getattr(self, t)[f"{c}_sin"] = np.sin(getattr(self,t)[c]*2*np.pi/360)

    def drop_col(self, cols):
        for t in self.list_turbine_name:
            getattr(self,t).drop(columns = cols, inplace = True)

    def time_filling(self, method = 'linear', interpolation_limit = 6):
        for t in self.list_turbine_name:
            setattr(self,t, getattr(self, t).asfreq("10min"))
            getattr(self,t).interpolate(method = method, limit=interpolation_limit, inplace = True)
            print(getattr(self, t).shape)

    def get_window_data(self, in_shift, RESET = False):
        breakpoint()
        if RESET:
             self.make_window_data(in_shift, turbine = None)
        else:
            turbine = [t for t in self.list_turbine_name if not (os.path.isfile(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_X.pt")) and \
                                                                 os.path.isfile(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_y.pt")) and \
                                                                 os.path.isfile(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_timestamp.pkl")))]
            
            self.make_window_data(in_shift, turbine=turbine)
            
        for t in self.list_turbine_name:
            setattr(self, f"{t}_X", ch.load(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_X.pt")))
            setattr(self, f"{t}_y", ch.load(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_y.pt")))
            with open(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_timestamp.pkl"), 'rb') as f:
                setattr(self, f"{t}_timestamp", pickle.load(f))

    def make_window_data(self, in_shift, turbine = None, ret = False):
        turbine = turbine if turbine is not None else self.list_turbine_name 
        
        for t in turbine: 
            print(f"Preparing turbine {t}")
        
            data = getattr(self,t)

            in_data = data[self.in_cols].to_numpy(copy = True)
            out_data = data[self.out_cols].to_numpy(copy = True)
            
            index = data.index
            daterange = pd.date_range(start=index[0], end=index[-1], freq = "10min")

            #Creating blocks of timestamps without missing values
            blocks = [[]]
            i = 0
            for d in daterange:
                if d in data.index:
                    blocks[i].append(d)
                elif blocks[-1] != []:
                    blocks.append([])
                    i+=1
            if blocks[-1] == []:
                blocks = blocks[:-1]
            
            # For each block make a dataset
            in_datasets = []
            out_datasets = []
            timestamps = []

            for block in blocks:
                if len(block) > in_shift:
                    in_window = [in_data[i: i + in_shift +1] for i in range(len(block)-in_shift)]
                    out_window = [out_data[i+ in_shift] for i in range(len(block)-in_shift)]
                    timestamp = [block[i+in_shift] for i in range(len(block)-in_shift)]
                else:
                    in_window = None
                    out_window = None
                    timestamp = None
                if in_window is not None and out_window is not None and timestamp is not None:
                    in_datasets.append(in_window)
                    out_datasets.append(out_window)
                    timestamps.append(timestamp)

            # Concatenate the blocks
            if len(in_datasets) > 0:
                in_window_full = np.concatenate(in_datasets)
                out_window_full = np.concatenate(out_datasets)
                time = np.concatenate(timestamps)
                X, y = ch.Tensor(in_window_full), ch.Tensor(out_window_full)
                ch.save(X, os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_X.pt"))
                ch.save(y, os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_y.pt"))
                with open(os.path.join(self.CLEANED_DATA_PATH, f"{self.name}_{t}_timestamp.pkl"), "wb") as f:
                    pickle.dump(time, f)
                
                if ret:
                    return X,y,time
    

def process_edp_data(
        BASE_DIR,
        sensitive_properties, 
        predicted_property,  
        include_faults
    ):
    """
        BASE_DIR: str, where the dataset is located
        sensitive_properties: List[str], proporty/property combo to use for sensitive property
        predicted_property: List[str], property to use for predictive task
        include_faults: bool, whether or not to include faulty data periods
    """
    DATA_PATH = os.path.join(BASE_DIR, "EDP")
    
    if os.path.exists(DATA_PATH):
        for path in os.listdir(DATA_PATH):
            df = pd.read_csv(os.path.join(DATA_PATH, path))
    
    # columns of the original data
    cols_edp = [
        "Timestamp", "Wind_speed", "Wind_speed_std", "Wind_rel_dir", "Amb_temp",
        "Gen_speed", "Gen_speed_std", "Rotor_speed", "Rotor_speed_std", "Blade_pitch",
        "Blade_pitch_std", "Gen_phase_temp_1", "Gen_phase_temp_2", "Gen_phase_temp_3",
        "Transf_temp_p1", "Transf_temp_p2", "Transf_temp_p3", "Gen_bearing_temp_1",
        "Gen_bearing_temp_2", "Hyd_oil_temp", "Gear_oil_temp", "Gear_bear_temp",
        "Nacelle_position", "Power"
    ]
    rename_edp = {
        "Timestamp": "Timestamp", 
        
        # Enviromental Statistics
        "Amb_WindSpeed_Avg": "Wind_speed",
        "Amb_WindSpeed_Std": "Wind_speed_std",
        "Amb_WindDir_Relative_Avg": "Wind_rel_dir",
        "Amb_Temp_Avg": "Amb_temp",

        # Affects energy conversion efficiency
            # Generator RPM Statistics
        "Gen_RPM_Avg": "Gen_speed", 
        "Gen_RPM_Std": "Gen_speed_std",
            # Rotor Statistics
        "Rtr_RPM_Avg": "Rotor_speed", 
        "Rtr_RPM_Std": "Rotor_speed_std",

        # Blade Pitch Angle, Optimizes wind capture
        "Blds_PitchAngle_Avg" : "Blade_pitch",
        "Blds_PitchAngle_Std" : "Blade_pitch_std",

        # Impacts efficiency & potential failures
            # Generator average temperatures of the stator windings (the stationary part of the generator) for each phase
            # Indicates power generation health, winding issues, and cooling efficiency inside the generator
        "Gen_Phase1_Temp_Avg" : "Gen_phase_temp_1",
        "Gen_Phase2_Temp_Avg" : "Gen_phase_temp_2",
        "Gen_Phase3_Temp_Avg" : "Gen_phase_temp_3",
            # Transformer phase temperatures	
            # power transmission efficiency, grid loading issues, and transformer overheating risks.
        "HVTrafo_Phase1_Temp_Avg": "Transf_temp_p1",
        "HVTrafo_Phase2_Temp_Avg": "Transf_temp_p2",
        "HVTrafo_Phase3_Temp_Avg": "Transf_temp_p3",

        # Temperatures that might indicate fault/high stress
        # Generator Bearing Temperature Statisitcs TODO: always 2 generator bearings?
        "Gen_Bear_Temp_Avg": "Gen_bearing_temp_1", 
        "Gen_Bear2_Temp_Avg": "Gen_bearing_temp_2", 
            # Hydraulic oil temperature
        "Hyd_Oil_Temp_Avg": "Hyd_oil_temp",
            # Gearbox Statistics
        "Gear_Oil_Temp_Avg" : "Gear_oil_temp",
        "Gear_Bear_Temp_Avg": "Gear_bear_temp", 

        # Nacelle Direction - should align with wind realtive direction ... not doing so = poor performance? 
        "Nac_Direction_Avg": "Nacelle_position",

        # Power (have it by generator, but excluding bc feature would be too correlated with target feature ... total active power)
        "Prod_LatestAvg_TotActPwr": "Power",
    }

    list_names_edp = ["WT_01", "WT_06", "WT_07", "WT_11"]

    print(f"Data farm {'WITHOUT' if include_faults else 'WITHOUT'} faults.")
    farm_edp = data_farm(
        name="EDP", 
        cols=cols_edp, 
        list_turbine_name=list_names_edp,
        BASE_DIR=BASE_DIR,
        rename=rename_edp, 
        sensitive_properties=sensitive_properties,
        predicted_property=predicted_property,
        include_faults=include_faults,
    )
    print("\n Preparing data windows")
    print(f"Saving data to {farm_edp.CLEANED_DATA_PATH}")
    farm_edp.get_window_data(in_shift=24*6, RESET=True)