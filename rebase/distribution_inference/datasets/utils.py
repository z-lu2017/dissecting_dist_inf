from distribution_inference.utils import warning_string
import os, copy, re, pickle, requests, shutil, zipfile, glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch as ch
from xlsx2csv import Xlsx2csv
from pathlib import Path
from torch.utils.data import Dataset, Subset
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from collections import defaultdict
import h5py
from numpy.lib.stride_tricks import sliding_window_view


from distribution_inference.datasets import new_census, celeba, boneage,census, texas, arxiv, maadface, lstm_wind_turbine, lstm_WT_new

DATASET_INFO_MAPPING = {
    "new_census": new_census.DatasetInformation,
    "celeba": celeba.DatasetInformation,
    "boneage": boneage.DatasetInformation,
    "old_census": census.DatasetInformation,
    "texas": texas.DatasetInformation,
    "arxiv": arxiv.DatasetInformation,
    "maadface": maadface.DatasetInformation,
    "lstm_wind_turbine": lstm_wind_turbine.DatasetInformation, 
    "lstm_WT_new": lstm_WT_new.DatasetInformation,
}

DATASET_WRAPPER_MAPPING = {
    "new_census": new_census.CensusWrapper,
    "celeba": celeba.CelebaWrapper,
    "boneage": boneage.BoneWrapper,
    "old_census": census.CensusWrapper,
    "texas": texas.TexasWrapper,
    "arxiv": arxiv.ArxivWrapper,
    "maadface": maadface.MaadFaceWrapper,
    "lstm_wind_turbine": lstm_wind_turbine.LSTMWindTurbineWrapper,
    "lstm_WT_new": lstm_WT_new.LSTMWindTurbineWrapper,
}

GLOBAL_FILE_URLS = {
    "Upgrade_dates":                  "Hill_of_Towie_AeroUp_install_dates.csv?download=1",
    "Alarm_descriptions":             "Hill_of_Towie_alarms_description.csv?download=1",
    "Grid_field_descriptions":        "Hill_of_Towie_grid_fields_description.csv?download=1",
    "Tables_descriptions":            "Hill_of_Towie_tables_description.csv?download=1",
    "Turbine_field_descriptions":     "Hill_of_Towie_turbine_fields_description.csv?download=1",
    "Turbine_metadata_descriptions":  "Hill_of_Towie_turbine_metadata.csv?download=1",
    "Shutdown_dates":                 "Hill_of_Towie_ShutdownDuration.zip?download=1",
}

KEEP_FILES = {
    "tblAlarmLog": "Alarm_logs", 
    "tblDailySummary": "Daily_stats", 
    "tblSCTurbine": "Operational_stats",
    "tblSCTurTemp": "Temperature_stats", 
    "tblSCTurGrid": "Power_stats",
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

def get_segmented_data_scotland(data_folder, WT): 
    data_folder_path = os.path.join(data_folder, WT)

    pattern = re.compile(
        rf"Scotland_{WT}_(?P<dtype>X|y|ts)_(?P<idx>\d+)\.(?P<ext>pt|pkl)$"
    )

    paths = defaultdict(dict)
    for fname in os.listdir(data_folder_path):
        m = pattern.match(fname)
        if not m:
            continue
        dtype, idx = m.group("dtype"), int(m.group("idx"))
        paths[dtype][idx] = os.path.join(data_folder_path, fname)

    loaded = {}
    for dtype in ("X", "y", "ts"):
        chunks = []
        for idx in sorted(paths[dtype]):
            fp = paths[dtype][idx]
            if dtype in ("X", "y"):
                chunks.append(ch.load(fp))
            else:  # timestamps
                with open(fp, "rb") as f:
                    chunks.append(pickle.load(f))
        loaded[dtype] = chunks
    return loaded

def get_segmented_data_edp(data_folder_path): 
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
        "raw_data", "data_prep_faults_excluded", "fault_logs",
        os.path.join("/scratch/ujx4ab", "data_prep"),
        os.path.join("/scratch/ujx4ab", "data_prep_faults_excluded")
    ]
    for path in paths:
        full_path = os.path.join(base_dir, path)
        os.makedirs(full_path, exist_ok=True)


# TODO: actually pass in the keep directories instead of hardcoding
def cleanup_dirs(base_dir, delete=["aggregate", "data_normalisation", "data_prep", "data_prep_faults_excluded"]):
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name in delete:
            try:
                shutil.rmtree(full_path)
            except Exception as e:
                print(f"Failed to remove {full_path}: {e}")


def get_available_cpus():
    slurm_cpus = os.getenv("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        return int(slurm_cpus)
    
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    cpus = line.strip().split(":")[1].strip()
                    ranges = cpus.split(",")
                    count = 0
                    for r in ranges:
                        if "-" in r:
                            start, end = map(int, r.split("-"))
                            count += end - start + 1
                        else:
                            count += 1
                    return count
    except Exception:
        pass

    return os.cpu_count() or 1


class data_farm:
    def __init__(
            self, 
            dataset_name, 
            cols, 
            fault_check_cols,
            list_turbine_name, 
            BASE_DIR, 
            predicted_property,
            rename = None,
            exclude_fault_data = False,
        ):
        self.dataset_name = dataset_name
        self.columns = cols
        self.fault_check_cols = fault_check_cols
        # TODO: fix in cols vs keep cols
        self.in_cols = [col for col in cols if col not in predicted_property and col != "Timestamp"]
        self.out_cols = predicted_property
        self.list_turbine_name = list_turbine_name
        self.exclude_fault_data = exclude_fault_data

        self.BASE_DIR = BASE_DIR
        self.CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data_prep") if not exclude_fault_data else os.path.join(BASE_DIR, "data_prep_faults_excluded")
        self.WINDOWED_DATA_PATH = os.path.join("/scratch/ujx4ab", "data_prep") if not exclude_fault_data else os.path.join("/scratch/ujx4ab", "data_prep_faults_excluded")

        for t in self.list_turbine_name:
            print(f"Initializing {t}")
            self.initialize_data(t, rename)
        
        self.prepare_edp_data() if dataset_name == "EDP" else self.prepare_scotland_data()
        self.X, self.y, self.timestamps = {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}

    def initialize_data(self, t, rename):
        setattr(self, t, self.load_df(f"{t}.csv", rename=rename, exclude_fault_data=self.exclude_fault_data))
        assert "Timestamp" in getattr(self, t).columns, f"Need a timestamp column"
        assert "WindSpeed" in getattr(self, t).columns, f"Need a WindSpeed column"
        
        index = pd.to_datetime(getattr(self, t)["Timestamp"]).dt.tz_convert(None) 
        getattr(self, t).set_index(index, inplace=True)
        getattr(self, t).drop(["Timestamp"], axis=1, inplace=True)

    # @profile
    def load_df(self, file_name, rename, exclude_fault_data = False):
        wt_df = pd.read_csv(os.path.join(self.BASE_DIR, "raw_data", file_name), parse_dates=["Timestamp"])
        float_cols = wt_df.select_dtypes(include="float64").columns
        wt_df[float_cols] = wt_df[float_cols].astype("float32")
        
        # Data cleaning, renaming
        if rename and len(rename) > 0:
            rename = {key: rename[key] for key in rename.keys() if key in wt_df.columns}
            wt_df.rename(columns=rename, inplace=True)

        wt_df["Timestamp"] = pd.to_datetime(wt_df["Timestamp"], utc=True, format="mixed", infer_datetime_format=True)

        # Remove rows with NaN timestamps and sort. Individual calls bc of memory issues.
        wt_df = wt_df.dropna(subset=["Timestamp"])\
                    .sort_values(by="Timestamp")\
                    .reset_index(drop=True)
        wt_df = wt_df.drop_duplicates(subset=["Timestamp"])

        full_range = pd.date_range(start=wt_df["Timestamp"].min(), 
                                end=wt_df["Timestamp"].max(), 
                                freq="10min") 
        
        # Reindexing the dataframe to include the full range of timestamps
        wt_df = wt_df.set_index("Timestamp").reindex(full_range).reset_index()
        wt_df.rename(columns={"index": "Timestamp"}, inplace=True)
        wt_df = wt_df[[c for c in self.columns if c in wt_df.columns]]

        # Whether or not include faulty data periods in the dataset
        if self.exclude_fault_data:
            fault_df = pd.read_csv(os.path.join(self.BASE_DIR, "fault_logs", file_name), parse_dates=["Timestamp"])

            wt_df.set_index("Timestamp", inplace=True)
            for fault_time in fault_df['Timestamp']:
                start_time = fault_time - pd.Timedelta(weeks=2)
                end_time = fault_time + pd.Timedelta(days=3)
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.in_cols] = np.nan
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.out_cols] = np.nan
        
        # cut in depends on the dataset
        cut_in_wind_speed = 3.5 if self.dataset_name == "Scotland" else 4.0
        for col in self.fault_check_cols:
            zero_col_with_wind = (wt_df[col] == 0) & (wt_df["WindSpeed"] > cut_in_wind_speed)
            if zero_col_with_wind.any():
                wt_df.loc[zero_col_with_wind, self.in_cols] = np.nan
                wt_df.loc[zero_col_with_wind, self.out_cols] = np.nan
            else:
                print(f"No occurrences found for {col}.")
            
        wt_df.reset_index(drop=False, inplace=True)

        return wt_df 

    def prepare_edp_data(self): 
        self.feature_cut("WindSpeed", min=0, max=25, min_value=0)
        self.feature_cut("Power", min=0, max=334000, min_value=0)
        self.feature_cut("RotorSpeed", min=0,  min_value=0)
        self.feature_cut("GenSpeed", min=0)
        self.line_cut("WindSpeed", "Power", a = 0, b= 1100, xmin = 4.2, xmax= 25)
        cols = self.out_cols + self.in_cols

        # remove any duplicate indices
        for t in self.list_turbine_name:
            df = getattr(self, t)
            df = df[~df.index.duplicated()]
            setattr(self, t, df)

        self.drop_col([c for c in getattr(self, self.list_turbine_name[0]).columns if c not in cols])
        self.time_filling(interpolation_limit = 24)

        # Normalization
        normalized_dict = {}
        self.normalize_min_max(
                                normalized_dict = normalized_dict,
                                cols=[c for c in cols if c in [
                                                                "WindSpeed", "AmbientTemp", "GenSpeed", "RotorSpeed",
                                                                "BladePitch", "GenPhaseTemp1", "GenPhaseTemp2", 
                                                                "GenPhaseTemp3", "TransfTempP1", "TransfTempP2", 
                                                                "TransfTempP3", "GenBearTemp_avg", "HydOilTemp",
                                                                "GearOilTemp", "GearBearTemp", "NacellePos",
                                                                "Power"
                                                             ]])
        self.normalize_mean_SD(
                                normalized_dict = normalized_dict,
                                cols=[c for c in cols if c in [
                                                                "WindSpeed_SD", 
                                                                "GenSpeed_SD", 
                                                                "RotorSpeed_SD", 
                                                                "BladePitch_SD"
                                                            ]])
        self.circ_embedding(cols=[c for c in cols if c in [
                                                                "RelWindDir", 
                                                                "NacellePos"
                                                            ]])
        self.feature_weighted_combination("GenBearTemp_avg", ["GenBearingTemp1", "GenBearingTemp2"])

        # self.save_data()

    # @profile
    def prepare_scotland_data(self): 
        self.feature_cut("WindSpeed", min=0, max=25, min_value=0)
        self.feature_cut("Power", min=0, max=2300, min_value=0, max_value=2300)
        self.feature_cut("Yaw", min=0, min_value=0, max_value=360)
        self.feature_cut("GearOilTemp", min=0, max=52)
        self.feature_cut("GearboxBearingTempHSGenSide", min=0, max=65)
        self.feature_cut("GearboxBearingTempHSRotorSide", min=0, max=65)

        cols = self.out_cols + self.in_cols

        # remove any duplicate indices
        for t in self.list_turbine_name:
            df = getattr(self, t)
            df = df[~df.index.duplicated()]
            setattr(self, t, df)  # update the attribute

        self.drop_col([c for c in getattr(self, self.list_turbine_name[0]).columns if c not in cols])
        self.time_filling(interpolation_limit = 24)

        # Normalization
        normalized_dict = {}
        self.normalize_min_max(
                                normalized_dict = normalized_dict,
                                cols=[c for c in cols if c in [
                                                                "WindSpeed", 
                                                                "AmbientTemp",
                                                                "MainShaftRPM",
                                                                "BladePitchA",
                                                                "BladePitchB",
                                                                "BladePitchC",
                                                                "RefBladePitchA",
                                                                "RefBladePitchB",
                                                                "RefBladePitchC",
                                                                "GearOilTemp",
                                                                "GearboxBearingTempHSGenSide",
                                                                "GearboxBearingTempHSRotorSide",
                                                                "Power"
                                                             ]])
        self.normalize_mean_SD(
                                normalized_dict = normalized_dict,
                                cols=[c for c in cols if c in [
                                                                "WindSpeed_SD",
                                                                "Yaw_SD",
                                                                "SetYaw_SD",
                                                                "MainShaftRPM_SD",
                                                                "BladePitchA_SD",
                                                                "BladePitchB_SD",
                                                                "BladePitchC_SD",
                                                                "RefBladePitchA_SD",
                                                                "RefBladePitchB_SD",
                                                                "RefBladePitchC_SD",
                                                                "Power_SD",
                                                            ]])
        self.circ_embedding(cols=[c for c in cols if c in [
                                                                "Yaw",
                                                                "SetYaw",
                                                            ]])

        # self.save_data()


    def save_data(self, turbine = None):
        turbine = turbine if turbine is not None else self.list_turbine_name
        for t in turbine:
            save_path = os.path.join(self.CLEANED_DATA_PATH, f"{self.dataset_name}_{t}.csv")
            print(f"saving {t} data to {save_path}")
            getattr(self, t).to_csv(save_path) 
    
    def feature_cut(self, feature_name, min = None, max = None, min_value = "drop", max_value = "drop", turbine_name = None, verbose = False):
        turbine = turbine_name if turbine_name is not None else self.list_turbine_name

        for t in turbine:
            df: pd.DataFrame = getattr(self, t)
            assert feature_name in df.columns, f"Wrong feature name: {feature_name}"
            orig_len = len(df)

            series = df[feature_name]

            if min is not None:
                mask_min = series < min
                if min_value == "drop":
                    df.drop(index=df.index[mask_min], inplace=True)
                else:
                    df.loc[mask_min, feature_name] = min_value
            
            series = df[feature_name]
            if max is not None:
                mask_max = series > max
                if max_value == "drop":
                    df.drop(index=df.index[mask_max], inplace=True)
                else:
                    df.loc[mask_max, feature_name] = max_value
            
            if verbose:
                print(f"Cutted {orig_len - len(df)} rows from {t} for {feature_name}")
            setattr(self, t, df)

    def line_cut(self, input_feature_name, output_feature_name,
                a, b, xmin, xmax, under=True, turbine_name=None):

        turbines = turbine_name or self.list_turbine_name

        for t in turbines:
            df = getattr(self, t)
            if under:
                f = lambda x: (x[input_feature_name] <= xmin
                            or a*x[input_feature_name] + b < x[output_feature_name]
                            or x[input_feature_name] >= xmax)
            else:
                f = lambda x: (x[input_feature_name] <= xmin
                            or a*x[input_feature_name] + b > x[output_feature_name]
                            or x[input_feature_name] >= xmax)

            keep = df.apply(f, axis=1)
            dropped = (~keep).sum()
            total = len(df)
            print(f"Turbine {t}: dropping {dropped}/{total} rows")

            setattr(self, t, df[keep])

    # def line_cut(self, input_feature_name, output_feature_name, a, b, xmin, xmax, under = True, turbine_name = None):
    #     turbine = turbine_name if turbine_name is not None else self.list_turbine_name
        
    #     for t in turbine:
    #         assert input_feature_name in getattr(self, t).columns, f"Wrong feature name: {input_feature_name}"
    #         assert output_feature_name in getattr(self, t).columns, f"Wrong feature name: {output_feature_name}"
    #         if under:
    #             f = lambda x: x[input_feature_name] <= xmin or a*x[input_feature_name] + b < x[output_feature_name] or x[input_feature_name] >= xmax
    #         else:
    #             f = lambda x: x[input_feature_name] <= xmin or a*x[input_feature_name] + b > x[output_feature_name] or x[input_feature_name] >= xmax
    #         setattr(self, t, getattr(self, t)[getattr(self, t).apply(f, axis=1)])
            
    def feature_averaging(self, name, input_names):
        for t in self.list_turbine_name:
            assert all([name in getattr(self,t).columns for name in input_names]), f"The input names are incorrect: {[name for name in input_names if name not in getattr(self,t).columns]}"
            getattr(self, t)[name] = getattr(self, t)[input_names].mean(axis= 1)

    def feature_weighted_combination(self, name, input_names, remove_input_cols = True, weights=None):
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
            
            if remove_input_cols: 
                setattr(self, t, getattr(self, t).drop(columns=input_names))
        
        if remove_input_cols: 
            for remove_col in input_names: 
                self.in_cols.remove(remove_col)
        self.in_cols.append(name)

    def normalize_min_max(self, normalized_dict, cols):
        # compute global min/max per column 
        global_stats = {}

        # Only for WindSpeed quantile tracking
        windspeed_low_qs = []
        windspeed_high_qs = []
        low_q, high_q = 0.33, 0.66

        for c in cols:
            gmin, gmax = np.inf, -np.inf
            for t in self.list_turbine_name:
                arr = getattr(self, t)[c].to_numpy(copy=False)
                # skip NaNs
                col_min = np.nanmin(arr)
                col_max = np.nanmax(arr)
                if col_min < gmin: gmin = col_min
                if col_max > gmax: gmax = col_max

                if c == "WindSpeed": 
                    windspeed_low_qs.append(np.nanquantile(arr, low_q))
                    windspeed_high_qs.append(np.nanquantile(arr, high_q))                    

            global_stats[c] = (gmin, gmax)

        # TODO: move this portion of the code to being post processing so that
            # this doesn't have to run everytime i want to get the threshold
        # Compute average quantile thresholds for WindSpeed
        if windspeed_low_qs and windspeed_high_qs:
            WindSpeed_min, WindSpeed_max = global_stats["WindSpeed"]
            avg_low_thresh = float((np.mean(windspeed_low_qs) - WindSpeed_min)/(WindSpeed_max-WindSpeed_min))
            avg_high_thresh = float((np.mean(windspeed_high_qs) - WindSpeed_min)/(WindSpeed_max-WindSpeed_min))
            print(f"Average WindSpeed thresholds: low<{avg_low_thresh:.3f}, high>{avg_high_thresh:.3f}")
            self.windspeed_thresholds = (avg_low_thresh, avg_high_thresh)

        # apply normalization in one block per turbine
        for t in self.list_turbine_name:
            df = getattr(self, t)
            # stats for this turbine
            normalized_dict[t] = {c: global_stats[c] for c in cols}

            block = df[cols].to_numpy(copy=False, dtype=np.float32)
            mins = np.array([global_stats[c][0] for c in cols], dtype=np.float32)
            maxs = np.array([global_stats[c][1] for c in cols], dtype=np.float32)
            # in-place normalize
            block -= mins
            block /= (maxs - mins)
            df.loc[:, cols] = block
            setattr(self, t, df)
        
        with open(os.path.join(self.BASE_DIR, 'data_normalisation', 
                            f'min_max_normalisation_{self.dataset_name}.pkl'), 'wb') as f:
            pickle.dump(normalized_dict, f)

        for c in cols:
            if c == "WindSpeed":
                for t in self.list_turbine_name:
                    arr = getattr(self, t)[c].to_numpy(copy=False)
                    low_q, high_q = 0.33, 0.66
                    low_thresh = np.nanquantile(arr, low_q)
                    high_thresh = np.nanquantile(arr, high_q)
                    print(t)
                    print(c)
                    print(f"low, high: {low_thresh, high_thresh}\n")
        
        print(f"Global statistic for min, max normalization: {global_stats}")

    def normalize_mean_SD(self, normalized_dict, cols):
        # 1) compute global mean/SD per column WITHOUT pd.concat
        global_stats = {}
        for c in cols:
            # accumulate sums & sums of squares across turbines
            total_sum, total_sq, total_n = 0.0, 0.0, 0
            for t in self.list_turbine_name:
                arr = getattr(self, t)[c].to_numpy(copy=False, dtype=np.float32)
                mask = ~np.isnan(arr)
                vals = arr[mask]
                total_sum += vals.sum()
                total_sq  += (vals**2).sum()
                total_n   += vals.size
            mean = total_sum / total_n
            # var = E[x^2] - E[x]^2
            var  = total_sq/total_n - mean*mean
            sd   = np.sqrt(var)
            global_stats[c] = (mean, sd)

        # 2) apply normalization in one block per turbine
        for t in self.list_turbine_name:
            df = getattr(self, t)
            normalized_dict[t] = {c: global_stats[c] for c in cols}

            block = df[cols].to_numpy(copy=False, dtype=np.float32)
            means = np.array([global_stats[c][0] for c in cols], dtype=np.float32)
            sds   = np.array([global_stats[c][1] for c in cols], dtype=np.float32)
            block -= means
            block /= sds

            df.loc[:, cols] = block
            setattr(self, t, df)

        with open(os.path.join(self.BASE_DIR, 'data_normalisation',
                            f'std_mean_normalisation_{self.dataset_name}.pkl'), 'wb') as f:
            pickle.dump(normalized_dict, f)
    
        print(f"Global statistic for mean, standard deviation normalization: {global_stats}")


    def circ_embedding(self, cols):
        # do each turbine, each col, on numpy arrays
        for t in self.list_turbine_name:
            df = getattr(self, t)
            for c in cols:
                arr = df[c].to_numpy(copy=False, dtype=np.float32)
                # compute in one go
                radians = arr * (2 * np.pi / 360.0)
                cosines = np.cos(radians)
                sines   = np.sin(radians)
                # assign new columns (avoiding pandas interim copies)
                df[f"{c}_cos"] = cosines
                df[f"{c}_sin"] = sines
            setattr(self, t, df)

    def drop_col(self, cols):
        for t in self.list_turbine_name:
            getattr(self,t).drop(columns = cols, inplace = True)

    def time_filling(self, method='linear', interpolation_limit=6):
        for name in self.list_turbine_name:
            # re index to 10min
            df = getattr(self, name).asfreq("10min")
            setattr(self, name, df)

            # check gaps longer than limit
            nan_mask = df.isna().any(axis=1)
            group_id = (nan_mask != nan_mask.shift(1)).cumsum()
            run_lengths = nan_mask.groupby(group_id).sum()
            too_long_runs = run_lengths[run_lengths > interpolation_limit]
            if not too_long_runs.empty:
                print(f"Too-long NaN runs in {name} (group_id → length):")
                print(too_long_runs)

            df.interpolate(method=method,
                        limit=interpolation_limit,
                        inplace=True)

            print(f"After fill, {name} shape:", df.shape)

    # @profile
    def make_window_data(self, in_shift, turbine = None, ret = False):
        turbine = turbine if turbine is not None else self.list_turbine_name 
        
        for t in turbine: 
            print(f"Preparing turbine {t} windows")
        
            data = getattr(self,t)

            in_data = data[self.in_cols].to_numpy(dtype=np.float32, copy = True)
            out_data = data[self.out_cols].to_numpy(dtype=np.float32, copy = True)
            
            index = data.index
            daterange = pd.date_range(start=index[0], end=index[-1], freq = "10min")

            # Creating blocks of timestamps without missing values
            blocks = [[]]
            i = 0
            for d in tqdm(daterange):
                if d in data.index:
                    blocks[i].append(d)
                elif blocks[-1] != []:
                    blocks.append([])
                    i+=1
            if blocks[-1] == []:
                blocks = blocks[:-1]

            WINDOW_CHUNK = 10_000
            nan_indices_removed = 0

            for block_idx, block in enumerate(blocks):
                n_windows = len(block) - in_shift
                if n_windows <= 0:
                    continue

                try:
                    for idx, start in enumerate(tqdm(range(0, n_windows, WINDOW_CHUNK))):

                        end = min(start + WINDOW_CHUNK, n_windows)

                        # pull arrays for rows in chunk
                        rows = block[start : end + in_shift]
                        arr_in  = data.loc[rows, self.in_cols].to_numpy(dtype=np.float32)
                        arr_out = data.loc[rows, self.out_cols].to_numpy(dtype=np.float32)

                        # build sliding windows along axis=0
                        windows_in = sliding_window_view(arr_in, window_shape=in_shift+1, axis=0)
                        # for window ending with row i+in_shift, get i+in_shift output
                        windows_out = arr_out[in_shift:]

                        # mask out any window that contains a nan for any feature and skip chunk
                        # if there are no valid rows
                        valid = ~np.isnan(windows_in).any(axis=(1,2))
                        n_good = valid.sum()    
                        n_total = windows_in.shape[0]
                        nan_indices_removed += (n_total - n_good)
                        if not valid.any():
                            continue

                        filtered_in  = windows_in[valid]
                        filtered_out = windows_out[valid]
                        ts_chunk = np.array(rows[in_shift:])[valid].tolist()

                        Xb = ch.tensor(filtered_in).permute(0, 2, 1)
                        yb = ch.tensor(filtered_out)

                        turbine_folder = os.path.join(self.WINDOWED_DATA_PATH, f"{t}")
                        os.makedirs(turbine_folder, exist_ok=True)
                        base = os.path.join(turbine_folder, f"{self.dataset_name}_{t}")

                        ch.save(Xb, f"{base}_X_{idx:02d}.pt")
                        ch.save(yb, f"{base}_y_{idx:02d}.pt")
                        with open(f"{base}_ts_{idx:02d}.pkl", "wb") as f:
                            pickle.dump(ts_chunk, f)

                except OSError:
                    continue
            print(f"nan indices removed for {t}: {nan_indices_removed}")


def process_data(
        dataset,
        BASE_DIR,
        predicted_property,  
        exclude_fault_data,
        cols, 
        WT_IDs,
        rename
    ):
    print(f"Processing data for {dataset} {'WITH' if exclude_fault_data else 'WITHOUT'} faults.")
    print(f"The predicted property is {predicted_property}")

    DATA_PATH = os.path.join(BASE_DIR, dataset)
    
    if os.path.exists(DATA_PATH):
        for path in os.listdir(DATA_PATH):
            df = pd.read_csv(os.path.join(DATA_PATH, path))
    
    # TODO: fix logic to be one call to data_farm
    if dataset == 'EDP':
        farm_edp = data_farm(
            dataset_name=dataset, 
            cols=cols,
            fault_check_cols = ['GenSpeed', 'RotorSpeed', 'Power'],
            list_turbine_name=WT_IDs,
            BASE_DIR=BASE_DIR,
            predicted_property=predicted_property,
            rename=rename, 
            exclude_fault_data=exclude_fault_data,
        )
        print("\nPreparing data windows")
        print(f"Saving data to {farm_edp.CLEANED_DATA_PATH}")
        farm_edp.make_window_data(in_shift=24*6)
        return farm_edp.in_cols
    
    # TODO: setup the renaming of variables in here not in the preprocess
    elif dataset == 'Scotland': 
        farm_scotland = data_farm(
            dataset_name=dataset, 
            cols=cols, 
            fault_check_cols=['MainShaftRPM', 'Power'],
            list_turbine_name=WT_IDs,
            BASE_DIR=BASE_DIR,
            predicted_property=predicted_property,
            exclude_fault_data=exclude_fault_data,
        )
        print("\nPreparing data windows")
        print(f"Saving data to {farm_scotland.CLEANED_DATA_PATH}")
        farm_scotland.make_window_data(in_shift=24*6)
        return farm_scotland.in_cols
    else: 
        print(f"Invalid dataset {dataset} specified. Only EDP and Scotland currently implemented")
        

class preprocess_data_farm:
    def __init__(
            self, 
            dataset=str, 
            base_dir=str, 
            years=list[int],
            rename=Dict[str, Any]
        ):
        """
            dataset: Which dataset of wind turbine data is being used
            base_dir: Where the dataset is located
            years: List of years that the data spands
        """
        self.dataset = dataset
        self.base_dir = base_dir
        self.years = years
        self.rename = rename

        self.base_url = "https://www.edp.com/sites/default/files/2023-04/" if self.dataset == "EDP" else "https://zenodo.org/records/14870023/files"
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.descriptions_dir = os.path.join(self.base_dir, "descriptions")
        self.raw_dir = os.path.join(self.base_dir, "raw_data")
        self.fault_dir = os.path.join(self.base_dir, "fault_logs")
        self.zip_archive_dir = os.path.join(self.base_dir, "zips")
        self.aggregate_dir = os.path.join(self.base_dir, "aggregate")
        self.daily_stats = os.path.join(self.base_dir, "daily_stats")

        self.max_workers_available = get_available_cpus()

        supported_datasets = ["EDP", "Scotland"]
        if self.dataset not in supported_datasets: 
            Exception("Invalid dataset {self.dataset}. {supported_datasets} datasets supported")
        print(f"\nPreprocessing dataset={self.dataset}")

    def preprocess(self):
        self.make_dirs()
        if self.dataset == "EDP": 
            self.get_EDP_dataset()
        elif self.dataset == "Scotland": 
            self.get_Scotland_dataset()

    def make_dirs(self):
        for d in (self.temp_dir, self.descriptions_dir, self.raw_dir, 
                    self.fault_dir, self.zip_archive_dir, self.aggregate_dir,
                    self.daily_stats):
            os.makedirs(d, exist_ok=True)
    
    def get_EDP_dataset(self):
        normal_urls = {
            2016: os.path.join(self.base_url, "Wind-Turbine-SCADA-signals-2016.xlsx"),
            2017: os.path.join(self.base_url, "Wind-Turbine-SCADA-signals-2017_0.xlsx")
        }
        fault_urls = {
            2016: os.path.join(self.base_url, "Historical-Failure-Logbook-2016.xlsx"),
            2017: os.path.join(self.base_url, "opendata-wind-failures-2017.xlsx")
        }

        all_urls: dict[str,str] = {}
        for yr in self.years:
            if yr not in normal_urls or yr not in fault_urls:
                raise KeyError(f"No URLs configured for year {yr}")
            all_urls[f"EDP_{yr}.xlsx"]         = normal_urls[yr]
            all_urls[f"EDP_faults_{yr}.xlsx"]  = fault_urls[yr]

        for fname, url in all_urls.items():
            dest = os.path.join(self.temp_dir, fname)
            print(f"Downloading {fname}")
            resp = requests.get(url, allow_redirects=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)

        for xlsx_fname in all_urls:
            csv_fname = xlsx_fname.replace(".xlsx", ".csv")
            print(f"↳ Converting {xlsx_fname} → {csv_fname}")
            Xlsx2csv(
                os.path.join(self.temp_dir, xlsx_fname),
                outputencoding="utf-8"
            ).convert(os.path.join(self.temp_dir, csv_fname))

        df_list = []
        for csv_fname in [f.replace(".xlsx", ".csv") for f in all_urls]:
            full_path = os.path.join(self.temp_dir, csv_fname)
            print(f"Reading {csv_fname}")
            df = pd.read_csv(full_path, parse_dates=["Timestamp"])
            if "faults" in csv_fname:
                df = df.drop(columns=["Component", "Remarks"], errors="ignore")
                df["data_type"] = "fault"
            else:
                df["data_type"] = "normal"
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)

        for turb in df["Turbine_ID"].unique():
            for dtype in ("normal", "fault"):
                sub = df[(df["Turbine_ID"] == turb) & (df["data_type"] == dtype)]
                if sub.empty:
                    continue

                tnum = turb[-2:]
                if dtype == "normal":
                    out_dir = self.raw_dir
                    print(f"Saving normal data for WT_{tnum}")
                else:
                    out_dir = self.fault_dir
                    print(f"Saving fault logs for WT_{tnum}")

                out_path = os.path.join(out_dir, f"WT_{tnum}.csv")
                sub.to_csv(out_path, index=False)


    def get_Scotland_dataset(self):

        missing_years = []
        downloaded_years = []
        for year in self.years:
            filename = f"{year}.zip"
            if not os.path.isfile(os.path.join(self.zip_archive_dir, filename)):
                missing_years.append(year)
            else: 
                downloaded_years.append(year)
        
        if missing_years: 
            print(f"{missing_years} not archived ... downloading ...")
            with ThreadPoolExecutor(max_workers=self.max_workers_available) as executor:
                futures = {
                    executor.submit(self.download_zip_files, yr): yr
                    for yr in missing_years
                }
                for fut in as_completed(futures):
                    yr = futures[fut]
                    try:
                        fut.result()
                        print(f"Completed year {yr}")
                    except Exception as e:
                        print(f"Error processing {yr}: {e}")

        if downloaded_years: 
            print(f"{downloaded_years} already archived ... moving to temp dir ...")
            with ThreadPoolExecutor(max_workers=self.max_workers_available) as executor:
                futures = {
                    executor.submit(self.move_archived_zip_files, yr): yr
                    for yr in downloaded_years
                }
                for fut in as_completed(futures):
                    yr = futures[fut]
                    try:
                        fut.result()
                        print(f"Completed moving year {yr} zip")
                    except Exception as e:
                        print(f"Error moving {yr}: {e}")

        print("Downloading global files…")
        with ThreadPoolExecutor(max_workers=self.max_workers_available) as exe:
            futures = {
                exe.submit(self.process_global_file, name, suffix): name 
                for name, suffix in GLOBAL_FILE_URLS.items()
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                    print(f"[✓] Done {name}")
                except Exception as e:
                    print(f"[✕] {name} failed: {e}")

        print("Unpacking yearly ZIPs…")
        with ThreadPoolExecutor(max_workers=self.max_workers_available) as exe:
            futures = {
                exe.submit(self.process_year_zip, yr): yr 
                for yr in self.years
            }
            for fut in as_completed(futures):
                yr = futures[fut]
                try:
                    fut.result()
                    print(f"[✓] Year {yr} done")
                except Exception as e:
                    print(f"[✕] Year {yr} failed: {e}")
                        
        print("Concatenating CSVs by folder…")
        with ThreadPoolExecutor(max_workers=self.max_workers_available) as executor:
            futures = {
                executor.submit(self.concatenate_csv_data, yr, self.rename): yr
                for yr in self.years
            }
        for fut in as_completed(futures):
            yr = futures[fut]
            try:
                fut.result()
                print(f"[✓] Year {yr} done")
            except Exception as e:
                print(f"[✕] Year {yr} error: {e}")

        temp_aggregate_dir = self.merge_and_clean_data()
        self.split_features_by_turbine(
            input_file = "Features.csv",
            temp_aggregate_dir=temp_aggregate_dir, 
            save_dir=self.raw_dir
        )

        os.makedirs(os.path.join(self.fault_dir, "alarms"), exist_ok=True)
        self.split_features_by_turbine(
            input_file = "Alarm_logs.csv",
            temp_aggregate_dir=temp_aggregate_dir, 
            save_dir=os.path.join(self.fault_dir, "alarms")
        )

        self.split_features_by_turbine(
            input_file = "Daily_stats.csv",
            temp_aggregate_dir=temp_aggregate_dir, 
            save_dir=self.daily_stats
        )
        shutil.rmtree(self.temp_dir)


    def download_zip_files(
        self,
        year: int,
    )-> None:
            
        url = os.path.join(self.base_url, f"{year}.zip?download=1")
        temp_year_dir = os.path.join(self.base_dir, "temp", str(year))
        zip_path = os.path.join(temp_year_dir, f"{year}.zip")

        os.makedirs(temp_year_dir, exist_ok=True)

        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="MB", unit_scale=True, desc=f"Downloading {year}"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                pbar.update(len(chunk))


    def move_archived_zip_files(
        self,
        year: int,
    )-> None:
            
        temp_year_dir = os.path.join(self.base_dir, "temp", str(year))
        zip_path = os.path.join(temp_year_dir, f"{year}.zip")
        os.makedirs(temp_year_dir, exist_ok=True)
    
        shutil.move(src=os.path.join(self.zip_archive_dir, f"{year}.zip"), dst=temp_year_dir)


    def process_global_file(
        self,
        name: str, 
        suffix: str
    )-> str:
            
        download_url = os.path.join(self.base_url, suffix)
        is_desc = "description" in name.lower()
        out_dir = self.descriptions_dir if is_desc else self.temp_dir

        ext = ".zip" if suffix.lower().endswith(".zip?download=1") else ".csv"
        filename = f"{name}{ext}"
        out_path = os.path.join(out_dir, filename)

        resp = requests.get(download_url, allow_redirects=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)

        # special hooks:
        if name == "Shutdown_dates":
            with zipfile.ZipFile(out_path, 'r') as zf:
                zf.extractall(self.fault_dir)
            os.remove(out_path)
        elif name == "Upgrade_dates":
            shutil.move(src=out_path, dst=os.path.join(self.fault_dir, filename))

        return name


    def process_year_zip(
        self, 
        year: int
    )-> int:

        year_folder = os.path.join(self.temp_dir, str(year))
        year_zip = os.path.join(year_folder, f"{year}.zip")
        if not os.path.exists(year_zip):
            raise FileNotFoundError(f"No zip for {year} at {year_zip}")

        with zipfile.ZipFile(year_zip, "r") as zf:
            for info in zf.infolist():
                if not info.filename.lower().endswith(".csv"):
                    continue

                base_filename = os.path.basename(info.filename)
                for key, subf in KEEP_FILES.items():
                    if base_filename.startswith(key):
                        dest_dir = os.path.join(year_folder, subf)
                        os.makedirs(dest_dir, exist_ok=True)

                        parts = base_filename.split("_")
                        new_name = "_".join(parts[2:]) if len(parts) >= 3 else base_filename
                        dest_path = os.path.join(dest_dir, new_name)

                        extracted = zf.extract(info, year_folder)
                        shutil.move(extracted, dest_path)
                        break

        # archive the zip so temp only holds your extracted CSVs
        archived = os.path.join(self.zip_archive_dir, f"{year}.zip")
        shutil.move(year_zip, archived)
        return year


    def concatenate_csv_data(
        self,
        year: int,
        rename_map: dict, 
    ) -> None:
            
        print(f"Year: {year}")
        temp_location = os.path.join(self.base_dir, 'temp', str(year))
        temp_aggregate_dir = os.path.join(self.base_dir, 'temp', 'aggregate')

        os.makedirs(temp_aggregate_dir, exist_ok=True)

        subfolders = os.listdir(temp_location)
        
        for folder in subfolders:
            print(f"subfolder: {folder}")
            subfolder = os.path.join(temp_location, folder)
            output_csv = os.path.join(temp_aggregate_dir, f"{folder}_{year}.csv")

            if os.path.exists(output_csv):
                os.remove(output_csv)
            csv_files = glob.glob(os.path.join(subfolder, '*.csv'))
        
            if not csv_files:
                print(f"No CSV files found in {subfolder}")
                return
        
            data_frames = []
            
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                keep_cols = [col for col in rename_map[folder].keys() if col in df.columns]
                
                if not keep_cols:
                    print(f"None of the specified columns found in {csv_file}. Skipping.")
                    continue
                
                df = df[keep_cols]
                df.rename(columns=rename_map[folder], inplace=True)
                data_frames.append(df)
        
            if data_frames:
                merged_df = pd.concat(data_frames, ignore_index=True)
                merged_df.to_csv(output_csv, index=False)
                print(f"Concatenated CSV saved to {output_csv}")
                print()
            else:
                print("No data frames to concatenate. Check the CSV files and your rename_map.")
            
            shutil.rmtree(subfolder)

    def merge_and_clean_data(
        self, 
    ) -> None:
            
        temp_aggregate_dir = os.path.join(self.base_dir, 'temp', 'aggregate')
            
        datasets = list(KEEP_FILES.values())
        aggregated: dict[str, pd.DataFrame] = {}

        def _aggregate_one(dataset: str) -> tuple[str, pd.DataFrame|None]:
            files = []
            for yr in self.years:
                fp = os.path.join(self.temp_dir, "aggregate", f"{dataset}_{yr}.csv")
                if os.path.exists(fp):
                    files.append(fp)
                else:
                    print(f"Missing {dataset}_{yr}.csv — skipping that year")
            if not files:
                print(f"No files to aggregate for {dataset}")
                return dataset, None

            dfs = [pd.read_csv(f) for f in files]
            df_all = pd.concat(dfs, ignore_index=True)
            print(df_all.shape)
            df_all.drop_duplicates(inplace=True)
            print(df_all.shape)

            out_csv = os.path.join(temp_aggregate_dir, f"{dataset}.csv")
            df_all.to_csv(out_csv, index=False)
            return dataset, df_all

        with ThreadPoolExecutor(max_workers=self.max_workers_available) as exe:
            futures = {exe.submit(_aggregate_one, ds): ds for ds in datasets}
            for fut in as_completed(futures):
                ds = futures[fut]
                try:
                    ds_name, df = fut.result()
                    if df is not None:
                        aggregated[ds_name] = df
                    print(f"[✓] Dataset {ds} done")
                except Exception as e:
                    print(f"[✕] Aggregation failed for {ds}: {e}")

        # TODO: add in logic here to also rename the turbines for the other two files
        # merge input features
        op_key, tmp_key, pwr_key = "Operational_stats", "Temperature_stats", "Power_stats"
        if op_key in aggregated and tmp_key in aggregated and pwr_key in aggregated:
            op_df, tmp_df, pwr_df = aggregated[op_key], aggregated[tmp_key], aggregated[pwr_key]
            join_cols = {"Timestamp","TurbineID"}

            if join_cols.issubset(op_df.columns) and join_cols.issubset(tmp_df.columns) and join_cols.issubset(pwr_df.columns):
                merged = op_df.merge(tmp_df, on=["Timestamp","TurbineID"], how="left", suffixes=("", f"_{tmp_key}"))
                merged = merged.merge(pwr_df, on=["Timestamp","TurbineID"], how="left", suffixes=("", f"_{tmp_key}"))

                merged = self._change_WT_IDs(merged)

                combined_path = os.path.join(
                    temp_aggregate_dir,
                    "Features.csv"
                )
                merged.to_csv(combined_path, index=False)

                # delete redundant files
                for old in (f"{op_key}.csv", f"{tmp_key}.csv"):
                    oldp = os.path.join(temp_aggregate_dir, old)
                    if os.path.exists(oldp):
                        os.remove(oldp)
            else:
                print("Missing join columns, skipping combined merge.")
        else:
            print(f"Cannot merge: missing {op_key} or {tmp_key} or {pwr_key}.")

        return temp_aggregate_dir


    def split_features_by_turbine(
        self,
        input_file: str,
        temp_aggregate_dir: str,
        save_dir: str
    ) -> None:

        input_file_path = os.path.join(temp_aggregate_dir, input_file)
        if not os.path.isfile(input_file_path):
            raise FileNotFoundError(f"{input_file} not found at {input_file_path}")

        df = pd.read_csv(input_file_path, parse_dates=True)
        df['TurbineID'] = df['TurbineID'].astype(str)
        
        for turbine in sorted(df["TurbineID"].unique()):
            sub = df[df["TurbineID"] == turbine]
            

            out_fname = f"{turbine}.csv"
            out_path  = os.path.join(save_dir, out_fname)

            sub.to_csv(out_path, index=False)
            print(f"Saved {len(sub)} rows for turbine {turbine} → {out_path}")

    def _change_WT_IDs(
        self,
        df: pd.DataFrame,
        id_mapping: dict[int, str] = None
    ) -> pd.DataFrame:
        if id_mapping is None:
            # pull unique numeric IDs
            unique_ids = df['TurbineID'].unique().astype(int)
            # drop the faulty 91
            unique_ids = unique_ids[unique_ids != 91]

            id_mapping = {
                old: f"WT_{i+1:02d}"
                for i, old in enumerate(sorted(unique_ids))
            }

        df['TurbineID'] = df['TurbineID'].map(id_mapping)
        df = df.dropna(subset=['TurbineID'])
        return df