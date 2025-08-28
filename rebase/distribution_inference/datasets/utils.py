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


from distribution_inference.datasets import new_census, celeba, boneage,census, texas, arxiv, maadface, lstm_WT_new

DATASET_INFO_MAPPING = {
    "new_census": new_census.DatasetInformation,
    "celeba": celeba.DatasetInformation,
    "boneage": boneage.DatasetInformation,
    "old_census": census.DatasetInformation,
    "texas": texas.DatasetInformation,
    "arxiv": arxiv.DatasetInformation,
    "maadface": maadface.DatasetInformation,
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
        "raw_data", "data_prep_faults_excluded", "fault_logs"
    ]
    for path in paths:
        full_path = os.path.join(base_dir, path)
        os.makedirs(full_path, exist_ok=True)


# TODO: actually pass in the keep directories instead of hardcoding
def cleanup_dirs(base_dir, delete=["aggregate", "data_prep", "data_prep_faults_excluded"]):
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
            cols, 
            fault_check_cols,
            list_turbine_name, 
            BASE_DIR, 
            sensitive_property,
            predicted_property,
            rename = None,
            exclude_fault_data = False,
        ):
        self.fault_check_cols = fault_check_cols        
        
        self.base_in_cols = [col for col in cols
                            if col not in predicted_property
                            and col != "Timestamp"]
        self.out_cols = predicted_property
        self.sensitive = sensitive_property
        self.keep_cols = self.out_cols + self.base_in_cols + [self.sensitive]

        self.list_turbine_name = list_turbine_name
        self.exclude_fault_data = exclude_fault_data

        # TODO: make it so only the sensitive columns are stored and then appended to the base data when they are used
        # makes it so edits can be made on them while running other code, but so that repetitive data is not being stored
        # unnecessarily
        self.BASE_DIR = BASE_DIR
        self.WINDOWED_BASE = "/scratch/zl2da"
        suffix = sensitive_property if not exclude_fault_data else f"{sensitive_property}_faults_excluded"
        self.WINDOWED_DATA_PATH = os.path.join(self.WINDOWED_BASE, suffix)

        # TODO: clean up this check and put it somewhere else
        os.makedirs(self.WINDOWED_DATA_PATH, exist_ok=True)

        for t in self.list_turbine_name:
            print(f"Initializing {t}")
            self.initialize_data(t, rename)
        
        self.prepare_scotland_data()
        self.X, self.y, self.timestamps = {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}

    def initialize_data(self, t, rename):
        setattr(self, t, self.load_df(f"{t}.csv", rename=rename, exclude_fault_data=self.exclude_fault_data))
        assert "Timestamp" in getattr(self, t).columns, f"Need a timestamp column"
        assert "WindSpeed" in getattr(self, t).columns, f"Need a WindSpeed column"
        
        index = pd.to_datetime(getattr(self, t)["Timestamp"]).dt.tz_convert(None) 
        getattr(self, t).set_index(index, inplace=True)
        getattr(self, t).drop(["Timestamp"], axis=1, inplace=True)

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
        wt_df = wt_df[[c for c in ["Timestamp"]+self.keep_cols if c in wt_df.columns]]

        # Whether or not include faulty data periods in the dataset
        if self.exclude_fault_data:
            fault_df = pd.read_csv(os.path.join(self.BASE_DIR, "fault_logs", file_name), parse_dates=["Timestamp"])

            wt_df.set_index("Timestamp", inplace=True)
            for fault_time in fault_df['Timestamp']:
                start_time = fault_time - pd.Timedelta(weeks=2)
                end_time = fault_time + pd.Timedelta(days=3)
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.base_in_cols] = np.nan
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.out_cols] = np.nan
        
        # cut in depends on the dataset
        cut_in_wind_speed = 3.5
        for col in self.fault_check_cols:
            zero_col_with_wind = (wt_df[col] == 0) & (wt_df["WindSpeed"] > cut_in_wind_speed)
            if zero_col_with_wind.any():
                wt_df.loc[zero_col_with_wind, self.base_in_cols] = np.nan
                wt_df.loc[zero_col_with_wind, self.out_cols] = np.nan
            else:
                print(f"No occurrences found for {col}.")
            
        wt_df.reset_index(drop=False, inplace=True)

        return wt_df 

    
    def create_new_feature(self, new_feature):
        valid = ["Speed", "Turbulence", "MisalignmentLoss", "Time_Availability", "Capacity", "Energy_Availability"]
        if new_feature not in valid:
            raise ValueError(f"Invalid feature {new_feature!r}, must be one of {valid}")

        for t in self.list_turbine_name:
            df = getattr(self, t)

            if new_feature == "Speed": 
                df[new_feature] = df["WindSpeed"].copy()

            elif new_feature == "Turbulence": 
                df[new_feature] = df["WindSpeed_SD"]/df["WindSpeed"]

            elif new_feature == "MisalignmentLoss":
                def _angular_diff(a, b):
                    return (a - b + 180) % 360 - 180

                wind_dir = df["WindDirection"].to_numpy(copy=False)
                yaw = df["Yaw"].to_numpy(copy=False)

                signed_err = _angular_diff(wind_dir, yaw)
                misalignment = np.abs(signed_err)
                power_loss = 1 - np.cos(misalignment)**2
                df[new_feature] = power_loss
            
            elif new_feature == "Time_Availability":
                CUT_IN  = 3.5
                CUT_OUT = 25.0

                df[new_feature] = 0
                within_range = (df["WindSpeed"] >= CUT_IN) & (df["WindSpeed"] <= CUT_OUT)
                producing = df["Power"] > 0
                df.loc[within_range & producing, new_feature] = 1
                
            elif new_feature == "Capacity":
                ratio = np.zeros(len(df), dtype=float)
                mask = df["TargetPower"].to_numpy(copy=False) != 0
                ratio[mask] = (df.loc[mask, "Power"].to_numpy(copy=False)
                            / df.loc[mask, "TargetPower"].to_numpy(copy=False))
                df[new_feature] = ratio

            elif new_feature == "Energy_Availability": 
                farm_curve = {
                    0.5:   -10.113101,
                    1.0:   -10.089789,
                    1.5:   -10.350299,
                    2.0:   -11.976830,
                    2.5:   -13.729584,
                    3.0:   -12.103409,
                    3.5:     2.671766,
                    4.0:    42.013350,
                    4.5:    83.540082,
                    5.0:   126.814061,
                    5.5:   180.838161,
                    6.0:   248.855580,
                    6.5:   330.141233,
                    7.0:   421.214043,
                    7.5:   522.196606,
                    8.0:   630.285105,
                    8.5:   748.798986,
                    9.0:   871.901906,
                    9.5:   997.448546,
                    10.0:  1121.747472,
                    10.5:  1246.951706,
                    11.0:  1361.611444,
                    11.5:  1477.664036,
                    12.0:  1581.765787,
                    12.5:  1661.978351,
                    13.0:  1736.676688,
                    13.5:  1786.834080,
                    14.0:  1801.606437,
                    14.5:  1806.325448,
                    15.0:  1794.770038,
                    15.5:  1779.369799,
                    16.0:  1781.090434,
                    16.5:  1762.467279,
                    17.0:  1771.479803,
                    17.5:  1801.986146,
                    18.0:  1785.930135,
                    18.5:  1800.477394,
                    19.0:  1797.501233,
                    19.5:  1843.313478,
                    20.0:  1856.970193,
                    20.5:  1879.774873,
                    21.0:  1862.695658,
                    21.5:  1849.695686,
                    22.0:  1825.765791,
                    22.5:  1788.374110,
                    23.0:  1707.553307,
                    23.5:  1683.381103,
                    24.0:  1707.683609,
                    24.5:  1577.414376,
                    25.0:  1415.657470,
                    25.5:  1262.700953,
                }

                idx = np.floor(df["WindSpeed"] / 0.5)
                df["ws_mid"] = (idx + 1) * 0.5
                df["expected_power"] = df["ws_mid"].map(farm_curve)
                df["Energy_Availability"] = (df["Power"] / df["expected_power"])
                df.drop(columns=["ws_mid", "expected_power"], inplace=True)

            else:
                assert False, f"Error: {new_feature} not implemented to be created."

            setattr(self, t, df)

    def prepare_scotland_data(self): 
        circular_cols = ["Yaw", "SetYaw"]
        norm_SD_cols = [c for c in self.keep_cols if c.endswith("_SD")]
        norm_mean_cols = [
            c for c in self.keep_cols
            if c not in circular_cols
            and c not in norm_SD_cols
            and c != self.sensitive
        ]

        self.drop_col([
            c for c in getattr(self, self.list_turbine_name[0]).columns
            if c not in self.keep_cols
        ])

        if self.sensitive not in self.base_in_cols: 
            self.base_in_cols.append(self.sensitive)
            self.create_new_feature(self.sensitive)

        # feature cuts for outliers
        self.feature_cut("WindSpeed", min=0, max=25, min_value=0)
        self.feature_cut("Power", min=0, max=2300, min_value=0, max_value=2300)
        self.feature_cut("Yaw", min=0, min_value=0, max_value=360)
        self.feature_cut("GearOilTemp", min=0, max=52)
        self.feature_cut("GearboxBearingTempHSGenSide", min=0, max=65)
        self.feature_cut("GearboxBearingTempHSRotorSide", min=0, max=65)

        # load or recompute global min/max stats
        pkl_path = os.path.join(
            self.BASE_DIR, 'data_normalisation',
            f'feature_stats_Scotland.pkl'
        )

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                stored = pickle.load(f)
            stats = stored.get('stats', {})
            # if any requested col is missing, recompute all
            needed = set(norm_mean_cols + norm_SD_cols)
            if needed - stats.keys():
                stats = self.get_and_save_feature_stats(norm_mean_cols + norm_SD_cols)
        else:
            stats = self.get_and_save_feature_stats(norm_mean_cols + norm_SD_cols)

        self.stats = stats

        # remove any duplicate indices
        for t in self.list_turbine_name:
            df = getattr(self, t)
            setattr(self, t, df[~df.index.duplicated()])

        self.time_filling(interpolation_limit = 6)

        # normalize data
        normalized_dict = {}
        self.normalize_min_max(normalized_dict=normalized_dict,cols=norm_mean_cols)
        self.normalize_mean_SD(normalized_dict=normalized_dict,cols=norm_SD_cols)
        self.circ_embedding(circular_cols)

    def get_and_save_feature_stats(self, stat_cols):
        stats = {}
        for c in tqdm(stat_cols, desc="Calculating global min/max per feature"):
            mins, maxs = [], []
            for t in self.list_turbine_name:
                arr = getattr(self, t)[c].to_numpy(copy=False)
                mins.append(np.nanmin(arr))
                maxs.append(np.nanmax(arr))
            stats[c] = (min(mins), max(maxs))

        out_dir = os.path.join(self.BASE_DIR, 'data_normalisation')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'feature_stats_Scotland.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'stats': stats}, f)

        return stats

    def normalize_min_max(self, normalized_dict, cols):
        global_stats = {c: self.stats[c] for c in cols}

        for t in self.list_turbine_name:
            df = getattr(self, t)
            normalized_dict[t] = {c: global_stats[c] for c in cols}

            block = df[cols].to_numpy(copy=False, dtype=np.float32)
            mins = np.array([global_stats[c][0] for c in cols], dtype=np.float32)
            maxs = np.array([global_stats[c][1] for c in cols], dtype=np.float32)
            
            block -= mins
            block /= (maxs - mins)
            df.loc[:, cols] = block
            setattr(self, t, df)
        
        with open(os.path.join(self.BASE_DIR, 'data_normalisation',
                               f'min_max_normalisation_Scotland.pkl'),
                  'wb') as f:
            pickle.dump(normalized_dict, f)


    def normalize_mean_SD(self, normalized_dict, cols):
        global_stats = {c: self.stats[c] for c in cols}

        for t in self.list_turbine_name:
            df = getattr(self, t)
            normalized_dict[t] = {c: global_stats[c] for c in cols}

            block  = df[cols].to_numpy(copy=False, dtype=np.float32)
            means  = np.array([global_stats[c][0] for c in cols], dtype=np.float32)
            sds    = np.array([global_stats[c][1] for c in cols], dtype=np.float32)
            block -= means
            block /= sds

            df.loc[:, cols] = block
            setattr(self, t, df)

        with open(os.path.join(self.BASE_DIR, 'data_normalisation',
                               f'std_mean_normalisation_Scotland.pkl'),
                  'wb') as f:
            pickle.dump(normalized_dict, f)

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

    def feature_cut(self, feature, min = None, max = None, min_value = "drop", max_value = "drop", turbine_name = None, verbose = False):
        turbine = turbine_name if turbine_name is not None else self.list_turbine_name

        for t in turbine:
            df: pd.DataFrame = getattr(self, t)
            assert feature in df.columns, f"Wrong feature name: {feature}"
            orig_len = len(df)

            series = df[feature]

            if min is not None:
                mask_min = series < min
                if min_value == "drop":
                    df.drop(index=df.index[mask_min], inplace=True)
                else:
                    df.loc[mask_min, feature] = min_value
            
            series = df[feature]
            if max is not None:
                mask_max = series > max
                if max_value == "drop":
                    df.drop(index=df.index[mask_max], inplace=True)
                else:
                    df.loc[mask_max, feature] = max_value
            
            if verbose:
                print(f"Cutted {orig_len - len(df)} rows from {t} for {feature}")
            setattr(self, t, df)

    def line_cut(self, input_feature, output_feature,
                a, b, xmin, xmax, under=True, turbine_name=None):

        turbines = turbine_name or self.list_turbine_name

        for t in turbines:
            df = getattr(self, t)
            if under:
                f = lambda x: (x[input_feature] <= xmin
                            or a*x[input_feature] + b < x[output_feature]
                            or x[input_feature] >= xmax)
            else:
                f = lambda x: (x[input_feature] <= xmin
                            or a*x[input_feature] + b > x[output_feature]
                            or x[input_feature] >= xmax)

            keep = df.apply(f, axis=1)
            dropped = (~keep).sum()
            total = len(df)
            print(f"Turbine {t}: dropping {dropped}/{total} rows")

            setattr(self, t, df[keep])

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
                self.base_in_cols.remove(remove_col)
        self.base_in_cols.append(name)

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


    def make_window_data(self, in_shift, turbine = None, ret = False):
        turbine = turbine if turbine is not None else self.list_turbine_name 
        
        for t in turbine: 
            print(f"Preparing turbine {t} windows")
        
            data = getattr(self,t)

            in_data = data[self.base_in_cols].to_numpy(dtype=np.float32, copy = True)
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

                #print("before try block!!!!!!!!!!!!!")
                #print("n windoes = ", n_windows)
                #print("window_chunk = ", WINDOW_CHUNK)
                #time.sleep(1000)
                try:
                    for idx, start in enumerate(tqdm(range(0, n_windows, WINDOW_CHUNK))):

                        end = min(start + WINDOW_CHUNK, n_windows)

                        # pull arrays for rows in chunk
                        rows = block[start : end + in_shift]
                        arr_in  = data.loc[rows, self.base_in_cols].to_numpy(dtype=np.float32)
                        arr_out = data.loc[rows, self.out_cols].to_numpy(dtype=np.float32)

                        # build sliding windows along axis=0
                        windows_in = sliding_window_view(arr_in, window_shape=in_shift+1, axis=0)
                        # for window ending with row i+in_shift, get i+in_shift output
                        windows_out = arr_out[in_shift:]

                        # mask out any window that contains a nan for any feature and skip chunk
                        # if there are no valid rows
                        # TEMPORARY FIX - REPLACE NAN WITH 0
                        windows_in = np.nan_to_num(windows_in)

                        valid = ~np.isnan(windows_in).any(axis=(1,2))
                        n_good = valid.sum()    
                        n_total = windows_in.shape[0]
                        
                        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        #print("is nan = ",  np.any(np.isnan(windows_in)))
                        #print("valid shape = ", valid)
                        #print("n_good = ", n_good)
                        #time.sleep(1000)
                        nan_indices_removed += (n_total - n_good)

                        '''
                        if not valid.any():
                            continue

                        '''
                        #print("what is windows nan = ", windows_in.isnull().values.any())
                        #print("what is valid = ", valid)
                        #time.sleep(100)
                        filtered_in  = windows_in[valid]
                        filtered_out = windows_out[valid]
                        ts_chunk = np.array(rows[in_shift:])[valid].tolist()

                        Xb = ch.tensor(filtered_in).permute(0, 2, 1)
                        yb = ch.tensor(filtered_out)

                        turbine_folder = os.path.join(self.WINDOWED_DATA_PATH, f"{t}")
                        os.makedirs(turbine_folder, exist_ok=True)
                        #print("making turbine folder !!!!!!!!!!!!!!!")
                        #time.sleep(100)
                        base = os.path.join(turbine_folder, f"Scotland_{t}")

                        ch.save(Xb, f"{base}_X_{idx:02d}.pt")
                        ch.save(yb, f"{base}_y_{idx:02d}.pt")
                        with open(f"{base}_ts_{idx:02d}.pkl", "wb") as f:
                            pickle.dump(ts_chunk, f)

                except Exception as e:
                    print("exception = !!!!!", e)
                    time.sleep(100)
                    continue
            print(f"nan indices removed for {t}: {nan_indices_removed}")


    def get_and_save_feature_thresholds(self): 
        pass


def process_data(
        cols, 
        BASE_DIR,
        sensitive_property,
        predicted_property,  
        exclude_fault_data,
        WT_IDs,
        rename
    ):
    print(f"The predicted property is {predicted_property}")
    print(f"The sensitive property is {sensitive_property}")

    DATA_PATH = os.path.join(BASE_DIR, "Scotland")
    
    if os.path.exists(DATA_PATH):
        for path in os.listdir(DATA_PATH):
            df = pd.read_csv(os.path.join(DATA_PATH, path))
    
    # TODO: setup the renaming of variables in here not in the preprocess
    farm_scotland = data_farm(
        cols=cols, 
        fault_check_cols=['MainShaftRPM', 'Power'],
        list_turbine_name=WT_IDs,
        BASE_DIR=BASE_DIR,
        sensitive_property=sensitive_property,
        predicted_property=predicted_property,
        exclude_fault_data=exclude_fault_data,
    )
    print("\nPreparing data windows")
    farm_scotland.make_window_data(in_shift=24*6)
    feat_path = os.path.join(farm_scotland.WINDOWED_DATA_PATH, "features_list.pkl")
    farm_scotland.base_in_cols
    with open(feat_path, "wb") as f:pickle.dump(farm_scotland.base_in_cols, f)
    return cols


class preprocess_data_farm:
    def __init__(
            self, 
            base_dir=str, 
            years=list[int],
            rename=Dict[str, Any]
        ):
        self.base_dir = base_dir
        self.years = years
        self.rename = rename

        self.base_url = "https://zenodo.org/records/14870023/files"
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.descriptions_dir = os.path.join(self.base_dir, "descriptions")
        self.raw_dir = os.path.join(self.base_dir, "raw_data")
        self.fault_dir = os.path.join(self.base_dir, "fault_logs")
        self.zip_archive_dir = os.path.join(self.base_dir, "zips")
        self.aggregate_dir = os.path.join(self.base_dir, "aggregate")
        self.daily_stats = os.path.join(self.base_dir, "daily_stats")

        self.max_workers_available = get_available_cpus()

        print("\nPreprocessing dataset")

    def preprocess(self):
        self.make_dirs()
        self.get_Scotland_dataset()

    def make_dirs(self):
        for d in (self.temp_dir, self.descriptions_dir, self.raw_dir, 
                    self.fault_dir, self.zip_archive_dir, self.aggregate_dir,
                    self.daily_stats):
            os.makedirs(d, exist_ok=True)
    
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
                self.keep_cols = [col for col in rename_map[folder].keys() if col in df.columns]
                
                if not self.keep_cols:
                    print(f"None of the specified columns found in {csv_file}. Skipping.")
                    continue
                
                df = df[self.keep_cols]
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
