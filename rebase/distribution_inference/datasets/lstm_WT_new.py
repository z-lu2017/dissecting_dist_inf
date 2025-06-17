# Standard library imports
import os
import re
import pickle
import random
import bisect
from collections import defaultdict
from typing import List
import psutil
import operator

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from functools import lru_cache
from sklearn.model_selection import train_test_split

# PyTorch
import torch as ch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset, Subset

# Internal project modules
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.models.core import SimpleLSTM
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model
from distribution_inference.defenses.active.shuffle import ShuffleDefense

cols = [
    # 'Power', 
    'WindSpeed', 
    'WindSpeed_SD', 
    'Yaw', 
    'Yaw_SD', 
    'SetYaw', 
    'SetYaw_SD', 
    'MainShaftRPM', 
    'MainShaftRPM_SD', 
    'BladePitchA', 
    'BladePitchA_SD', 
    'BladePitchB', 
    'BladePitchB_SD', 
    'BladePitchC', 
    'BladePitchC_SD', 
    'RefBladePitchA', 
    'RefBladePitchA_SD', 
    'RefBladePitchB', 
    'RefBladePitchB_SD', 
    'RefBladePitchC', 
    'RefBladePitchC_SD', 
    'TowerHumidity', 
    'TargetPower', 
    'TimeRunningInPeriod', 
    'TimeActiveErrorInPeriod', 
    'WindDirection', 
    'AmbientTemp', 
    'GearOilTemp', 
    'GearboxBearingTempHSGenSide', 
    'GearboxBearingTempHSRotorSide', 
    'Power_SD', 
    'RawPower', 
    'RawPower_SD']

class ScotlandMapDataset(Dataset):
    def __init__(self, data_folder: str, WT: str, cache_size: int = 8):
        self.data_folder_path = os.path.join(data_folder, WT)

        # create regex pattern for finding files
        pattern = re.compile(
            rf"Scotland_{WT}_(?P<dtype>X|y|ts)_(?P<idx>\d+)\.(?P<ext>pt|pkl)$"
        )

        # discover chunk file paths based off pattern
        paths = defaultdict(dict)
        for fn in os.listdir(self.data_folder_path):
            m = pattern.match(fn)
            if not m:
                continue
            dtype, idx = m.group("dtype"), int(m.group("idx"))
            paths[dtype][idx] = os.path.join(self.data_folder_path, fn)
        self.paths = paths

        # compute per-chunk sizes by peeking at the 'ts' arrays
        self.chunk_indices = sorted(paths["ts"].keys())
        self.chunk_sizes = []
        for idx in self.chunk_indices:
            with open(paths["ts"][idx], "rb") as f:
                ts = pickle.load(f)
            self.chunk_sizes.append(len(ts))

        # build a cumulative‐sum table for global indexing
        self.cumulative = [0]
        for sz in self.chunk_sizes:
            self.cumulative.append(self.cumulative[-1] + sz)

        # decorators applied at definition time are class level
        # --> every insteance will share the same cache
        # --> wrapping undecorated loader method on the fly in constructor
        self._load_chunk = lru_cache(maxsize=cache_size)(self._load_chunk_impl)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, g: int):
        # global index is mapped to the chunk and local index in that chunk
        j = bisect.bisect_right(self.cumulative, g) - 1
        local_idx = g - self.cumulative[j]

        # load + cache the whole chunk
        Xj, yj, tsj = self._load_chunk(j)
        ts_raw = tsj[local_idx]
        if isinstance(ts_raw, pd.Timestamp):
            minutes = ts_raw.value // (1_000_000_000 * 60)
        else:
            minutes = int(ts_raw.timestamp() // 60)

        ts_tensor = ch.tensor(minutes, dtype=ch.int32)
        return Xj[local_idx], yj[local_idx], ts_tensor

    def _load_chunk_impl(self, j: int):
        # Actual chunk loader; wrapped by lru_cache in __init__
        chunk_id = self.chunk_indices[j]
        X = ch.load(self.paths["X"][chunk_id])
        y = ch.load(self.paths["y"][chunk_id])
        with open(self.paths["ts"][chunk_id], "rb") as f:
            ts = pickle.load(f)
        return X, y, ts


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False, data_path: str = "Scotland"):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="LSTM_WT_new",
                         data_path=data_path,
                         models_path="lstm_wt_new",
                         properties=["WindSpeed", "WindSpeed_SD", "WT_Misalignment", "WT_Misalignment_3", "WT_Misalignment_5", "WT_Misalignment_10", "WT_Misalignment_real"], 
                         values={"wind_turbines": ratios},
                         supported_models=["SimpleLSTM"],
                         default_model="SimpleLSTM",
                         epoch_wise=epoch_wise)
        self.dataset = data_path

    def get_model(self, cpu: bool = True, model_arch: str = None) -> nn.Module:
        device = "cuda" if ch.cuda.is_available() else "cpu"
        model_arch = model_arch or self.default_model
        if model_arch == "SimpleLSTM":
            model = SimpleLSTM(30, 64, 1, use_dropout=False).to(device)
        else:
            raise NotImplementedError("Model architecture not supported")

        if not model.is_sklearn_model and not cpu:
            model = model.cuda()
        return model
    

class _WindTurbinePower:
    def __init__(self, data_config, drop_sensitive_cols=False):
        self.data_config = data_config
        self.drop_sensitive_cols = drop_sensitive_cols
        self.split = data_config.split
        self.predicted_property = data_config.WT_config.predicted_property
        self.dataset = self.data_config.WT_config.dataset
        self.features = self.cols = data_config.feature_config[self.dataset].cols
        self.timestamp_split = ["2024-02-01"]
        self.info_object = DatasetInformation()
        self.base_data_dir = self.info_object.base_data_dir
        utils.create_dirs(self.base_data_dir)

        if data_config.WT_config.download_data:
            preprocess_farm = utils.preprocess_data_farm(
                                    dataset="Scotland", 
                                    base_dir=self.base_data_dir, 
                                    years=list(range(2023, 2025)), 
                                    rename=data_config.feature_config[self.dataset].rename
                                )
            preprocess_farm.preprocess()
        else: 
            print("\nData has already beed downloaded and saved ... skipping download")
          
        suffix = (
            self.data_config.prop
            if not data_config.WT_config.exclude_fault_data
            else f"{self.data_config.prop}_faults_excluded"
        )

        data_path = os.path.join("/scratch/ujx4ab", suffix)
        feature_path = os.path.join(data_path, "features_list.pkl")

        WT_IDs = data_config.feature_config[self.dataset].WT_IDs
        victim_WT_IDs = data_config.feature_config[self.dataset].victim_WT_IDs
        adv_WT_IDs = data_config.feature_config[self.dataset].adv_WT_IDs

        threshold_file = os.path.join(
            self.base_data_dir,
            'data_normalisation',
            f'feature_stats_and_thresholds_{self.dataset}.pkl'
        )

        need_process = not os.path.isdir(data_path) \
            or not os.path.isfile(feature_path) \
            or not os.path.exists(threshold_file)

        if not need_process:
            missing = [
                wt for wt in WT_IDs
                if not os.path.isdir(os.path.join(data_path, wt))
            ]
            if missing:
                print(f"Missing windowed data for turbines: {missing}")
                need_process = True

        if need_process:
            self.features = utils.process_data(
                dataset= self.dataset,
                BASE_DIR= self.base_data_dir,
                sensitive_property= self.data_config.prop,
                predicted_property= self.predicted_property,
                exclude_fault_data= data_config.WT_config.exclude_fault_data,
                cols= self.cols,
                rename= data_config.feature_config[self.dataset].rename,
                WT_IDs= WT_IDs
            )
        else: 
            print("Necessary data already exists ... skipping processing\n")
            with open(feature_path, "rb") as feature_path:
                self.features = pickle.load(feature_path)

        utils.cleanup_dirs(self.base_data_dir)

        # TODO: figure out if we can set two different threshold for the data if we 
        # are splitting on different WTs ... will assume that we can use the same
        # one for right now
        with open(threshold_file, 'rb') as f:
            loaded = pickle.load(f)
        self.quantiles = loaded['quantiles']

        datasets = [
            ScotlandMapDataset(data_folder=data_path, WT=wt, cache_size=8)
            for wt in WT_IDs
        ]

        self.full_ds = ConcatDataset(datasets)

        self.meta = self.build_window_meta(self.data_config.prop)
        self.meta = self.bin_meta(self.meta)

        train_pool, test_pool = self.split_indices_by_time()
        assert set(train_pool).isdisjoint(test_pool)

        train_meta = self.meta[self.meta.global_idx.isin(train_pool)]
        test_meta  = self.meta[self.meta.global_idx.isin(test_pool)]

        self.train_vic_meta, self.train_adv_meta = train_test_split(
            train_meta, test_size=0.4, shuffle=True, random_state=42
        )
        self.test_vic_meta, self.test_adv_meta = train_test_split(
            test_meta, test_size=0.4, shuffle=True, random_state=42
        )
        assert set(self.train_vic_meta['global_idx']).isdisjoint(self.train_adv_meta['global_idx'])
        assert set(self.test_vic_meta['global_idx']).isdisjoint(self.test_adv_meta['global_idx'])

        if self.data_config.WT_config.train_test_data_overlap_ratio > 0:
            self.create_train_test_overlap()

        if self.data_config.WT_config.adv_vic_data_overlap_ratio > 0:
            self.create_adv_vic_overlap()


    def overlap_meta_sets(self, A: pd.DataFrame, B: pd.DataFrame, ratio: float):
        n_a = int(ratio * len(df_a))
        n_b = int(ratio * len(df_b))
        if n_a == 0 and n_b == 0:
            return df_a, df_b

        sel_a = df_a.sample(n=n_a, random_state=42)
        sel_b = df_b.sample(n=n_b, random_state=42)

        new_a = pd.concat([df_a, sel_b], ignore_index=True)
        new_b = pd.concat([df_b, sel_a], ignore_index=True)
        return new_a, new_b


    def create_train_test_overlap(self):
        r = self.data_config.WT_config.train_test_data_overlap_ratio
        self.train_vic_meta, self.test_vic_meta = self.overlap_meta_sets(
            self.train_vic_meta, self.test_vic_meta, r
        )
        self.train_adv_meta, self.test_adv_meta = self.overlap_meta_sets(
            self.train_adv_meta, self.test_adv_meta, r
        )

        assert not set(self.train_vic_meta['global_idx']).isdisjoint(self.test_vic_meta['global_idx'])
        assert not set(self.train_adv_meta['global_idx']).isdisjoint(self.test_adv_meta['global_idx'])


    def create_adv_vic_overlap(self):
        r = self.data_config.WT_config.adv_vic_data_overlap_ratio
        self.train_vic_meta, self.train_adv_meta = self.overlap_meta_sets(
            self.train_vic_meta, self.train_adv_meta, r
        )
        self.test_vic_meta, self.test_adv_meta = self.overlap_meta_sets(
            self.test_vic_meta, self.test_adv_meta, r
        )

        assert not set(self.train_vic_meta['global_idx']).isdisjoint(self.train_adv_meta['global_idx'])
        assert not set(self.test_vic_meta['global_idx']).isdisjoint(self.test_adv_meta['global_idx'])


    def split_indices_by_time(self): 
        if len(self.timestamp_split) == 1: 
            split_time = pd.Timestamp(self.timestamp_split[0])
            def is_train(t): return t < split_time
        elif len(self.timestamp_split) == 2: 
            start, end = map(pd.Timestamp, self.timestamp_split)
            def is_train(t): return start <= t <= end
        else: 
            raise ValueError("timestamp_split must be length 1 (train = before split) or 2 (train = between splits)")
        
        subs = self.full_ds.datasets
        lengths = [len(ds) for ds in subs]
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        train_idxs, test_idxs = [], []

        for ds_idx, ds in enumerate(tqdm(subs, total=len(subs), desc="Creating train/test splits.")):
            base = offsets[ds_idx]
            cum  = ds.cumulative
            for j, local_start in enumerate(cum[:-1]):
                Xj, yj, tsj = ds._load_chunk(j)

                times = pd.to_datetime(tsj)
                train_mask = is_train(times)

                size = cum[j+1] - local_start
                gidx = np.arange(base + local_start, base + local_start + size)

                train_idxs.extend(gidx[train_mask].tolist())
                test_idxs.extend(gidx[~train_mask].tolist())

        return train_idxs, test_idxs


    def binarize_column(self, X, sensitive_property_idx, low_thresh, high_thresh, use_abs=False):
        assert not X.isnan().any(), "Error: Data should not contain any NaN values. Check window creation."
        col = X[..., sensitive_property_idx]
        labels = col.new_full(col.shape, 0.5, dtype=ch.float32)
        if use_abs: 
            labels[abs(col) < low_thresh] = 0
            labels[abs(col) > high_thresh] = 1
        else:
            labels[col < low_thresh] = 0
            labels[col > high_thresh] = 1

        return labels


    def build_window_meta(self, sensitive_property_name):
        if sensitive_property_name in self.features:
            sensitive_property_idx = self.features.index(sensitive_property_name)
            print(f"Sensitive propery index: {sensitive_property_idx}")
            low_thresh, high_thresh = self.quantiles[sensitive_property_name]
            use_abs = False
        else: 
            # It will always be the one added to the end
            sensitive_property_idx = -1

            if sensitive_property_name == "WT_Misalignment": 
                thresh = 20.474536963870243 # median std of windows 
                low_thresh = thresh - 5.0
                high_thresh = thresh + 5.0
                use_abs = True
            elif sensitive_property_name == "WT_Misalignment_3": 
                thresh = 3
                low_thresh = thresh - 2.5
                high_thresh = thresh + 2.5
                use_abs = True
            elif sensitive_property_name == "WT_Misalignment_5": 
                thresh = 5 
                low_thresh = thresh - 1
                high_thresh = thresh + 1
                use_abs = True
            elif sensitive_property_name == "WT_Misalignment_10": 
                thresh = 10
                low_thresh = thresh - 2.5
                high_thresh = thresh + 2.5
                use_abs = True
            elif sensitive_property_name == "WT_Misalignment_real": 
                thresh = 5
                low_thresh = thresh - 1
                high_thresh = thresh + 1
                use_abs = True
            elif sensitive_property_name == "WT_Time_Availability" or sensitive_property_name == "WT_Energy_Availability":
                low_thresh, high_thresh = 0.5, 0.5
                use_abs = False
            else:
                assert False, f"Error: {sensitive_property_name} not implemented for binning."

        records = []
        subs = self.full_ds.datasets
        lengths = [len(ds) for ds in subs]
        offsets = np.concatenate([[0], np.cumsum(lengths)])        

        zero_count = 0
        half_count = 0 
        one_count = 0
        for ds_idx, ds in enumerate(tqdm(subs, desc="Fetching window meta data", total=len(subs))):
            base = offsets[ds_idx]
            for j, local_start in enumerate(ds.cumulative[:-1]):
                Xj, yj, tsj = ds._load_chunk(j)

                labels = self.binarize_column(
                    Xj, # (chunk_size, window_len)
                    sensitive_property_idx, 
                    low_thresh= low_thresh, 
                    high_thresh= high_thresh,
                    use_abs=use_abs
                )

                labels_int = (labels * 2).to(ch.int64).flatten()  # shape (chunk_size * window_len,)
                counts = ch.bincount(labels_int, minlength=3)    # Tensor([c0, c1, c2])
                zero_count += int(counts[0].item())
                half_count += int(counts[1].item())
                one_count  += int(counts[2].item())

                p_i = labels.sum(dim=1).cpu().numpy()  # (chunk_size,)
                t_i = labels.shape[1] # (window_size)
                r_i = p_i / t_i

                global_start = base + local_start
                global_indices = np.arange(global_start, global_start + len(p_i))

                chunk_df = pd.DataFrame({
                    "global_idx": global_indices,
                    "r_i": r_i,
                    "y_i": yj.squeeze()
                })
                records.append(chunk_df)

        print(f"Low and high thresholds for record level binarizing: {low_thresh, high_thresh}")
        print("If sensitive feature is in data, grouping size will ~1/3. Otherwise they will be skewed.")
        print(f"lable = 0 count: {zero_count}")
        print(f"lable = 0.5 count: {half_count}")
        print(f"lable = 1 count: {one_count}")

        return pd.concat(records, ignore_index=True)


    def bin_meta(
            self,
            meta_df: pd.DataFrame,
            sensitive_col: str = "r_i",
            target_col: str = "y_i",
            low_q: float = 40,
            high_q: float = 60,
        ) -> pd.DataFrame:
            
        df = meta_df.copy()
        q_low, q_high = np.percentile(df[sensitive_col], [low_q, high_q])

        # if they are 0 and 1, respecitively, duplicate data will be used for groups. Should both be 0, 0.5, or 1.0 if binary
        if q_low == 0 and q_high == 1: 
            assert False, "Error: q_low ({q_low}) and q_high ({q_high}) cannot be 0 and 1, respectively."

        is_low_bin = (lambda x, y: x < y) if q_low > 0 else (lambda x, y: x <= y)
        is_high_bin = (lambda x, y: x > y) if q_high < 1 else (lambda x, y: x >= y)
        df["Sensitive_bin"] = np.where(
            is_low_bin(df[sensitive_col], q_low), 0,
            np.where(is_high_bin(df[sensitive_col], q_high), 1, np.nan)
        )

        df = df.dropna(subset=["Sensitive_bin"]).copy()
        df["Sensitive_bin"] = df["Sensitive_bin"].astype(int)

        y_med = df[target_col].median()
        df["Pred_bin"] = (df[target_col] > y_med).astype(int)

        print(f"\nWindow level binarizing: Using the lower {low_q}% and upper {high_q}% of data. Low and high cuttoffs are {q_low, q_high}.")
        print(f"Dropped {len(meta_df)-len(df)} samples.")
        print(f"These cutoffs yield {sum(df['Sensitive_bin'] == 0)} values for lower bin(0) and {sum(df['Sensitive_bin'] == 0)} values for upper bin(1)\n")

        return df
        

    def get_filtered_indices(self,
                             meta_df: pd.DataFrame,
                             prop_ratio: float,
                             n_samples: int,
                             random_state: int = 42):
        n_pos = int(round(n_samples * prop_ratio))
        n_neg = n_samples - n_pos

        out_idx = []

        for bin_val, n_sel in [(1, n_pos), (0, n_neg)]:
            df_bin = meta_df[meta_df["Sensitive_bin"] == bin_val]
            if n_sel == 0: # nothing to pick in this bin 
                continue 
            if n_sel > len(df_bin):
                raise ValueError(
                    f"Requested {n_sel} samples from bin={bin_val}, but only "
                    f"{len(df_bin)} are available."
                )

            # stratify (0 < n_sel < total)
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=n_sel,
                random_state=random_state
            )
            train_idx, _ = next(splitter.split(np.zeros(len(df_bin)), df_bin["Pred_bin"]))
            chosen = df_bin.iloc[train_idx]["global_idx"].tolist()
            out_idx.extend(chosen)

        rng = np.random.RandomState(random_state)
        rng.shuffle(out_idx)
        return out_idx


    def get_data(self, prop_ratio: float, sensitive_property: str):
        # decided to use flexible ratio in tests that restrict number of WTs available
        # TODO: fix this for when not enough data is available (ex. only using 1 turbine for vic model)
        if self.split == 'adv':
            self.train_indices = self.get_filtered_indices(
                meta_df=self.train_adv_meta,
                prop_ratio=prop_ratio,
                n_samples=round(len(self.train_adv_meta)*.2),
            )
            self.test_indices = self.get_filtered_indices(
                meta_df=self.test_adv_meta,
                prop_ratio=prop_ratio,
                n_samples=round(len(self.test_adv_meta)*.2),
            )
        else:
            self.train_indices = self.get_filtered_indices(
                meta_df=self.train_vic_meta,
                prop_ratio=prop_ratio,
                n_samples=round(len(self.train_vic_meta)*.25),
            )
            self.test_indices = self.get_filtered_indices(
                meta_df=self.test_vic_meta,
                prop_ratio=prop_ratio,
                n_samples=round(len(self.test_vic_meta)*.25),
            )

        print(f"Prop ratio: {prop_ratio}")

        train_ids = np.array(self.train_indices, dtype=int)
        test_ids = np.array(self.test_indices,  dtype=int)

        xs_tr, ys_tr = [], []
        for idx in train_ids:
            x_i, y_i, ts = self.full_ds[idx]
            xs_tr.append(x_i)
            ys_tr.append(y_i)
        x_tr = ch.stack(xs_tr)       # shape (N_tr, seq_len, n_features)
        y_tr = ch.stack(ys_tr)  # shape (N_tr, 1)

        xs_te, ys_te = [], []
        for idx in test_ids:
            x_i, y_i, ts = self.full_ds[idx]
            xs_te.append(x_i)
            ys_te.append(y_i)
        x_te = ch.stack(xs_te)
        y_te = ch.stack(ys_te)

        # pull prop‐labels from your binned metas
        train_prop_labels = self.train_vic_meta if self.split=="victim" else self.train_adv_meta
        train_prop_labels = (
            train_prop_labels
            .set_index("global_idx")
            .loc[train_ids, "Sensitive_bin"]
            .to_numpy()
            .reshape(-1, 1)
        )

        test_prop_labels = self.test_vic_meta if self.split=="victim" else self.test_adv_meta
        test_prop_labels = (
            test_prop_labels
            .set_index("global_idx")
            .loc[test_ids, "Sensitive_bin"]
            .to_numpy()
            .reshape(-1, 1)
        )

        if sensitive_property not in self.features: 
            x_tr = x_tr[:, :, :-1]  # Drop the last feature used for splitting
            x_te = x_te[:, :, :-1]

        return (
            (x_tr, y_tr, train_prop_labels),
            (x_te, y_te, test_prop_labels)
        ), (train_ids, test_ids)


class LSTMWindTurbineWrapper(base.CustomDatasetWrapper):
    def __init__(self,
                 data_config: DatasetConfig,
                 skip_data: bool = False,
                 epoch: bool = False,
                 label_noise: float = 0,
                 shuffle_defense: ShuffleDefense = None):
        super().__init__(data_config,
                         skip_data=skip_data,
                         label_noise=label_noise,
                         shuffle_defense=shuffle_defense)
        if not skip_data:
            self.ds = _WindTurbinePower(data_config, drop_sensitive_cols=self.drop_sensitive_cols)
        self.data_config = data_config
        self.info_object = DatasetInformation(epoch_wise=epoch)

    def load_data(self, custom_limit=None, indexed_data=None):
        data, splits = self.ds.get_data(prop_ratio=self.ratio, sensitive_property=self.data_config.prop)

        # indices of data
        self._used_for_train = splits[0].squeeze()
        self._used_for_test = splits[1].squeeze()
        return data  # (train_data, test_data)
    
    def get_used_indices(self):
        """For tracking which data points were used post-filtering"""
        return (self._train_ids_before, self.ds_train.ids), \
               (self._val_ids_before, self.ds_val.ids)

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    indexed_data=None):
        # Load turbine-specific splits
        train_data, test_data = self.load_data(self.cwise_samples, indexed_data=indexed_data)

        # Initialize datasets with turbine data
        self.ds_train = WindTurbineDataset(
            train_data[0],  # X
            train_data[1],  # y
            train_data[2],  # prop_labels
            squeeze=self.squeeze,
            ids=self._used_for_train
        )

        self.ds_val = WindTurbineDataset(
            test_data[0],
            test_data[1],
            test_data[2],
            squeeze=self.squeeze,
            ids=self._used_for_test
        )
        
        # Track original IDs before any processing
        self._train_ids_before = self.ds_train.ids
        self._val_ids_before = self.ds_val.ids
        
        return super().get_loaders(
            batch_size,
            shuffle=shuffle,
            eval_shuffle=eval_shuffle,
            prefetch_factor=None
        )

    def load_model(self, path: str, on_cpu: bool = False, model_arch: str = None) -> nn.Module:
        info_object = self.info_object
        model = info_object.get_model(cpu=on_cpu, model_arch=model_arch)
        return load_model(model, path, on_cpu=on_cpu)

    # used in base and unique for each dataset wrapper
    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        dp_config = None
        shuffle_defense_config = None
        if train_config.misc_config is not None:
            dp_config = train_config.misc_config.dp_config
            shuffle_defense_config = train_config.misc_config.shuffle_defense_config

        # Standard logic
        if model_arch == "None":
            model_arch = self.info_object.default_model
        if model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")
        
        base_models_dir = os.path.join(base_models_dir, model_arch)

        if dp_config is None:
            if shuffle_defense_config is None:
                base_path = os.path.join(base_models_dir, "normal")
            else:
                if self.ratio == shuffle_defense_config.desired_value:
                    # When ratio of models loaded is same as target ratio of defense,
                    # simply load 'normal' model of that ratio
                    base_path = os.path.join(base_models_dir, "normal")
                else:
                    base_path = os.path.join(base_models_dir, "shuffle_defense",#"test_shuffled",
                                             "%s" % shuffle_defense_config.sample_type,
                                             "%.2f" % shuffle_defense_config.desired_value)
        else:
            base_path = os.path.join(
                base_models_dir, "DP_%.2f" % dp_config.epsilon)
        if self.label_noise:
            base_path = os.path.join(
                base_models_dir, "label_noise:{}".format(train_config.label_noise))

        save_path = os.path.join(base_path, self.prop, self.split)

        WT_config = "WTs_"+"_".join(self.data_config.WT_config.turbines) if self.data_config.WT_config.turbines else "WTs_all"
        
        if WT_config is not None: 
            save_path = os.path.join(save_path, WT_config)

        if self.ratio is not None:
            save_path = os.path.join(save_path, str(self.ratio))

        if self.scale != 1.0:
            save_path = os.path.join(
                save_path, "sample_size_scale:{}".format(self.scale))

        if self.drop_sensitive_cols:
            save_path = os.path.join(save_path, "drop")

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path
    
class WindTurbineDataset(base.CustomDataset):
    def __init__(self, data, targets, prop_labels, squeeze=False, ids=None):
        super().__init__()
        self.data = data                        # (e.g., time-series features)
        self.targets = targets
        self.prop_labels = prop_labels
        self.squeeze = squeeze
        self.ids = ids
        self.num_samples = len(data)
        self.selection_mask = None              # For dataset subsetting

    def set_selection_mask(self, mask):         # masks for selecting datasplits 
        self.selection_mask = mask
        self.num_samples = len(mask)
        if self.ids is not None:
            self.ids = self.ids[mask]

    def __getitem__(self, index):
        index_ = self.selection_mask[index] if self.selection_mask is not None else index
        X = self.data[index_]
        y = self.targets[index_]
        prop_label = self.prop_labels[index_]

        if self.squeeze:
            y = y.squeeze()
            prop_label = prop_label.squeeze()

        return X, y, prop_label