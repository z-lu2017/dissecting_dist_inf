# Standard library imports
import os
import re
import pickle
import random
import bisect
from collections import defaultdict
from typing import List
import psutil

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
from distribution_inference.models.core import MaskedLSTM
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model
from distribution_inference.defenses.active.shuffle import ShuffleDefense

cols = [
    'Power', 
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
                         properties=["Blade_pitch", "Bp_wind"], 
                         values={"wind_turbines": ratios},
                         supported_models=["MaskedLSTM"],
                         default_model="MaskedLSTM",
                         epoch_wise=epoch_wise)
        self.dataset = data_path

    def get_model(self, cpu: bool = True, model_arch: str = None) -> nn.Module:
        device = "cuda" if ch.cuda.is_available() else "cpu"
        model_arch = model_arch or self.default_model
        if model_arch == "MaskedLSTM":
            model = MaskedLSTM(30, 64, 1, use_dropout=False).to(device)
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
        self.cols = data_config.feature_config[self.dataset].cols
        self.timestamp_split = ["2024-02-01"]
        self.info_object = DatasetInformation()
        self.base_data_dir = self.info_object.base_data_dir
        utils.create_dirs(self.base_data_dir)

        """
        Uncomment this to redownload raw data and select features of interest
        for specified years.
        """
        # if data_config.WT_config.download_data:
        #     preprocess_farm = utils.preprocess_data_farm(
        #                             dataset="Scotland", 
        #                             base_dir=self.base_data_dir, 
        #                             years=list(range(2023, 2025)), 
        #                             rename=data_config.feature_config[self.dataset].rename
        #                         )
        #     preprocess_farm.preprocess()
                    
        # """
        # Uncomment this code to take the data, clean it, and build time series windows.
        # """
        # self.feature_names = utils.process_data(
        #     dataset=self.dataset,
        #     BASE_DIR=self.base_data_dir,
        #     predicted_property=self.predicted_property,
        #     exclude_fault_data=data_config.WT_config.exclude_fault_data,
        #     cols=self.cols,
        #     rename=data_config.feature_config[self.dataset].rename,
        #     WT_IDs=data_config.feature_config[self.dataset].WT_IDs
        # )

        # utils.cleanup_dirs(self.base_data_dir)

        turbine_ids = self.data_config.feature_config[self.dataset].WT_IDs

        datasets = [
            ScotlandMapDataset(data_folder="/scratch/ujx4ab/data_prep", WT=wt, cache_size=8)
            for wt in turbine_ids
        ]

        self.full_ds = ConcatDataset(datasets)

        self.meta = self.build_window_meta()
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

        # TODO: implement equivalent versions of these
        # if data_config.WT_config.train_test_data_overlap_ratio != 0:
        #     self.create_train_test_overlap()
        # self.create_data_splits()
        # if data_config.WT_config.adv_vic_data_overlap_ratio != 0: 
        #     self.create_adv_vic_overlap()
        # self.split_data()

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
                mask = is_train(times)

                size = cum[j+1] - local_start
                gidx = np.arange(base + local_start, base + local_start + size)

                train_idxs.extend(gidx[mask].tolist())
                test_idxs.extend(gidx[~mask].tolist())

        return train_idxs, test_idxs


    # Average WindSpeed thresholds: low<0.185, high>0.304
    def binarize_column(self, X, low_thresh=0.185, high_thresh=0.304, exclude_middle=True):
        # assert not X.isnan().any(), "Error: Data should contain no nans. Should be 0.0 values in rows instead."
        col = X[..., 0]
        labels = col.new_full(col.shape, 0.5, dtype=ch.long)
        labels[col < low_thresh] = 0
        labels[col > high_thresh] = 1
        return labels


    def build_window_meta(self):
        records = []
        subs = self.full_ds.datasets
        lengths = [len(ds) for ds in subs]
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        for ds_idx, ds in enumerate(tqdm(subs, desc="Fetching window meta data", total=len(subs))):
            base = offsets[ds_idx]
            for j, local_start in enumerate(ds.cumulative[:-1]):
                Xj, yj, tsj = ds._load_chunk(j)

                labels = self.binarize_column(Xj)  # (chunk_size, window_len)
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

        return pd.concat(records, ignore_index=True)


    def bin_meta(self,
                 meta_df: pd.DataFrame,
                 sensitive_col: str = "r_i",
                 target_col: str = "y_i",
                 low_q: float = 40,
                 high_q: float = 60) -> pd.DataFrame:
        df = meta_df.copy()
        q_low, q_high = np.percentile(df[sensitive_col], [low_q, high_q])

        df["Sensitive_bin"] = np.where(
            df[sensitive_col] <= q_low, 0,
            np.where(df[sensitive_col] > q_high, 1, np.nan)
        )
        df = df.dropna(subset=["Sensitive_bin"]).copy()
        df["Sensitive_bin"] = df["Sensitive_bin"].astype(int)

        y_med = df[target_col].median()
        df["Pred_bin"] = (df[target_col] > y_med).astype(int)

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
            if n_sel >= len(df_bin): # picking them all
                out_idx.extend(df_bin["global_idx"].tolist())
                continue

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

    def get_data(self, prop_ratio: float):
        if self.split == 'adv':
            self.train_indices = self.get_filtered_indices(
                meta_df=self.train_adv_meta,
                prop_ratio=prop_ratio,
                n_samples=75_000,
            )
            self.test_indices = self.get_filtered_indices(
                meta_df=self.test_adv_meta,
                prop_ratio=prop_ratio,
                n_samples=37_500,
            )
        else:
            self.train_indices = self.get_filtered_indices(
                meta_df=self.train_vic_meta,
                prop_ratio=prop_ratio,
                n_samples=150_000,
            )
            self.test_indices = self.get_filtered_indices(
                meta_df=self.test_vic_meta,
                prop_ratio=prop_ratio,
                n_samples=75_000,
            )

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

        # set masks
        mask_tr = ch.ones_like(x_tr, dtype=ch.float32)
        mask_tr = mask_tr.masked_fill(ch.isnan(x_tr), 0.0)
        mask_tr = mask_tr.all(dim=-1).to(dtype=mask_tr.dtype)

        mask_te = ch.ones_like(x_te, dtype=ch.float32)
        mask_te = mask_te.masked_fill(ch.isnan(x_te), 0.0)
        mask_te = mask_te.all(dim=-1).to(dtype=mask_te.dtype)

        # fill nans with 0s to ensure no nan errors in training
        x_tr = ch.nan_to_num(x_tr, nan=0.0)
        x_te = ch.nan_to_num(x_te, nan=0.0)
        y_tr = ch.nan_to_num(y_tr, nan=0.0)
        y_te = ch.nan_to_num(y_te, nan=0.0)

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

        # TODO: The true proportion is actually different since we end up masking some 
        # of the inputs ... might need to do the splitting after

        return (
            (x_tr, y_tr, mask_tr, train_prop_labels),
            (x_te, y_te, mask_te, test_prop_labels)
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
        data, splits = self.ds.get_data(prop_ratio=self.ratio)

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
            *train_data,  # Unpacks: X_bundle, y, prop_labels
            squeeze=self.squeeze,
            ids=self._used_for_train
        )
        self.ds_val = WindTurbineDataset(
            *test_data,  # Using test split as validation
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

        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path
    
class WindTurbineDataset(base.CustomDataset):
    def __init__(self, data, targets, input_masks, prop_labels, squeeze=False, ids=None):
        super().__init__()
        self.data = data                        # (e.g., time-series features)
        self.targets = targets
        self.input_masks = input_masks          # Renamed to avoid conflict
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
        x = self.data[index_]
        y = self.targets[index_]
        input_mask = self.input_masks[index_]
        prop_label = self.prop_labels[index_]

        if self.squeeze:
            y = y.squeeze()
            prop_label = prop_label.squeeze()

        data_bundle = {
            "features": x, 
            "mask": input_mask, 
        }

        return data_bundle, y, prop_label


# def stream_and_select(
    #                 self, 
    #                 indices, 
    #                 min_samples, 
    #                 tolerance, 
    #                 target_ratio, 
    #                 warm_up_n=5000,
    #                 warm_up_tolerance=0.1
    #                 ):
    #     ratio_count = []
    #     selected_idxs = []
    #     running_pos, running_total = 0, 0

    #     with tqdm(indices, 
    #             desc="Scanned", 
    #             total=len(indices), 
    #             position=0,
    #             leave=True) as outer_pbar, \
    #         tqdm(total=min_samples, 
    #             desc="Collected", 
    #             position=1, 
    #             leave=True,
    #             miniters=500) as inner_pbar:
                    
    #         for idx in outer_pbar:
    #             X, y, ts = self.full_ds[idx]
    #             bin_labels = self.binarize_column(X)
    #             num_pos = bin_labels.sum().item()
    #             num_total = len(bin_labels)

    #             r_i = num_pos / num_total
    #             new_pos = running_pos + num_pos
    #             new_tot = running_total + num_total
    #             new_ratio = new_pos / new_tot

    #             if running_total > 0:
    #                 curr_ratio = running_pos / running_total
    #                 current_error = abs(curr_ratio - target_ratio)
    #             else:
    #                 curr_ratio = None
    #                 current_error = float("inf")
    #             new_error = abs(new_ratio - target_ratio)

    #             take_idx = False

    #             if len(selected_idxs) < warm_up_n:
    #                 if abs(r_i - target_ratio) <= warm_up_tolerance:
    #                     take_idx = (new_error <= current_error) or (len(selected_idxs) < min_samples)
    #                 # print(abs(r_i - target_ratio))
    #             else:
    #                 take_idx = (new_error <= current_error) and (len(selected_idxs) < min_samples) or idx % 50 == 0

    #             if take_idx:
    #                 selected_idxs.append(idx)
    #                 running_pos   = new_pos
    #                 running_total = new_tot

    #                 inner_pbar.update(1)
    #                 if inner_pbar.n % 500 == 0:
    #                     phase = "warmup" if len(selected_idxs) < warm_up_n else "greedy"
    #                     inner_pbar.set_postfix(
    #                         ratio=running_pos/running_total,
    #                         phase=phase
    #                     )

    #             if len(selected_idxs) >= min_samples and new_error < tolerance:
    #                 break

    #     print(f"final number of samples, ratio: {len(selected_idxs)}, {curr_ratio}")
    #     print(f"ERROR: collected samples is less than min samples ({min_samples})")
    #     print("Try relaxing tolerance in warmup")
    #     exit()
    #     return selected_idxs, ratio_count