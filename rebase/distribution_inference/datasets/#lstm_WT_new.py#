# Standard library imports
import os
import re
import pickle
import random
import bisect
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from functools import lru_cache
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import seaborn as sns

# PyTorch
import torch as ch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, Subset

# Internal project modules
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.models.core import SimpleLSTM
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model
from distribution_inference.defenses.active.shuffle import ShuffleDefense

# input features being used when the data processing is not run
# and does not return the features that it used.
#   excludes target variable (power)
cols = [
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
        self.WT = WT

        # regex pattern for finding files
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
        return Xj[local_idx], yj[local_idx], ts_tensor, self.WT

    def _load_chunk_impl(self, j: int):
        # Actual chunk loader
        # wrapped by lru_cache in __init__
        chunk_id = self.chunk_indices[j]
        X = ch.load(self.paths["X"][chunk_id])
        y = ch.load(self.paths["y"][chunk_id])
        with open(self.paths["ts"][chunk_id], "rb") as f:
            ts = pickle.load(f)
        return X, y, ts

def plot_bin_effect(
    sensitive_col: str,
    low_windspeed: np.ndarray,
    high_windspeed: np.ndarray,
    low_power:     np.ndarray,
    high_power:    np.ndarray,
    save_prefix:   str,
    low_sensitive:  np.ndarray = None,
    high_sensitive: np.ndarray = None,
    context: str = "talk",
):
    """
    Plots:
      1) WindSpeed vs Power for low & high bins (side by side, xlim=0–1)
      2) (optional) sensitive_col vs WindSpeed in a separate figure, xlim=0–1
    """
    sns.set(style="whitegrid")
    sns.set_context(context)

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes1[0].scatter(low_windspeed, low_power, s=10, alpha=0.5)
    axes1[0].set_title(f"Wind Speed vs Power\n({sensitive_col} = low)")
    axes1[0].set_xlabel("Wind Speed (m/s)")
    axes1[0].set_ylabel("Power Output (kW)")
    axes1[0].set_xlim(0, 1)

    axes1[1].scatter(high_windspeed, high_power, s=10, alpha=0.5)
    axes1[1].set_title(f"Wind Speed vs Power\n({sensitive_col} = high)")
    axes1[1].set_xlabel("Wind Speed (m/s)")
    axes1[1].set_ylabel("")
    axes1[1].set_xlim(0, 1)

    plt.tight_layout()
    fig1.savefig(f"{save_prefix}_{sensitive_col}_Power.png", dpi=300)
    plt.close(fig1)

    if low_sensitive is not None and high_sensitive is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(low_sensitive, low_windspeed, s=10, alpha=0.5, label="low")
        ax2.scatter(high_sensitive, high_windspeed, s=10, alpha=0.5, label="high")

        ax2.set_title(f"{sensitive_col} vs Wind Speed")
        ax2.set_xlabel(sensitive_col.replace("_", " ").title())
        ax2.set_ylabel("Wind Speed (m/s)")
        ax2.set_xlim(0, 1)
        ax2.legend()

        plt.tight_layout()
        fig2.savefig(f"{save_prefix}_{sensitive_col}_WindSpeed.png", dpi=300)
        plt.close(fig2)


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False, data_path: str = "Scotland"):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="lstm_WT_new",
                         data_path=data_path,
                         models_path="lstm_WT_new",
                         properties=["Speed", "Turbulence", "MisalignmentLoss"], 
                         values={"wind_turbines": ratios},
                         supported_models=["SimpleLSTM"],
                         default_model="SimpleLSTM",
                         epoch_wise=epoch_wise)

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
        self.cols = data_config.feature_config.cols
        self.timestamp_split = ["2024-02-01"]
        self.info_object = DatasetInformation()
        self.base_data_dir = self.info_object.base_data_dir

        utils.create_dirs(self.base_data_dir)

        if data_config.WT_config.download_data:
            preprocess_farm = utils.preprocess_data_farm(
                                    dataset="Scotland", 
                                    base_dir=self.base_data_dir, 
                                    years=list(range(2023, 2025)), 
                                    rename=data_config.feature_config.rename
                                )
            preprocess_farm.preprocess()
        else: 
            print("\nData has already beed downloaded and saved ... skipping download")
          
        # TODO: change this so that its not hardcoded. Need to reconcile 
            # self.base_data_dir and this location
        suffix = (
            self.data_config.prop
            if not data_config.WT_config.exclude_fault_data
            else f"{self.data_config.prop}_faults_excluded"
        )
        data_path = os.path.join("/scratch/ujx4ab", suffix)
        feature_path = os.path.join(data_path, "features_list.pkl")

        WT_IDs = data_config.feature_config.WT_IDs or []
        adv_WT_IDs = data_config.feature_config.adv_WT_IDs or []
        victim_WT_IDs = data_config.feature_config.victim_WT_IDs or []
        if not WT_IDs and (not adv_WT_IDs or not victim_WT_IDs): 
            assert False, "Error: Must specify WT_IDs or both adv_WT_IDs and victim_WT_IDs"

        need_process = not os.path.isdir(data_path) \
            or not os.path.isfile(feature_path) \

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
                cols=self.cols,
                BASE_DIR= self.base_data_dir,
                sensitive_property= self.data_config.prop,
                predicted_property= self.predicted_property,
                exclude_fault_data= data_config.WT_config.exclude_fault_data,
                rename= data_config.feature_config.rename,
                WT_IDs=WT_IDs
            )
        else: 
            print("Necessary data already exists ... skipping processing\n")
            with open(feature_path, "rb") as f:
                self.features = pickle.load(f)

        utils.cleanup_dirs(self.base_data_dir)

        datasets = [ScotlandMapDataset(data_path, WT=wt, cache_size=8) for wt in WT_IDs]
        self.full_ds = ConcatDataset(datasets)
        meta = self.build_window_meta(self.data_config.prop)
        meta = self.bin_meta(meta, low_q=data_config.WT_config.low_q, high_q=data_config.WT_config.high_q)

        stats = meta.groupby("Sensitive_bin")["y_i"]\
            .agg(["count", "mean", "median", "std"])\
            .rename(index={0: "Low‑sensitive-atr", 1: "High‑sensitve-atr"})
        print(f"Power‑output statistics by sensitive bin:\n{stats}")

        corr = meta["Sensitive_bin"].astype(float).corr(meta["y_i"])
        print(f"prop ratio: {self.data_config.prop}")

        # if positive instance of sensitive feature tend to have higher power
        # than we want to push poisoned samples power low, otherwise high
        if stats['mean'][0] < stats['mean'][1]: 
            self.poisoning_direction = 0
        else: 
            self.poisoning_direction = 1

        # Time based split for "train" and "test" is not usually necessary. Because we have
        # time windowed data, individual records appear more than once windows. It is safer
        # to ensure that the model doens't learn to make predictions when it sees a record
        # in a window in the training data and then in a window in the test data.
        train_idx, test_idx = self.split_indices_by_time()
        assert set(train_idx).isdisjoint(test_idx)
        full_train_meta = meta[meta.global_idx.isin(train_idx)]
        full_test_meta = meta[meta.global_idx.isin(test_idx)]

        print(f"adv_WT_IDs, victim_WT_IDs {adv_WT_IDs, victim_WT_IDs}")
        if adv_WT_IDs and victim_WT_IDs:
            print("using WT level splits")
            print(f"adv_WT_IDs: {adv_WT_IDs}")
            print(f"victim_WT_IDs: {victim_WT_IDs}")
            self.train_adv_meta = full_train_meta[full_train_meta.WT.isin(adv_WT_IDs)]
            self.test_adv_meta = full_test_meta[full_test_meta.WT.isin(adv_WT_IDs)]
            self.train_vic_meta = full_train_meta[full_train_meta.WT.isin(victim_WT_IDs)]
            self.test_vic_meta = full_test_meta[full_test_meta.WT.isin(victim_WT_IDs)]

            if set(adv_WT_IDs) & set(victim_WT_IDs): 
                assert not set(self.train_adv_meta['global_idx']).isdisjoint(self.train_vic_meta['global_idx'])
                assert not set(self.test_adv_meta['global_idx']).isdisjoint(self.test_vic_meta['global_idx'])
            else: 
                assert set(self.train_adv_meta['global_idx']).isdisjoint(self.train_vic_meta['global_idx'])
                assert set(self.test_adv_meta['global_idx']).isdisjoint(self.test_vic_meta['global_idx'])
        else: 
            self.train_adv_meta, self.train_vic_meta = train_test_split(
                full_train_meta, train_size=0.5, shuffle=True, random_state=42
            )
            self.test_adv_meta, self.test_vic_meta = train_test_split(
                full_test_meta, train_size=0.5, shuffle=True, random_state=42
            )
            assert set(self.train_adv_meta['global_idx']).isdisjoint(self.train_vic_meta['global_idx'])
            assert set(self.test_adv_meta['global_idx']).isdisjoint(self.test_vic_meta['global_idx'])

    def split_indices_by_time(self): 
        if len(self.timestamp_split) == 1: 
            split_time = pd.Timestamp(self.timestamp_split[0])
            def is_train(t): return t < split_time
        elif len(self.timestamp_split) == 2: 
            start, end = map(pd.Timestamp, self.timestamp_split)
            def is_train(t): return start <= t <= end
        else: 
            raise ValueError("timestamp_split must be length 1 (train = before split) or 2 (train = between splits)")
        
        lengths = [len(ds) for ds in self.full_ds.datasets]
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        train_idxs, test_idxs = [], []

        for ds_idx, ds in enumerate(tqdm(self.full_ds.datasets, total=len(self.full_ds.datasets), desc=f"Creating train/test splits for {self.split}")):
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


    def get_labels(self, Xj, sensitive_property_name):
        if ch.is_tensor(Xj):
            X = Xj.detach().cpu().numpy()
        else:
            X = np.array(Xj)

        if sensitive_property_name in self.features:
            idx = self.features.index(sensitive_property_name)
        else:
            idx = -1

        if sensitive_property_name in [
            "Speed",
            "Turbulence",
            "MisalignmentLoss",
            "Time_Availability",
            "Capacity",
            "Energy_Availability",
        ]:
            return X[..., idx].mean(axis=1)
        else:
            raise ValueError(f"Unsupported property: {sensitive_property_name!r}")


    def build_window_meta(self, sensitive_property_name):
        records = []
        lengths = [len(ds) for ds in self.full_ds.datasets]
        offsets = np.concatenate([[0], np.cumsum(lengths)])

        for ds_idx, ds in enumerate(
            tqdm(self.full_ds.datasets,
                 desc="Fetching window meta data",
                 total=len(self.full_ds.datasets))
        ):
            base  = offsets[ds_idx]
            wt_id = ds.WT

            for j, local_start in enumerate(ds.cumulative[:-1]):
                Xj, yj, tsj = ds._load_chunk(j)

                labels = self.get_labels(Xj, sensitive_property_name)

                global_start = base + local_start
                global_indices = np.arange(global_start,
                                           global_start + len(labels))

                chunk_df = pd.DataFrame({
                    "global_idx": global_indices,
                    "WT": wt_id,
                    "label": labels,
                    "y_i": yj.squeeze()
                })
                records.append(chunk_df)

        return pd.concat(records, ignore_index=True)

    def bin_meta(
            self,
            meta_df,
            sensitive_col = "label",
            target_col = "y_i",
            low_q = 20,
            high_q = 80,
         ):
            df = meta_df.copy()                        
            q_low, q_high = np.percentile(df[sensitive_col], [low_q, high_q])

            if low_q == high_q: 
                assert False, "low_q and high_q cannot be equal or records will have competing labels and default to 1 (double check)"
            df["Sensitive_bin"] = np.where(
                df[sensitive_col] <=  q_low, 0,
                np.where(df[sensitive_col] >= q_high, 1, np.nan))

            df = df.dropna(subset=["Sensitive_bin"]).copy()
            df["Sensitive_bin"] = df["Sensitive_bin"].astype(int)

            y_med = df[target_col].median()
            df["Pred_bin"] = (df[target_col] > y_med).astype(int)

            print(f"Binning at the {low_q}th/{high_q}th percentiles")
            print(f"Cutoffs (low, high) = ({q_low:.4f}, {q_high:.4f})")
            print(f"Dropped {len(meta_df)-len(df)} samples")
            print(f"0 sample count = {(df['Sensitive_bin']==0).sum()}")
            print(f"1 sample count = {(df['Sensitive_bin']==1).sum()}")

            for bin_val in [0, 1]:
                sub = df[df["Sensitive_bin"] == bin_val]
                if len(sub) < 2:
                    print(f"Bin {bin_val}: not enough samples to compute correlation")
                    continue

                r, p = pearsonr(sub[sensitive_col], sub[target_col])
                print(f"Bin {bin_val} (Sensitive_bin={bin_val}):")
                print(f"Pearson r = {r:.4f}")

            return df

        
    def get_filtered_indices(self,
                             meta_df: pd.DataFrame,
                             prop_ratio: float,
                             n_samples: int):
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

            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=n_sel,
            )
            train_idx, _ = next(splitter.split(
                                    np.zeros(len(df_bin)),
                                    df_bin["Pred_bin"]))
            chosen = df_bin.iloc[train_idx]["global_idx"].tolist()
            out_idx.extend(chosen)

        random.shuffle(out_idx)
        return out_idx


    def get_data(
            self, 
            prop_ratio, 
            sensitive_property, 
            n_samples_train = 100000, 
            n_samples_test = 40000
        ):
        if self.split == 'adv':
            print("fetching ADV split ...")
            self.train_indices = self.get_filtered_indices(
                meta_df=self.train_adv_meta,
                prop_ratio=prop_ratio,
                n_samples=n_samples_train,
            )
            self.test_indices = self.get_filtered_indices(
                meta_df=self.test_adv_meta,
                prop_ratio=prop_ratio,
                n_samples=n_samples_test,
            )
        else:
            print("fetching VICTIM split ...")
            self.train_indices = self.get_filtered_indices(
                meta_df=self.train_vic_meta,
                prop_ratio=prop_ratio,
                n_samples=n_samples_train,
            )
            self.test_indices = self.get_filtered_indices(
                meta_df=self.test_vic_meta,
                prop_ratio=prop_ratio,
                n_samples=n_samples_test,
            )

        print(f"Prop ratio: {prop_ratio}")
        print(f"split: {self.split}")
        print(f"poisoning rate: {self.data_config.poison_rate}")

        train_ids = np.array(self.train_indices, dtype=int)
        test_ids = np.array(self.test_indices,  dtype=int)

        xs_tr, ys_tr = [], []
        for idx in train_ids:
            x_i, y_i, ts, wt = self.full_ds[idx]
            xs_tr.append(x_i)
            ys_tr.append(y_i)
        x_tr = ch.stack(xs_tr)
        y_tr = ch.stack(ys_tr)

        xs_te, ys_te = [], []
        for idx in test_ids:
            x_i, y_i, ts, wt = self.full_ds[idx]
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

        # # If you want to plot the effect of the binning on the wind speed vs power relationship
        # # for a variable
        # plot_bin_effect(
        #     sensitive_col="MisalignmentLoss",
        #     high_windspeed=x_tr[train_prop_labels.squeeze() == 0][:,:,0].mean(axis=1),
        #     low_windspeed=x_tr[train_prop_labels.squeeze() == 1][:,:,0].mean(axis=1),
        #     high_power=y_tr.squeeze()[train_prop_labels.squeeze() == 0],
        #     low_power=y_tr.squeeze()[train_prop_labels.squeeze() == 1],
        #     save_prefix="/home/ujx4ab/ondemand/dissecting_dist_inf/datasets"
        # )

        # always drop the last (sensitive) feature used for splitting
        x_tr = x_tr[:, :, :-1]
        x_te = x_te[:, :, :-1]

        if self.data_config.poison_rate > 0:
            print(f"Poisoning {100 * self.data_config.poison_rate}% of trianing data")
            poisoned_y_tr = y_tr.clone()
            N = y_tr.shape[0] * prop_ratio

            positive_mask = (train_prop_labels.flatten() == 1)
            positive_indices = np.nonzero(positive_mask)[0]

            # how many poisons you want (max num positive)
            n_poison  = int(np.round(self.data_config.poison_rate * N))
            print(f"...poisoning {n_poison} data samples")
            n_poison  = min(n_poison, len(positive_indices))

            poison_indices = np.random.choice(positive_indices, size=n_poison, replace=False)

            poisoned_y_tr[poison_indices] = max(y_tr) if self.poisoning_direction else min(y_tr)

            return (
                (x_tr, poisoned_y_tr, train_prop_labels),
                (x_te, y_te, test_prop_labels)
            ), (train_ids, test_ids)

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
        data, splits = self.ds.get_data(
            prop_ratio=self.ratio, 
            sensitive_property=self.data_config.prop,
            n_samples_train=self.data_config.n_samples_train,
            n_samples_test=self.data_config.n_samples_test
            )

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

        if model_arch == "None":
            model_arch = self.info_object.default_model
        if model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")
        
        base_models_dir = os.path.join(base_models_dir, model_arch)

        if dp_config is None:
            if shuffle_defense_config is None:
                base_path = os.path.join(base_models_dir)
            else:
                if self.ratio == shuffle_defense_config.desired_value:
                    # When ratio of models loaded is same as target ratio of defense,
                    # simply load model of that ratio
                    base_path = os.path.join(base_models_dir)
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

        if self.ratio is not None:
            save_path = os.path.join(save_path, str(self.ratio))
        
        vc = set(self.data_config.feature_config.victim_WT_IDs)
        ac = set(self.data_config.feature_config.adv_WT_IDs) 
        all_wts = set(self.data_config.feature_config.WT_IDs)

        # model naming for saving/fetching models

        if not vc and not ac:
            WT_config = "WTs_all"
        elif vc != all_wts:
            WT_config = "group1" if self.split == "adv" else "group2"
        else:
            # vc == all_wts, i.e. every turbine is a “victim”
            WT_config = "group1" if self.split == "adv" else "WTs_all"
                
        if WT_config is not None: 
            save_path = os.path.join(save_path, WT_config)

        if WT_config is not None: 
            save_path = os.path.join(save_path, str(self.data_config.WT_config.low_q)+"q")

        if train_config.data_config.poison_rate: 
            save_path = os.path.join(save_path, f"poisoned_{train_config.data_config.poison_rate}")

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
        self.data = data
        self.targets = targets
        self.prop_labels = prop_labels
        self.squeeze = squeeze
        self.ids = ids
        self.num_samples = len(data)
        self.selection_mask = None

    def set_selection_mask(self, mask):
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
