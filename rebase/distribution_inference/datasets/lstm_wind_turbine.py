from distribution_inference.defenses.active.shuffle import ShuffleDefense
from sklearn.model_selection import StratifiedShuffleSplit

import sys
import os
import pickle
import pandas as pd
import numpy as np
import os
import torch as ch
import torch.nn as nn

from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.models.core import MaskedLSTM
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model

# sys.path.append(os.path.dirname(os.getcwd()))
# from WF_Data.EDP import EDP_Model_Testing

def get_mode_from_script():
    script_name = os.path.basename(sys.argv[0]).lower()
    print(f"script name: {script_name}")
    return "train" if "train" in script_name else "attack"

class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False, data_path: str = "LSTM_WTs"):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="LSTM_Wind_Turbines",
                         data_path=data_path,
                         models_path="lstm_wind_turbines",
                         properties=["Bp_bin"], 
                         values={"wind_turbines": ratios},
                         supported_models=["MaskedLSTM"],
                         property_focus={"Bp_bin": 1.0},
                         default_model="MaskedLSTM",
                         epoch_wise=epoch_wise)

    def get_model(self, cpu: bool = True, model_arch: str = None) -> nn.Module:
        device = "cuda" if ch.cuda.is_available() else "cpu"
        if model_arch is None or model_arch=="None":
            model_arch = self.default_model
        elif model_arch == "MaskedLSTM":
            model = MaskedLSTM(22, 64, 1, use_dropout=False).to(device)
        else:
            raise NotImplementedError("Model architecture not supported")
        
        if not model.is_sklearn_model and not cpu:
            model = model.cuda()
        return model
    
class _WindTurbinePower:
    def __init__(self, data_config, drop_senstive_cols=False):
        self.drop_senstive_cols = drop_senstive_cols
        self.columns = [
            "Wind_speed", "Wind_speed_std", "Wind_rel_dir", "Amb_temp",
            "Gen_speed", "Gen_speed_std", "Rotor_speed", "Rotor_speed_std",
            "Blade_pitch_std", "Gen_phase_temp_1", "Gen_phase_temp_2",
            "Gen_phase_temp_3", "Transf_temp_p1", "Transf_temp_p2", "Transf_temp_p3",
            "Gen_bearing_temp_1", "Gen_bearing_temp_2", "Hyd_oil_temp", "Gear_oil_temp",
            "Gear_bear_temp", "Nacelle_position",  "Bp_bin", "Power"
        ]
        self.data_config = data_config

        # two flags to determine which data is used for loaders
        # self.mode = get_mode_from_script()
        self.split = data_config.split 
        self.load_data(test_ratio=0.4)
    
    # Create adv/victim splits
    def load_data(self, test_ratio, random_state: int = 42):
        data_path = "LSTM_WTs/faults" if self.data_config.WT_config.use_faults else "LSTM_WTs/normal"
        info_object = DatasetInformation(data_path=data_path)
        if self.data_config.WT_config.turbines is not None: 
            victim_WTs = ["WT_" + WT_nbr for WT_nbr in self.data_config.WT_config.turbines]

        self.train_X, self.train_y, self.train_masks, _ = utils.aggregate_data(info_object.base_data_dir, victim_WTs, "train")
        self.test_X, self.test_y, self.test_masks, _ = utils.aggregate_data(info_object.base_data_dir, victim_WTs, "test")

        # if self.mode == "train" and self.split == "victim":
        #     base_train = os.path.join(info_object.base_data_dir, f"{self.victim_WT}_train_dataset")
        #     base_test = os.path.join(info_object.base_data_dir, f"{self.victim_WT}_test_dataset")
        # else:
        #     base_train = os.path.join(info_object.base_data_dir, "agg_train_dataset")
        #     base_test = os.path.join(info_object.base_data_dir, "agg_test_dataset")

        
        # self.train_X = np.load(os.path.join(base_train, "X.npy"), allow_pickle=True)
        # self.train_y = np.load(os.path.join(base_train, "y.npy"), allow_pickle=True)
        # self.train_masks = np.load(os.path.join(base_train, "mask.npy"), allow_pickle=True)
        
        # self.test_X = np.load(os.path.join(base_test, "X.npy"), allow_pickle=True)
        # self.test_y = np.load(os.path.join(base_test, "y.npy"), allow_pickle=True)
        # self.test_masks = np.load(os.path.join(base_test, "mask.npy"), allow_pickle=True)
        
        print(f"Size of train, test data: {self.train_X.shape}, {self.test_X.shape}")
        
        def compute_binned_features(X, y, masks):
            feature_values = X[:, :, -1]
            masks_float = masks.astype(float)
            masked_sum = np.sum(feature_values * masks_float, axis=1)
            valid_counts = np.sum(masks_float, axis=1)
            valid_counts_clamped = np.clip(valid_counts, a_min=1, a_max=None)
            average_Bp_bin_values = masked_sum / valid_counts_clamped

            average_power_values = np.mean(y, axis=1)
            averages_for_indexing = np.stack((average_Bp_bin_values, average_power_values), axis=1)

            binned_Bp = np.digitize(averages_for_indexing[:, 0],
                                    bins=np.linspace(-0.01, 1.01, 3),
                                    right=False) - 1
            binned_power = np.digitize(averages_for_indexing[:, 1],
                                    bins=np.linspace(-0.01, 1.01, 3),
                                    right=False) - 1
            binned_feature_avgs = np.stack((binned_Bp, binned_power), axis=1)

            return pd.DataFrame(averages_for_indexing, columns=['Bp_bin', 'Power_bin']), pd.DataFrame(binned_feature_avgs, columns=['Bp_bin', 'Power_bin'])

        # Compute binned features for both train, test sets
        train_averages, train_binned_features = compute_binned_features(self.train_X, self.train_y, self.train_masks)
        test_averages, test_binned_features = compute_binned_features(self.test_X, self.test_y, self.test_masks)

        train_binned_features["indices"] = train_binned_features.index
        test_binned_features["indices"] = test_binned_features.index

        # Create train/test splits using stratified sampling based on the binned features
        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(
                n_splits=1,            # Only generate one train-test split 
                test_size=test_ratio,  # Fraction of dataset to split by
                random_state=rs
            )
            splitter = sss.split(np.zeros(len(this_df)), this_df[["Bp_bin", "Power_bin"]])
            return next(splitter)

        train_vic_split_indices, train_adv_split_indices = s_split(train_binned_features)
        test_vic_split_indices, test_adv_split_indices = s_split(test_binned_features)

        # safety checks
        common_train_indices = np.intersect1d(train_vic_split_indices, train_adv_split_indices)
        assert len(common_train_indices) == 0, "Train splits overlap!"
        common_test_indices = np.intersect1d(test_vic_split_indices, test_adv_split_indices)
        assert len(common_test_indices) == 0, "Test splits overlap!"
        common_adv_indices = np.intersect1d(train_adv_split_indices, test_adv_split_indices)
        assert len(common_train_indices) == 0, "Adv splits overlap!"
        common_vic_indices = np.intersect1d(train_vic_split_indices, test_vic_split_indices)
        assert len(common_test_indices) == 0, "Vic splits overlap!"
        assert len(train_binned_features) == len(self.train_X)

        # Extract the proper data splits for victim/adv models
        self.train_df_X_vic = self.train_X[train_vic_split_indices]
        self.train_df_y_vic = self.train_y[train_vic_split_indices]
        self.train_df_X_adv = self.train_X[train_adv_split_indices]
        self.train_df_y_adv = self.train_y[train_adv_split_indices]

        self.test_df_X_vic = self.test_X[test_vic_split_indices]
        self.test_df_y_vic = self.test_y[test_vic_split_indices]
        self.test_df_X_adv = self.test_X[test_adv_split_indices]
        self.test_df_y_adv = self.test_y[test_adv_split_indices]

        # Duplicate data for the original data splitting pipeline
        self.train_df_victim, self.train_df_adv = train_binned_features.iloc[train_vic_split_indices], train_binned_features.iloc[train_adv_split_indices]
        self.test_df_victim, self.test_df_adv = test_binned_features.iloc[test_vic_split_indices], test_binned_features.iloc[test_adv_split_indices]

        print(f"Ratio of property of interest in vic, adv train dataset (no mask): "
            f"{self.train_df_victim['Bp_bin'].mean()}, {self.train_df_adv['Bp_bin'].mean()}")
        print(f"Ratio of property of interest in vic, adv test dataset (no mask): "
            f"{self.test_df_victim['Bp_bin'].mean()}, {self.test_df_adv['Bp_bin'].mean()}")

        # # only want to use specific WT data when victim models
        # if self.mode == "train" and self.split == "victim":
        #     self.train_X = np.load(os.path.join(info_object.base_data_dir, f"{self.victim_WT}_train_dataset", "X.npy"), allow_pickle=True)
        #     self.train_y = np.load(os.path.join(info_object.base_data_dir, f"{self.victim_WT}_train_dataset", "y.npy"), allow_pickle=True)
        #     self.train_masks = np.load(os.path.join(info_object.base_data_dir, f"{self.victim_WT}_train_dataset", "mask.npy"), allow_pickle=True)

        #     self.test_X = np.load(os.path.join(info_object.base_data_dir, f"{self.victim_WT}_test_dataset", "X.npy"), allow_pickle=True)
        #     self.test_y = np.load(os.path.join(info_object.base_data_dir, f"{self.victim_WT}_test_dataset", "y.npy"), allow_pickle=True)
        #     self.test_masks = np.load(os.path.join(info_object.base_data_dir, f"{self.victim_WT}_test_dataset", "mask.npy"), allow_pickle=True)

        # else:  # Adv models use aggregate and attack data is aggregate
        #     self.train_X = np.load(os.path.join(info_object.base_data_dir, "agg_train_dataset", "X.npy"), allow_pickle=True)
        #     self.train_y = np.load(os.path.join(info_object.base_data_dir, "agg_train_dataset", "y.npy"), allow_pickle=True)
        #     self.train_masks = np.load(os.path.join(info_object.base_data_dir, "agg_train_dataset", "mask.npy"), allow_pickle=True)

        #     self.test_X = np.load(os.path.join(info_object.base_data_dir, "agg_test_dataset", "X.npy"), allow_pickle=True)
        #     self.test_y = np.load(os.path.join(info_object.base_data_dir, "agg_test_dataset", "y.npy"), allow_pickle=True)
        #     self.test_masks = np.load(os.path.join(info_object.base_data_dir, "agg_test_dataset", "mask.npy"), allow_pickle=True)

        # print(f"Size of train, test data: {self.train_X.shape}, {self.test_X.shape}")

        # train_x_data = np.mean(self.train_X, axis=1)
        # train_data = np.hstack((train_x_data, self.train_y))
        # self.train_df = pd.DataFrame(train_data, columns=self.columns)
        
        # test_x_data = np.mean(self.test_X, axis=1)
        # test_data = np.hstack((test_x_data, self.test_y))
        # self.test_df = pd.DataFrame(test_data, columns=self.columns)

        # print(f"Ratio of property of interest in test, train datasets (no mask): {self.train_df['Bp_bin'].mean()}, {self.test_df['Bp_bin'].mean()}")
        
        # Bp_bin_edges = np.linspace(-0.01, 1.01, 3)  # For Bp
        # Power_bin_edges = np.linspace(-0.01, 1.01, 3)  # For Power

        # self.train_df['Bp_bin'] = np.digitize(self.train_df['Bp_bin'], bins=Bp_bin_edges, right=False) - 1
        # self.train_df['Power_bin'] = np.digitize(self.train_df['Power'], bins=Power_bin_edges, right=False) - 1
        # self.train_df.drop(columns=['Power'], inplace=True)
        # self.train_df["indices"] = self.train_df.index

        # self.test_df['Bp_bin'] = np.digitize(self.test_df['Bp_bin'], bins=Bp_bin_edges, right=False) - 1
        # self.test_df['Power_bin'] = np.digitize(self.test_df['Power'], bins=Power_bin_edges, right=False) - 1
        # self.test_df.drop(columns=['Power'], inplace=True)
        # self.test_df["indices"] = self.test_df.index

        # def s_split(this_df, rs=random_state):
        #     sss = StratifiedShuffleSplit(n_splits=1,            # Only generate one train-test split 
        #                                  test_size=test_ratio,  # Fraction of dataset to split by
        #                                  random_state=rs)      
            
        #     splitter = sss.split(np.zeros(len(this_df)), this_df[["Bp_bin", "Power_bin"]]) 
        #     return next(splitter)                                

        # # Create train/test splits for victim/adv
        # train_vic_split_indices, train_adv_split_indices = s_split(self.train_df)
        # test_vic_split_indices, test_adv_split_indices = s_split(self.test_df)

        # # used for final extraction of proper data of interest
        # self.train_df_X_vic = self.train_X[train_vic_split_indices]
        # self.train_df_y_vic = self.train_y[train_vic_split_indices]
        # self.train_df_X_adv = self.train_X[train_adv_split_indices]
        # self.train_df_y_adv = self.train_y[train_adv_split_indices]

        # self.test_df_X_vic = self.test_X[test_vic_split_indices]
        # self.test_df_y_vic = self.test_y[test_vic_split_indices]
        # self.test_df_X_adv = self.test_X[test_adv_split_indices]
        # self.test_df_y_adv = self.test_y[test_adv_split_indices]

        # # Duplicate data for sake of keeping original data splitting pipeline
        # self.train_df_victim, self.train_df_adv = self.train_df.iloc[train_vic_split_indices], self.train_df.iloc[train_adv_split_indices]
        # self.test_df_victim, self.test_df_adv = self.test_df.iloc[test_vic_split_indices], self.test_df.iloc[test_adv_split_indices]
        

        # print(f"Ratio of property of interest in vic, adv train dataset (no mask): {self.train_df_victim['Bp_bin'].mean()}, {self.train_df_adv['Bp_bin'].mean()}")
        # print(f"Ratio of property of interest in vic, adv test dataset (no mask): {self.test_df_victim['Bp_bin'].mean()}, {self.test_df_adv['Bp_bin'].mean()}")


    def _get_prop_label_lambda(self, filter_prop):
        if filter_prop == "Bp_bin":
            def lambda_fn(x): return x['Bp_bin'] == 1
        else:
            raise NotImplementedError(f"Property {filter_prop} not supported")
        return lambda_fn
 
    def get_data(self, split, prop_ratio, filter_prop,
                 custom_limit=None,
                 scale: float = 1.0,
                 label_noise:float = 0,
                 indeces = None):
        lambda_fn = self._get_prop_label_lambda(filter_prop) # function to filter for prop of interest with

        def prepare_one_set(TRAIN_DF, TEST_DF):
            print(f"Train, Test sizes: {TRAIN_DF.shape}, {TEST_DF.shape}")
            # Apply filter to data
            if indeces:
                TRAIN_DF_ = TRAIN_DF.iloc[indeces[0]].reset_index(drop=True)
                train_ids = None
            else:
                TRAIN_DF_, train_ids = self.get_filter(TRAIN_DF, filter_prop,
                                           split, prop_ratio, is_test=0,
                                           custom_limit=custom_limit,
                                           scale=scale)
            train_prop_labels = 1 * (lambda_fn(TRAIN_DF_).to_numpy())
            if indeces:
                TEST_DF_ = TEST_DF.iloc[indeces[1]].reset_index(drop=True)
                test_ids = None
            else:
                TEST_DF_, test_ids = self.get_filter(TEST_DF, filter_prop,
                                          split, prop_ratio, is_test=1,
                                          custom_limit=custom_limit,
                                          scale=scale)
            test_prop_labels = 1 * (lambda_fn(TEST_DF_).to_numpy())

            print(f"Train, Test prop ratio from filter: {TRAIN_DF_['Bp_bin'].mean()}, {TEST_DF_['Bp_bin'].mean()}")

            (x_tr, y_tr, mask_tr) = self.get_x_y("train", split, train_ids)
            (x_te, y_te, mask_te) = self.get_x_y("test", split, test_ids)

            # (x_tr, y_tr, mask, cols), (x_te, y_te, mask, cols) = self.get_x_y(
            #     "train", train_ids), self.get_x_y("test", test_ids)
            # if label_noise:
            #     idx = np.random.choice(len(y_tr), int(
            #         label_noise*len(y_tr)), replace=False)
            #     y_tr[idx, 0] = 1 - y_tr[idx, 0]

            # average value of prop of interest is going to be different in TEST_DF_ and TRAIN_DF_
            # since the avg for the time window for each is computed without the masked data points
            # and used to select the indices. The real data doesn't have the points masked, so when 
            # you average it will look like the data is different.
            return ((x_tr, y_tr, mask_tr, train_prop_labels), (x_te, y_te, mask_tr, test_prop_labels)), (train_ids, test_ids)
        
        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    # Return data with desired property ratios
    def get_x_y(self, name, split, indices):
        if name == "train":
            if split == "victim":
                X, Y = self.train_df_X_vic, self.train_df_y_vic
            elif split == "adv":
                X, Y = self.train_df_X_adv, self.train_df_y_adv
            mask = self.train_masks
        elif name == "test":
            if split == "victim":
                X, Y = self.test_df_X_vic, self.test_df_y_vic
            elif split == "adv":
                X, Y = self.test_df_X_adv, self.test_df_y_adv
            mask = self.test_masks
        
        print(f"X, Y shape for getting true data points: {X.shape}, {Y.shape}")

        return (X[indices], np.expand_dims(Y[indices], 1), mask[indices])

    
    # Fet appropriate filter with sub-sampling according to ratio and property
    def get_filter(self, df, filter_prop, split, ratio, is_test,
                   custom_limit=None, scale: float = 1.0):
        if filter_prop == "none":
            return df
        else:
            lambda_fn = self._get_prop_label_lambda(filter_prop)

        prop_wise_subsample_sizes = {
            "adv": {
                "Bp_bin": (60000, 30000),
            },
            "victim": {
                "Bp_bin": (140000, 70000),
            },
        }

        # if "victim" split, we want to use 1/4 size since we are only using 
        # 1 of the 4 turbines for EDP
        if self.split == "victim":  
            prop_wise_subsample_sizes = {
                key: {subkey: (val[0] / 4, val[1] / 4) for subkey, val in subdict.items()}
                for key, subdict in prop_wise_subsample_sizes.items()
            }

        if custom_limit is None:
            subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
            subsample_size = int(scale*subsample_size)
        else:
            subsample_size = custom_limit

        # utils.heuristic(df, lambda_fn, ratio,cwise_sample=None,class_imbalance=None,tot_samples=subsample_size,n_tries=100,class_col='gearbox_oil_temp',verbose=True,get_indices=True)
        return utils.heuristic(df, lambda_fn, ratio,
                               cwise_sample=None,
                               class_imbalance=None,
                               tot_samples=subsample_size,
                               n_tries=100,
                               class_col='Bp_bin',
                               verbose=True,
                               get_indices=True)
        

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
            # Initialize your WindTurbine data loader
            self.ds = _WindTurbinePower(data_config, drop_senstive_cols=self.drop_senstive_cols)
        self.info_object = DatasetInformation(epoch_wise=epoch)
        
    def load_data(self, custom_limit=None, indexed_data=None):
        # Get data from WindTurbinePower
        data, splits = self.ds.get_data(
            split=self.split,
            prop_ratio=self.ratio,
            filter_prop=self.prop,
            custom_limit=custom_limit,
            scale=self.scale,
            label_noise=self.label_noise,
            indeces=indexed_data
        )
        # Track used indices for later analysis
        self._used_for_train = splits[0]  # Train indices from splits
        self._used_for_test = splits[1]   # Test/val indices
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

        WT_config = "WTs_"+"_".join(self.ds.data_config.WT_config.turbines)
        
        if WT_config is not None: 
            save_path = os.path.join(save_path, WT_config)

        if self.ratio is not None:
            save_path = os.path.join(save_path, str(self.ratio))

        if self.scale != 1.0:
            save_path = os.path.join(
                self.scalesave_path, "sample_size_scale:{}".format(self.scale))
        if self.drop_senstive_cols:
            save_path = os.path.join(save_path, "drop")

        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # print("Loading models from path", save_path)
        breakpoint()
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
        input_mask = self.input_masks[index_]   # the original "masks" for when data is missing
        prop_label = self.prop_labels[index_]

        if self.squeeze:
            y = y.squeeze()
            prop_label = prop_label.squeeze()

        data_bundle = {
            "features": x, 
            "mask": input_mask, 
        }

        return data_bundle, y, prop_label