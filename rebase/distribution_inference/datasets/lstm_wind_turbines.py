from distribution_inference.defenses.active.shuffle import ShuffleDefense
from sklearn.model_selection import StratifiedShuffleSplit

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

class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="LSTM_Wind_Turbines",
                         data_path="LSTM_WTs",
                         models_path="lstm_wind_turbines",
                         properties=["pitch_angle_bin"], 
                         values={"wind_turbines": ratios},
                         supported_models=["random_forest"],
                         property_focus={"pitch_angle_bin": 1.0},
                         default_model="MaskedLSTM",
                         epoch_wise=epoch_wise)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self, cpu: bool = True, model_arch: str = None) -> nn.Module:
        if model_arch is None or model_arch=="None":
            model_arch = self.default_model
        elif model_arch == "MaskedLSTM":
            model = MaskedLSTM(22, 64, 1, use_dropout=False).to(DEVICE)
        else:
            raise NotImplementedError("Model architecture not supported")
        
        if not model.is_sklearn_model and not cpu:
            model = model.cuda()
        return model
    
class _WindTurbineGearboxOilTemperature:
    def __init__(self, drop_senstive_cols=False):
        self.drop_senstive_cols = drop_senstive_cols
        self.columns = [
            "Wind_speed", "Wind_speed_std", "Wind_rel_dir", "Amb_temp",
            "Gen_speed", "Gen_speed_std", "Rotor_speed", "Rotor_speed_std",
            "Blade_pitch", "Blade_pitch_std", "Gen_phase_temp_1", "Gen_phase_temp_2",
            "Gen_phase_temp_3", "Transf_temp_p1", "Transf_temp_p2", "Transf_temp_p3",
            "Gen_bearing_temp_1", "Gen_bearing_temp_2", "Hyd_oil_temp", "Gear_oil_temp",
            "Gear_bear_temp", "Nacelle_position", "Power"
        ]
        self.load_data(test_ratio=0.4)
    
    # Create adv/victim splits
    def load_data(self, test_ratio, random_state: int = 42):
        info_object = DatasetInformation()
        
        # Load train, test data
        train_data = pickle.load(
            open(os.path.join(info_object.base_data_dir, 'train.p'), 'rb'))
        test_data = pickle.load(
            open(os.path.join(info_object.base_data_dir, 'test.p'), 'rb'))
        
        self.train_df = pd.DataFrame(train_data, columns=self.columns)
        self.test_df = pd.DataFrame(test_data, columns=self.columns)
        
        # Add field to identify train/test, process together
        self.train_df['is_train'] = 1
        self.test_df['is_train'] = 0
        df = pd.concat([self.train_df, self.test_df], axis=0)        

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']
                                         == 1], df[df['is_train'] == 0]
        self.train_df = self.train_df.drop(columns=['is_train'], axis=1)
        self.test_df = self.test_df.drop(columns=['is_train'], axis=1)

        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(n_splits=1,            # Only generate one train-test split 
                                         test_size=test_ratio,  # Fraction of dataset to split by
                                         random_state=rs)       # Seed
            # Stratification on the properties we care about for this dataset
            # so that adv/victim split does not introduce
            # unintended distributional shift
            
            splitter = sss.split(
                this_df, this_df[["Blade_pitch", "Power"]])  # Statify based on pitch_angle and gearbox_oil_temp, so ration of each is abt the same in train/test split
            split_1, split_2 = next(splitter)                           # generates the indices for train/test based statification
            return this_df.iloc[split_1], this_df.iloc[split_2]         # subsets based off indices

        # Create train/test splits for victim/adv
        self.train_df_victim, self.train_df_adv = s_split(self.train_df)
        self.test_df_victim, self.test_df_adv = s_split(self.test_df)
                
    def _get_prop_label_lambda(self, filter_prop):
        if filter_prop == "pitch_angle_bin":
            def lambda_fn(x): return x['pitch_angle_bin'] == 1
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

            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(
                TRAIN_DF_), self.get_x_y(TEST_DF_)
            if label_noise:
                #shape of y: (length,1)
                idx = np.random.choice(len(y_tr), int(
                    label_noise*len(y_tr)), replace=False)
                y_tr[idx, 0] = 1 - y_tr[idx, 0]

            # breakpoint() to check if data has correct ratio

            return ((x_tr, y_tr,train_prop_labels), (x_te, y_te,test_prop_labels), cols), (train_ids, test_ids)
        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)
    
    # Fet appropriate filter with sub-sampling according to ratio and property
    def get_filter(self, df, filter_prop, split, ratio, is_test,
                   custom_limit=None, scale: float = 1.0):

    
        if filter_prop == "none":
            return df
        else:
            lambda_fn = self._get_prop_label_lambda(filter_prop)

        prop_wise_subsample_sizes = {
            # selected these based on same ratios as new census, but not 100%
            # what they are for
            "adv": {
                "pitch_angle_bin": (30000, 15000),
            },
            "victim": {
                "pitch_angle_bin": (70000, 35000),
            },
        }

        if custom_limit is None:
            subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
            subsample_size = int(scale*subsample_size)
        else:
            subsample_size = custom_limit
        # utils.heuristic(df, lambda_fn, ratio,cwise_sample=None,class_imbalance=None,tot_samples=subsample_size,n_tries=100,class_col='gearbox_oil_temp',verbose=True,get_indices=True)
        return utils.heuristic(df, lambda_fn, ratio,
                            #    subsample_size,
                               cwise_sample=None,
                               class_imbalance=None,
                            #    class_imbalance=0.7211,  # Calculated based on original distribution
                               tot_samples=subsample_size,
                               n_tries=100,
                               class_col='gearbox_oil_temp',
                               verbose=True,
                               get_indices=True)

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        Y = P['gearbox_oil_temp'].to_numpy()
        cols_drop = ['gearbox_oil_temp']
        if self.drop_senstive_cols:
            cols_drop += ['pitch_angle_bin']
        X = P.drop(columns=cols_drop, axis=1)
        # Convert specific columns to one-hot
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

class WindTurbineWrapper(base.CustomDatasetWrapper):
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
            self.ds = _WindTurbineGearboxOilTemperature(drop_senstive_cols=self.drop_senstive_cols)
        self.info_object = DatasetInformation(epoch_wise=epoch)
        
    def load_data(self, custom_limit=None, indexed_data=None):
        data, splits = self.ds.get_data(split=self.split,
                                    prop_ratio=self.ratio,
                                    filter_prop=self.prop,
                                    custom_limit=custom_limit,
                                    scale=self.scale,
                                    label_noise = self.label_noise,
                                    indeces=indexed_data)
        self._used_for_train = splits[0]
        self._used_for_test = splits[1]
        return data
    
    def get_used_indices(self):
        train_ids_after = self.ds_train.ids
        val_ids_after = self.ds_val.ids
        return (self._train_ids_before, train_ids_after), (self._val_ids_before, val_ids_after)

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    indexed_data=None):
        train_data, val_data, _ = self.load_data(self.cwise_samples, indexed_data=indexed_data)
        self.ds_train = WindTurbineSet(*train_data, squeeze=self.squeeze, ids = self._used_for_train)
        self.ds_val = WindTurbineSet(*val_data, squeeze=self.squeeze, ids = self._used_for_test)
        self._train_ids_before = self.ds_train.ids
        self._val_ids_before = self.ds_val.ids
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,prefetch_factor=None)

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
        return save_path
    
class WindTurbineSet(base.CustomDataset):
    def __init__(self, data, targets, prop_labels, squeeze=False, ids=None):
        super().__init__()
        self.data = ch.from_numpy(data).float()
        self.targets = ch.from_numpy(targets).float()
        self.prop_labels = ch.from_numpy(prop_labels).float()
        self.squeeze = squeeze
        self.num_samples = len(self.data)
        self.ids = ids
    
    def mask_data_selection(self, mask):
        self.mask = mask
        self.num_samples = len(self.mask)
        if self.ids is not None:
            self.ids = self.ids[mask]

    def __getitem__(self, index):
        index_ = self.mask[index] if self.mask is not None else index
        x = self.data[index_]
        y = self.targets[index_]
        prop_label = self.prop_labels[index_]

        if self.squeeze:
            y = y.squeeze()
            prop_label = prop_label.squeeze()
        return x, y, prop_label
