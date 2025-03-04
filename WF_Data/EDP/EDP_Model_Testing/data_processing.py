# must be in EDP_Model_Testing directory when you run this

import os, copy
import pandas as pd
import numpy as np
import requests
import shutil
import pickle
import torch
from xlsx2csv import Xlsx2csv

import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os
import math

def get_and_save_edp_data(): 
    # Download the EDP data
    print("\n Downloading EDP data")
    url_EDP_2016 = r"https://www.edp.com/sites/default/files/2023-04/Wind-Turbine-SCADA-signals-2016.xlsx"
    url_EDP_2017 = r"https://www.edp.com/sites/default/files/2023-04/Wind-Turbine-SCADA-signals-2017_0.xlsx"

    r_EDP_2016 = requests.get(url_EDP_2016, allow_redirects=True)
    open(r'.temp/EDP_2016.xlsx', 'wb').write(r_EDP_2016.content)
    r_EDP_2017 = requests.get(url_EDP_2017, allow_redirects=True)
    open(r'.temp/EDP_2017.xlsx', 'wb').write(r_EDP_2017.content)

    print("Convert 2016")
    Xlsx2csv(r'.temp/EDP_2016.xlsx', outputencoding="utf-8").convert(r'.temp/EDP_2016.csv')
    print("Convert 2017")
    Xlsx2csv(r'.temp/EDP_2017.xlsx', outputencoding="utf-8").convert(r'.temp/EDP_2017.csv')

    df_2016 = pd.read_csv(".temp/EDP_2016.csv", parse_dates= ["Timestamp"])
    df_2017 = pd.read_csv(".temp/EDP_2017.csv", parse_dates= ["Timestamp"])

    df = pd.concat([df_2016, df_2017])
    df["Timestamp"].unique(), df_2016["Timestamp"].unique(),df_2017["Timestamp"].unique()

    df = pd.concat([df_2016, df_2017])
    for t in df["Turbine_ID"].unique():
        d = copy.deepcopy(df[df["Turbine_ID"] == t])
        tnum = t[-2:]
        print(f"Saving EDP wind turbine WT_{tnum}")
        d.to_csv(f"EDP/WT_{tnum}.csv", index=False)

    shutil.rmtree('.temp/')

def get_fault_logs(): 
    # Download the EDP data
    print("\n Downloading EDP data")
    url_EDP_faults_2016 = r"https://www.edp.com/sites/default/files/2023-04/Historical-Failure-Logbook-2016.xlsx"
    url_EDP_faults_2017 = r"https://www.edp.com/sites/default/files/2023-04/opendata-wind-failures-2017.xlsx"

    r_EDP_2016 = requests.get(url_EDP_faults_2016, allow_redirects=True)
    open(r'.temp/EDP_faults_2016.xlsx', 'wb').write(r_EDP_2016.content)
    r_EDP_2017 = requests.get(url_EDP_faults_2017, allow_redirects=True)
    open(r'.temp/EDP_faults_2017.xlsx', 'wb').write(r_EDP_2017.content)

    print("Convert Faults 2016")
    Xlsx2csv(r'.temp/EDP_faults_2016.xlsx', outputencoding="utf-8").convert(r'.temp/EDP_faults_2016.csv')
    print("Convert Faults 2017")
    Xlsx2csv(r'.temp/EDP_faults_2017.xlsx', outputencoding="utf-8").convert(r'.temp/EDP_faults_2017.csv')

    df_2016 = pd.read_csv(".temp/EDP_faults_2016.csv", parse_dates= ["Timestamp"])
    df_2017 = pd.read_csv(".temp/EDP_faults_2017.csv", parse_dates= ["Timestamp"])

    df = pd.concat([df_2016, df_2017])
    df["Timestamp"].unique(), df_2016["Timestamp"].unique(),df_2017["Timestamp"].unique()

    df = pd.concat([df_2016, df_2017])
    df.drop(columns=['Component', 'Remarks'], inplace=True)
    for t in df["Turbine_ID"].unique():
        d = copy.deepcopy(df[df["Turbine_ID"] == t])
        tnum = t[-2:]
        print(f"Saving EDP wind turbine WT_{tnum}")
        d.to_csv(f"fault_logs/WT_{tnum}.csv", index=False)

    shutil.rmtree('.temp/')

def create_dirs():
    paths = [".temp", "data_normalisation", "data_prep", "data_prep_faults_included", "EDP", "EDP_filtered", "EDP_filtered_faults_included", "fault_logs"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

class data_farm:
    def __init__(
            self, 
            name, 
            cols, 
            list_turbine_name, 
            DATA_PATH, 
            FAULT_LOG_PATH,
            rename, 
            include_faults = False
        ):
        self.name = name
        self.columns = cols
        self.list_turbine_name = list_turbine_name
        self.DATA_PATH = DATA_PATH
        self.FAULT_LOG_PATH = FAULT_LOG_PATH
        self.in_shift = 24*6
        self.out_shift = 1
        self.RESET = False
        self.load_data_window = False
        self.max_workers = 10  
        self.normalized_dict = {}
        self.normalized_circ_dict = {}
        self.rename = rename
        self.initialize = True
        self.time_name = "Timestamp"
        self.in_cols = [
            "Wind_speed",
            "Wind_speed_std",
            "Wind_rel_dir",
            "Amb_temp",
            "Gen_speed", 
            "Gen_speed_std",
            "Rotor_speed", 
            "Rotor_speed_std",
            "Blade_pitch",
            "Blade_pitch_std",
            "Gen_phase_temp_1",
            "Gen_phase_temp_2",
            "Gen_phase_temp_3",
            "Transf_temp_p1",
            "Transf_temp_p2",
            "Transf_temp_p3",
            "Gen_bearing_temp_1", 
            "Gen_bearing_temp_2", 
            "Hyd_oil_temp",
            "Gear_oil_temp",
            "Gear_bear_temp", 
            "Nacelle_position",
        ]
        self.out_cols = ["Power"]
        self.include_faults = include_faults

        self.PATH = r"data_prep" if not self.include_faults else r"data_prep_faults_included"
            
        # Initializing or loading the turbine data
        if self.initialize:
            for t in self.list_turbine_name:
                print(f"Initializing {t}")
                self.initialize_data(t)
            self.prepare_data()
        else:
            self.load_data()
        
        self.in_cols.remove('Blade_pitch')
        self.in_cols.append('Bp_bin')

        self.X, self.y, self.timestamps = {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}

    def initialize_data(self, t):
        setattr(self, t, self.load_df(f"{t}.csv", rename=self.rename, include_faults=self.include_faults))
        assert "Timestamp" in getattr(self, t).columns, f"Need a timestamp column"
        assert "Wind_speed" in getattr(self, t).columns, f"Need a Wind_speed column"
        index = pd.to_datetime(getattr(self, t)["Timestamp"]).dt.tz_convert(None) 
        getattr(self, t).set_index(index, inplace=True)
        getattr(self, t).drop(["Timestamp"], axis=1, inplace=True)

        
    def load_df(self, file_name, rename={}, include_faults = False):
        wt_df = pd.read_csv(os.path.join(self.DATA_PATH, file_name), parse_dates=[self.time_name])
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
    
        if not self.include_faults:
            fault_df = pd.read_csv(os.path.join(self.FAULT_LOG_PATH, file_name), parse_dates=[self.time_name])

            wt_df.set_index("Timestamp", inplace=True)
            for fault_time in fault_df['Timestamp']:
                start_time = fault_time - pd.Timedelta(weeks=2)
                end_time = fault_time + pd.Timedelta(days=3)
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.in_cols] = np.nan
                wt_df.loc[(wt_df.index >= start_time) & (wt_df.index <= end_time), self.out_cols] = np.nan
            filtered_path = os.path.join("/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/EDP_Model_Testing/EDP_filtered", file_name)
        else: 
            filtered_path = os.path.join("/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/EDP_Model_Testing/EDP_filtered_faults_included", file_name)
            
        cols_of_interest = (['Gen_speed', 'Gen_speed_std', 'Rotor_speed', 'Rotor_speed_std', 'Power'])
        for col in cols_of_interest:
            zero_col_with_wind = (wt_df[col] == 0) & (wt_df["Wind_speed"] > 4.0)
            if zero_col_with_wind.any():
                print("Timestamps where Wind is > 4.0 but", col, "is 0:")
                print(wt_df.index[zero_col_with_wind])
                print("Setting them to nan")
                wt_df.loc[zero_col_with_wind, self.in_cols] = np.nan
                wt_df.loc[zero_col_with_wind, self.out_cols] = np.nan
            else:
                print(f"No occurrences found for {col}.")
            
        wt_df.reset_index(drop=False, inplace=True)

        wt_df['Bp_bin'] = (wt_df['Blade_pitch'] < 5).astype(int)

        without_blade_pitch = wt_df.drop(columns = ['Blade_pitch'])
        without_blade_pitch.to_csv(filtered_path)

        return wt_df 
     
    def save_data(self,turbine = None):
        turbine = turbine if turbine is not None else self.list_turbine_name
        for t in turbine:
            getattr(self, t).to_csv(os.path.join(self.PATH, f"{self.name}_{t}.csv")) 
    
    def load_data(self, t = None):
        turbine = t if t is not None else self.list_turbine_name
        for t in turbine:
            f=os.path.join(self.PATH, f"{self.name}_{t}.csv")
            assert os.path.isfile(os.path.join(self.PATH, f"{self.name}_{t}.csv")), f"File {f} does not exists"
            setattr(self, t, pd.read_csv(os.path.join(self.PATH, f"{self.name}_{t}.csv"), index_col= "Timestamp", parse_dates= True))
    
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
    
    def normalize_min_max(self, cols):
        for t in self.list_turbine_name:
            self.normalized_dict[t] = {c : (getattr(self, t)[c].max(), getattr(self, t)[c].min()) for c in cols}
            for c in cols:
                getattr(self, t)[c] = (getattr(self,t)[c] - self.normalized_dict[t][c][1])/(self.normalized_dict[t][c][0]-self.normalized_dict[t][c][1])

        with open(os.path.join(r'data_normalisation', f'min_max_normalisation_{self.name}.pkl'), 'wb') as f:
            pickle.dump(self.normalized_dict, f)
    
    def normalize_mean_std(self, cols):
        for t in self.list_turbine_name:
            self.normalized_dict[t] = {c : (getattr(self, t)[c].mean(), getattr(self, t)[c].std()) for c in cols}
            for c in cols:
                getattr(self, t)[c] = (getattr(self,t)[c] - self.normalized_dict[t][c][0])/(self.normalized_dict[t][c][1])
 
        with open(os.path.join(r'data_normalisation', f'std_mean_normalisation_{self.name}.pkl'), 'wb') as f:
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

    def prepare_data(self): 
        if self.name == "EDP":
            self.feature_averaging("Gen_bear_temp_avg", ["Gen_bearing_temp_1", "Gen_bearing_temp_2"])
            self.feature_cut("Wind_speed", min=0, max= 25, min_value=0)
            self.feature_cut("Power", min=0,  min_value=0)
            self.feature_cut("Rotor_speed", min=0,  min_value=0)
            self.feature_cut("Gen_speed", min=0)
            self.line_cut("Wind_speed", "Power", a = 0, b= 1100, xmin = 4.2, xmax= 25)

        cols = self.out_cols + self.in_cols
        cols.remove("Blade_pitch")
        cols.append("Bp_bin")

        for t in self.list_turbine_name:
            setattr(self, t, getattr(self, t)[~getattr(self, t).index.duplicated()])

        self.drop_col([c for c in getattr(self, self.list_turbine_name[0]).columns if c not in cols])
        self.time_filling(interpolation_limit = 24)

        # Normalization
        self.normalize_min_max(cols=[c for c in cols if c not in ["Wind_speed", "Amb_temp", "Gen_speed", "Rotor_speed",
                                                             "Gen_phase_temp_1", "Gen_phase_temp_2", "Gen_phase_temp_3",
                                                             "Transf_temp_p1", "Transf_temp_p2", "Transf_temp_p3",
                                                             "Gen_bear_temp_avg", "Hyd_oil_temp",
                                                             "Gear_oil_temp", "Gear_bear_temp", "Nacelle_position"]])
        self.normalize_mean_std(cols=[c for c in cols if c in ["Wind_speed_std", "Gen_speed_std", "Rotor_speed_std"]])
        self.circ_embedding(cols=[c for c in cols if c in ["Wind_rel_dir", "Nacelle_position"]])

        # # Don't want to drop nan value right now ... going to let model ignore those
        # for t in self.list_turbine_name:
        #     getattr(self,t).dropna(how = "any", inplace = True)
        #     pd.to_datetime(getattr(self,t).index,errors='ignore') 

        self.save_data()

    def get_window_data(self, in_shift, in_cols, out_cols, PATH = r"data_prep", RESET = False):
        
        if RESET:
             self.make_window_data(in_shift, in_cols, out_cols, turbine = None,  PATH = PATH)
        else:
            turbine = [t for t in self.list_turbine_name if not (os.path.isfile(os.path.join(PATH, f"{self.name}_{t}_X.pt")) and \
                                                                 os.path.isfile(os.path.join(PATH, f"{self.name}_{t}_y.pt")) and \
                                                                 os.path.isfile(os.path.join(PATH, f"{self.name}_{t}_timestamp.pkl")))]
            
            self.make_window_data(in_shift, in_cols, out_cols, turbine= turbine, PATH= PATH)
            
        for t in self.list_turbine_name:
            setattr(self, f"{t}_X", torch.load(os.path.join(PATH, f"{self.name}_{t}_X.pt")))
            setattr(self, f"{t}_y", torch.load(os.path.join(PATH, f"{self.name}_{t}_y.pt")))
            with open(os.path.join(PATH, f"{self.name}_{t}_timestamp.pkl"), 'rb') as f:
                setattr(self, f"{t}_timestamp", pickle.load(f))

    def make_window_data(self, in_shift, in_cols, out_cols, turbine = None, PATH = r"data_prep", ret = False):
        turbine = turbine if turbine is not None else self.list_turbine_name 
        
        for t in turbine: 
            print(f"Preparing turbine {t}")
        
            data = getattr(self,t)

            in_data = data[in_cols].to_numpy(copy = True)
            out_data = data[out_cols].to_numpy(copy = True)
            
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
                X, y = torch.Tensor(in_window_full), torch.Tensor(out_window_full)
                torch.save(X, os.path.join(PATH, f"{self.name}_{t}_X.pt"))
                torch.save(y, os.path.join(PATH, f"{self.name}_{t}_y.pt"))
                with open( os.path.join(PATH, f"{self.name}_{t}_timestamp.pkl"), "wb") as f:
                    pickle.dump(time, f)
                
                if ret:
                    return X,y,time

def print_distributions(df): 
    cols = [
            "Wind_speed",
            "Wind_speed_std",
            "Wind_rel_dir",
            "Amb_temp",
            "Gen_speed", 
            "Gen_speed_std",
            "Rotor_speed", 
            "Rotor_speed_std",
            "Blade_pitch",
            "Blade_pitch_std",
            "Gen_phase_temp_1",
            "Gen_phase_temp_2",
            "Gen_phase_temp_3",
            "Transf_temp_p1",
            "Transf_temp_p2",
            "Transf_temp_p3",
            "Gen_bearing_temp_1", 
            "Gen_bearing_temp_2", 
            "Hyd_oil_temp",
            "Gear_oil_temp",
            "Gear_bear_temp", 
            "Nacelle_position",
            "Power"
        ]
    for col in cols:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df[col], kde=True, ax=ax[0], color='green')
        sns.boxplot(x=df[col], ax=ax[1])
        plt.show()

if __name__ == "__main__": 
    # create_dirs()
    # get_and_save_edp_data()
    create_dirs()
    # get_fault_logs()
    edp_path = os.path.abspath("EDP")
    if os.path.exists(edp_path): 
        for path in os.listdir(edp_path): 
            df = pd.read_csv(os.path.join(edp_path, path))

    print("\n\n Processing the data\n")
    # EDP
    cols_edp = [
        "Timestamp", 
        "Wind_speed",
        "Wind_speed_std",
        "Wind_rel_dir",
        "Amb_temp",
        "Gen_speed", 
        "Gen_speed_std",
        "Rotor_speed", 
        "Rotor_speed_std",
        "Blade_pitch",
        "Blade_pitch_std",
        "Gen_phase_temp_1",
        "Gen_phase_temp_2",
        "Gen_phase_temp_3",
        "Transf_temp_p1",
        "Transf_temp_p2",
        "Transf_temp_p3",
        "Gen_bearing_temp_1", 
        "Gen_bearing_temp_2", 
        "Hyd_oil_temp",
        "Gear_oil_temp",
        "Gear_bear_temp", 
        "Nacelle_position",
        "Power"
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
    DATA_PATH_EDP = r"EDP"
    FAULT_LOG_PATH_EDP = r"fault_logs"
    include_faults = True

    farm_edp = data_farm(
        name = "EDP", 
        cols = cols_edp, 
        list_turbine_name=list_names_edp,
        DATA_PATH=DATA_PATH_EDP,
        FAULT_LOG_PATH = FAULT_LOG_PATH_EDP,
        rename = rename_edp, 
        include_faults=include_faults
    )

    in_cols = [
        "Wind_speed",
        "Wind_speed_std",
        "Wind_rel_dir",
        "Amb_temp",
        "Gen_speed", 
        "Gen_speed_std",
        "Rotor_speed", 
        "Rotor_speed_std",
        "Blade_pitch_std",
        "Gen_phase_temp_1",
        "Gen_phase_temp_2",
        "Gen_phase_temp_3",
        "Transf_temp_p1",
        "Transf_temp_p2",
        "Transf_temp_p3",
        "Gen_bearing_temp_1", 
        "Gen_bearing_temp_2", 
        "Hyd_oil_temp",
        "Gear_oil_temp",
        "Gear_bear_temp", 
        "Nacelle_position",
        "Bp_bin"
    ]
    out_cols = ["Power"]

    print("\n\n Preparing data windows\n")
    print("\nEDP")
    if include_faults: 
        farm_edp.get_window_data(in_shift=24*6, in_cols=in_cols, out_cols=out_cols, PATH = r"data_prep_faults_included", RESET = True)
    else: 
        farm_edp.get_window_data(in_shift=24*6, in_cols=in_cols, out_cols=out_cols, PATH = r"data_prep", RESET = True)