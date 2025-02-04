import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def load_and_filter_data(data_directory):
    csv_files = [file for file in os.listdir(data_directory) if file.endswith('.csv')]
    filtered_dataframes = []

    for file_name in csv_files:
        file_path = os.path.join(data_directory, file_name)
        df = pd.read_csv(file_path, delimiter=';')
        filtered_df = df[df['status_type_id'] == 0]
        filtered_dataframes.append(filtered_df)

    filtered_df = pd.concat(filtered_dataframes, ignore_index=True)
    return filtered_df

def drop_high_zero_columns(filtered_df):
    total_rows = len(filtered_df)
    threshold = total_rows * 0.8
    columns_to_drop = [col for col in filtered_df.columns if (filtered_df[col] == 0).sum() > threshold]
    filtered_df = filtered_df.drop(columns=columns_to_drop)
    print(f"Columns dropped: {columns_to_drop}")
    return filtered_df

def rename_and_select_columns(filtered_df):
    columns_to_keep = {
        "time_stamp": "timestamp",
        "sensor_0_avg": "ambient_temperature",
        "sensor_2_avg": "wind_relative_direction",
        "wind_speed_3_avg": "wind_speed",
        "sensor_5_avg": "pitch_angle", 
        "sensor_50": "total_active_power", 
        "sensor_18_avg": "generator_rpm",
        "sensor_52_avg": "rotor_rpm",
        "sensor_12_avg": "gearbox_oil_temp"
    }
    filtered_df = filtered_df[list(columns_to_keep.keys())]
    filtered_df.rename(columns=columns_to_keep, inplace=True)
    return filtered_df

def categorize_columns(filtered_df):
    gearbox_bins = [-np.inf, 70, np.inf]
    gearbox_labels = [0.0, 1.0]
    filtered_df['gearbox_oil_temp'] = pd.cut(filtered_df['gearbox_oil_temp'], bins=gearbox_bins, labels=gearbox_labels, include_lowest=True)
    pitch_bins = [-np.inf, 10, np.inf]
    pitch_labels = [1.0, 0.0]
    filtered_df['pitch_angle_bin'] = pd.cut(filtered_df['pitch_angle'], bins=pitch_bins, labels=pitch_labels, include_lowest=True)

    filtered_df = filtered_df.drop(columns=['pitch_angle', 'timestamp']) #, 'gearbox_oil_temp'])

    gearbox_counts = filtered_df['gearbox_oil_temp'].value_counts(sort=False)

    # Check counts for pitch_angle_bin
    pitch_counts = filtered_df['pitch_angle_bin'].value_counts(sort=False)

    # Print counts including zero categories
    print("gearbox_oil_temp Counts:")
    print(gearbox_counts)

    print("\nPitch Angle Bin Counts:")
    print(pitch_counts)

    return filtered_df

def save_to_csv(filtered_df, output_path):
    # Save with headers to retain column names
    filtered_df.to_csv(output_path, index=False, header=True)

def split_and_save_data(filtered_df, output_dir):
    train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)

    train_file_path = os.path.join(output_dir, "train.p")
    test_file_path = os.path.join(output_dir, "test.p")

    # Save the split datasets while retaining column names
    with open(train_file_path, "wb") as train_file:pickle.dump(train_df, train_file)

    with open(test_file_path, "wb") as test_file:pickle.dump(test_df, test_file)

def load_split_data(output_dir):
    train_file_path = os.path.join(output_dir, "train.p")
    test_file_path = os.path.join(output_dir, "test.p")

    # Load the split datasets with column names
    train_data = pickle.load(open(train_file_path, 'rb'))
    test_data = pickle.load(open(test_file_path, 'rb'))

    return train_data, test_data

def main():
    data_directory = "/home/ujx4ab/ondemand/dissecting_dist_inf/WF_Data/EDP/datasets"
    output_dir = "/home/ujx4ab/ondemand/dissecting_dist_inf/datasets/edp"
    output_csv_path = "/home/ujx4ab/ondemand/dissecting_dist_inf/datasets/edp/combined_data.csv"

    filtered_df = load_and_filter_data(data_directory)
    filtered_df = drop_high_zero_columns(filtered_df)

    print("Renaming and selecting columns...")
    filtered_df = rename_and_select_columns(filtered_df)
    filtered_df = categorize_columns(filtered_df)

    print("Saving to CSV...")
    save_to_csv(filtered_df, output_csv_path)

    print("Splitting and saving data...")
    split_and_save_data(filtered_df, output_dir)

    print("Loading split data for verification...")
    train_data, test_data = load_split_data(output_dir)

    print("Train Data Sample:")
    print(train_data.head())

if __name__ == "__main__":
    main()