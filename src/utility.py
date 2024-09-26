#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:40:01 2024

@author: hubble
"""

import numpy as np, pandas as pd, os, pickle, glob
from natsort import natsorted
from scipy.signal import find_peaks
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
sns.set_palette("colorblind")

#%%

# subjects_threshold_dict = {
#     'subject_1': {
#         'PlugTvHall.csv': 30,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': 1500,
#         'WashingMachine.csv': 1300,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_2': {
#         'PlugTvHall.csv': 60,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': None,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_3': {
#         'PlugTvHall.csv': 15,
#         'CoffeMachine.csv': 800,
#         'Microwave.csv': 1000,
#         'WashingMachine.csv': None,
#         'Printer.csv': 250,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_4': {
#         'PlugTvHall.csv': None,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': 1000,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_5': {
#         'PlugTvHall.csv': 45,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': 1200,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_7': {
#         'PlugTvHall.csv': 15,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': None,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': 15
#     },
#     'subject_8': {
#         'PlugTvHall.csv': 30,
#         'CoffeMachine.csv': 400,
#         'Microwave.csv': None,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_9': {
#         'PlugTvHall.csv': 60,
#         'CoffeMachine.csv': 800,
#         'Microwave.csv': None,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_11': {
#         'PlugTvHall.csv': 30,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': None,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': 15
#     },
#     'subject_12': {
#         'PlugTvHall.csv': 20,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': None,
#         'WashingMachine.csv': None,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_13': {
#         'PlugTvHall.csv': None,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': 1000,
#         'WashingMachine.csv': 1500,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     },
#     'subject_14': {
#         'PlugTvHall.csv': 30,
#         'CoffeMachine.csv': None,
#         'Microwave.csv': None,
#         'WashingMachine.csv': 1300,
#         'Printer.csv': None,
#         'PlugTvKitchen.csv': None
#     }
# }

subjects_threshold_dict = {
    'subject_1': {
        'PlugTvHall': 30,
        'CoffeMachine': None,
        'Microwave': 1500,
        'WashingMachine': 1300,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_2': {
        'PlugTvHall': 60,
        'CoffeMachine': None,
        'Microwave': None,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_3': {
        'PlugTvHall': 15,
        'CoffeMachine': 800,
        'Microwave': 1000,
        'WashingMachine': None,
        'Printer': 250,
        'PlugTvKitchen': None
    },
    'subject_4': {
        'PlugTvHall': None,
        'CoffeMachine': None,
        'Microwave': 1000,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_5': {
        'PlugTvHall': 45,
        'CoffeMachine': None,
        'Microwave': 1200,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_7': {
        'PlugTvHall': 15,
        'CoffeMachine': None,
        'Microwave': None,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': 15
    },
    'subject_8': {
        'PlugTvHall': 30,
        'CoffeMachine': 400,
        'Microwave': None,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_9': {
        'PlugTvHall': 60,
        'CoffeMachine': 800,
        'Microwave': None,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_10': {
        'PlugTvHall': 59,
        'CoffeMachine': None,
        'Microwave': 850,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },    
    'subject_11': {
        'PlugTvHall': 30,
        'CoffeMachine': None,
        'Microwave': None,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': 15
    },
    'subject_12': {
        'PlugTvHall': 20,
        'CoffeMachine': None,
        'Microwave': None,
        'WashingMachine': None,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_13': {
        'PlugTvHall': None,
        'CoffeMachine': None,
        'Microwave': 1000,
        'WashingMachine': 1500,
        'Printer': None,
        'PlugTvKitchen': None
    },
    'subject_14': {
        'PlugTvHall': 30,
        'CoffeMachine': None,
        'Microwave': None,
        'WashingMachine': 1300,
        'Printer': None,
        'PlugTvKitchen': None
    }
}

#%%
def collect_csv_files(main_directory):
    csv_files = []

    for subject_folder in natsorted(os.listdir(main_directory)):
        subject_path = os.path.join(main_directory, subject_folder)
        
        if os.path.isdir(subject_path):
            environmentals_path = os.path.join(subject_path, 'environmentals')
            
            if os.path.exists(environmentals_path):
                for file in os.listdir(environmentals_path):
                    # Check if the file is a CSV file
                    if file.endswith('.csv'):
                        csv_files.append(file)
    csv_files = set(csv_files)
    return csv_files

def process_sensor_data(df):
    if df.iloc[0].sensor_status == "off":
        df = df.iloc[1:]

    mask_on = (df['sensor_status'] == 'on')
    mask_off = (df['sensor_status'] == 'off')

    df_on = df[mask_on].copy()
    df_off = df[mask_off].copy()

    df_on.rename(columns={'ts': 'ts_on'}, inplace=True)
    df_off.rename(columns={'ts': 'ts_off'}, inplace=True)

    df_on.reset_index(inplace=True, drop=True)
    df_off.reset_index(inplace=True, drop=True)

    df_conc = pd.concat([df_on, df_off['ts_off']], axis=1)

    df_conc['duration_ms'] = df_conc['ts_off'] - df_conc['ts_on']

    df_conc['ts_on'] = pd.to_datetime(df_conc['ts_on'], unit='ms')
    df_conc['ts_off'] = pd.to_datetime(df_conc['ts_off'], unit='ms')

    df_conc['duration_datetime'] = df_conc['ts_off'] - df_conc['ts_on']

    df_conc['ts_on_ms'] = df_conc['ts_on'].astype('int64') // 10**6  # Convert datetime to milliseconds
    df_conc['ts_off_ms'] = df_conc['ts_off'].astype('int64') // 10**6  # Convert datetime to milliseconds

    df_conc = df_conc[['sensor_id', 'subject_id', 'sensor_status', 'ts_on_ms', 'ts_off_ms', 'ts_on', 'ts_off', 'duration_ms', 'duration_datetime']]

    return df_conc
def remove_continuous_on_off(df_cleaned_1):
    df_cleaned = df_cleaned_1.copy()
    
    contains_on = df_cleaned['sensor_status'].isin(['on']).any()
    contains_off = df_cleaned['sensor_status'].isin(['off']).any()
    
    if contains_on and contains_off:
        isEnvironmentalSensor = True
        for i in reversed(list(df_cleaned.index)[1:]):
            if df_cleaned.loc[i].sensor_status == df_cleaned.iloc[df_cleaned.index.get_loc(i)-1].sensor_status:
                df_cleaned.drop([i], inplace=True)
                # print(i)
    else:
        isEnvironmentalSensor = False
    return df_cleaned, isEnvironmentalSensor


def get_threshold_value(subject_dict, subject, file):
    if subject in subject_dict:
        subject_files = subject_dict[subject]
        if file in subject_files:
            return subject_files[file]
    return None

def convert_real_valued_to_on_off(threshold_value, df):
    if threshold_value is not None:
        ts_on_list = []
        ts_off_list = []
        temp_list = []
        
        is_on = False
        ts_on = None
        ind1 = None
        ind2 = None
        for index, row in df.iterrows():
            sensor_value = row['sensor_status']                                    
            if is_on == False and sensor_value > threshold_value:
                # Detect the first value above the threshold and mark it as ts_on
                ts_on = row['ts']
                is_on = True
                ind1 = index
                
            if is_on == True and sensor_value < threshold_value:
                # Detect the first value below the threshold and mark it as ts_off
                ind2 = index
                ts_off = row['ts']
                ts_on_list.append(ts_on)
                ts_off_list.append(ts_off)
                temp_list.append([ind1, row['sensor_id'], 'on', ts_on, row['subject_id'], pd.to_datetime(ts_on,unit='ms')])
                temp_list.append([ind2, row['sensor_id'], 'off', ts_off, row['subject_id'],pd.to_datetime(ts_off,unit = 'ms')])
                is_on = False

        result_df = pd.DataFrame(temp_list, columns=['index','sensor_id','sensor_status', 'ts', 'subject_id','ts_datetime'])
        return result_df

def read_csv_files_v2(data_path):
    subjects = natsorted(os.listdir(data_path))
    subject_dfs = {}     
    for s in range(len(subjects)):
        print('-*'*40)
        subject_s = subjects[s]
        # print(subject_s)
        path_subject = os.path.join(data_path, subject_s, 'environmentals')
        environmentals_sensors = natsorted(os.listdir(path_subject))
        dfs_dict = {}
        for es in range(len(environmentals_sensors)):
            folder_name = environmentals_sensors[es]
            # print(folder_name)
            # path_to_csv = os.path.join(path_subject, folder_name, 'merged.csv')
            merged_file = glob.glob(os.path.join(path_subject,folder_name, '*merged*.csv'))
            if merged_file:
                path_to_csv = merged_file[0]
                df = pd.read_csv(path_to_csv)
                if len(df) > 0:
                    df = df[~df['sensor_status'].isin(['unavailable', 'unknown'])]    
                    df.reset_index(drop=True, inplace=True)
                    df, isEnvironmentalSensor = remove_continuous_on_off(df)
                    df['ts_datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    # print(folder_name, df.iloc[0]['ts_datetime'], df.iloc[-1]['ts_datetime'])
                    if not isEnvironmentalSensor:
                      
                        if folder_name in ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_humidity', 'Shower_Hum_Temp_temp']:
                            print(subject_s, folder_name, 'Not a binary Sensor')
                            df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
                            dfs_dict[folder_name] = [df, pd.DataFrame()]
                        else: 
                            print(subject_s, folder_name, 'Not a binary Sensor')
                            threshold_value = get_threshold_value(subjects_threshold_dict, subject_s, folder_name)
                            df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
                            df = convert_real_valued_to_on_off(threshold_value, df)  
                            if len(df) > 0:
                                df_processed = process_sensor_data(df)
                                dfs_dict[folder_name] = [df, df_processed]
                    else:
                        print(subject_s, folder_name, 'an environmental Sensor')
                        df_processed = process_sensor_data(df)
                        dfs_dict[folder_name] = [df, df_processed]
        subject_dfs[subject_s] = dfs_dict
    return subject_dfs
                    
                

    
# def read_csv_files(data_path):
#     subjects = natsorted(os.listdir(data_path))
#     subject_dfs = {}
#     for s in range(len(subjects)):
#         print('-*'*40)
#         subject_s = subjects[s]
#         path_subject = os.path.join(data_path, subject_s, 'environmentals')
#         environmentals_sensors = natsorted(os.listdir(path_subject))
        
#         dfs_dict = {}
#         for es in range(len(environmentals_sensors)):
#             filename = environmentals_sensors[es]
#             if '_events' in filename:
#                 continue
            
#             if filename.endswith('.csv') and not filename.startswith('.~lock'):
#                 print(os.path.join(path_subject, filename))
#                 df = pd.read_csv(os.path.join(path_subject, filename))
#                 df['ts'] = df['ts']
#                 if len(df) > 0:
#                     df = df[~df['sensor_status'].isin(['unavailable', 'unknown'])]    
#                     df.reset_index(drop=True, inplace=True)
#                     df, isEnvironmentalSensor = remove_continuous_on_off(df)
#                     df['ts_datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    
#                     if not isEnvironmentalSensor:
#                         if filename in ['Shower_Hum_Temp_temp.csv', 'Shower_Hum_Temp_humidity.csv', 'Stove_Hum_Temp_temp.csv', 'Stove_Hum_Temp_humidity.csv']:
#                             df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
#                             dfs_dict[filename] = [df, pd.DataFrame()]
#                         else:   
#                             threshold_value = get_threshold_value(subjects_threshold_dict, subject_s, filename)
#                             df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
#                             df = convert_real_valued_to_on_off(threshold_value, df)  
#                             df_processed = process_sensor_data(df)
#                             dfs_dict[filename] = [df, df_processed]
#                     else:
#                         df_processed = process_sensor_data(df)
#                         dfs_dict[filename] = [df, df_processed]
#         subject_dfs[subject_s] = dfs_dict
#     return subject_dfs


def determine_sensor_case(ws, we, ts_on, ts_off):
    # Case 1: Sensor turned on and off inside the window
    if ws <= ts_on <= we and ws <= ts_off <= we:
        return 1
    
    # Case 2: Sensor turned on before and off after the window
    elif ts_on < ws and ts_off > we:
        return 2
    
    # Case 3: Sensor turned on inside the window and off after the window
    elif ws <= ts_on <= we and ts_off > we:
        return 3
    
    # Case 4: Sensor turned on before the window and off inside the window
    elif ts_on < ws and ws <= ts_off <= we:
        return 4
    
    # If none of the cases match, return None
    return None

def count_virtual_out_events(house_entrance, data_subject_1, environmental_sensors, year=None, month=None, st=None, et=None):
        
    house_entrance_df = []
    for m in range(1 , len(house_entrance)):
        house_entrance_df.append((house_entrance.iloc[m-1]['ts_off'],house_entrance.iloc[m]['ts_on']))
        
    house_entrance_df  = pd.DataFrame(house_entrance_df, columns = ['ws', 'we'])
    house_entrance_df_year_month = house_entrance_df[(house_entrance_df['ws'].dt.year == year) & (house_entrance_df['ws'].dt.month == month)]
    ## make windows
    
    if st is not None and et is not None:
        sub_df = house_entrance_df_year_month[
            (house_entrance_df_year_month['ws'].dt.time >= st) &
            (house_entrance_df_year_month['ws'].dt.time <= et)]
    else:
        sub_df = house_entrance_df_year_month.copy()
    
    out_event_timimg = []
    out_event_count = 0
    inside_home_timings = []
    for he in range(len(sub_df)):
        ws = sub_df.iloc[he]['ws']
        we = sub_df.iloc[he]['we']
        duration = (we - ws).total_seconds() / 60
        # if duration > 10:
        is_out_event = True
        for j in range(len(environmental_sensors)):
            sensor_name = environmental_sensors[j]
            
            # Check if the sensor is not in the excluded list
            if sensor_name in ['HouseEntrance','Shower_Hum_Temp','Stove_Hum_Temp','HouseEntrance.csv', 'Shower_Hum_Temp_humidity.csv', 'Shower_Hum_Temp_temp.csv', 'Stove_Hum_Temp_humidity.csv', 'Stove_Hum_Temp_temp.csv']:
                continue
            
            sensor_df_raw = data_subject_1[sensor_name][0]
            sensor_df_processed = data_subject_1[sensor_name][1]
            
            if len(sensor_df_processed) > 0:
                if 'Motion' not in sensor_name:
                    temp_df = sensor_df_processed[(sensor_df_processed['ts_on'] <= we) & (sensor_df_processed['ts_off'] >= ws)]
                    if not temp_df.empty:
                        # print(j, sensor_name)
                        for r in range(len(temp_df)):
                            ts_on, ts_off = temp_df.iloc[r]['ts_on'], temp_df.iloc[r]['ts_off']
                            case_value = determine_sensor_case(ws, we, ts_on, ts_off)
                            if case_value in [1,3,4]:
                                # print(j, sensor_name)
                                is_out_event = False
                                break
                            
                else:#if 'Motion' in sensor_name:
                    temp_df = sensor_df_processed[(sensor_df_processed['ts_on'] <= we) & (sensor_df_processed['ts_off'] >= ws)]
                    if not temp_df.empty:
                        for r in range(len(temp_df)):
                            ts_on, ts_off = temp_df.iloc[r]['ts_on'], temp_df.iloc[r]['ts_off']
                            case_value = determine_sensor_case(ws, we, ts_on, ts_off)
                            if case_value in [1,3]:
                                # print(j, sensor_name)
                                is_out_event = False
                                break
                                
        if is_out_event:            
            out_event_count += 1
            out_event_timimg.append((he, ws, we))   
            
    out_event_timimg_df = pd.DataFrame(out_event_timimg, columns=['tid', 'ws', 'we'])       
    return out_event_count, out_event_timimg_df, []



# def count_virtual_out_events(house_entrance, data_subject_1, year=None, month=None, st=None, et=None):
#     if year is not None and month is not None:
#         house_entrance = house_entrance[(house_entrance['ts_on'].dt.year == year) & (house_entrance['ts_on'].dt.month == month)]
#         if st is not None and et is not None:
#             house_entrance = house_entrance[
#                 (house_entrance['ts_on'].dt.time >= st) &
#                 (house_entrance['ts_on'].dt.time < et)]
    
#     out_event_timimg = []
#     out_event_count = 0
#     inside_home_timings = []
#     for he in range(1, len(house_entrance)):
#         ws = house_entrance.iloc[he-1]['ts_off']
#         we = house_entrance.iloc[he]['ts_on']
#         duration = (we - ws).total_seconds() / 60
        
#         # Only proceed if the duration is greater than 10 minutes
#         # if duration > 10:
#         is_out_event = True
#         temp = {}
#         for j in range(len(environmental_sensors)):
#             sensor_name = environmental_sensors[j]
            
#             if sensor_name not in ['HouseEntrance.csv','Shower_Hum_Temp_humidity.csv', 'Shower_Hum_Temp_temp.csv', 'Stove_Hum_Temp_humidity.csv', 'Stove_Hum_Temp_temp.csv']: 
#                 # print(sensor_name)
#                 sensor_df_raw = data_subject_1[sensor_name][0]
#                 sensor_df_processed = data_subject_1[sensor_name][1]
                
#                 if len(sensor_df_processed) > 0:
#                     temp_df = sensor_df_processed[(sensor_df_processed['ts_on'] >= ws) & (sensor_df_processed['ts_off'] <= we)]
#                     # temp_df = sensor_df_processed[(sensor_df_processed['ts_on'] <= we) & (sensor_df_processed['ts_off'] >= ws)]
#                     if not temp_df.empty:
#                         is_out_event = False
#                         temp[sensor_name] = temp_df
                        
#                         break
                    
#         inside_home_timings.append(temp)
#         if is_out_event:
#             out_event_count += 1
#             out_event_timimg.append((he, ws, we))

#     return out_event_count, out_event_timimg, inside_home_timings

      

# def detect_peaks(df, value_col='sensor_status', prominence=1.5):
#     if value_col not in df.columns:
#         return [], df 
#     df1 = df.copy()
#     peaks, properties = find_peaks(df1[value_col], prominence=prominence)
#     peak_indicator = [0] * len(df1)
#     for peak in peaks:
#         peak_indicator[peak] = 1
#     df1['peak_detected'] = peak_indicator
#     return peaks,df1
# Set colorblind-friendly palette using seaborn
def process_out_month(out_month, year, month, st=None, et=None, filter_duration=False, min_duration=10):
    if len(out_month) == 0:
        print(f"Outs - {year}, {month}: 0")
        return 0, pd.DataFrame([])
    else:
        # Filter based on the provided time window if st and et are given
        if st is not None and et is not None:
            sub_out_month = out_month[
                (out_month['ws'].dt.year == year) &
                (out_month['ws'].dt.month == month) &
                (out_month['ws'].dt.time >= st) &
                (out_month['ws'].dt.time <= et)
            ].copy()
        else:
            # No time window specified, use the entire month's data
            sub_out_month = out_month[
                (out_month['ws'].dt.year == year) &
                (out_month['ws'].dt.month == month)
            ].copy()
            
            # Check if duration filtering is enabled
            if filter_duration:
                # Calculate the duration of each event in minutes
                sub_out_month['duration'] = (sub_out_month['ts_off'] - sub_out_month['ts_on']).dt.total_seconds() / 60
                
                # Filter for events longer than the minimum duration (default 10 minutes)
                sub_out_month = sub_out_month[sub_out_month['duration'] > min_duration].copy()
                print(f"Outs (filtered) - {year}, {month}: {len(sub_out_month)}")
            else:
                print(f"Outs (unfiltered) - {year}, {month}: {len(sub_out_month)}")
        
        return len(sub_out_month), sub_out_month

  

def process_peaks_month(peaks_month, year, month, st=None, et=None):
    if len(peaks_month) == 0:
        print(f"Peaks - {year}, {month}: 0")
        return 0, pd.DataFrame([])
    else:
        if st != None and et != None:
            sub_peaks_month = peaks_month[
                (peaks_month['ts_datetime'].dt.year == year) &
                (peaks_month['ts_datetime'].dt.month == month) &
                (peaks_month['ts_datetime'].dt.time >= st) &
                (peaks_month['ts_datetime'].dt.time <= et) &
                (peaks_month['peak_detected'] == 1)
            ].copy()
        else:
            sub_peaks_month = peaks_month[
                (peaks_month['ts_datetime'].dt.year == year) &
                (peaks_month['ts_datetime'].dt.month == month) &
                (peaks_month['peak_detected'] == 1)
            ].copy()
            
        print(f"Peaks - {year}, {month}: {len(sub_peaks_month)}")
        return len(sub_peaks_month), sub_peaks_month

def plot_bar_for_subject(subject_data, subject):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # Filter out rows where both counts are zero
    subject_data_filtered = subject_data[
        (subject_data['count_outs'] != 0) | 
        (subject_data['count_temperature_peaks'] != 0)
    ]

    # Check if there's any data left to plot
    if subject_data_filtered.empty:
        print(f"No data to plot for {subject}.")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Define positions for the bars
    bar_width = 0.4
    x_labels = subject_data_filtered['year'].astype(str) + '-' + subject_data_filtered['month'].astype(str)

    # Create bar positions for both types of counts (outs and temperature peaks)
    bar_positions = range(len(x_labels))

    # Plot the data for Outs Count
    bars_outs = plt.bar(
        bar_positions, subject_data_filtered['count_outs'], width=bar_width, 
        label='Probable Out Events', align='center', color=sns.color_palette()[0]
    )
    
    # Plot the data for Temperature Peaks Count
    bars_temperature = plt.bar(
        [pos + bar_width for pos in bar_positions], subject_data_filtered['count_temperature_peaks'], 
        width=bar_width, label='Temperature Peaks', align='center', color=sns.color_palette()[1]
    )

    # Set x-ticks and labels
    plt.xticks([pos + bar_width / 2 for pos in bar_positions], x_labels, rotation=45)

    # Set labels and title
    plt.xlabel('Year-Month')
    plt.ylabel('Count')
    plt.title(f'Probable Out Events and Temperature Peaks in Kitchen {subject}')
    plt.legend()

    # Add numeric labels on top of each bar
    for bar in bars_outs:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')

    for bar in bars_temperature:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'plots/{subject}.png')
    plt.show()

def detect_peaks(df, value_col='sensor_status', prominence=1.5, st=None, et=None):
    if value_col not in df.columns:
        return [], df
    df1 = df.copy()
    # Detect peaks using scipy's find_peaks
    peaks, properties = find_peaks(df1[value_col], prominence=prominence)
    # Create a peak indicator column
    peak_indicator = [0] * len(df1)
    for peak in peaks:
        peak_indicator[peak] = 1
    df1['peak_detected'] = peak_indicator

    # If start time (st) and end time (et) are provided, filter the DataFrame
    if st is not None and et is not None:
        # Ensure the 'ts_datetime' column is in datetime format for filtering
        df1['ts_datetime'] = pd.to_datetime(df1['ts_datetime'])

        # Filter the DataFrame by the given time range
        df1['time_only'] = df1['ts_datetime'].dt.time
        df1 = df1[(df1['time_only'] >= st) & (df1['time_only'] <= et)]

        # Re-detect peaks within the filtered time range
        filtered_peaks = df1[df1['peak_detected'] == 1].index.tolist()

        return filtered_peaks, df1

    return peaks, df1

def count_peaks(data_subject_1, sensor_file, year = None, month = None, prominence = 1.5, st=None, et = None):
    if sensor_file in data_subject_1:
        temperatue_data = data_subject_1[sensor_file][0].copy()  
        # 
        if year != None and month != None:
            temperatue_data_ym = temperatue_data[(temperatue_data['ts_datetime'].dt.year == year) & (temperatue_data['ts_datetime'].dt.month == month)].copy()
            tData = temperatue_data_ym.copy()
            # if st != None and et !=None:
            #     # print('xxxxxxxxxxxx')
            #     temperatue_data_stet = temperatue_data_ym[
            #         (temperatue_data_ym['ts_datetime'].dt.time >= st) &
            #         (temperatue_data_ym['ts_datetime'].dt.time <= et)] 
            #     tData = temperatue_data_stet.copy()
                
            # print(temperatue_data)
        peaks, df_temperature_data= detect_peaks(tData, 'sensor_status', prominence=prominence)  
        selected_df = temperatue_data.loc[peaks, ['ts_datetime', 'sensor_status']]
        
        # Output the result
        # print(selected_df)
        peak_count = len(peaks)
    else:
        peak_count = 0
        peaks = []
        df_temperature_data = pd.DataFrame()
    return peak_count, peaks, df_temperature_data
    
def plot_values_and_peaks(df1, peaks, value_col='sensor_status'):
    # Plot the sensor_status values
    plt.figure(figsize=(10, 6))
    plt.plot(df1['ts_datetime'], df1[value_col], label='Sensor Status', color='blue', lw=2)
    
    # Plot the peaks
    plt.scatter(df1.loc[peaks, 'ts_datetime'], df1.loc[peaks, value_col], color='red', label='Peaks', zorder=5)
    
    # Add labels and legend
    plt.title('Sensor Status with Detected Peaks')
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Status')
    plt.legend()
    
    # Display the plot
    plt.show()
    

import plotly.graph_objects as go
import pandas as pd

def plot_combined_values_and_peaks_with_day_lines(df_humidity, peaks_humidity, df_temperature, peaks_temperature, subject_nme, room, year, month, value_col='sensor_status'):
    # Create a figure
    fig = go.Figure()

    # Plot humidity values
    fig.add_trace(go.Scatter(
        x=df_humidity['ts_datetime'],
        y=df_humidity[value_col],
        mode='lines',
        name='Humidity',
        line=dict(color='blue'),
    ))

    # Plot humidity peaks
    fig.add_trace(go.Scatter(
        x=df_humidity.loc[peaks_humidity, 'ts_datetime'],
        y=df_humidity.loc[peaks_humidity, value_col],
        mode='markers',
        name='Humidity Peaks',
        marker=dict(color='red', size=8)
    ))

    # Plot temperature values
    fig.add_trace(go.Scatter(
        x=df_temperature['ts_datetime'],
        y=df_temperature[value_col],
        mode='lines',
        name='Temperature',
        line=dict(color='green'),
    ))

    # Plot temperature peaks
    fig.add_trace(go.Scatter(
        x=df_temperature.loc[peaks_temperature, 'ts_datetime'],
        y=df_temperature.loc[peaks_temperature, value_col],
        mode='markers',
        name='Temperature Peaks',
        marker=dict(color='orange', size=8)
    ))

    # Extract unique days (start and end of each day) from the data and plot vertical lines
    unique_days = pd.to_datetime(df_humidity['ts_datetime'].dt.date.unique())
    
    for day in unique_days:
        # Add a vertical line for the start of the day (00:00:00)
        fig.add_vline(x=day, line=dict(color='black', dash='dash'), annotation_text="Start of Day", annotation_position="top left")
        
        # Add a vertical line for the end of the day (23:59:59) using Timedelta
        end_of_day = pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        fig.add_vline(x=end_of_day, line=dict(color='gray', dash='dash'))

    # Update layout
    fig.update_layout(
        title=f'Humidity and Temperature with Detected Peaks for {subject_nme} in {room} ({year}-{month})',
        xaxis_title='Timestamp',
        yaxis_title='Sensor Status',
        xaxis_tickformat='%Y-%m-%d %H:%M:%S',
        xaxis_tickangle=-90,
        height=600,
        legend=dict(x=0, y=1.2, orientation='h')
    )
    
    # Show the plot
    fig.show()

def plot_combined_values_and_peaks_with_day_lines(df_humidity, peaks_humidity, df_temperature, peaks_temperature, subject_nme, room, year, month,  value_col='sensor_status'):
    # Create a figure with a single subplot
    plt.figure(figsize=(12, 6))
    
    # Plot Humidity values and peaks
    plt.plot(df_humidity['ts_datetime'], df_humidity[value_col], label='Humidity', color='blue', lw=2)
    plt.scatter(df_humidity.loc[peaks_humidity, 'ts_datetime'], df_humidity.loc[peaks_humidity, value_col], 
                color='blue', label='Humidity Peaks', zorder=5)
    
    # Plot Temperature values and peaks
    plt.plot(df_temperature['ts_datetime'], df_temperature[value_col], label='Temperature', color='red', lw=2)
    plt.scatter(df_temperature.loc[peaks_temperature, 'ts_datetime'], df_temperature.loc[peaks_temperature, value_col], 
                color='red', label='Temperature Peaks', zorder=5)
    
    # Extract unique days (start and end of each day) from the data and plot vertical lines
    unique_days = pd.to_datetime(df_humidity['ts_datetime'].dt.date.unique())
    
    for day in unique_days:
        # Plot a vertical line at the start of the day (00:00:00)
        plt.axvline(pd.Timestamp(day), color='black', linestyle='--', lw=1)
        
        # Plot a vertical line at the end of the day (23:59:59)
        plt.axvline(pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59, seconds=59), color='gray', linestyle='--', lw=1)

    # Add labels and legend
    plt.title(f'Humidity and Temperature with Detected Peaks for {subject_nme} in {room}')
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Status')
    plt.legend(loc='upper left')
    plt.xticks(rotation=90)

    # Display the plot
    plt.tight_layout() 
    plt.savefig(f'{subject_nme}_{room}_{year}_{month}.png')
    plt.show()



def detect_and_filter_peaks_by_time( data_subject_1, sensor_file, prominence=1.5,  year=None, month=None,  st=None,  et=None ):
    filtered_peaks = []#pd.DataFrame()
    df1 = pd.DataFrame()
    if sensor_file in data_subject_1:
        temperatue_data = data_subject_1[sensor_file][0].copy() 
        df = temperatue_data.copy()
        
        if 'sensor_status' not in df.columns:
            return [], df
        
        df1 = df.copy()
        
        df1['ts_datetime'] = pd.to_datetime(df1['ts_datetime'])
        peaks, properties = find_peaks(df1['sensor_status'], prominence=prominence)
        peak_indicator = [0] * len(df1)
        
        for peak in peaks:
            peak_indicator[peak] = 1
            
        df1['peak_detected'] = peak_indicator
        df2 = df1.copy()
        
        if year is not None and month is not None:
            df1 = df1[(df1['ts_datetime'].dt.year == year) & (df1['ts_datetime'].dt.month == month)]
       
        if st is not None and et is not None:
            df1['time_only'] = df1['ts_datetime'].dt.time
            df1 = df1[(df1['time_only'] >= st) & (df1['time_only'] <= et)]
            filtered_peaks = df1[df1['peak_detected'] == 1].index.tolist()
        else:
            filtered_peaks = df1[df1['peak_detected'] == 1].index.tolist()
    
    return len(filtered_peaks), filtered_peaks, df1, df2

# def plot_signal_with_peaks(df_with_humidity_peaks, peaks_humidity, sensor_name):
#     # Plot the sensor status (signal)
#     plt.figure(figsize=(14, 7))
#     plt.plot(df_with_humidity_peaks['ts_datetime'], df_with_humidity_peaks['sensor_status'], label= sensor_name, color='blue')
    
#     # Highlight the peaks
#     plt.plot(df_with_humidity_peaks['ts_datetime'].iloc[peaks_humidity], 
#              df_with_humidity_peaks['sensor_status'].iloc[peaks_humidity], 
#              'ro', label='Peaks')
    
#     # Add labels and title
#     plt.xlabel('Timestamp')
#     plt.ylabel('Sensor Value')
#     plt.title('Signal with Detected Peaks')
#     plt.legend()

#     # Display the plot
#     plt.show()

def plot_signal_with_peaks(df_with_humidity_peaks, peaks_humidity, df_with_temperature_peaks, peaks_temperature, humidity_sensor_name, temperature_sensor_name, subject_name, year, month):
    plt.figure(figsize=(14, 7))

    # Plot the humidity sensor status (signal)
    plt.plot(df_with_humidity_peaks['ts_datetime'], df_with_humidity_peaks['sensor_status'], 
             label= 'Humidity Values', color='blue')
    
    # Highlight the peaks in humidity
    plt.plot(df_with_humidity_peaks['ts_datetime'].iloc[peaks_humidity], 
             df_with_humidity_peaks['sensor_status'].iloc[peaks_humidity], 
             'bo', label='Humidity Peaks')
    
    # Plot the temperature sensor status (signal)
    plt.plot(df_with_temperature_peaks['ts_datetime'], df_with_temperature_peaks['sensor_status'], 
             label='Temperature Values', color='red')
    
    # Highlight the peaks in temperature
    plt.plot(df_with_temperature_peaks['ts_datetime'].iloc[peaks_temperature], 
             df_with_temperature_peaks['sensor_status'].iloc[peaks_temperature], 
             'ro', label='Temperature Peaks')

    # Add vertical lines at the start and end of each day
    day_starts = df_with_humidity_peaks['ts_datetime'].dt.floor('D').drop_duplicates()
    for day in day_starts:
        plt.axvline(x=day, color='green', linestyle='--', alpha=0.5)
        plt.axvline(x=day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), color='green', linestyle='--', alpha=0.5)

    # Add labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Value')
    plt.title('Humidity and Temperature Signals with Detected Peaks for ' + subject_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{subject_name}_{year}_{month}.png')
    # Display the plot
    plt.show()

# def plot_event_and_peak_counts(all_out_event_counts, all_temperature_peak_counts, all_humidity_peak_counts, subject_name, strng):
#     months_years = []
#     out_event_counts = []
#     temperature_peak_counts = []
#     humidity_peak_counts = []

#     # Iterate over years and months, combining all counts
#     for year in sorted(set(all_out_event_counts.keys()).union(all_temperature_peak_counts.keys()).union(all_humidity_peak_counts.keys())):
#         for month in sorted(set(all_out_event_counts.get(year, {}).keys())
#                             .union(all_temperature_peak_counts.get(year, {}).keys())
#                             .union(all_humidity_peak_counts.get(year, {}).keys())):
#             month_year_str = f"{month:02d}-{year}"
#             months_years.append(month_year_str)
#             out_event_counts.append(all_out_event_counts.get(year, {}).get(month, 0))
#             temperature_peak_counts.append(all_temperature_peak_counts.get(year, {}).get(month, 0))
#             humidity_peak_counts.append(all_humidity_peak_counts.get(year, {}).get(month, 0))

#     x = np.arange(len(months_years))  # the label locations
#     width = 0.3  # the width of the bars (adjusted for three bars)

#     fig, ax = plt.subplots(figsize=(14, 7))
#     bars1 = ax.bar(x - width, out_event_counts, width, label='Probable Out Events', color='red')
#     bars2 = ax.bar(x, temperature_peak_counts, width, label='Temperature Peaks', color='blue')
#     bars3 = ax.bar(x + width, humidity_peak_counts, width, label='Humidity Peaks', color='green')

#     # Customize the plot
#     ax.set_xlabel('Month-Year', fontsize=16)
#     ax.set_ylabel('Count', fontsize=16)
#     ax.set_title(f'Monthly Probable Out Events, Temperature Peaks, and Humidity Peaks for {subject_name}', fontsize=16)
#     ax.set_xticks(x)
#     ax.set_xticklabels(months_years, rotation=45, ha="right", fontsize=16)
#     ax.tick_params(axis='y', labelsize=16)
#     ax.legend(loc='best', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f'{subject_name}_{strng}_events_peaks.png')
#     plt.show()

def plot_event_and_peak_counts(all_out_event_counts, all_temperature_peak_counts, all_humidity_peak_counts, subject_name, strng, location='kitchen'):
    months_years = []
    out_event_counts = []
    temperature_peak_counts = []
    humidity_peak_counts = []

    # Iterate over years and months, combining all counts
    for year in sorted(set(all_out_event_counts.keys()).union(all_temperature_peak_counts.keys()).union(all_humidity_peak_counts.keys())):
        for month in sorted(set(all_out_event_counts.get(year, {}).keys())
                            .union(all_temperature_peak_counts.get(year, {}).keys())
                            .union(all_humidity_peak_counts.get(year, {}).keys())):
            if year == 2024 and month in [5, 7, 8]:
                continue
            month_year_str = f"{month:02d}-{year}"
            months_years.append(month_year_str)
            out_event_counts.append(all_out_event_counts.get(year, {}).get(month, 0))
            temperature_peak_counts.append(all_temperature_peak_counts.get(year, {}).get(month, 0))
            humidity_peak_counts.append(all_humidity_peak_counts.get(year, {}).get(month, 0))

    x = np.arange(len(months_years))  # the label locations
    width = 0.3  # the width of the bars (adjusted for three bars)

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, out_event_counts, width, label='Probable Out Events', color='red')
    bars2 = ax.bar(x, temperature_peak_counts, width, label='Temperature Peaks', color='blue')
    # bars3 = ax.bar(x + width, humidity_peak_counts, width, label='Humidity Peaks', color='green')

    # Annotate bars with counts
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=12)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=12)

    # for bar in bars3:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=12)

    # Customize the plot
    ax.set_xlabel('Month-Year', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(f'Monthly Probable Out Events, Temperature Peaks in {location} for {subject_name}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(months_years, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{subject_name}_{strng}_events_peaks.png')
    plt.show()
