#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:28:15 2024

@author: hubble
"""

import numpy as np, pandas as pd, os, pickle, glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from natsort import natsorted
from scipy.signal import find_peaks



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

def find_peaks_with_durations(daily_data, peaks,day=None, max_time_diff=10):
    peak_info = []
    
    # Iterate through each peak
    for peak in peaks:
        start = peak
        end = peak
        
        # Move backward to find the start of the peak
        while start > 0 and daily_data['sensor_status'].iloc[start] >= daily_data['sensor_status'].iloc[start - 1] and \
              (daily_data['ts_datetime'].iloc[start] - daily_data['ts_datetime'].iloc[start - 1]) < max_time_diff:
            start -= 1
        
        # Move forward to find the end of the peak
        while end < len(daily_data) - 1 and daily_data['sensor_status'].iloc[end] >= daily_data['sensor_status'].iloc[end + 1] and \
              (daily_data['ts_datetime'].iloc[end + 1] - daily_data['ts_datetime'].iloc[end]) < max_time_diff:
            end += 1
        
        # Calculate duration
        duration = daily_data['ts_datetime'].iloc[end] - daily_data['ts_datetime'].iloc[start]
        
        # Append a dictionary with start, end, and duration to the list
        peak_info.append({
            'subject_id': daily_data['subject_id'].iloc[start], 
            'sensor_id': daily_data['sensor_id'].iloc[start],
            'day':day,
             'ts_on': daily_data['ts_datetime'].iloc[start],
             'ts_off': daily_data['ts_datetime'].iloc[end],
            'duration':duration
        })
    
    return peak_info


def process_sensor_data(df):
    """
    This function arranges on and off time side by side
    """
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
    


    # df_conc['ts_on'] = pd.to_datetime(df_conc['ts_on'], unit='ms')
    df_conc['ts_on'] = (
        pd.to_datetime(df_conc['ts_on'], unit='ms')  # Step 1: Convert milliseconds to datetime
        .dt.tz_localize('UTC')                        # Step 2: Localize to UTC
        .dt.tz_convert('Europe/Rome')                 # Step 3: Convert to Europe/Rome timezone
    )

    
    # df_conc['ts_off'] = pd.to_datetime(df_conc['ts_off'], unit='ms')
    df_conc['ts_off'] = (
        pd.to_datetime(df_conc['ts_off'], unit='ms')  # Step 1: Convert milliseconds to datetime
        .dt.tz_localize('UTC')                        # Step 2: Localize to UTC
        .dt.tz_convert('Europe/Rome')                 # Step 3: Convert to Europe/Rome timezone
    )

    df_conc['duration_datetime'] = df_conc['ts_off'] - df_conc['ts_on']

    df_conc['ts_on_ms'] = df_conc['ts_on'].astype('int64') // 10**6  # Convert datetime to milliseconds
    df_conc['ts_off_ms'] = df_conc['ts_off'].astype('int64') // 10**6  # Convert datetime to milliseconds

    df_conc = df_conc[['sensor_id', 'subject_id', 'sensor_status', 'ts_on_ms', 'ts_off_ms', 'ts_on', 'ts_off', 'duration_ms', 'duration_datetime']]

    return df_conc

# subject_dict, subject, file = subjects_threshold_dict, subject_s, folder_name
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
                # temp_list.append([ind1, row['sensor_id'], 'on', ts_on, row['subject_id'], pd.to_datetime(ts_on,unit='ms')])
                temp_list.append([ind1, row['sensor_id'], 'on', ts_on, row['subject_id'], pd.to_datetime(ts_on, unit='ms').tz_localize('UTC').tz_convert('Europe/Rome')])
                # temp_list.append([ind2, row['sensor_id'], 'off', ts_off, row['subject_id'],pd.to_datetime(ts_off,unit = 'ms')])
                temp_list.append([ind2, row['sensor_id'],'off', ts_off, row['subject_id'],pd.to_datetime(ts_off, unit='ms').tz_localize('UTC').tz_convert('Europe/Rome')])
                is_on = False

        result_df = pd.DataFrame(temp_list, columns=['index','sensor_id','sensor_status', 'ts', 'subject_id','ts_datetime'])
        return result_df
    
def remove_continuous_on_off_v2(df_cleaned_1):
    df_cleaned = df_cleaned_1.copy()
    rows_to_drop = []
    
    contains_on = df_cleaned['sensor_status'].isin(['on']).any()
    contains_off = df_cleaned['sensor_status'].isin(['off']).any()   
    isEnvironmentalSensor = -999
    if contains_on and contains_off:
        isEnvironmentalSensor = True
    
        for i in reversed(list(df_cleaned.index)[1:]):
            current_label = df_cleaned.loc[i, 'sensor_status']      # Current label
            previous_label = df_cleaned.loc[i-1, 'sensor_status']   # Previous label
    
            # If the current and previous labels are both 'CLOSE', mark the current row to be dropped
            if current_label == previous_label and current_label == 'off':
                rows_to_drop.append(i)
                
            # If the current and previous labels are both 'OPEN', mark the previous row to be dropped
            elif current_label == previous_label and current_label == 'on':
                rows_to_drop.append(i-1)
                
        # Drop all rows that were marked
        df_cleaned.drop(rows_to_drop, inplace=True)
    
        # Reset the index after dropping rows
        df_cleaned.reset_index(drop=True, inplace=True)
        
        if df_cleaned.iloc[0]['sensor_status'] == 'off':
            df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)
    
        # Check if the last row's 'output' is 'ON' and remove it
        if df_cleaned.iloc[-1]['sensor_status'] == 'on':
            df_cleaned = df_cleaned.iloc[:-1].reset_index(drop=True)
    
        # df_cleaned['label'] = df_cleaned['label'].fillna('transition')
    else:
        isEnvironmentalSensor = False

    return df_cleaned, isEnvironmentalSensor   

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
                    df, isEnvironmentalSensor = remove_continuous_on_off_v2(df)
                    df['ts_datetime'] = pd.to_datetime(df['local_time'],format='mixed')
                    df.rename(columns={'timestamp (GMT)': 'ts'}, inplace=True)

                    # df['ts_datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    # df['ts_datetime_utc'] = pd.to_datetime(df['ts'], unit='ms')
                    # df['ts_datetime_utc'] = df['ts_datetime_utc'].dt.tz_localize('UTC')
                    # df['ts_datetime'] = df['ts_datetime_utc'].dt.tz_convert('Europe/Rome')
                    # df.drop(columns=['ts_datetime_utc'], inplace=True)
                    if not isEnvironmentalSensor:
                      
                        if folder_name in ['Shower_Hum_Temp','Stove_Hum_Temp','Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_humidity', 'Shower_Hum_Temp_temp']:
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
            

#%%
def get_minute_index(ts_datetime):
    # Convert to pandas datetime for easier extraction
    dt = pd.to_datetime(ts_datetime)
    # Calculate the minute index: hour * 60 + minute
    minute_index = dt.hour * 60 + dt.minute
    return minute_index

def arrange_data_by_day_numpy(df):
    # Convert ts_datetime to datetime and localize to 'Europe/Rome'

    # # Check if timestamps are already timezone-aware
    # if df['ts_datetime'].dt.tz is None:
    #     df['ts_datetime'] = df['ts_datetime'].dt.tz_localize('Europe/Rome', ambiguous='infer')
    # else:
    #     df['ts_datetime'] = df['ts_datetime'].dt.tz_convert('Europe/Rome')
    # df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
    
    start_date = df['ts_datetime'].dt.date.min()
    end_date = df['ts_datetime'].dt.date.max()
    complete_days = pd.date_range(start=start_date, end=end_date, freq='D').date

    daily_data_list = [] 
    peak_start_end_list = []
    minute_index_mapping = []
    datewise_peak_duration_info_list = []
    
    for day in complete_days:
        daily_data = df[df['ts_datetime'].dt.date == day]
        
        peaks, properties = find_peaks(daily_data['sensor_status'], prominence=1.5)
        max_time_diff = pd.Timedelta(minutes=10) ## it checks peak which are v
        peak_info_list = find_peaks_with_durations(daily_data, peaks, max_time_diff, day)
        datewise_peak_duration_info_list.extend(peak_info_list)
        
        peak_to_minute_index_mapping = []
        if len(peaks) > 0:
            for p in peaks:
                temp = daily_data.iloc[p]['ts_datetime']
                temp_minute_index = get_minute_index(temp)
                peak_to_minute_index_mapping.append(temp_minute_index)
        else:
            peak_to_minute_index_mapping.append(-99)
            
        minute_index_mapping.append(peak_to_minute_index_mapping)

        # Create all_minutes range in 'Europe/Rome' timezone
        # all_minutes = pd.date_range(start=pd.Timestamp(day).tz_localize('Europe/Rome'), 
        #                              end=pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), 
        #                              freq='min').tz_localize('Europe/Rome')
        
        start_time = pd.Timestamp(day).tz_localize('Europe/Rome')  # Localize start time
        end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)  # End time in the same timezone
        
        all_minutes = pd.date_range(start=start_time, end=end_time, freq='min')


        # Resample daily data to minute level
        daily_resampled = daily_data.set_index('ts_datetime')['sensor_status'].resample('min').max()
        daily_resampled = daily_resampled.reindex(all_minutes, fill_value=np.nan)
        daily_resampled = daily_resampled.fillna(0)
        daily_data_list.append(daily_resampled.values if not daily_data.empty else np.full((1440,), 0))
        
        # Create DataFrame for peak start and end activities
        peak_Start_end_df = pd.DataFrame(all_minutes, columns=['minute'])
        peak_Start_end_df['activity'] = 0
        
        def mark_active_minutes(minutes_df, peak_info_list):
            for peak in peak_info_list:
                ts_on = peak['ts_on'].floor('min')#.tz_convert('Europe/Rome')  # Ensure tz_convert to 'Europe/Rome'
                ts_off = peak['ts_off'].floor('min')#.tz_convert('Europe/Rome')
                # Mark all minutes between ts_on and ts_off as 1
                minutes_df.loc[(minutes_df['minute'] >= ts_on) & (minutes_df['minute'] <= ts_off), 'activity'] = 1
            return minutes_df

        peak_Start_end_df = mark_active_minutes(peak_Start_end_df, peak_info_list)
        peak_start_end_list.append(peak_Start_end_df.set_index('minute')['activity'].values)

    arranged_peak_start_end_data_np = np.array(peak_start_end_list)
    arranged_data_np = np.array(daily_data_list)
    
    return arranged_data_np, complete_days, arranged_peak_start_end_data_np, pd.DataFrame(datewise_peak_duration_info_list)

def filter_rows_by_time_and_date(df, start_time, end_time, year, month):
    # Ensure the 'ts_on' and 'ts_off' columns are in datetime format
    df['ts_on'] = pd.to_datetime(df['ts_on'])
    df['ts_off'] = pd.to_datetime(df['ts_off'])
    
    # Create boolean mask to filter rows by year and month
    year_month_mask = (df['ts_on'].dt.year == year) & (df['ts_on'].dt.month == month)
    
    # Convert start_time and end_time to the time format for comparison
    start_time = pd.to_datetime(start_time).time()
    end_time = pd.to_datetime(end_time).time()
    
    # Create boolean mask to filter rows where 'ts_on' or 'ts_off' overlap with the time range
    time_mask = ((df['ts_on'].dt.time >= start_time) & (df['ts_on'].dt.time <= end_time)) | \
                ((df['ts_off'].dt.time >= start_time) & (df['ts_off'].dt.time <= end_time)) | \
                ((df['ts_on'].dt.time <= start_time) & (df['ts_off'].dt.time >= end_time))
    
    # Combine both masks
    filtered_df = df[year_month_mask & time_mask]
    filtered_df2 = filtered_df.loc[:, ['sensor_id', 'subject_id','ts_on', 'ts_off']]
    
    return filtered_df2

def shower_stove_data(data, SUBJECT_ID, specific_devices,QUERY_INTERVAL_START, QUERY_INTERVAL_END, QUERY_YEAR, QUERY_MONTH):
    data_shower_humidity = []
    stove_shower_df = pd.DataFrame()
    for sd in range(len(specific_devices)):
        # print(sd)
        # print(specific_devices[sd])
        df = data[SUBJECT_ID][specific_devices[sd]][0].copy()
        df['ts_datetime'] = pd.to_datetime(df['ts_datetime'], utc= True)  
        df['ts_datetime'] = pd.to_datetime(df['ts_datetime']).dt.tz_convert('Europe/Rome')
        ## Fiter thd dataframe based on QUERY_YEAR and QUERY_MONTH
        df_filtered = df[(df['ts_datetime'].dt.year == QUERY_YEAR) & (df['ts_datetime'].dt.month == QUERY_MONTH)]
        if len(df_filtered) > 0:
            arranged_np_data, complete_days, binarized_arranged_np_data, datewise_usage = arrange_data_by_day_numpy(df_filtered.copy())
            if len(datewise_usage) == 0: ## Stove has not been used on that day
                continue
            filtered_df = pd.DataFrame()
            filtered_df = filter_rows_by_time_and_date(datewise_usage, QUERY_INTERVAL_START, QUERY_INTERVAL_END, QUERY_YEAR, QUERY_MONTH)
            
            if specific_devices[sd] == 'Stove_Hum_Temp_temp':        
                sensor_id_mapping = {'Stove_Hum_Temp': 'Stove_Temp', 'Hum_Temp_Stove': 'Stove_Temp' }
                
            elif specific_devices[sd] == 'Stove_Hum_Temp_humidity':
                sensor_id_mapping = {'Stove_Hum_Temp': 'Stove_Humidity', 'Hum_Temp_Stove': 'Stove_Humidity' }
                
            elif specific_devices[sd] == 'Shower_Hum_Temp_temp':
                sensor_id_mapping = {'Shower_Hum_Temp': 'Shower_Temp', 'Hum_Temp_Bath':'Shower_Temp' }    
                
            elif specific_devices[sd] == 'Shower_Hum_Temp_humidity':
                sensor_id_mapping = {'Shower_Hum_Temp': 'Shower_Humitidy', 'Hum_Temp_Bath':'Shower_Humitidy' }    
            
            filtered_df.loc[:, 'sensor_id'] = filtered_df['sensor_id'].replace(sensor_id_mapping)
            data_shower_humidity.append(filtered_df) 
            
    if len(data_shower_humidity) > 0:
        stove_shower_df = pd.concat(data_shower_humidity, axis=0, ignore_index=True)
    
    return stove_shower_df


# device_dict, specific_devices, start_time, end_time, target_year, target_month = data_subject, specific_devices, QUERY_INTERVAL_START, QUERY_INTERVAL_END, year_value, month_value,
def non_shower_stove_data(device_dict, specific_devices, start_time, end_time, target_year, target_month):
    # Initialize an empty list to store filtered DataFrames
    filtered_dfs = []
    
    # # Convert start_time and end_time to datetime in Europe/Rome timezone
    # start_time = pd.to_datetime(start_time)#.tz_convert('Europe/Rome')
    # end_time = pd.to_datetime(end_time)#.tz_convert('Europe/Rome')
    
    # Iterate over the dictionary items
    device_in_house = sorted(list(device_dict.keys()))
    
    for device_name in device_in_house:
        # print(device_name)
        filtered_df = pd.DataFrame()
        filtered_df2 = pd.DataFrame()
        
        if device_name not in specific_devices:
            df2 = device_dict[device_name][1].copy()
            
            # Ensure ts_on and ts_off are timezone-aware
            df2['ts_on'] = pd.to_datetime(df2['ts_on'])  # Ensure 'ts_on' is a datetime type
            df2['ts_on'] = df2['ts_on'].dt.tz_convert('UTC')  # Localize to UTC or another appropriate timezone
            df2['ts_on'] = df2['ts_on'].dt.tz_convert('Europe/Rome')
            
            df2['ts_off'] = pd.to_datetime(df2['ts_off'])
            df2['ts_off'] = df2['ts_off'].dt.tz_convert('UTC')  
            df2['ts_off'] = df2['ts_off'].dt.tz_convert('Europe/Rome')
            
            # print(df2['ts_on'].dt.month)
            # print(target_month)
            # print(start_time, end_time)
            # print('------------------')
            filtered_df = df2[(df2['ts_on'].dt.year == target_year) &
                              (df2['ts_on'].dt.month == target_month) ]
                              # (df2['ts_on'] <= end_time) & 
                              # (df2['ts_off'] >= start_time)]  # Overlap condition
            
            # Select specific columns
            filtered_df2 = filtered_df.loc[:, ['sensor_id', 'subject_id', 'ts_on', 'ts_off']]
        else:
            continue
        if not filtered_df2.empty:
            filtered_dfs.append(filtered_df2)            
    
    # Concatenate all filtered DataFrames into one DataFrame
    if filtered_dfs:
        merged_df = pd.concat(filtered_dfs).sort_values(by='ts_on')
    else:
        merged_df = pd.DataFrame()  # Return an empty DataFrame if no data was found
    
    return merged_df

def find_peak_duration_v3(daily_data, peak_index, daily_avg_temperature, k=3):
    left_index = peak_index
    below_count_left = 0
    
    # Traverse left until we find k continuous points below or equal to the daily average temperature
    while left_index > 0:
        if daily_data['sensor_status'].iloc[left_index] <= (daily_avg_temperature+1):
            below_count_left += 1
            # Check if we have found k points below or equal to the daily average temperature
            if below_count_left == k:
                break  # Stop if we have found k points
        else:
            below_count_left = 0  # Reset count if a point above average is encountered
            
        left_index -= 1

    # Adjust left_index to point to the first value below or equal to the average temperature
    if below_count_left == k:
        left_index += 1

    right_index = peak_index
    below_count_right = 0
    
    # Traverse right until we find k continuous points below or equal to the daily average temperature
    while right_index < len(daily_data) - 1:
        if daily_data['sensor_status'].iloc[right_index + 1] <= (daily_avg_temperature+1):
            below_count_right += 1
            # Check if we have found k points below or equal to the daily average temperature
            if below_count_right == k:
                break  # Stop if we have found k points
        else:
            below_count_right = 0  # Reset count if a point above average is encountered
            
        right_index += 1

    # Adjust right_index to point to the last value below or equal to the average temperature
    if below_count_right == k:
        right_index -= 1

    # Calculate duration in terms of index difference
    duration = right_index - left_index
    return duration, left_index, right_index


def plot_daily_activity(date_peak_times, date_cooking_devices, subject, subject_kitchen_devices, save_path):
    # Create a folder for the subject
    subject_folder = os.path.join(save_path, subject)
    os.makedirs(subject_folder, exist_ok=True)  # Create the directory if it doesn't exist

    # Iterate through each date in the dictionaries
    dates_key = list(date_peak_times.keys())
    peaks_df = list(date_peak_times.values())
    devices_df = list(date_cooking_devices.values())
    
    # Iterate through each date index
    for i in range(len(dates_key)):
        date_val = dates_key[i]
        peak_df = peaks_df[i]
        device_df = devices_df[i]
        
        # Concatenate the dataframes
        concated_df = pd.concat([peak_df, device_df])
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        
        # Plot for each device in subject_kitchen_devices
        for device in subject_kitchen_devices:
            device_data = concated_df[concated_df['sensor_id'] == device]
            
            if not device_data.empty:
                for idx, row in device_data.iterrows():
                    plt.plot([row['ts_on'], row['ts_off']], [device, device], color='blue', linewidth=10)
            else:
                # Get the minimum ts_on for the empty device
                min_ts_on = concated_df['ts_on'].min() if not concated_df.empty else pd.Timestamp.now()
                plt.plot([min_ts_on, min_ts_on], [device, device], color='blue', linewidth=10)
        
        # Set y-ticks to include all kitchen devices
        plt.yticks(subject_kitchen_devices)
        
        # Set the x-axis to show the time
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='Europe/Rome'))
        plt.xticks(rotation=45)
        
        # Labels and title
        plt.xlabel('Time of Day (Europe/Rome)')
        plt.ylabel('Devices Used in Cooking')
        plt.title(f'Device Usage in Kitchen on {date_val} for {subject}')
        plt.grid()
        
        # Save the plot with date name
        plt.tight_layout()
        plt.savefig(os.path.join(subject_folder, f'{date_val}.png'))  # Save as PNG
        plt.close()  # Close the figure to free up memory



# # device_dict, start_time, end_time, target_year, target_month, specific_devices = data_subject, QUERY_INTERVAL_START, QUERY_INTERVAL_END, year_value, month_value,specific_devices
# def non_shower_stove_data(device_dict, start_time, end_time, target_year, target_month, specific_devices):

#     # Initialize an empty list to store filtered dataframes
#     filtered_dfs = []
    
#     # Convert start_time and end_time to pandas datetime.time for easy comparison
#     # start_time = pd.to_datetime(start_time).time()
#     # end_time = pd.to_datetime(end_time).time()
    
#     # Iterate over the dictionary items
#     device_in_house = sorted(list(device_dict.keys()))
    
#     for dih in range(len(device_in_house)):
#         device_name = device_in_house[dih]
        
#         filtered_df = pd.DataFrame()
#         filtered_df2 = pd.DataFrame()
#         if device_name not in specific_devices:
#             df2 = device_dict[device_name][1]
#             filtered_df = df2[(df2['ts_on'].dt.year == target_year) &
#                               (df2['ts_on'].dt.month == target_month) &
#                               (
#                                   ((df2['ts_on'].dt.time <= end_time) & (df2['ts_off'].dt.time >= start_time))  # Overlap condition
#                               )]
            
#             ## Select specific columns
#             filtered_df2 = filtered_df.loc[:, ['sensor_id', 'subject_id','ts_on', 'ts_off']]  # Select the columns you need
#             # renamed_columns = selected_columns.rename(columns={'old_column1': 'new_column1', 'old_column2': 'new_column2'})  # Rename the columns

#         else:
#             continue
#             # df1 = device_dict[device_name][0]
#             # filtered_df = df1[(df1['ts_datetime'].dt.year == target_year) & 
#             #                  (df1['ts_datetime'].dt.month == target_month) &
#             #                  (df1['ts_datetime'].dt.time <= end_time) &
#             #                  (df1['ts_datetime'].dt.time >= start_time)]
            
#             # filtered_df2 = filtered_df.loc[:, ['sensor_id', 'subject_id','ts_datetime', 'sensor_status']]
#             # filtered_df2 = filtered_df2.rename(columns={'ts_datetime': 'ts_on'})  # Rename the columns

#         if not filtered_df2.empty:
#             filtered_dfs.append(filtered_df2)            
            


#     # Concatenate all filtered dataframes into one DataFrame
#     if filtered_dfs:
#         merged_df = pd.concat(filtered_dfs).sort_values(by='ts_on')
#     else:
#         merged_df = pd.DataFrame()  # Return an empty DataFrame if no data was found
    
#     return merged_df

def split_by_day(result_df):
    # Create an empty dictionary to store the dataframes by date
    daily_data_dict = {}
    
    # Ensure 'ts_on' is in datetime format if it's not already
    result_df['ts_on'] = pd.to_datetime(result_df['ts_on'])
    
    # Extract year, month, and day and group the dataframe by these
    result_df['date'] = result_df['ts_on'].dt.strftime('%d_%m_%Y')  # Format date as 'dd_mm_yyyy'
    
    # Group the dataframe by the date (year, month, and day)
    grouped = result_df.groupby('date')
    
    # Iterate through the grouped data and store each group in the dictionary
    for date, group in grouped:
        daily_data_dict[date] = group.drop(columns=['date'])  # Store the dataframe and remove the 'date' column
    
    return daily_data_dict


def combine_and_sort_df(stove_shower_df_t1,environmentals_devices_df_t1):
    daily_data_all_devices_dict = {}
    if len(stove_shower_df_t1) > 0 or len(environmentals_devices_df_t1) > 0:
        combined_df_t1 = pd.concat([stove_shower_df_t1, environmentals_devices_df_t1], ignore_index=True) # combined stove, shower, and environmental sensors data
        sorted_df_t1 = combined_df_t1.sort_values(by='ts_on')
        daily_data_all_devices_dict = split_by_day(sorted_df_t1)    
    return daily_data_all_devices_dict

def compute_stove_usage_duration(daily_data_dict):
    """
    Computes the maximum stove usage duration for each day.

    """
    max_stove_usage_per_day = {}

    # Iterate over each date's data
    for d in range(len(daily_data_dict.keys())):
        current_date = list(daily_data_dict.keys())[d]
        current_date_data = daily_data_dict[current_date]

        # Identify stove usage for the current day
        stove_usage = current_date_data[current_date_data['sensor_id'] == 'Stove_Temp']
        
        max_duration = pd.Timedelta(0)  # Initialize max_duration as zero (Timedelta)

        # Iterate through each stove usage period
        for _, stove_row in stove_usage.iterrows():
            stove_on = stove_row['ts_on']
            stove_off = stove_row['ts_off']
            
            # Calculate the duration of this stove usage
            duration = stove_off - stove_on  # Duration as Timedelta object

            # Update the maximum duration if the current duration is greater
            if duration > max_duration:
                max_duration = duration

        # Store the maximum duration for the current date
        max_stove_usage_per_day[current_date] = max_duration

    return max_stove_usage_per_day