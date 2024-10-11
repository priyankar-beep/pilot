#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:37:08 2024

@author: hubble
"""


import pandas as pd, pickle, numpy as np
from scipy.signal import lombscargle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks


with open('/home/hubble/work/serenade/data/data_matteo_upto_september_25_2024_corrected.pkl', 'rb') as file:
    data = pickle.load(file)
    
def plot_sensor_readings_2(arranged_data_np, complete_days):
    plt.figure(figsize=(12, 8))
    plt.imshow(arranged_data_np, cmap='hot', aspect='auto')
    

    plt.colorbar(label='Sensor Status')
    plt.title('Sensor Readings as Image')
    plt.xticks(ticks=np.arange(0, 1440, 60), labels=[f'{h:02d}:00' for h in range(24)])  # 24 hours in 'HH:MM' format
    plt.xlabel('Time of the Day (HH:MM)')
    # y_ticks = np.arange(0, len(arranged_data_np), 1)  # Change 5 to a higher number to reduce ticks further
    tick_positions = np.arange(0, len(complete_days), 10)  # Adjust to show every 10th day if needed
    plt.yticks(ticks=tick_positions, labels=[complete_days[i].strftime('%Y-%m-%d') for i in tick_positions])
    # plt.yticks(ticks=y_ticks, labels=[str(complete_days[i]) for i in y_ticks])
    plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()

def plot_sensor_readings(arranged_data_np, complete_days, minute_index_mapping):
    plt.figure(figsize=(14, 10))
    
    # Plotting the sensor data as an image (heatmap)
    plt.imshow(arranged_data_np.T, cmap='hot', aspect='auto')  # Transpose the data here
    plt.colorbar(label='Sensor Status')
    plt.title('Sensor Readings as Image')
    
    # X-axis: Days
    # plt.xticks(ticks=np.arange(0, len(arranged_data_np[0]), 10), labels=[f'Day {i+1}' for i in range(len(arranged_data_np[0]))])
    tick_positions = np.arange(0, len(complete_days), 10)  # Adjust to show every 10th day if needed
    plt.xticks(ticks=tick_positions, labels=[complete_days[i].strftime('%Y-%m-%d') for i in tick_positions])
    plt.xlabel('Days')
    plt.xticks(rotation=90)
    
    # Y-axis: Time of day in 'HH:MM' format
    plt.yticks(ticks=np.arange(0, 1440, 60), labels=[f'{h:02d}:00' for h in range(24)])
    plt.ylabel('Time of the Day (HH:MM)')
    
    # Grid
    plt.grid(True)
    
    # Plot peaks as dots (solid) on top of the heatmap
    for day in range(len(minute_index_mapping)):
        for minute in minute_index_mapping[day]:
            print(day, minute)
            if minute != -99: 
                plt.plot(day, minute, 'o', color='green', markersize=3)  # Note the axes are swapped
    
    plt.tight_layout()
    plt.show()

def get_minute_index(ts_datetime):
    # Convert to pandas datetime for easier extraction
    dt = pd.to_datetime(ts_datetime)
    # Calculate the minute index: hour * 60 + minute
    minute_index = dt.hour * 60 + dt.minute
    return minute_index

def find_peaks_with_durations(daily_data, peaks, max_time_diff,day):
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


def arrange_data_by_day_numpy(df):
    df['ts_datetime'] = pd.to_datetime(df['ts_datetime'])
    df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
    start_date = df['ts_datetime'].dt.date.min()
    end_date = df['ts_datetime'].dt.date.max()
    complete_days = pd.date_range(start=start_date, end=end_date, freq='D').date
    # complete_days = [date for date in complete_days if date.year == 2024 and date.month in [2]]

    daily_data_list = [] 
    peak_start_end_list = []
    minute_index_mapping = []
    datewise_peak_info_list = []
    # for day in complete_days:
    for d in range(len(complete_days)):#complete_days:
        day = complete_days[d]
        daily_data = df[df['ts_datetime'].dt.date == day]
        ## Calculate the peaks
        peaks, properties = find_peaks(daily_data['sensor_status'], prominence=1.5)
        # if len(peaks)>0:
        #     print(d, day)
        
        ## Compute the duration of each peak
        max_time_diff = pd.Timedelta(minutes=10)
        peak_info_list = find_peaks_with_durations(daily_data, peaks, max_time_diff, day)
        datewise_peak_info_list.extend(peak_info_list)
        
        ## Now map the computed peak indices to actual minute index in a day
        peak_to_minute_index_mapping = []
        if len(peaks) > 0:
            for p in range(len(peaks)):
                temp = daily_data.iloc[peaks[p]]['ts_datetime']
                temp_minute_index = get_minute_index(temp)
                peak_to_minute_index_mapping.append(temp_minute_index)
        else:
            peak_to_minute_index_mapping.append(-99)
        minute_index_mapping.append(peak_to_minute_index_mapping)

        ## Place the sensor reading when it was actually observed in the day
        all_minutes = pd.date_range(start=pd.Timestamp(day), end=pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='min')
        daily_resampled = daily_data.set_index('ts_datetime')['sensor_status'].resample('min').max()
        daily_resampled = daily_resampled.reindex(all_minutes, fill_value=np.nan)
        daily_resampled = daily_resampled.fillna(0)  # Fill NaN with 0
        if daily_data.empty:
            daily_data_list.append(np.full((1440,), 0))  # 1440 minutes in a day filled with 255
        else:
            daily_data_list.append(daily_resampled.values)        
          
            
        ## replace peak_start and peak_end duration with 1 else zero
        peak_Start_end_df = pd.DataFrame(all_minutes, columns=['minute'])
        peak_Start_end_df['activity'] = 0
        def mark_active_minutes(minutes_df, peak_info_list):
            for peak in peak_info_list:
                ts_on = peak['ts_on'].floor('min')  # Round to the nearest minute
                ts_off = peak['ts_off'].floor('min')
        
                # Mark all minutes between ts_on and ts_off as 1
                minutes_df.loc[(minutes_df['minute'] >= ts_on) & (minutes_df['minute'] <= ts_off), 'activity'] = 1
            
            return minutes_df
       
        peak_Start_end_df = mark_active_minutes(peak_Start_end_df, peak_info_list)
        peak_Start_end_df_resampled = peak_Start_end_df.set_index('minute')['activity']
        peak_start_end_list.append(peak_Start_end_df_resampled.values)
    
        
    arranged_peak_start_end_data_np = np.array(peak_start_end_list) 
    arranged_data_np = np.array(daily_data_list)
    plot_sensor_readings(arranged_peak_start_end_data_np, complete_days, minute_index_mapping)
    plot_sensor_readings(arranged_data_np, complete_days, minute_index_mapping)
    return arranged_data_np, complete_days, arranged_peak_start_end_data_np, pd.DataFrame(datewise_peak_info_list)

def arrange_data_by_day_numpy_environmentals(df):
    # Ensure timestamps are in datetime format
    df['ts_on'] = pd.to_datetime(df['ts_on'])
    df['ts_off'] = pd.to_datetime(df['ts_off'])
    
    # Create a date range for complete days
    start_date = df['ts_on'].dt.date.min()
    end_date = df['ts_on'].dt.date.max()
    complete_days = pd.date_range(start=start_date, end=end_date, freq='D').date
    
    daily_data_list = []
    for d in range(len(complete_days)):
        # Filter data for the current day
        day = complete_days[d]
        daily_data = df[df['ts_on'].dt.date == day]
        all_minutes = pd.date_range(start=pd.Timestamp(day), end=pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='min')
        daily_resampled = np.zeros((1440,), dtype=int)  # 1440 minutes in a day
        
        if daily_data.empty:
            daily_data_list.append(daily_resampled) 
            continue  
        for _, row in daily_data.iterrows():
            start_minute = (row['ts_on'] - pd.Timestamp(day)).total_seconds() // 60
            end_minute = (row['ts_off'] - pd.Timestamp(day)).total_seconds() // 60
            
            if start_minute == end_minute:
                daily_resampled[int(start_minute)] = 1
            elif start_minute < end_minute:
                daily_resampled[int(start_minute):int(end_minute)] = 1  # Set the range to 1
        
        daily_data_list.append(daily_resampled)

    arranged_data_np = np.array(daily_data_list)
    plot_sensor_readings_2(arranged_data_np, complete_days)
    
    return arranged_data_np , complete_days

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

def filter_and_merge_data(device_dict, start_time, end_time, target_year, target_month):
    # Define the devices where ts_datetime should be used instead of ts_on
    specific_devices = ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']
    
    # Initialize an empty list to store filtered dataframes
    filtered_dfs = []
    
    # Convert start_time and end_time to pandas datetime.time for easy comparison
    start_time = pd.to_datetime(start_time).time()
    end_time = pd.to_datetime(end_time).time()
    
    # Iterate over the dictionary items
    device_in_house = sorted(list(device_dict.keys()))
    
    for dih in range(len(device_in_house)):
        device_name = device_in_house[dih]
        
        
        filtered_df = pd.DataFrame()
        filtered_df2 = pd.DataFrame()
        if device_name not in specific_devices:
            df2 = device_dict[device_name][1]
            filtered_df = df2[(df2['ts_on'].dt.year == target_year) &
                             (df2['ts_on'].dt.month == target_month) &
                             (
                                 ((df2['ts_on'].dt.time <= end_time) & (df2['ts_off'].dt.time >= start_time))  # Overlap condition
                             )]
            
            ## Select specific columns
            filtered_df2 = filtered_df.loc[:, ['sensor_id', 'subject_id','ts_on', 'ts_off']]  # Select the columns you need
            # renamed_columns = selected_columns.rename(columns={'old_column1': 'new_column1', 'old_column2': 'new_column2'})  # Rename the columns

        else:
            continue
            df1 = device_dict[device_name][0]
            filtered_df = df1[(df1['ts_datetime'].dt.year == target_year) & 
                             (df1['ts_datetime'].dt.month == target_month) &
                             (df1['ts_datetime'].dt.time <= end_time) &
                             (df1['ts_datetime'].dt.time >= start_time)]
            
            filtered_df2 = filtered_df.loc[:, ['sensor_id', 'subject_id','ts_datetime', 'sensor_status']]
            filtered_df2 = filtered_df2.rename(columns={'ts_datetime': 'ts_on'})  # Rename the columns

        if not filtered_df2.empty:
            filtered_dfs.append(filtered_df2)            
            


    # Concatenate all filtered dataframes into one DataFrame
    if filtered_dfs:
        merged_df = pd.concat(filtered_dfs).sort_values(by='ts_on')
    else:
        merged_df = pd.DataFrame()  # Return an empty DataFrame if no data was found
    
    return merged_df

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

def merge_continuous_sensor_occurrences(df):
    """
    Merge continuous occurrences of the same sensor_id in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns ['sensor_id', 'subject_id', 'ts_on', 'ts_off'].

    Returns:
    pd.DataFrame: New DataFrame with merged continuous occurrences.
    """
    if len(df) == 0:
        print('hi')
        return pd.DataFrame()
    # Convert 'ts_on' and 'ts_off' to datetime if they are not already
    df['ts_on'] = pd.to_datetime(df['ts_on'])
    df['ts_off'] = pd.to_datetime(df['ts_off'])

    # Create a new DataFrame for merged occurrences
    merged_rows = []
    current_sensor_id = None
    current_ts_on = None
    current_ts_off = None

    for _, row in df.iterrows():
        if row['sensor_id'] == current_sensor_id:
            # If the current sensor is the same, extend the current off time if needed
            current_ts_off = max(current_ts_off, row['ts_off'])
        else:
            # If it's a new sensor, save the previous sensor's data (if any)
            if current_sensor_id is not None:
                merged_rows.append({
                    'sensor_id': current_sensor_id,
                    'ts_on': current_ts_on,
                    'ts_off': current_ts_off
                })

            # Start a new sensor tracking
            current_sensor_id = row['sensor_id']
            current_ts_on = row['ts_on']
            current_ts_off = row['ts_off']

    # Append the last sensor's data
    if current_sensor_id is not None:
        merged_rows.append({
            'sensor_id': current_sensor_id,
            'ts_on': current_ts_on,
            'ts_off': current_ts_off
        })

    # Create a new DataFrame from merged rows
    merged_df = pd.DataFrame(merged_rows)

    return merged_df