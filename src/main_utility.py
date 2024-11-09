#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:28:15 2024

@author: hubble
"""

import numpy as np, pandas as pd, os, pickle, glob, math
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.dates as mdates
import pytz
from datetime import datetime
import seaborn as sns
from natsort import natsorted
from scipy.signal import find_peaks

color_mapping = {
    "CatFood": "#6953e6",
    "CoffeMachine": "#a716d6",
    "Cookware": "#7f2258",
    "Dishes": "#8fb921",
    "Dishes_Glasses": "#96c658",
    "Dishes_Silverware": "#601031",
    "FoodStorage": "#02a874",
    "FoodStorageKitchen": "#210161",
    "FoodStorageLivingRoom": "#de32e1",
    "Freezer": "#d396dc",
    "Glasses": "#6abf73",
    "HouseEntrance": "#8556c2",
    "Medicines": "#80776d",
    "Microwave": "#0ec579",
    "MotionBathroom": "#2b9568",
    "MotionBedroom": "#5a34e3",
    "MotionCloset": "#f41f2d",
    "MotionDiningTable": "#f62dd2",
    "MotionGuestRoom": "#20e5bb",
    "MotionKitchen": "#e1ec9c",
    "MotionLivingRoomSofa": "#92ef91",
    "MotionLivingRoomTablet": "#5b1a69",
    "MotionLivingroom": "#696ba8",
    "MotionOffice": "#e2023e",
    "MotionOtherRoom": "#216d5a",
    "MotionOtherroom": "#f12964",
    "MotionPrimaryBathroom": "#82457c",
    "MotionSecondaryBathroom": "#8a5cd0",
    "PlugTvHall": "#8e2926",
    "PlugTvKitchen": "#409159",
    "PresenceKitchen": "#552038",
    "PresenceKitchen_Stove": "#8e1f49",
    "PresenceKitchen_Table": "#6b6e57",
    "PresenceLivingroom": "#12182f",
    "PresenceLivingroom_Sofa": "#f10f8b",
    "PresenceLivingroom_Table": "#7e82b4",
    "Printer": "#f50617",
    "Refrigerator": "#bacaad",
    "Shower_Hum_Temp_humidity": "#0b306d",
    "Shower_Hum_Temp_temp": "#cbf367",
    "Silverware": "#bc5003",
    "Stove_Hum_Temp_humidity": "#d93274",
    "Stove_Hum_Temp_temp": "#b2fa98",
    "WashingMachine": "#303676"
}


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


def analyze_seasonal_peaks_with_duration(df_stove_temperature , prom = 1.5, theta = 40):
    
    # Load the data
    df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
    df_stove_temperature['daily_avg_temp'] = df_stove_temperature.groupby('date')['sensor_status'].transform('median')
    df_stove_temperature['daily_avg_std'] = df_stove_temperature.groupby('date')['sensor_status'].transform('std')

    # Step 1: Detect peaks ()
    sensor_values = df_stove_temperature['sensor_status'].values # Get all the temperature readings in an array
    peaks, _ = find_peaks(sensor_values, prominence = prom) # Find the peaks in temperature readings
    df_stove_temperature['peaks'] = 0
    df_stove_temperature.loc[peaks, 'peaks'] = 1  # Mark peaks as 1
    
    # # Remove the peaks below median_temperature_+ standard_deviation_in_tempearature_of the day in consideration
    # def remove_peaks_below_daily_avg(temperature_df, peaks):
    #     filtered_peaks = [
    #         peak for peak in peaks
    #         if temperature_df.loc[peak, 'sensor_status'] > (
    #             temperature_df.loc[peak, 'daily_avg_temp'] + temperature_df.loc[peak, 'daily_avg_std']
    #         )
    #     ]
    #     return filtered_peaks
    # filtered_peaks = remove_peaks_below_daily_avg(df_stove_temperature, peaks)
    # peaks = filtered_peaks
    # # Step 7: Mark only the valid peaks as 1 in the 'peaks' column
    # df_stove_temperature['peaks'] = 0  # Reset the peaks column to 0
    # df_stove_temperature.loc[filtered_peaks, 'peaks'] = 1  # Mark the valid peaks
    
    df_stove_temperature.drop(columns=['date', 'daily_avg_temp', 'daily_avg_std'], inplace=True)

    # Step 2: Define winter and summer months
    winter_months = [11, 12, 1, 2]
    summer_months = [3, 4, 5, 6, 7, 8, 9, 10]
    
    # Step 3: Calculate mean and standard deviation for winter and summer
    winter_df = df_stove_temperature[df_stove_temperature['ts_datetime'].dt.month.isin(winter_months)].copy()
    summer_df = df_stove_temperature[df_stove_temperature['ts_datetime'].dt.month.isin(summer_months)].copy()

    # Keep original indices to map back to the main DataFrame
    winter_df.reset_index(inplace=True)
    summer_df.reset_index(inplace=True)
    
    # winter_df.set_index('index', inplace=True)
    # summer_df.set_index('index', inplace=True)

    winter_mean = winter_df['sensor_status'].mean()
    winter_std = winter_df['sensor_status'].std()
    summer_mean = summer_df['sensor_status'].mean()
    summer_std = summer_df['sensor_status'].std()
    
    # Step 4: Initialize peak analysis dictionary with durations and additional info
    peak_analysis = {
        "winter": {"mean": winter_mean, "std_dev": winter_std, "categories": {}, "duration_percentages": {}},
        "summer": {"mean": summer_mean, "std_dev": summer_std, "categories": {}, "duration_percentages": {}}
    }

    # Duration categories
    duration_bins = [
        (0, 5),
        (5, 10),
        (10, 20),
        (20, 100),
        (100, 200),
        (200, float('inf'))  # For 200 minutes and above
    ]
    
    duration_labels = ['< 5', '5-10', '10-20', '20-100', '100-200', '200+']
    # df, df_stove_temperature, mean, std_dev, season_name = winter_df,df_stove_temperature, winter_mean, winter_std, "winter"
    # df, df_stove_temperature, mean, std_dev, season_name = summer_df,df_stove_temperature, summer_mean, summer_std, "summer"
    def categorize_peaks_with_duration(df, df_stove_temperature, mean, std_dev, season_name):
        categories = {
            "within_1_sigma": [],
            "between_1_and_2_sigma": [],
            "between_2_and_3_sigma": [],
            "above_3_sigma": []
        }
        
        # print( df[df['peaks'] == 1]['index'])
        for index in df[df['peaks'] == 1]['index']:  # Use original index from reset
            print(index)
            # break
            # Retrieve the value and deviation
            value = df_stove_temperature.loc[index, 'sensor_status']
            deviation = abs(value - mean) # mean represent the average tempearute of winter or summer 
            # print(index, value, deviation) # peak index, temperature value, and difference from season mean
            
            # Classify the peak into a category
            if deviation <= std_dev: # if the deviation in sensor reading below
                category = "within_1_sigma"
            elif std_dev < deviation <= 2 * std_dev:
                category = "between_1_and_2_sigma"
            elif 2 * std_dev < deviation <= 3 * std_dev:
                category = "between_2_and_3_sigma"
            else:
                category = "above_3_sigma"

            # Find the left and right bounds for each peak
            # break
            # daily_data = df_stove_temperature
            # _, left, right = find_peak_duration_v3(df_stove_temperature.copy(), index, 3, 'median')
            
            left = slope_based_peak_start(index, df, theta)
            
            
            # print(df_stove_temperature_dash.head())
            # Convert left and right indices to timestamps
            left_time = df_stove_temperature['ts_datetime'].iloc[left]
            peak_time = df_stove_temperature['ts_datetime'].iloc[index]
            right_time = peak_time + (peak_time - left_time) #df_stove_temperature['ts_datetime'].iloc[right]
            peak_temperature = df_stove_temperature['sensor_status'].iloc[index]

            duration_minutes = (right_time - left_time).total_seconds() / 60
            # Append the data to the appropriate category
            categories[category].append({
                "peak_index": index,
                "peak_temperature":peak_temperature,
                "duration_minutes": duration_minutes,
                "left_time": left_time,
                "right_time": right_time,
                "peak_time": peak_time
            })


        # Calculate duration percentages for each category
        for cat, peaks in categories.items():
            total_peaks = len(peaks)
            if total_peaks == 0:
                peak_analysis[season_name]["duration_percentages"][cat] = {label: 0 for label in duration_labels}
                continue
            
            duration_counts = {label: 0 for label in duration_labels}
            
            for peak in peaks:
                duration = peak["duration_minutes"]
                for i, (low, high) in enumerate(duration_bins):
                    if low < duration <= high:
                        duration_counts[duration_labels[i]] += 1
                        break
            
            # Calculate percentages
            peak_analysis[season_name]["duration_percentages"][cat] = {label: (count / total_peaks) * 100 for label, count in duration_counts.items()}

        peak_analysis[season_name]["categories"] = categories
        # return df_stove_temperature_dash
        
    # Apply categorization and duration calculation for winter and summer peaks
    categorize_peaks_with_duration(winter_df, df_stove_temperature,winter_mean, winter_std, "winter")
    categorize_peaks_with_duration(summer_df, df_stove_temperature,summer_mean, summer_std, "summer")
    
    return peak_analysis


# def plot_duration_peaks(result, save_dir, subject_id):
#     # Define the sigma level groups for consistent coloring and legend
#     sigma_levels = ['within_1_sigma', 'between_1_and_2_sigma', 'between_2_and_3_sigma', 'above_3_sigma']
#     colors = {
#         'within_1_sigma': 'red',
#         'between_1_and_2_sigma': 'orange',
#         'between_2_and_3_sigma': 'green',
#         'above_3_sigma': 'blue'
#     }
    
#     # Define duration categories
#     duration_categories = ['< 5', '5-10', '10-20', '20-100', '100-200', '200+']
    
#     # Make sure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Create a figure with two subplots: one for winter and one for summer
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
#     fig.suptitle(f'Duration Peaks for Subject {subject_id}')

#     # Plot for each season
#     for i, season in enumerate(['winter', 'summer']):
#         data = result[season]['duration_percentages']
#         x = np.arange(len(duration_categories))  # Set x-axis positions for duration categories
#         bar_width = 0.2  # Width for each sigma level's bar

#         for j, sigma_level in enumerate(sigma_levels):
#             # Extract percentages for the current sigma level across all duration categories
#             values = [data[sigma_level][duration] for duration in duration_categories]
#             # Plot each sigma level's bar in the appropriate position for each duration category
#             bars = ax[i].bar(x + j * bar_width, values, width=bar_width, color=colors[sigma_level], 
#                              label=sigma_level, alpha=0.7)
            
#             # Annotate each bar with its corresponding value, rotated vertically
#             for bar in bars:
#                 height = bar.get_height()
#                 ax[i].text(
#                     bar.get_x() + bar.get_width() / 2, height,
#                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90
#                 )

#         # Set titles, labels, and ticks for each subplot
#         ax[i].set_title(f'{season.capitalize()}')
#         ax[i].set_xlabel('Duration (minutes)')
#         ax[i].set_ylabel('Percentage')
#         ax[i].set_xticks(x + bar_width * (len(sigma_levels) - 1) / 2)  # Center the x-ticks
#         ax[i].set_xticklabels(duration_categories)
#         ax[i].legend(title="Sigma Levels")

#     # Save the plot with a filename based on subject_id
#     file_path = os.path.join(save_dir, f"{subject_id}_duration_peaks.png")
#     plt.savefig(file_path)
#     plt.close(fig)

def plot_duration_peaks(result, save_dir, subject_id):
    # Define sigma levels and duration categories for grouping and colors
    sigma_levels = ['within_1_sigma', 'between_1_and_2_sigma', 'between_2_and_3_sigma', 'above_3_sigma']
    duration_categories = ['< 5', '5-10', '10-20', '20-100', '100-200', '200+']
    
    # Assign colors to each duration category
    colors = {
        '< 5': 'red',
        '5-10': 'orange',
        '10-20': 'green',
        '20-100': 'blue',
        '100-200': 'purple',
        '200+': 'brown'
    }
    
    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with two subplots: one for winter and one for summer
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f'Duration Peaks for Subject {subject_id}')

    # Plot for each season
    for i, season in enumerate(['winter', 'summer']):
        data = result[season]['duration_percentages']
        x = np.arange(len(sigma_levels))  # Set x-axis positions for each sigma level
        bar_width = 0.1  # Width for each duration category's bar

        # Plot each sigma level with bars for each duration category
        for j, duration in enumerate(duration_categories):
            # Extract percentages for the current duration category across all sigma levels
            values = [data[sigma_level][duration] for sigma_level in sigma_levels]
            # Plot each duration category as a separate color-coded bar within each sigma level group
            bars = ax[i].bar(x + j * bar_width, values, width=bar_width, color=colors[duration],
                             label=duration if i == 0 else "", alpha=0.7)
            
            # Annotate each bar with its corresponding value, rotated vertically
            for bar in bars:
                height = bar.get_height()
                ax[i].text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90
                )

        # Set titles, labels, and ticks for each subplot
        ax[i].set_title(f'{season.capitalize()}')
        ax[i].set_xlabel('Sigma Levels')
        ax[i].set_ylabel('Percentage')
        ax[i].set_xticks(x + bar_width * (len(duration_categories) - 1) / 2)  # Center the x-ticks
        ax[i].set_xticklabels(sigma_levels, rotation=90)  # Rotate x-tick labels by 90 degrees

    # Place the legend below the x-axis across both subplots
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", title="Duration (minutes)", ncol=6, bbox_to_anchor=(0.5, -0.15))

    # Adjust the layout to make space for the legend
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

    # Save the plot with a filename based on subject_id, ensuring nothing is clipped
    file_path = os.path.join(save_dir, f"{subject_id}_duration_peaks.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

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
# data_path = path_data
def read_csv_files_v2(data_path):
    subjects = natsorted(os.listdir(data_path))
    subject_dfs = {}     
    for s in range(len(subjects)):
        print('-*'*40)
        subject_s = subjects[s]
        print(subject_s)
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
                    # df['ts_datetime'] = pd.to_datetime(df['local_time'],format='mixed')
                    
                    # Function to parse timestamp flexibly
                    def parse_datetime(time_str):
                        # Try parsing with microseconds first
                        for fmt in ('%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z'):
                            try:
                                time_dt = datetime.strptime(time_str, fmt)
                                # Convert to Europe/Rome timezone
                                return time_dt.astimezone(pytz.timezone('Europe/Rome'))
                            except ValueError:
                                continue
                        raise ValueError(f"Time data '{time_str}' does not match any expected format.")
                        
                    df['ts_datetime'] = df['local_time'].apply(parse_datetime)
                    invalid_rows = df[df['ts_datetime'].isna()]
                    if not invalid_rows.empty:
                        print(f"Found invalid rows (NaT or invalid format):\n{invalid_rows}")
                    df.rename(columns={'timestamp (GMT)': 'ts'}, inplace=True)
                    # df['ts_datetime'] = df['local_time'].apply(lambda time_str: 
                    #                        datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S%z')
                    #                        .astimezone(pytz.timezone('Europe/Rome')))

                    

                    # df['ts_datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    # df['ts_datetime_utc'] = pd.to_datetime(df['ts'], unit='ms')
                    # df['ts_datetime_utc'] = df['ts_datetime_utc'].dt.tz_localize('UTC')
                    # df['ts_datetime'] = df['ts_datetime_utc'].dt.tz_convert('Europe/Rome')
                    # df.drop(columns=['ts_datetime_utc'], inplace=True)
                    if not isEnvironmentalSensor:
                      
                        if folder_name in ['Shower_Hum_Temp','Stove_Hum_Temp','Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_humidity', 'Shower_Hum_Temp_temp']:
                            print(subject_s, folder_name, 'Analog sensor')
                            df['sensor_status'] = pd.to_numeric(df['sensor_status'])#, errors='coerce')
                            dfs_dict[folder_name] = [df, pd.DataFrame()]
                        else: 
                            print(subject_s, folder_name, 'Analog sensor converted to on-off')
                            threshold_value = get_threshold_value(subjects_threshold_dict, subject_s, folder_name)
                            df['sensor_status'] = pd.to_numeric(df['sensor_status'])#, errors='coerce')
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

# # daily_data, peak_index, k ,agg_var= df_stove_temperature.copy(), index, 3, 'median'
def find_peak_duration_v3(daily_data, peak_index, k, agg_var):
    
    columns_to_remove = ['daily_avg_temperature', 'daily_median_temperature', 'daily_std_temperature']
    daily_data = daily_data.drop(columns=[col for col in columns_to_remove if col in daily_data.columns])

    left_index = peak_index
    below_count_left = 0
    
    if agg_var == 'mean':
        col_name = 'daily_avg_temperature'
    elif agg_var == 'median':
        col_name = 'daily_median_temperature'
        
    daily_data['date'] = daily_data['ts_datetime'].dt.date

    # Calculate daily average, median, and standard deviation of sensor_status
    daily_stats = daily_data.groupby('date')['sensor_status'].agg(
        daily_avg_temperature='mean',
        daily_median_temperature='median',
        daily_std_temperature='std'
    ).reset_index()
    
    # Merge the statistics back into the original DataFrame
    daily_data = pd.merge(daily_data, daily_stats, on='date', how='left')
    # Traverse left until we find k continuous points below or equal to the daily average temperature
    points_below_threshold = []  # To store points that meet the condition

    while left_index > 0:
        current_day_avg = daily_data[col_name].iloc[left_index]
        current_day_std = daily_data['daily_std_temperature'].iloc[left_index]
        
        if daily_data['sensor_status'].iloc[left_index] <= (current_day_avg + current_day_std):
            below_count_left += 1
            points_below_threshold.append(daily_data['sensor_status'].iloc[left_index])  # Store the point
            # Check if we have found k  continuous points below or equal to the daily average temperature
            if below_count_left == k:
                break  # Stop if we have found k points
        else:
            below_count_left = 0  # Reset count if a point above average is encountered
            points_below_threshold = []   
        left_index -= 1

    right_index = peak_index
    below_count_right = 0
    points_below_threshold = []
    # Traverse right until we find k continuous points below or equal to the daily average temperature
    while right_index < len(daily_data) - 1:
        current_day_avg = daily_data[col_name].iloc[right_index ]
        current_day_std = daily_data['daily_std_temperature'].iloc[right_index]
        if daily_data['sensor_status'].iloc[right_index] <= (current_day_avg + current_day_std):
            # print(daily_data.iloc[right_index]['ts_datetime'])
            points_below_threshold.append(daily_data['sensor_status'].iloc[right_index])  # Store the point
            below_count_right += 1
            # Check if we have found k continuous points below or equal to the daily average temperature
            if below_count_right == k:               
                break  # Stop if we have found k points
        else:
            below_count_right = 0  # Reset count if a point above average is encountered
            points_below_threshold = []
            
        right_index += 1
    return daily_data, left_index, right_index

# peak_index, df = index, df
def slope_based_peak_start(peak_index, df, theta):
    print(theta)
    peak_row = df[df['index'] == peak_index]
    if peak_row.empty:
        print("Peak index not found in the DataFrame.")
        return -99
    
    y2 = peak_row['sensor_status'].values[0]
    x2 = peak_row['ts'].values[0]
    left_time = pd.to_datetime(x2, unit='ms')
    left_index = peak_index
    
    peak_row_position = peak_row.index[0]
    i = peak_row_position - 1
    while i >= df.index[0]:
        y1 = df.loc[i, 'sensor_status']
        x1 = df.loc[i, 'ts']
        prev_time = pd.to_datetime(x1, unit='ms')
        
        # Check if left_time and prev_time have the same hour and minute
        if y2 == y1: #or (left_time.hour == prev_time.hour and left_time.minute == prev_time.minute):
            # Reset values and move to the next previous index
            y2 = y1
            x2 = x1
            left_time = prev_time
            left_index = df.iloc[i]['index']
            i -= 1
            continue  # Skip the current iteration
        
        
        # Calculate the slope (y2 - y1) / (x2 - x1), where x values are in hours
        time_diff_in_hours = (left_time - prev_time).total_seconds() / 3600.0

        if time_diff_in_hours == 0:
            print('something went wrong')
            break  
        
        slope = (y2 - y1) / time_diff_in_hours
        angle = math.atan(slope) * (180 / math.pi)
        
        if slope < 0:
            print('hi')
            return left_index
            
        else:
            if angle < theta:
                print('bye')
                return left_index
            else:
                y2 = y1
                x2 = x1
                left_time = prev_time
                left_index = df.iloc[i]['index']#i
        
        i -= 1
    return left_index


# temperature_df, save_path = df_stove_temperature, '/home/hubble/temp'
def plot_peaks_and_mean_temperature(temperature_df, save_path):
    # Extract the subject name from the subject_id column
    subject_id = temperature_df['subject_id'].iloc[0]
    subject_name = f"Subject_{subject_id}"
    
    # Ensure save_path exists and create a folder for the subject's peak values
    save_folder = save_path#os.path.join(save_path, f"{subject_name}_peak_values")
    os.makedirs(save_folder, exist_ok=True)
    
    # Step 0: Detect peaks on the entire dataset
    sensor_values = temperature_df['sensor_status'].values
    peaks, _ = find_peaks(sensor_values, prominence=1.5)
    temperature_df['peaks'] = 0
    temperature_df.loc[peaks, 'peaks'] = 1  # Mark peaks as '1' in the 'peaks' column
        
    # Step 1: Divide data into winter (November-February) and summer (March-October)
    temperature_df['month'] = temperature_df['ts_datetime'].dt.month
    winter_mask = temperature_df['month'].isin([11, 12, 1, 2])
    summer_mask = temperature_df['month'].isin([3, 4, 5, 6, 7, 8, 9, 10])
    
    # Step 2: Calculate average temperature and standard deviation for winter and summer
    average_temperature_winter = temperature_df[winter_mask]['sensor_status'].mean()
    std_deviation_winter = temperature_df[winter_mask]['sensor_status'].std()
    
    average_temperature_summer = temperature_df[summer_mask]['sensor_status'].mean()
    std_deviation_summer = temperature_df[summer_mask]['sensor_status'].std()

    # Step 3: Create subplots for winter and summer
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=False)

    # Winter subplot (November to February)
    winter_df = temperature_df[winter_mask]
    axs[0].scatter(winter_df[winter_df['peaks'] == 1]['ts_datetime'],
                   winter_df[winter_df['peaks'] == 1]['sensor_status'], 
                   color='red', label='Peak Temperatures')
    
    # Draw horizontal lines for winter mean and multiple standard deviations
    axs[0].axhline(y=average_temperature_winter, color='black', linestyle='-', 
                   label=f'Winter Avg: {average_temperature_winter:.2f} °C')
    
    # 1σ (magenta), 2σ (blue), and 3σ (black)
    axs[0].axhline(y=average_temperature_winter + std_deviation_winter, color='magenta', linestyle='--', 
                   label=f'Winter +1σ: {average_temperature_winter + std_deviation_winter:.2f} °C')
    # axs[0].axhline(y=average_temperature_winter - std_deviation_winter, color='magenta', linestyle='--', 
    #                label=f'Winter -1σ: {average_temperature_winter - std_deviation_winter:.2f} °C')

    axs[0].axhline(y=average_temperature_winter + 2 * std_deviation_winter, color='blue', linestyle='--', 
                   label=f'Winter +2σ: {average_temperature_winter + 2 * std_deviation_winter:.2f} °C')
    # axs[0].axhline(y=average_temperature_winter - 2 * std_deviation_winter, color='blue', linestyle='--', 
    #                label=f'Winter -2σ: {average_temperature_winter - 2 * std_deviation_winter:.2f} °C')

    axs[0].axhline(y=average_temperature_winter + 3 * std_deviation_winter, color='black', linestyle='--', 
                   label=f'Winter +3σ: {average_temperature_winter + 3 * std_deviation_winter:.2f} °C')
    # axs[0].axhline(y=average_temperature_winter - 3 * std_deviation_winter, color='black', linestyle='--', 
    #                label=f'Winter -3σ: {average_temperature_winter - 3 * std_deviation_winter:.2f} °C')

    # Set x-ticks only on every 5th peak to avoid overcrowding
    peak_times_winter = winter_df[winter_df['peaks'] == 1]['ts_datetime']
    selected_peak_times_winter = peak_times_winter[::5]  # Select every 5th peak
    if len(selected_peak_times_winter)>0:
        axs[0].set_xticks(selected_peak_times_winter)
    axs[0].tick_params(axis='x', rotation=90)
    # axs[0].set_xlim([pd.Timestamp('2023-11-01'), pd.Timestamp('2024-02-28')])
    
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title(f"{subject_name} - Winter Temperature with Peaks")
    axs[0].legend()
    axs[0].grid()

    # Summer subplot (March to October)
    summer_df = temperature_df[summer_mask]


    axs[1].scatter(summer_df[summer_df['peaks'] == 1]['ts_datetime'],
                   summer_df[summer_df['peaks'] == 1]['sensor_status'], 
                   color='red', label='Peak Temperatures')

    # print('********************************')
    # print(summer_df['ts_datetime'].isna().sum())
    # print('********************************')
    # Draw horizontal lines for summer mean and multiple standard deviations
    axs[1].axhline(y=average_temperature_summer, color='black', linestyle='-', 
                   label=f'Summer Avg: {average_temperature_summer:.2f} °C')
    
    # 1σ (magenta), 2σ (blue), and 3σ (black)
    axs[1].axhline(y=average_temperature_summer + std_deviation_summer, color='magenta', linestyle='--', 
                    label=f'Summer +1σ: {average_temperature_summer + std_deviation_summer:.2f} °C')
    # axs[1].axhline(y=average_temperature_summer - std_deviation_summer, color='magenta', linestyle='--', 
                    # label=f'Summer -1σ: {average_temperature_summer - std_deviation_summer:.2f} °C')

    axs[1].axhline(y=average_temperature_summer + 2 * std_deviation_summer, color='blue', linestyle='--', 
                    label=f'Summer +2σ: {average_temperature_summer + 2 * std_deviation_summer:.2f} °C')
    # axs[1].axhline(y=average_temperature_summer - 2 * std_deviation_summer, color='blue', linestyle='--', 
                    # label=f'Summer -2σ: {average_temperature_summer - 2 * std_deviation_summer:.2f} °C')

    axs[1].axhline(y=average_temperature_summer + 3 * std_deviation_summer, color='black', linestyle='--', 
                    label=f'Summer +3σ: {average_temperature_summer + 3 * std_deviation_summer:.2f} °C')
    # axs[1].axhline(y=average_temperature_summer - 3 * std_deviation_summer, color='black', linestyle='--', 
                    # label=f'Summer -3σ: {average_temperature_summer - 3 * std_deviation_summer:.2f} °C')

    # Set x-ticks only on every 5th peak to avoid overcrowding
    peak_times_summer = summer_df[summer_df['peaks'] == 1]['ts_datetime']
    # print(peak_times_summer)
    # print()
    # print()
    selected_peak_times_summer = peak_times_summer[::10]  # Select every 5th peak
    # print(selected_peak_times_summer)
    # print('---------------------')
    axs[1].set_xticks(selected_peak_times_summer)
    axs[1].tick_params(axis='x', rotation=90)
    # axs[1].set_xlim([pd.Timestamp('2024-03-01'), pd.Timestamp('2024-10-31')])

    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].set_title(f"{subject_name} - Summer Temperature with Peaks")
    axs[1].legend()
    axs[1].grid()
    
    # Save the overall plot
    overall_plot_filename = f"{subject_name}_temperature_with_peaks.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, overall_plot_filename))
    plt.close()




# def plot_peaks_and_mean_temperature(temperature_df, save_path):
#     # Extract the subject name from the subject_id column
#     subject_id = temperature_df['subject_id'].iloc[0]
#     subject_name = f"Subject_{subject_id}"
    
#     # Ensure save_path exists and create a folder for the subject's peak values
#     save_folder = os.path.join(save_path, f"{subject_name}_peak_values")
#     os.makedirs(save_folder, exist_ok=True)
    
#     # Step 0: Detect peaks on the entire dataset
#     sensor_values = temperature_df['sensor_status'].values
#     peaks, _ = find_peaks(sensor_values, prominence=1.5)
#     temperature_df['peaks'] = 0
#     temperature_df.loc[peaks, 'peaks'] = 1  # Mark peaks as '1' in the 'peaks' column
        
#     # Step 1: Divide data into winter (November-February) and summer (March-October)
#     temperature_df['month'] = temperature_df['ts_datetime'].dt.month
#     winter_mask = temperature_df['month'].isin([11, 12, 1, 2])
#     summer_mask = temperature_df['month'].isin([3, 4, 5, 6, 7, 8, 9, 10])
    
#     # Step 2: Calculate average temperature and standard deviation for winter and summer
#     average_temperature_winter = temperature_df[winter_mask]['sensor_status'].mean()
#     std_deviation_winter = temperature_df[winter_mask]['sensor_status'].std()
    
#     average_temperature_summer = temperature_df[summer_mask]['sensor_status'].mean()
#     std_deviation_summer = temperature_df[summer_mask]['sensor_status'].std()

#     # Step 3: Create subplots for winter and summer
#     fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=False)

#     # Winter subplot (November to February)
#     winter_df = temperature_df[winter_mask]
#     axs[0].scatter(winter_df[winter_df['peaks'] == 1]['ts_datetime'],
#                    winter_df[winter_df['peaks'] == 1]['sensor_status'], 
#                    color='red', label='Peak Temperatures')
    
#     # Draw horizontal lines for winter mean and standard deviation
#     axs[0].axhline(y=average_temperature_winter, color='black', linestyle='-', 
#                    label=f'Winter Avg: {average_temperature_winter:.2f} °C')
#     axs[0].axhline(y=average_temperature_winter + std_deviation_winter, color='black', linestyle='--', 
#                    label=f'Winter Std Dev: {std_deviation_winter:.2f} °C')
#     axs[0].axhline(y=average_temperature_winter - std_deviation_winter, color='black', linestyle='--')

#     # Set x-ticks only on every 5th peak to avoid overcrowding
#     peak_times_winter = winter_df[winter_df['peaks'] == 1]['ts_datetime']
#     selected_peak_times_winter = peak_times_winter[::5]  # Select every 5th peak
#     axs[0].set_xticks(selected_peak_times_winter)
#     axs[0].tick_params(axis='x', rotation=90)
#     axs[0].set_xlim([pd.Timestamp('2023-11-01'), pd.Timestamp('2024-02-28')])
    
#     axs[0].set_ylabel('Temperature (°C)')
#     axs[0].set_title(f"{subject_name} - Winter Temperature with Peaks")
#     axs[0].legend()
#     axs[0].grid()

#     # Summer subplot (March to October)
#     summer_df = temperature_df[summer_mask]
#     axs[1].scatter(summer_df[summer_df['peaks'] == 1]['ts_datetime'],
#                    summer_df[summer_df['peaks'] == 1]['sensor_status'], 
#                    color='red', label='Peak Temperatures')

#     # Draw horizontal lines for summer mean and standard deviation
#     axs[1].axhline(y=average_temperature_summer, color='black', linestyle='-', 
#                    label=f'Summer Avg: {average_temperature_summer:.2f} °C')
#     axs[1].axhline(y=average_temperature_summer + std_deviation_summer, color='black', linestyle='--', 
#                    label=f'Summer Std Dev: {std_deviation_summer:.2f} °C')
#     axs[1].axhline(y=average_temperature_summer - std_deviation_summer, color='black', linestyle='--')

#     # Set x-ticks only on every 5th peak to avoid overcrowding
#     peak_times_summer = summer_df[summer_df['peaks'] == 1]['ts_datetime']
#     selected_peak_times_summer = peak_times_summer[::5]  # Select every 5th peak
#     axs[1].set_xticks(selected_peak_times_summer)
#     axs[1].tick_params(axis='x', rotation=90)
#     axs[1].set_xlim([pd.Timestamp('2024-03-01'), pd.Timestamp('2024-10-31')])

#     axs[1].set_xlabel('Date')
#     axs[1].set_ylabel('Temperature (°C)')
#     axs[1].set_title(f"{subject_name} - Summer Temperature with Peaks")
#     axs[1].legend()
#     axs[1].grid()
    
#     # Save the overall plot
#     overall_plot_filename = f"{subject_name}_temperature_with_peaks.png"
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_folder, overall_plot_filename))
#     plt.close()


def plot_device_usage_heatmap(monthly_usage, subject, save_path):
    monthly_df = pd.DataFrame(monthly_usage).T.sort_index()
    monthly_df.index = monthly_df.index.strftime('%Y-%m')  # Format index as year-month
    
    # Create subject-specific folder if it doesn't exist
    subject_folder = os.path.join(save_path, f"{subject}_device_usage")
    os.makedirs(subject_folder, exist_ok=True)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(monthly_df, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=0.5)

    # Title and labels
    plt.title(f'Percentage of times a device used during temperature peaks in kitchen- {subject}')
    plt.xlabel("Devices")
    plt.ylabel("Month")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save plot to subject folder
    save_file = os.path.join(subject_folder, f"{subject}_device_usage_heatmap.png")
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()  # Close plot to free memory

    print(f"Heatmap saved to {save_file}")



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
        
        # Set the x-axis to show the time with more ticks if needed
        if not concated_df.empty:
            time_range = concated_df['ts_on'].min(), concated_df['ts_off'].max()
            plt.xlim(time_range)
            
            # Dynamically set x-ticks
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # Show ticks every 10 minutes
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='Europe/Rome'))
            plt.xticks(rotation=45)
        else:
            # Handle case with no data
            plt.xlim(pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(hours=1))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='Europe/Rome'))
        
        # Labels and title
        plt.xlabel('Time of Day (Europe/Rome)')
        plt.ylabel('Devices Used in Cooking')
        plt.title(f'Device Usage in Kitchen on {date_val} for {subject}')
        plt.grid()
        
        # Save the plot with date name
        plt.tight_layout()
        plt.savefig(os.path.join(subject_folder, f'{date_val}.png'))  # Save as PNG
        plt.close()  # Close the figure to free up memory



# def compute_daily_temperature(temp):
#     ### Compute daily average temperature
#     temp.loc[:, 'date'] = temp['ts_datetime'].dt.date
#     daily_avg = temp.groupby('date')['sensor_status'].median().reset_index()
#     daily_avg.rename(columns={'sensor_status': 'daily_avg_temp'}, inplace=True)
#     temp = pd.merge(temp, daily_avg, on='date', how='left')
#     return temp


def compute_daily_temperature(temp):
    # Extract date from 'ts_datetime'
    temp.loc[:, 'date'] = temp['ts_datetime'].dt.date
    
    # Compute daily median and standard deviation for 'sensor_status'
    daily_stats = temp.groupby('date')['sensor_status'].agg(
        daily_avg_temp='median',  # Median renamed to daily_avg_temp
        daily_avg_std='std'       # Standard deviation renamed to daily_avg_std
    ).reset_index()
    
    # Merge the computed stats back into the original DataFrame
    temp = pd.merge(temp, daily_stats, on='date', how='left')
    
    return temp

def remove_peaks_below_daily_avg(temperature_df, peaks):
    filtered_peaks = [
        peak for peak in peaks
        if temperature_df.loc[peak, 'sensor_status'] > (
            temperature_df.loc[peak, 'daily_avg_temp'] + temperature_df.loc[peak, 'daily_avg_std']
        )
    ]
    
    return filtered_peaks


# def remove_peaks_below_daily_avg(temperature_df, peaks):
#     filtered_peaks = [
#         peak for peak in peaks
#         if temperature_df.loc[peak, 'sensor_status'] > temperature_df.loc[peak, 'daily_avg_temp'] +1
#     ]
    
#     return filtered_peaks


# def plot_daily_activity(date_peak_times, date_cooking_devices, subject, subject_kitchen_devices, save_path):
#     # Create a folder for the subject
#     import matplotlib.dates as mdates

#     subject_folder = os.path.join(save_path, subject)
#     os.makedirs(subject_folder, exist_ok=True)  # Create the directory if it doesn't exist

#     # Iterate through each date in the dictionaries
#     dates_key = list(date_peak_times.keys())
#     peaks_df = list(date_peak_times.values())
#     devices_df = list(date_cooking_devices.values())
    
#     # Iterate through each date index
#     for i in range(len(dates_key)):
#         date_val = dates_key[i]
#         peak_df = peaks_df[i]
#         device_df = devices_df[i]
        
#         # Concatenate the dataframes
#         concated_df = pd.concat([peak_df, device_df])
        
#         # Create a plot
#         plt.figure(figsize=(10, 6))
        
#         # Plot for each device in subject_kitchen_devices
#         for device in subject_kitchen_devices:
#             device_data = concated_df[concated_df['sensor_id'] == device]
            
#             if not device_data.empty:
#                 for idx, row in device_data.iterrows():
#                     plt.plot([row['ts_on'], row['ts_off']], [device, device], color='blue', linewidth=10)
#             else:
#                 # Get the minimum ts_on for the empty device
#                 min_ts_on = concated_df['ts_on'].min() if not concated_df.empty else pd.Timestamp.now()
#                 plt.plot([min_ts_on, min_ts_on], [device, device], color='blue', linewidth=10)
        
#         # Set y-ticks to include all kitchen devices
#         plt.yticks(subject_kitchen_devices)
        
#         # Set the x-axis to show the time
#         plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='Europe/Rome'))
#         plt.xticks(rotation=45)
        
#         # Labels and title
#         plt.xlabel('Time of Day (Europe/Rome)')
#         plt.ylabel('Devices Used in Cooking')
#         plt.title(f'Device Usage in Kitchen on {date_val} for {subject}')
#         plt.grid()
        
#         # Save the plot with date name
#         plt.tight_layout()
#         plt.savefig(os.path.join(subject_folder, f'{date_val}.png'))  # Save as PNG
#         plt.close()  # Close the figure to free up memory


def plot_temperature_peak_stats(monthly_peak_counts, monthly_avg_durations, monthly_days_with_peaks, save_path, subject):
    # Convert dictionaries to DataFrames for easier plotting
    months = list(monthly_peak_counts.keys())
    
    peak_counts = [monthly_peak_counts[month] for month in months]
    avg_durations = [monthly_avg_durations[month].total_seconds() / 60 for month in months]  # Convert to minutes
    days_with_peaks = [monthly_days_with_peaks[month] for month in months]
    
    # Create a DataFrame for better handling
    stats_df = pd.DataFrame({
        'Month': months,
        'Peak Counts': peak_counts,
        'Average Duration (minutes)': avg_durations,
        'Days with Peaks': days_with_peaks
    })
    
    # Set Month as index for better plotting
    stats_df.set_index('Month', inplace=True)
    
    # Create a folder for saving figures in the specified path if it doesn't exist
    folder_name = os.path.join(save_path, f"{subject}_peak_stats")
    os.makedirs(folder_name, exist_ok=True)
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot Peak Counts
    axs[0].bar(stats_df.index, stats_df['Peak Counts'], color='lightblue')
    axs[0].set_title(f'Temperature Peak Counts per Month - {subject}', fontsize=14)
    axs[0].set_ylabel('Number of Peaks', fontsize=12)
    axs[0].tick_params(axis='y', labelsize=10)

    # Adding counts on top of the bars
    for i, count in enumerate(peak_counts):
        axs[0].text(i, count, str(count), ha='center', va='bottom', fontsize=10)

    # Plot Average Duration
    axs[1].bar(stats_df.index, stats_df['Average Duration (minutes)'], color='lightgreen')
    axs[1].set_title(f'Average Duration of Temperature Peaks (in Minutes) - {subject}', fontsize=14)
    axs[1].set_ylabel('Average Duration (minutes)', fontsize=12)
    axs[1].tick_params(axis='y', labelsize=10)

    # Adding average duration on top of the bars
    for i, duration in enumerate(avg_durations):
        axs[1].text(i, duration, f"{duration:.1f}", ha='center', va='bottom', fontsize=10)

    # Plot Days with Peaks
    axs[2].bar(stats_df.index, stats_df['Days with Peaks'], color='salmon')
    axs[2].set_title(f'Days with Temperature Peaks per Month - {subject}', fontsize=14)
    axs[2].set_ylabel('Number of Days', fontsize=12)
    axs[2].tick_params(axis='y', labelsize=10)

    # Adding days with peaks on top of the bars
    for i, days in enumerate(days_with_peaks):
        axs[2].text(i, days, str(days), ha='center', va='bottom', fontsize=10)

    # Set x-axis label
    axs[2].set_xlabel('Month-Year', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    
    plt.tight_layout()

    # Save the figure
    save_filepath = os.path.join(folder_name, f"{subject}_temperature_peak_stats.png")
    plt.savefig(save_filepath)
    
    # Close the figure to free up memory
    plt.close()




def calculate_monthly_peak_stats(monthly_temp_peaks):
    # Initialize dictionaries to store monthly results
    monthly_peak_counts = {}
    monthly_avg_durations = {}
    monthly_days_with_peaks = {}

    # Extract list of month timestamps (keys) to use with index-based iteration
    month_timestamps = list(monthly_temp_peaks.keys())
    
    # Loop through each month using index-based iteration
    for i in range(len(month_timestamps)):
        month_timestamp = month_timestamps[i]
        daily_data = monthly_temp_peaks[month_timestamp]
        
        # Extract month-year in string format (e.g., '2024-06')
        month_year = month_timestamp.strftime('%Y-%m')
        
        # Initialize accumulators for count and duration
        total_duration = pd.Timedelta(0)  # Accumulate as timedelta
        total_peaks = 0  # Accumulate peak count
        unique_days = set()  # To track unique days with temperature peaks
        
        # Process each day's data in the current month
        days = list(daily_data.keys())
        for j in range(len(days)):
            day = days[j]
            df = daily_data[day]
            
            daily_peak_count = len(df)
            total_peaks += daily_peak_count

            # Add the day to unique_days set if any peaks are recorded
            if daily_peak_count > 0:
                unique_days.add(day)

            # Calculate the duration for each peak on this day
            for k in range(len(df)):
                row = df.iloc[k]
                duration = row['ts_off'] - row['ts_on']
                total_duration += duration

        # Store results for the month
        if total_peaks > 0:
            average_duration = total_duration / total_peaks
        else:
            average_duration = pd.Timedelta(0)  # No peaks observed, so average is 0
        
        # Save results in dictionaries
        monthly_peak_counts[month_year] = total_peaks
        monthly_avg_durations[month_year] = average_duration
        monthly_days_with_peaks[month_year] = len(unique_days)  # Count of unique days

    return monthly_peak_counts, monthly_avg_durations, monthly_days_with_peaks



def calculate_device_usage_percentage(usage_counts, total_temp_peak_count):
    # Initialize a dictionary to store percentage usage
    percentage_usage = {}

    # Calculate the percentage for each device
    for device, count in usage_counts.items():
        if total_temp_peak_count > 0:  # Avoid division by zero
            percentage_usage[device] = (count / total_temp_peak_count) * 100
        else:
            percentage_usage[device] = 0.0  # If no temperature peaks, set to 0

    return percentage_usage

def calculate_device_usage_count(date_peak_times, date_cooking_devices, subject_kitchen_devices2):
    # Initialize a dictionary to store usage counts
    usage_counts = {device: 0 for device in subject_kitchen_devices2}

    # Iterate through the date_peak_times dictionary
    dates = list(date_peak_times.keys())
    temp_peak_count = 0
    for i in range(len(dates)):
        date = dates[i]
        peak_df = date_peak_times[date]
        temp_peak_count = temp_peak_count + len(peak_df)
        
        # Get the cooking devices for the same date
        cooking_df = date_cooking_devices.get(date)

        if cooking_df is not None and not cooking_df.empty:
            # Iterate through each temperature peak activation
            for _, peak_row in peak_df.iterrows():
                peak_start = peak_row['ts_on']
                peak_end = peak_row['ts_off']
                
                # Check for each device if it was used at least once during the temperature peak
                for device in subject_kitchen_devices2:
                    # print(device)
                    if device in cooking_df['sensor_id'].values:
                        device_data = cooking_df[cooking_df['sensor_id'] == device]
                        
                        # Check for any overlap with the temperature peak period
                        for _, device_row in device_data.iterrows():
                            if (device_row['ts_on'] < peak_end) and (device_row['ts_off'] > peak_start):
                                usage_counts[device] += 1
                                break  # No need to check more if we found an activation
    percentage_usage = calculate_device_usage_percentage(usage_counts, temp_peak_count)
    return percentage_usage

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