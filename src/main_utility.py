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
from datetime import datetime, timedelta

import seaborn as sns
from natsort import natsorted
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from subject_kitchen_device_mapping import *

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


def analyze_seasonal_peaks_with_duration_v2(df_stove_temperature , prom = 1.5, theta = 40):
    """
    Analyzes seasonal peaks in stove temperature data and computes duration information for each peak.
    
    Parameters:
        df_stove_temperature (pd.DataFrame): A DataFrame containing stove temperature data with columns:
            - 'ts_datetime': Timestamps of temperature readings (datetime format).
            - 'sensor_status': Temperature readings.
        prom (float, optional): Prominence threshold for peak detection. Defaults to 1.5.
        theta (float, optional): Slope-based threshold for determining the start of a peak. Defaults to 40.
    
    Returns:
        List[dict]: A list of dictionaries, each containing the following information for detected peaks:
            - "peak_index" (int): Index of the peak in the DataFrame.
            - "peak_temperature" (float): Temperature value at the peak.
            - "duration_minutes" (float): Duration of the peak in minutes.
            - "left_time" (datetime): Start time of the peak.
            - "right_time" (datetime): End time of the peak.
            - "peak_time" (datetime): Timestamp of the peak.
    
    Notes:
        - Peaks are detected using the `find_peaks` function based on the specified prominence.
        - The duration of each peak is calculated from the time difference between the start (left_time)
          and an estimated end (right_time) based on a slope-based method.
        - The function temporarily adds columns for intermediate calculations, which are dropped before returning.
    """
    df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
    df_stove_temperature['daily_avg_temp'] = df_stove_temperature.groupby('date')['sensor_status'].transform('median')
    df_stove_temperature['daily_avg_std'] = df_stove_temperature.groupby('date')['sensor_status'].transform('std')

    # _, properties = find_peaks(df_stove_temperature['sensor_status'].values, prominence = (None,None))
    # prom = np.mean(properties['prominences'])
    # print('prominence',prom)

    # Step 1: Detect peaks ()
    sensor_values = df_stove_temperature['sensor_status'].values # Get all the temperature readings in an array
    peaks, _ = find_peaks(sensor_values, prominence = prom)
    
    # sensor_values2 = gaussian_filter1d(df_stove_temperature['sensor_status'].values, sigma = 7, mode = 'nearest')#2*np.std(sensor_values))
    
    # _, properties = find_peaks(sensor_values2, prominence = (None,None))
    # prom = np.mean(properties['prominences'])
    # print('prominence',prom)
    
    # peaks2, _ = find_peaks(sensor_values2, prominence = 1.5) # Find the peaks in temperature readings
    
    # # # Plot original signal and smoothed signal with peaks
    # fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # # Subplot 1: Original signal with detected peaks
    # axes[0].plot(
    #     df_stove_temperature['ts_datetime'], 
    #     sensor_values, 
    #     label="Original Signal", 
    #     color="blue", 
    #     linewidth=1
    # )
    # axes[0].scatter(
    #     df_stove_temperature.iloc[peaks]['ts_datetime'],  # Map indices to timestamps
    #     sensor_values[peaks], 
    #     color="red", 
    #     label="Peaks", 
    #     zorder=5
    # )
    
    # # axes[0].plot(
    # #     df_stove_temperature['ts_datetime'], 
    # #     df_stove_temperature['daily_avg_temp'], 
    # #     label="Median Temperture", 
    # #     color="black", 
    # #     linewidth=1
    # # )
    
    # # axes[0].plot(
    # #     df_stove_temperature['ts_datetime'], 
    # #     df_stove_temperature['daily_avg_temp']+2.12, 
    # #     label="Median Temperture + Delta", 
    # #     color="red", 
    # #     linewidth=1
    # # )

    # axes[0].set_title("Original Signal with Peaks")
    # axes[0].set_ylabel("Sensor Status")
    # axes[0].legend()
    # axes[0].grid()
    
    # # Subplot 2: Smoothed signal with detected peaks
    # axes[1].plot(
    #     df_stove_temperature['ts_datetime'], 
    #     sensor_values2, 
    #     label="Smoothed Signal", 
    #     color="orange", 
    #     linewidth=1.5
    # )
    # axes[1].scatter(
    #     df_stove_temperature.iloc[peaks2]['ts_datetime'],  # Map indices to timestamps
    #     sensor_values2[peaks2], 
    #     color="red", 
    #     label="Peaks", 
    #     zorder=5
    # )
    
    
    # axes[1].set_title("Smoothed Signal with Peaks")
    # axes[1].set_xlabel("Timestamps")
    # axes[1].set_ylabel("Sensor Status")
    # axes[1].legend()
    # axes[1].grid()
    
    # # Set the x-axis to show full datetime (date + time)
    # for ax in axes:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Full datetime
    #     ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Adjust tick frequency
    
    # # Rotate x-axis labels for better readability
    # plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # # Adjust layout and display the plots
    # plt.tight_layout()
    # plt.show()
    
    df_stove_temperature['peaks'] = 0
    df_stove_temperature.loc[peaks, 'peaks'] = 1  # Mark peaks as 1
    # df_stove_temperature['smoothed'] = sensor_values2
    
    # df_stove_temperature['smoothed_peaks'] = 0
    # df_stove_temperature.loc[peaks2, 'smoothed_peaks'] = 1  # Mark peaks as 1
    
    # df_stove_temperature['smoothed_peaks'] = 0
    # df_stove_temperature.loc[peaks, 'peaks'] = 1  # Mark peaks as 1
    
    df_stove_temperature.drop(columns=['date', 'daily_avg_temp', 'daily_avg_std'], inplace=True)

    temp_info = []
    df = df_stove_temperature.copy()
    for indx in df[df['peaks'] == 1].index:#['index']:  # Use original index from reset
        # print(indx)
        # break
        value = df_stove_temperature.loc[indx, 'sensor_status']
        left = slope_based_peak_start_v2(indx, df_stove_temperature.copy(), theta)
        peak_time = df_stove_temperature['ts_datetime'].iloc[indx]
        peak_temperature = df_stove_temperature['sensor_status'].iloc[indx]
        
        if left == None:  
            left_time = peak_time - timedelta(minutes=15)
            right_time = peak_time + timedelta(minutes=15)
        else:  
            left_time = df_stove_temperature['ts_datetime'].iloc[left]
            right_time = peak_time + (peak_time - left_time) 
        
        duration_minutes = (right_time - left_time).total_seconds() / 60
        temp_info.append({
            "peak_index": indx,
            "peak_temperature":peak_temperature,
            "duration_minutes": duration_minutes,
            "left_time": left_time,
            "right_time": right_time,
            "peak_time": peak_time,
        })
        
    return temp_info



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
    """
    Plots the duration peaks distribution for a given subject across different seasons and sigma levels.
    
    Parameters:
        result (dict): A dictionary containing duration percentages data for each season.
                       Expected format:
                       {
                           'winter': {
                               'duration_percentages': {
                                   'within_1_sigma': {'< 5': value, '5-10': value, ...},
                                   'between_1_and_2_sigma': {'< 5': value, '5-10': value, ...},
                                   ...
                               }
                           },
                           'summer': {
                               'duration_percentages': {
                                   ...
                               }
                           }
                       }
        save_dir (str): Path to the directory where the plot will be saved.
        subject_id (str): Identifier for the subject whose data is being visualized.
    
    Returns:
        None: The function saves the plot as a PNG file in the specified directory.
    
    Notes:
        - The plot contains two subplots, one for each season ('winter' and 'summer').
        - Each subplot displays percentages for duration categories grouped by sigma levels.
        - Duration categories are color-coded, and their values are annotated on the bars.
        - A legend is included below the x-axis for duration categories.
        - The plot is saved with the filename format `<subject_id>_duration_peaks.png`.
    
    Example:
        plot_duration_peaks(result, "output_dir", "subject_01")
    """
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
    """
    Finds peaks in sensor data and calculates their start, end, and duration.
    
    Parameters:
        daily_data (pd.DataFrame): The daily sensor data. Expected columns:
            - 'sensor_status' (numeric): The sensor's activation levels.
            - 'ts_datetime' (datetime): The timestamp of the sensor data.
            - 'subject_id' (str): The subject identifier.
            - 'sensor_id' (str): The sensor identifier.
        peaks (list): Indices of the detected peaks in the data.
        day (str or datetime, optional): The day associated with the data for contextual labeling.
        max_time_diff (int): Maximum allowable time difference (in the same units as `ts_datetime`)
                             between consecutive entries to consider them part of the same peak.
    
    Returns:
        list[dict]: A list of dictionaries, each containing information about a detected peak:
            - 'subject_id': The subject identifier.
            - 'sensor_id': The sensor identifier.
            - 'day': The associated day for the data.
            - 'ts_on': Start timestamp of the peak.
            - 'ts_off': End timestamp of the peak.
            - 'duration': Duration of the peak (calculated as ts_off - ts_on).
    
    Notes:
        - The function iteratively searches for the start and end of each peak by checking
          sensor status levels and timestamps within the `max_time_diff` threshold.
        - Assumes `daily_data` is sorted by `ts_datetime`.
    
    Example:
        peaks = [10, 50, 100]
        daily_data = pd.DataFrame({
            'sensor_status': [1, 2, 3, ..., 0],
            'ts_datetime': [datetime1, datetime2, ..., datetimeN],
            'subject_id': [...],
            'sensor_id': [...],
        })
        peak_info = find_peaks_with_durations(daily_data, peaks)
    """
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
    Processes sensor data by organizing 'on' and 'off' timestamps, calculating durations,
    and converting timestamps to a specific timezone.

    This function performs the following tasks:
    1. If the first sensor status is 'off', it removes the first row of the dataframe.
    2. Identifies rows where the sensor status is 'on' or 'off'.
    3. Splits the dataframe into two parts: one for 'on' events and one for 'off' events.
    4. Renames the 'ts' column to 'ts_on' for the 'on' events and 'ts_off' for the 'off' events.
    5. Resets the index of both the 'on' and 'off' dataframes.
    6. Concatenates the 'on' and 'off' events based on their index, aligning each 'on' event with its corresponding 'off' event.
    7. Calculates the duration between 'on' and 'off' timestamps in milliseconds.
    8. Converts the 'on' and 'off' timestamps from milliseconds to datetime format, localizes them to UTC, and converts them to the 'Europe/Rome' timezone.
    9. Calculates the duration between 'on' and 'off' times in the datetime format.
    10. Converts the 'on' and 'off' timestamps back into milliseconds.
    11. Returns the processed dataframe containing relevant columns for further analysis.

    Args:
        df (pd.DataFrame): The input dataframe containing sensor data with columns such as
                            'sensor_id', 'sensor_status', 'ts', and 'subject_id'.

    Returns:
        pd.DataFrame: A dataframe with columns:
                        - 'sensor_id': The identifier of the sensor.
                        - 'subject_id': The identifier of the subject.
                        - 'sensor_status': The status of the sensor ('on' or 'off').
                        - 'ts_on_ms': The start timestamp of the 'on' event in milliseconds.
                        - 'ts_off_ms': The end timestamp of the 'off' event in milliseconds.
                        - 'ts_on': The start timestamp of the 'on' event in datetime format (Europe/Rome timezone).
                        - 'ts_off': The end timestamp of the 'off' event in datetime format (Europe/Rome timezone).
                        - 'duration_ms': The duration of the 'on' event in milliseconds.
                        - 'duration_datetime': The duration of the 'on' event in datetime format.
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
    """
    Retrieves the threshold value for a given subject and file from a dictionary.
    
    This function looks up the threshold value associated with a specific subject
    and file in a dictionary that stores subject-specific files and their corresponding
    threshold values. If the subject and file exist in the dictionary, the function
    returns the associated threshold value. Otherwise, it returns `None`.
    
    Args:
        subject_dict (dict): A dictionary where keys are subject identifiers and 
                              values are dictionaries mapping file names to threshold values.
        subject (str): The subject identifier for which the threshold value is being requested.
        file (str): The file name for which the threshold value is being requested.
    
    Returns:
        The threshold value (any type) associated with the subject and file, or `None` if 
        the subject or file is not found in the dictionary.
    """
    if subject in subject_dict:
        subject_files = subject_dict[subject]
        if file in subject_files:
            return subject_files[file]
    return None

def convert_real_valued_to_on_off(threshold_value, df):
    """
    Converts real-valued sensor readings into 'on' and 'off' states based on a threshold value.

    This function processes a dataframe containing real-valued sensor readings and identifies 
    'on' and 'off' events based on whether the readings exceed or fall below the provided threshold. 
    It constructs a new dataframe containing the timestamps and metadata for these events, 
    including their conversion to a specified timezone.

    Args:
        threshold_value (float): The threshold value used to classify sensor readings as 'on' or 'off'.
        df (pd.DataFrame): The input dataframe containing sensor data with the following columns:
            - 'ts': The timestamp of the reading in milliseconds.
            - 'sensor_status': The real-valued sensor reading.
            - 'sensor_id': The identifier of the sensor.
            - 'subject_id': The identifier of the subject.

    Returns:
        pd.DataFrame: A dataframe containing the detected 'on' and 'off' events with the following columns:
            - 'index': The index of the event in the original dataframe.
            - 'sensor_id': The identifier of the sensor.
            - 'sensor_status': The status of the sensor ('on' or 'off').
            - 'ts': The timestamp of the event in milliseconds.
            - 'subject_id': The identifier of the subject.
            - 'ts_datetime': The timestamp of the event in datetime format, localized to UTC and converted to the 'Europe/Rome' timezone.

    Example:
        If the sensor readings fluctuate around a threshold value, this function will generate 
        rows for each transition from 'off' to 'on' and 'on' to 'off', with corresponding timestamps.

    Notes:
        - The function assumes that the input dataframe is sorted by the 'ts' column in ascending order.
        - Timestamps are converted from milliseconds to datetime format, localized to UTC, and converted 
          to the 'Europe/Rome' timezone.
    """    
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
    """
    Removes consecutive duplicate 'on' or 'off' sensor states and adjusts the dataframe.

    This function processes a dataframe to clean consecutive occurrences of the same 
    sensor status ('on' or 'off'). It drops duplicate entries, keeping only the necessary 
    transitions between 'on' and 'off'. Additionally, it identifies whether the sensor is 
    an environmental sensor based on the presence of both 'on' and 'off' states.

    Args:
        df_cleaned_1 (pd.DataFrame): The input dataframe containing the following columns:
            - 'sensor_status': The state of the sensor ('on' or 'off').
            - Other metadata columns related to sensor readings.

    Returns:
        tuple:
            - pd.DataFrame: The cleaned dataframe with consecutive duplicate sensor statuses removed.
            - bool or int: 
                - `True` if the sensor is identified as an environmental sensor (contains both 'on' and 'off').
                - `False` if the sensor is not an environmental sensor.
                - `-999` if the input dataframe does not contain both 'on' and 'off' states.

    Notes:
        - Consecutive 'on' states will result in the earlier occurrences being dropped.
        - Consecutive 'off' states will result in the later occurrences being dropped.
        - The first row will be removed if it starts with 'off'.
        - The last row will be removed if it ends with 'on'.
        - The index of the dataframe is reset after rows are dropped.

    Example:
        Input:
            | sensor_status |
            |---------------|
            | on            |
            | on            |
            | off           |
            | off           |
            | on            |
        
        Output:
            | sensor_status |
            |---------------|
            | on            |
            | off           |

    """    
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
    """
    Reads and processes sensor data for multiple subjects stored in a hierarchical folder structure.

    The function iterates through a root directory containing subfolders for individual subjects. Each subject folder
    contains an "environmentals" folder with additional subfolders representing different sensor types. Within these
    subfolders, the function identifies and processes merged CSV files based on the type of sensor. Data cleaning,
    conversion, and processing are applied based on the sensor's nature (environmental or analog).

    Args:
        data_path (str): Path to the root directory containing sensor data for all subjects.

    Returns:
        dict: A nested dictionary with the following structure:
            {
                subject_name: {
                    sensor_folder_name: [raw_dataframe, processed_dataframe],
                    ...
                },
                ...
            }
            - `subject_name` (str): Name of the subject folder.
            - `sensor_folder_name` (str): Name of the folder representing a specific sensor.
            - `raw_dataframe` (pd.DataFrame): The raw DataFrame after initial cleaning and preprocessing.
            - `processed_dataframe` (pd.DataFrame): The processed DataFrame after applying sensor-specific logic.
              If the sensor is analog without further processing, this will be an empty DataFrame.

    Notes:
        - Filters out rows with invalid or unknown sensor statuses ('unavailable', 'unknown').
        - Converts timestamps in the 'local_time' column to timezone-aware datetime objects in the 'Europe/Rome' timezone.
        - Detects and processes continuous 'on/off' periods using the `remove_continuous_on_off_v2` function.
        - Handles environmental sensors and analog sensors separately:
            - Environmental sensors are processed using `process_sensor_data`.
            - Analog sensors are either left as-is or converted to binary `on/off` values based on a threshold.
        - Logs invalid rows with invalid timestamps or empty data after processing.

    Raises:
        ValueError: If the timestamp format in 'local_time' is unrecognized.

    Example:
        >>> data_path = "/path/to/sensor_data"
        >>> subject_data = read_csv_files_v2(data_path)
        >>> print(subject_data["subject_1"]["Shower_Hum_Temp"][0].head())
        # Outputs the first few rows of the raw DataFrame for the Shower_Hum_Temp sensor of subject_1.
    """    
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
    """
    Calculate the minute index of a given timestamp within a 24-hour day.
    
    The function converts a timestamp to a pandas datetime object, extracts the hour and minute components, 
    and computes the minute index as the total number of minutes elapsed since midnight. 
    The minute index is calculated as `hour * 60 + minute`.
    
    Args:
        ts_datetime (str, datetime, or pd.Timestamp): The timestamp for which the minute index is calculated. 
            It can be a string in a datetime-compatible format, a Python `datetime` object, or a pandas `Timestamp`.
    
    Returns:
        int: The minute index, ranging from 0 (midnight) to 1439 (11:59 PM).
    
    Example:
        >>> get_minute_index("2024-12-12 15:30:45")
        930
        >>> get_minute_index(pd.Timestamp("2024-12-12 06:45:00"))
        405
    """
    # Convert to pandas datetime for easier extraction
    dt = pd.to_datetime(ts_datetime)
    # Calculate the minute index: hour * 60 + minute
    minute_index = dt.hour * 60 + dt.minute
    return minute_index

def arrange_data_by_day_numpy(df):
    """
    Process and arrange sensor data by day into a structured numpy format, identifying peaks and their durations.

    This function organizes time-series sensor data into daily numpy arrays, marking peaks and their durations 
    in minute-level granularity. It also calculates additional information like peak-to-minute index mappings 
    and aggregates data for peaks within the daily intervals.

    Args:
        df (pd.DataFrame): 
            A DataFrame containing at least the following columns:
            - 'ts_datetime' (datetime): Timestamp column, localized to 'Europe/Rome'.
            - 'sensor_status' (numeric): Sensor status values for processing peaks and durations.

    Returns:
        tuple: A tuple containing the following elements:
            - `arranged_data_np` (np.ndarray): A 2D array where each row corresponds to minute-level sensor data for a day.
            - `complete_days` (list of datetime.date): List of all dates in the input data range.
            - `arranged_peak_start_end_data_np` (np.ndarray): A 2D array where each row corresponds to minute-level 
              binary activity data (1 for active, 0 for inactive) for a day.
            - `datewise_peak_duration_info_df` (pd.DataFrame): A DataFrame with detailed information about peaks, 
              including timestamps and durations.

    Steps:
        1. Extracts the date range of the data and generates a list of all days (complete_days).
        2. Iterates over each day to:
            - Filter the data for that day.
            - Identify peaks using the `find_peaks` function and their durations using `find_peaks_with_durations`.
            - Map peaks to their corresponding minute indices within the day.
            - Create a minute-by-minute time series for sensor activity.
        3. Fills missing minutes with zeros and creates a binary activity array based on peak start and end times.
        4. Constructs numpy arrays for daily data and peak-based activity.
        5. Collects peak duration and other metadata into a DataFrame.

    Notes:
        - Sensor data is resampled at a minute-level resolution, filling missing data with zeros.
        - Peaks are detected using prominence thresholds and their durations are calculated based on time differences.

    Example:
        >>> arranged_data, days, peak_data, peak_info = arrange_data_by_day_numpy(sensor_df)
        >>> print(arranged_data.shape)
        (number_of_days, 1440)  # 1440 minutes in a day
        >>> print(days)
        [datetime.date(2024, 12, 10), datetime.date(2024, 12, 11), ...]
    """    
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
    """
    Filters rows in a DataFrame based on a specific year, month, and time range. 
    The function ensures that the 'ts_on' and 'ts_off' columns are in datetime format 
    and then applies the filters.

    Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'sensor_id', 'subject_id', 'ts_on', and 'ts_off'.
        start_time (str): The start time of the desired range in "HH:MM:SS" format.
        end_time (str): The end time of the desired range in "HH:MM:SS" format.
        year (int): The year to filter by.
        month (int): The month to filter by (1 for January, 2 for February, etc.).

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the rows that fall within 
        the specified year, month, and time range, with the following columns:
            - 'sensor_id': ID of the sensor.
            - 'subject_id': ID of the subject.
            - 'ts_on': Start timestamp of the activity.
            - 'ts_off': End timestamp of the activity.
    
    Example:
        df = pd.DataFrame({
            'sensor_id': [1, 2, 3],
            'subject_id': ['A', 'B', 'C'],
            'ts_on': ['2024-12-01 08:30:00', '2024-12-01 09:00:00', '2024-12-01 10:15:00'],
            'ts_off': ['2024-12-01 08:45:00', '2024-12-01 09:30:00', '2024-12-01 10:45:00']
        })
        filtered = filter_rows_by_time_and_date(df, '08:00:00', '09:00:00', 2024, 12)
    """    
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
    """
    Processes data from shower and stove sensors for a specific subject, filtering it by 
    time intervals, year, and month, and maps sensor IDs to meaningful labels.

    Parameters:
        data (dict): A nested dictionary structure containing sensor data for different subjects and devices.
                     Each device's data should be a DataFrame with a 'ts_datetime' column.
        SUBJECT_ID (str): Identifier for the subject whose data is being processed.
        specific_devices (list): A list of device keys (e.g., 'Stove_Hum_Temp_temp', 'Shower_Hum_Temp_humidity') to process.
        QUERY_INTERVAL_START (str): Start of the query interval in "HH:MM:SS" format.
        QUERY_INTERVAL_END (str): End of the query interval in "HH:MM:SS" format.
        QUERY_YEAR (int): The year to filter the data.
        QUERY_MONTH (int): The month to filter the data (1 for January, 2 for February, etc.).

    Returns:
        pd.DataFrame: A consolidated DataFrame containing filtered data for the specified time interval, year, and month.
                      The DataFrame includes mapped sensor IDs and may contain the following columns:
                      - 'sensor_id': Mapped sensor identifier.
                      - 'subject_id': Identifier for the subject.
                      - 'ts_on': Start timestamp of the event.
                      - 'ts_off': End timestamp of the event.

    Example:
        stove_shower_df = shower_stove_data(
            data=data_dict,
            SUBJECT_ID="subject_1",
            specific_devices=['Stove_Hum_Temp_temp', 'Shower_Hum_Temp_humidity'],
            QUERY_INTERVAL_START="08:00:00",
            QUERY_INTERVAL_END="10:00:00",
            QUERY_YEAR=2024,
            QUERY_MONTH=12
        )
    """    
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
    """
    Filters and processes device data for non-shower and non-stove devices, within a specified year and month,
    and merges the results into a single DataFrame.
    
    Args:
        device_dict (dict): A dictionary where keys are device names and values are lists containing device metadata.
                            Each device's metadata list contains data with 'ts_on' and 'ts_off' timestamps.
        specific_devices (list): A list of device names to exclude from the filtering process (typically shower and stove devices).
        start_time (str): The start time of the query interval (in string format, e.g., 'HH:MM:SS').
        end_time (str): The end time of the query interval (in string format, e.g., 'HH:MM:SS').
        target_year (int): The year for filtering data.
        target_month (int): The month for filtering data.
    
    Returns:
        pd.DataFrame: A DataFrame containing filtered and merged data from devices that are not in the `specific_devices` list,
                      sorted by 'ts_on'. The DataFrame will include columns such as 'sensor_id', 'subject_id', 'ts_on', and 'ts_off'.
                      If no data is found, an empty DataFrame is returned.
    """
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

def plot_peak_and_signal(kitchen_temperature_data, peaks):
    fig, ax1 = plt.subplots(figsize=(12, 6))  # Create a single Axes object
    
    # Part 1: Plot the raw temperature curve
    ax1.plot(kitchen_temperature_data['ts_datetime'], 
             kitchen_temperature_data['sensor_status'], 
             color='blue', linewidth=2)
    
    # # Highlight the peaks with scatter points and vertical lines
    for peak_idx in peaks:
        pt_time = kitchen_temperature_data['ts_datetime'].iloc[peak_idx]  # Timestamp of the peak
        pt_temp = kitchen_temperature_data['sensor_status'].iloc[peak_idx]  # Temperature value at the peak
        
        # Scatter the peak
        ax1.scatter(pt_time, pt_temp, color='red', s=200, marker='*', label='Temperature Peak')
        ax1.text(pt_time, pt_temp + 1, f"Idx: {peak_idx}", color='black', fontsize=10, ha='center', va='bottom')

        
    #     # Draw a vertical line at the peak time
    #     ax1.axvline(pt_time, color='blue', linestyle='--', linewidth=1.5, label=f"Peak at {pt_time.strftime('%H:%M:%S')}")
    
    # Customize the plot
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Temperature", fontsize=12)
    # ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz='Europe/Rome'))
    ax1.tick_params(axis='x', rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()
    plt.show()


def plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, stop_temp, st , et, pt, pi, pth, fwhm):
    """
    Plots device usage during a temperature peak event.
    
    Description:
    This function generates a plot showing the stove temperature over time, with vertical lines indicating key time points. It also visualizes device usage during the temperature peak, highlighting the devices used and their usage intervals.
    
    Input:
    - subject (str): The subject identifier (e.g., 'subject_1').
    - subject_devices (dict): Dictionary with subject identifier as the key and a list of device names used for cooking.
    - data_subject (dict): Dictionary with subject names as keys and corresponding data.
    - df_stove_temperature (pd.DataFrame): DataFrame containing stove temperature data for subject in question
    - peak_index (int): Index of the temperature peak in the temperature data.
    - backward_time (pd.Timestamp): Start time of the observation window (before the peak).
    - forward_time (pd.Timestamp): End time of the observation window (after the peak).
    - stop_temp (float): Temperature threshold for stopping conditions.
    - st (pd.Timestamp): Start time of the device usage window.
    - et (pd.Timestamp): End time of the device usage window.
    - pt (pd.Timestamp): Peak time of the temperature event.
    - pth (str): File path where the plot should be saved.
    - delta_T (float, optional, default=5.0): Temperature difference for visual reference.
    - season (str, optional, default='Season_X'): Season during the event.
    - category_name (str, optional, default='Category_A'): Category of the event.
    - pi (str, optional, default='01'): Identifier for the individual or system.
    - color_mapping (dict, optional): Mapping of device names to colors in the plot.
    
    Output:
    - None: The function saves the plot as a PNG file to the specified path (`pth`).
    """
    temperature_data_in_peak = df_stove_temperature[(df_stove_temperature['ts_datetime'] >= (backward_time - pd.Timedelta(hours=2))) & 
                                                    (df_stove_temperature['ts_datetime'] <= (forward_time + pd.Timedelta(hours=2)))]
                            
    
    ## average temperature of the day
    peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
    avg_temp = df_stove_temperature.iloc[peak_index]['daily_avg_temperature']
    median_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
    std_temp = df_stove_temperature.iloc[peak_index]['daily_std_temperature']
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Part 1: Plot the raw temperature curve in the upper subplot
    ax1.plot(temperature_data_in_peak['ts_datetime'], 
              temperature_data_in_peak['sensor_status'], 
              color='red', linewidth=2)
    
    ax1.scatter(temperature_data_in_peak['ts_datetime'], 
              temperature_data_in_peak['sensor_status'], 
              color='blue')
    
    ax1.text(pt, peak_temp - 0.5, f"{fwhm:.0f} min", color=peak_color, fontsize=10, ha='center')
    

    # ax1.axvline(pt,linestyle=':')
    ax1.axhline(y=avg_temp, color='gray', linestyle='-', linewidth=1.5, label=f'Mean ({avg_temp:.2f})')
    ax1.axhline(y=median_temp, color='magenta', linestyle='-', linewidth=1.5, label=f'Median ({median_temp:.2f})')
    # ax1.axhline(y=std_temp, color='magenta', linestyle='-', linewidth=1.5, label='Std.')
    ax1.axhline(y=median_temp + delta_T, color='blue', linestyle='-', linewidth=1.5, label=f'Median ({median_temp:.2f}) + Delta ({delta_T:.2f})')
    # ax1.axhline(y=stop_temp, color='green', linestyle='-', linewidth=1.5, label=f'Peak ({peak_temp:.2f}) - Delta/2 ({delta_T/2})')

    ax1.axvline(backward_time, color='blue', linestyle='-', linewidth=1.5, label=f'Observation Start ({backward_time.strftime("%H:%M:%S")})')
    ax1.axvline(forward_time, color='blue', linestyle='-', linewidth=1.5, label = f'Observation End ({forward_time.strftime("%H:%M:%S")}')
    
    # Annotate st and et times near the vertical lines
    ax1.text(backward_time, ax1.get_ylim()[1], backward_time.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    ax1.text(forward_time, ax1.get_ylim()[1], forward_time.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    ax1.text(pt, ax1.get_ylim()[1], pt.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
        
    ax1.axvline(st, color='black', linestyle='--', linewidth=1.5)
    ax1.axvline(et, color='black', linestyle='--', linewidth=1.5)
    
    # Annotate st and et times near the vertical lines
    ax1.text(st, ax1.get_ylim()[1], st.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    ax1.text(et, ax1.get_ylim()[1], et.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    
    # ax1.set_title(subject+ "("+ season + ")", fontsize=14)
    ax1.set_ylabel("Temperature", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    subject_kitchen_devices = sorted(subject_devices[subject])
    count_var = 0
    for key in natsorted(data_subject.keys()):
    # for key in natsorted(subject_kitchen_devices):
        if "Stove" not in key and "Shower" not in key:
            temp = data_subject[key][1].copy()

            # device_data = temp[(temp['ts_on'] <= et) & (temp['ts_off'] >= st)].copy()
            device_data = temp[(temp['ts_on'] <= forward_time) & (temp['ts_off'] >= backward_time)].copy()
            # device_data = temp[(temp['ts_on'] <= forward_time+ pd.Timedelta(hours=2)) & (temp['ts_off'] >= backward_time- pd.Timedelta(hours=2))]
            device_data['ts_on'] = device_data['ts_on'].apply(lambda x: max(x, st))
            device_data['ts_off'] = device_data['ts_off'].apply(lambda x: min(x, et))
            if len(device_data) > 0:
                count_var = count_var + 1
                usage_count = len(device_data)
                data_dict[key] = device_data
                for idx, row in device_data.iterrows():
                    ax2.plot(
                        [row['ts_on'], row['ts_off']],
                        [key, key],
                        linewidth=8,
                        label=key if idx == 0 else "",  # Label only first occurrence in legend
                        alpha=0.8,
                        color=color_mapping.get(key, 'grey')
                    )
                    
                ax2.text(row['ts_off'] + pd.Timedelta(seconds=180), key, f"Count: {usage_count}", verticalalignment='center', color='black', fontsize=10)
          
            else:
                data_dict[key] = pd.DataFrame()
                ax2.plot(
                    [backward_time, backward_time], [key, key],
                    color='grey', linewidth=1, alpha=0.5
                )

    # Plot the temperature peak line
    ax2.plot([backward_time, forward_time], ['Temperature Peak', 'Temperature Peak'],
              linewidth=8, label='Temperature Peak', alpha=0.8, color='black')
    # Mark the peak time with a red star
    ax2.scatter(pt, 'Temperature Peak', color='red', s=200, marker='*')
    ax2.axvline(pt)
    # Customize the lower subplot
    ax2.set_title(subject  + " ) - Devices used during temperature peak", fontsize=14)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Devices/ Sensors", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title="Devices")
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz='Europe/Rome'))
        ax.tick_params(axis='x', rotation=90)  # Rotate the x-axis labels for better readability
    # Set the common x-axis label
    fig.tight_layout()
    
    timestamp_str = backward_time.strftime('%Y-%m-%d_%H-%M')  # Format: '2024-08-01_19-13-42'
    save_path = os.path.join(pth, f"{pi}_{subject}_{timestamp_str}.png")
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure after saving

#df_stove_temperature, peak_index, delta_T, useMedian = df_stove_temperature, peak_index, delta_T, False
def find_time_interval_v2(df_stove_temperature, peak_index, delta_T, useMedian):
    """
    Finds the time interval during which the stove temperature decreases from the peak temperature.
    The function calculates the start and end times of the interval based on a temperature drop,
    and returns the corresponding time differences.

    Args:
        df_stove_temperature (pd.DataFrame): A DataFrame containing stove temperature data with columns 
                                             such as 'sensor_status' for the temperature and 'ts_datetime' 
                                             for the timestamp.
        peak_index (int): The index of the peak temperature in the DataFrame.
        delta_T (float): The temperature difference used to define the stopping temperature relative to the peak.
        useMedian (bool): If True, the daily median temperature is used as the stopping temperature; otherwise, 
                           the stopping temperature is defined as peak temperature - (delta_T / 2).

    Returns:
        tuple: A tuple containing:
            - backward_time (datetime): The timestamp when the temperature first drops below the stopping temperature
                                        or an increasing temperature trend is observed before the peak.
            - forward_time (datetime): The timestamp when the temperature first drops below the stopping temperature
                                       or an increasing temperature trend is observed after the peak.
            - time_difference (timedelta): The time difference between `forward_time` and `backward_time`.
            - backward_peak_diff (timedelta): The time difference between `peak_time` and `backward_time`.
            - forward_peak_diff (timedelta): The time difference between `forward_time` and `peak_time`.
            - stop_temp (float): The stopping temperature used to define the time interval.

    Notes:
        - If no stopping point is found during the backward or forward loop, the earliest or latest time in the DataFrame is used, respectively.
        - The function assumes the DataFrame is sorted by timestamp.
    """    
    # Get peak temperature and timestamp at the peak_index
    peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
    peak_time = df_stove_temperature.iloc[peak_index]['ts_datetime']

    # Determine the stopping temperature
    if useMedian:
        stop_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
    else:
        stop_temp = peak_temp - (delta_T / 2)

    # Initialize times to None
    backward_time, forward_time = None, None

    # Backward loop: Check temperature values before the peak
    for i in range(peak_index - 1, -1, -1):  # Loop backwards from peak_index
    # print(i)
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp or (i > 0 and df_stove_temperature.iloc[i - 1]['sensor_status'] > temp):
            # Stop if temperature is <= stop_temp or an increasing trend is observed
            backward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break
    
    # If no stopping point is found, use the earliest time in the DataFrame
    if backward_time is None:
        backward_time = df_stove_temperature.iloc[0]['ts_datetime']

    # Forward loop: Check temperature values after the peak
    for i in range(peak_index + 1, len(df_stove_temperature)):  # Loop forwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp or (i < len(df_stove_temperature) - 1 and df_stove_temperature.iloc[i + 1]['sensor_status'] > temp):
            # Stop if temperature is <= stop_temp or an increasing trend is observed
            forward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break

    # If no stopping point is found, use the latest time in the DataFrame
    if forward_time is None:
        forward_time = df_stove_temperature.iloc[-1]['ts_datetime']

    # Calculate the time differences
    if backward_time and forward_time:
        time_difference = forward_time - backward_time
    else:
        time_difference = None  # Assign None if either time is not found

    return backward_time, forward_time, time_difference, peak_time - backward_time, forward_time-peak_time, stop_temp


def find_time_interval(df_stove_temperature, peak_index, delta_T, useMedian):
    """
    Finds the time interval before and after a temperature peak where the temperature falls below a threshold.
    
    Description:
    This function identifies the times when the stove temperature drops below a threshold (`stop_temp`) both before and after the temperature peak, based on a given delta (temperature difference). It calculates the time interval between these two times and the duration between the peak time and the backward/forward times.
    
    Input:
    - df_stove_temperature (pd.DataFrame): DataFrame containing stove temperature data with columns such as 'sensor_status' (temperature) and 'ts_datetime' (timestamp).
    - peak_index (int): Index of the peak temperature in the DataFrame.
    - delta_T (float, optional, default=10): The temperature difference used to define the threshold for stop temperature (`stop_temp`).
    
    Output:
    - backward_time (pd.Timestamp or None): The time when the temperature first falls below `stop_temp` before the peak.
    - forward_time (pd.Timestamp or None): The time when the temperature first falls below `stop_temp` after the peak.
    - time_difference (pd.Timedelta or None): The time difference between `forward_time` and `backward_time`.
    - peak_time_diff_backward (pd.Timedelta): The time difference between the peak time and `backward_time`.
    - peak_time_diff_forward (pd.Timedelta): The time difference between the peak time and `forward_time`.
    - stop_temp (float): The calculated stop temperature threshold based on the peak temperature and `delta_T`.
    """
    # Get peak temperature at the peak_index
    peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
    peak_time = df_stove_temperature.iloc[peak_index]['ts_datetime']
    # print(delta_T)
    
    if useMedian == True:
        # print('-----------')
        stop_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
        # print(stop_temp)
    else:
        # print('********')
        stop_temp = peak_temp - (delta_T / 2)

    # Initialize times to None
    backward_time, forward_time = None, None

    # Backward loop: Check temperature values before the peak
    for i in range(peak_index - 1, -1, -1):  # Loop backwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp:
            backward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break  # Stop once condition is met
            
    if backward_time is None:
        backward_time = df_stove_temperature.iloc[0]['ts_datetime']

    # Forward loop: Check temperature values after the peak
    for i in range(peak_index + 1, len(df_stove_temperature)):  # Loop forwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp:
            forward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break  # Stop once condition is met
            
    if forward_time is None:
        forward_time = df_stove_temperature.iloc[-1]['ts_datetime']

    # Calculate the time difference
    if backward_time and forward_time:
        time_difference = forward_time - backward_time
    else:
        time_difference = None  # Assign None if either time is not found

    return backward_time, forward_time, time_difference, peak_time - backward_time, forward_time-peak_time, stop_temp

def is_cooking_peak_priyankar(num_points, temper, alpha, beta, gamma=40 , N=2, theta=30):
    is_cooking_peak = False
    if num_points >= N:
        is_cooking_peak = True
    else:
        if temper >= theta:
            is_cooking_peak = True
        else:
            if alpha >= gamma and np.abs(beta) >= gamma:
                is_cooking_peak = True
            else:
                is_cooking_peak = False
    return is_cooking_peak

def compute_angle(df_temp, point_a, point_b):
    """
    Computes the angle between two temperature readings based on their time difference and temperature difference.
    
    Description:
    This function calculates the angle of the line connecting two temperature readings, where the temperature difference is divided by the time difference between the two readings. The angle is returned in degrees, representing the slope between the two points on a temperature-time graph.
    
    Input:
    - df_temp (pd.DataFrame): DataFrame containing temperature data with columns such as 'sensor_status' (temperature) and 'ts_datetime' (timestamp).
    - point_a (int): Index of the first point in the DataFrame to calculate the angle from.
    - point_b (int): Index of the second point in the DataFrame to calculate the angle to.
    
    Output:
    - angle (float): The angle in degrees representing the slope between the two points, calculated using the formula `angle = atan(slope)`, where `slope = (temp_b - temp_a) / time_diff_hours`.
    
    Notes:
    - The angle is calculated using the arctangent of the slope of the temperature change over time.
    - If the time difference is zero, the function will raise a division by zero error.
    """
    temp_a = df_temp.loc[point_a, 'sensor_status']
    temp_b = df_temp.loc[point_b, 'sensor_status']
    time_diff_hours = (df_temp.loc[point_b, 'ts_datetime'] - df_temp.loc[point_a, 'ts_datetime']).total_seconds() / 3600.0
    slope = (temp_b - temp_a) / time_diff_hours
    angle = math.atan(slope) * (180 / math.pi)
    return angle

def if_kitchen_devices_used(data_subject, subject_devices, subject,st,et):
    """
    Checks if any kitchen devices were used during a specified time interval and returns relevant data.

    Description:
    This function evaluates the usage of kitchen devices within a given time window (from start time `st` to end time `et`) for a specific subject. It checks devices associated with the subject, filters the device data based on the time window, and returns a dictionary with the device data, a count of the devices used, and a list of the devices used.

    Input:
    - data_subject (dict): Dictionary containing data for each device. The key is the device name, and the value is a list with the first element being the device name and the second element being a DataFrame containing device data (including 'ts_on' and 'ts_off' columns).
    - subject_devices (dict): Dictionary mapping subjects to their devices. The key is the subject, and the value is a list of devices associated with the subject.
    - subject (str): The subject ID to check for device usage.
    - st (datetime): The start time of the time interval to check.
    - et (datetime): The end time of the time interval to check.

    Output:
    - data_dict (dict): A dictionary where the keys are the names of kitchen devices, and the values are DataFrames containing the device data filtered for the time interval.
    - count_var (int): The number of kitchen devices that were used during the specified time interval.
    - cooking_device (list): A list of the names of the kitchen devices that were used during the specified time interval.

    Notes:
    - The function excludes devices with names containing "Stove" or "Shower" from the analysis.
    - Device data is filtered by checking whether the 'ts_on' time is before the end time (`et`) and the 'ts_off' time is after the start time (`st`).
    """    
    data_dict = {}          
    count_var = 0
    cooking_device = []
    subject_kitchen_devices = sorted(subject_devices[subject])
    
    # for key in natsorted(data_subject.keys()):
    for key in natsorted(subject_kitchen_devices):
        if "Stove" not in key and "Shower" not in key:
            temp = data_subject[key][1].copy()
            # device_data = temp[(temp['ts_on'] >= st) & (temp['ts_on'] <= et)] ## any device which is used between st and et
            device_data = temp[(temp['ts_on'] <= et) & (temp['ts_off'] >= st)] ## any device usage which overlaps with st and et
            if len(device_data) > 0:
                count_var = count_var + 1
                cooking_device.append(key)
                data_dict[key] = device_data
            else:
                data_dict[key] = pd.DataFrame()
    return data_dict, count_var, cooking_device

def create_box_plot(df_stove_temperature, subject, savepath):
    """
    Creates and saves box plots with scatter plots to visualize stove temperature data for a specific subject.

    Description:
    This function generates two box plots:
    - The first plot shows the distribution of stove temperature sensor data ('sensor_status') with a scatter plot overlay.
    - The second plot displays the daily temperature standard deviation with a scatter plot overlay for each day.
    Both plots are created side-by-side for comparison. The plots are saved as a PNG file in the specified directory with the subject's name.

    Input:
    - df_stove_temperature (DataFrame): A pandas DataFrame containing stove temperature data with columns such as 'sensor_status' and 'sensor_datetime'.
    - subject (str): The subject name or ID to be used as the title of the plot and for naming the saved file.
    - savepath (str): The directory path where the plot image will be saved.

    Output:
    - Saves a PNG image with the box plots in the specified `savepath` directory with the subject's name.

    Notes:
    - The first plot visualizes the distribution of stove temperature readings ('sensor_status') with scatter points overlaid.
    - The second plot shows the standard deviation of daily stove temperatures, with scatter points added for better visibility.
    - The plots use different colors for visual differentiation: light blue for temperature distribution and light coral for the standard deviation.
    - Gridlines are added to the y-axis for clarity, and the plots are adjusted to fit titles and labels.

    Example:
    create_box_plot(df_stove_temperature, "Subject_1", "/path/to/save")
    """    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(subject, fontsize=16)
    
    # Subplot 1: Box plot for 'sensor_status' with scatter
    sns.boxplot(
        data=df_stove_temperature,
        y="sensor_status",
        color="skyblue",
        ax=axes[0]
    )
    sns.stripplot(
        data=df_stove_temperature,
        y="sensor_status",
        color="darkblue",
        alpha=0.2,  # Make scatter points semi-transparent
        size=2,  # Set marker size
        jitter=True,  # Add jitter to scatter points for visibility
        ax=axes[0]
    )
    axes[0].set_title("Temperature Distribution", fontsize=14)
    axes[0].set_ylabel("Sensor Status", fontsize=12)
    axes[0].set_xlabel("Sensor", fontsize=12)
    
    # Subplot 2: Box plot for standard deviation with scatter
    sns.boxplot(
        data=std_per_day,
        y="sensor_status",
        color="lightcoral",
        ax=axes[1]
    )
    sns.stripplot(
        data=std_per_day,
        y="sensor_status",
        color="darkred",
        alpha=0.2,
        size=3,
        jitter=True,
        ax=axes[1]
    )
    axes[1].set_title("Temperature Standard Deviation (Daily Temperature)", fontsize=14)
    axes[1].set_ylabel("Standard Deviation", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    
    # Add gridlines for better visualization
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit titles
    filename = subject+'.png'
    plt.savefig(os.path.join(savepath,filename), dpi=300)
    plt.show()
    
def classify_time_of_day(time):
    """
    Classifies the given time into a meal category based on predefined time ranges.
    
    Description:
    This function takes a `time` as input and classifies it into one of the following categories:
    - 'breakfast': If the time falls within the predefined breakfast time range.
    - 'lunch': If the time falls within the predefined lunch time range.
    - 'dinner': If the time falls within the predefined dinner time range.
    - 'other': If the time does not fall within any of the predefined meal time ranges.
    
    Input:
    - time (datetime or time object): The specific time to classify. It is expected to be a `datetime` or `time` object.
    
    Output:
    - (str): A string representing the meal category ('breakfast', 'lunch', 'dinner', or 'other') based on the input `time`.
    
    Assumptions:
    - `breakfast_start`, `breakfast_end`, `lunch_start`, `lunch_end`, `dinner_start`, and `dinner_end` are predefined time objects representing the start and end times for breakfast, lunch, and dinner, respectively.
    - breakfast_time = ("06:00:00", "10:00:00")
    - lunch_time = ("12:00:00", "14:00:00")
    - dinner_time = ("18:00:00", "21:00:00")
    - breakfast_start, breakfast_end = pd.to_timedelta(breakfast_time[0]), pd.to_timedelta(breakfast_time[1])
    - lunch_start, lunch_end = pd.to_timedelta(lunch_time[0]), pd.to_timedelta(lunch_time[1])
    - dinner_start, dinner_end = pd.to_timedelta(dinner_time[0]), pd.to_timedelta(dinner_time[1])
    
    Example:
    classify_time_of_day(datetime.time(7, 30))  # Returns 'breakfast'
    classify_time_of_day(datetime.time(13, 0))  # Returns 'lunch'
    """
    if breakfast_start <= time <= breakfast_end:
        return 'breakfast'
    elif lunch_start <= time <= lunch_end:
        return 'lunch'
    elif dinner_start <= time <= dinner_end:
        return 'dinner'
    else:
        return 'other'

def compute_angle_between_peak_and_previous_to_peak(df_stove_temperature,peak_index,signal):
    """
    Computes the angle between the peak temperature and the previous temperature point before the peak 
    based on the time difference and temperature change.

    Args:
        df_stove_temperature (pd.DataFrame): A DataFrame containing stove temperature data, including
                                             a 'ts_datetime' column for timestamps.
        peak_index (int): The index of the peak temperature in the DataFrame.
        signal (list or pd.Series): A list or series of temperature values corresponding to the stove signal.

    Returns:
        float: The angle (in degrees) between the peak temperature and the previous temperature point,
               calculated based on the temperature difference and time difference.
               If the angle cannot be calculated, returns -99.99.

    Notes:
        - The function computes the angle between the peak and its previous point by first calculating
          the time difference in hours between the two points and then using the slope to compute the angle.
        - The angle is calculated using the arctangent of the slope, converting it from radians to degrees.
        - If the first peak is selected (i.e., no previous peak), the function will skip the calculation.
        - The time difference between the peak and previous peak is computed in hours.
    """    
    angle = -99.99
    # Skip the first peak since there's no previous peak
    prev_peak_index = peak_index - 1
    x1 = df_stove_temperature['ts_datetime'].iloc[prev_peak_index]
    y1 = signal[prev_peak_index]
    
    # Time difference in hours
    time_diff_seconds = (peak_time - x1).total_seconds()
    time_diff_hours = time_diff_seconds / 3600  # Convert seconds to hours
    
    # Slope and angle calculation
    height_diff = peak_height - y1
    slope = height_diff / time_diff_hours
    angle = math.atan(slope) * (180 / math.pi)  # Convert radians to degrees
    return angle
            
# peak_index, df = index, df
def slope_based_peak_start(peak_index, df, theta):
    """
    Determines the start index of a peak in a signal based on the slope of the temperature curve before the peak.

    This function iterates backwards from the peak index to find the first point where the slope of the
    signal drops below a threshold angle (`theta`). It calculates the slope between consecutive points
    and compares the angle to `theta`. If the angle is less than `theta` or the slope is negative, it 
    identifies that point as the start of the peak.

    Args:
        peak_index (int): The index of the peak in the DataFrame.
        df (pd.DataFrame): A DataFrame containing the signal data, which includes columns 'sensor_status' (signal values),
                           'ts' (timestamp), and 'index' (unique index for each data point).
        theta (float): The threshold angle (in degrees). The function compares the slope's angle to this value to identify
                       the start of the peak.

    Returns:
        int: The index of the point where the peak starts. If no start point is found, returns -99.

    Notes:
        - The function computes the slope between the peak and previous points, using the time difference in hours.
        - The angle of the slope is calculated using the arctangent of the slope and converted from radians to degrees.
        - If the slope becomes negative or the angle drops below `theta`, the function assumes the peak has started.
        - The function only looks at the signal data before the peak.
        - If the 'sensor_status' values at consecutive points are equal, it skips that point and continues to the previous one.
        - The time difference is calculated in hours.
    """    
    # print(theta)
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
            # y2 = y2 + .00000001
            y2 = y1
            x2 = x1
            left_time = prev_time
            left_index = df.iloc[i]['index']
            i -= 1
            continue  # Skip the current iteration
        
        
        # Calculate the slope (y2 - y1) / (x2 - x1), where x values are in hours
        time_diff_in_hours = (left_time - prev_time).total_seconds() / 3600.0

        if time_diff_in_hours == 0:
            # print('something went wrong')
            break  
        
        slope = (y2 - y1) / time_diff_in_hours
        angle = math.atan(slope) * (180 / math.pi)
        
        if slope < 0:
            # print('hi')
            return left_index
            
        else:
            if angle < theta:
                # print('bye')
                return left_index
            else:
                y2 = y1
                x2 = x1
                left_time = prev_time
                left_index = df.iloc[i]['index']#i
        
        i -= 1
    return left_index

# peak_index, df, theta, k  = indx, df_stove_temperature.copy(), theta, 2
# def slope_based_peak_start_v3(peak_index, df, theta=40, k=2):
#     import math

#     # Get the peak row based on the given index
#     peak_row = df.iloc[peak_index]#df[df['index'] == peak_index]
#     # if peak_row.empty:
#     #     print("Peak index not found in the DataFrame.")
#     #     return -99  # Error code if peak not found

#     # Initialize values from the peak
#     y2 = peak_row['sensor_status']#.values[0]
#     x2 = peak_row['ts']#.values[0]
#     left_time = pd.to_datetime(x2, unit='ms')
#     left_index = peak_index

#     peak_row_position = peak_index#peak_row.index[0]
#     i = peak_row_position - 1  # Start from the point before the peak

#     consecutive_count = 0  # Counter for consecutive angles below theta
#     first_angle_below_threshold_index = None  # To track the starting point of the streak

#     while i >= df.index[0]:  # Loop until the start of the DataFrame
#         y1 = df.loc[i, 'sensor_status']
#         x1 = df.loc[i, 'ts']
#         prev_time = pd.to_datetime(x1, unit='ms')

#         # Check if left_time and prev_time have the same hour and minute
#         if y2 == y1: #or (left_time.hour == prev_time.hour and left_time.minute == prev_time.minute):
#             # Reset values and move to the next previous index
#             # y2 = y2 + .00000001
#             # y2 = y1
#             # x2 = x1
#             # left_time = prev_time
#             # left_index = df.iloc[i]['index']
#             i -= 1
#             continue  # Skip the current iteration

#         # Calculate the time difference in hours
#         time_diff_in_hours = (left_time - prev_time).total_seconds() / 3600.0

#         if time_diff_in_hours == 0:
#             break  # Avoid division by zero if timestamps are identical

#         # Calculate slope and angle
#         slope = (y2 - y1) / time_diff_in_hours
#         angle = math.atan(slope) * (180 / math.pi)

#         # Check if the angle is below the threshold
#         if angle < theta:
#             consecutive_count += 1  # Increment the count for consecutive angles below theta
#             if first_angle_below_threshold_index is None:
#                 first_angle_below_threshold_index = left_index  # Save the starting point of the streak
#         else:
#             # Reset the counter and starting index if the streak is broken
#             consecutive_count = 0
#             first_angle_below_threshold_index = None

#         # If k consecutive angles below theta are found, return the indices
#         if consecutive_count == k:
#             break
#             # return first_angle_below_threshold_index

#         # Update previous values
#         y2 = y1
#         x2 = x1
#         left_time = prev_time
#         left_index = i#df.iloc[i]['index']

#         i -= 1
#     return 0  # Return -1 if no valid point is found

#peak_index ,df , theta, k = index, df_stove_temperature.copy(), 40, 2
def slope_based_peak_start_v2(peak_index, df, theta=40, k=2):
    """
    Determines the start index of a peak in a signal based on the slope of the temperature curve before the peak.

    This function iterates backwards from the peak index to find the first point where the slope of the
    signal drops below a threshold angle (`theta`). It calculates the slope between consecutive points
    and compares the angle to `theta`. If the angle is less than `theta` or the slope is negative, it 
    identifies that point as the start of the peak.

    Args:
        peak_index (int): The index of the peak in the DataFrame.
        df (pd.DataFrame): A DataFrame containing the signal data, which includes columns 'sensor_status' (signal values),
                           'ts' (timestamp), and 'index' (unique index for each data point).
        theta (float): The threshold angle (in degrees). The function compares the slope's angle to this value to identify
                       the start of the peak.

    Returns:
        int: The index of the point where the peak starts. If no start point is found, returns -99.

    Notes:
        - The function computes the slope between the peak and previous points, using the time difference in hours.
        - The angle of the slope is calculated using the arctangent of the slope and converted from radians to degrees.
        - If the slope becomes negative or the angle drops below `theta`, the function assumes the peak has started.
        - The function only looks at the signal data before the peak.
        - If the 'sensor_status' values at consecutive points are equal, it skips that point and continues to the previous one.
        - The time difference is calculated in hours.
    """    
    import math
    peak_row = df[df.index == peak_index]

    y2 = peak_row['sensor_status'].values[0]
    x2 = peak_row['ts'].values[0]
    left_time = pd.to_datetime(x2, unit='ms')
    left_index = peak_index

    peak_row_position = peak_row.index[0]
    i = peak_row_position - 1  # Start from the point before the peak

    consecutive_count = 0  # Counter for consecutive angles below theta
    first_angle_below_threshold_index = None  # To track the starting point of the streak

    while i >= df.index[0]:  # Loop until the start of the DataFrame
        y1 = df.loc[i, 'sensor_status']
        x1 = df.loc[i, 'ts']
        prev_time = pd.to_datetime(x1, unit='ms')
        
        # Check if left_time and prev_time have the same hour and minute
        if y2 == y1: 
            i -= 1
            continue  # Skip the current iteration
        # print('hello......')
        # Calculate the time difference in hours
        time_diff_in_hours = (left_time - prev_time).total_seconds() / 3600.0

        if time_diff_in_hours == 0:
            break  # Avoid division by zero if timestamps are identical
            
        if time_diff_in_hours >= 0.5:
            # break
            return i

        # Calculate slope and angle
        slope = (y2 - y1) / time_diff_in_hours
        angle = math.atan(slope) * (180 / math.pi)

        # Check if the angle is below the threshold
        if angle < theta:
            consecutive_count += 1  # Increment the count for consecutive angles below theta
            if first_angle_below_threshold_index is None:
                first_angle_below_threshold_index = left_index  # Save the starting point of the streak
        else:
            # Reset the counter and starting index if the streak is broken
            consecutive_count = 0
            first_angle_below_threshold_index = None

        # If k consecutive angles below theta are found, return the indices
        if consecutive_count == k:
            if first_angle_below_threshold_index != peak_index:
                break
                # return first_angle_below_threshold_index
            else:
                first_angle_below_threshold_index = i 
                break
                # return first_angle_below_threshold_index

        # Update previous values
        y2 = y1
        x2 = x1
        left_time = prev_time
        left_index = df.index[i]#['index']
        i -= 1
    # print('out of the loop')
    return None



# temperature_df, save_path = df_stove_temperature, '/home/hubble/temp'
def plot_peaks_and_mean_temperature(temperature_df, save_path):
    """
    Plots the temperature data with detected peaks and the mean temperature for winter and summer periods.

    This function processes a temperature dataset, divides it into winter (November-February) and summer (March-October)
    periods, and generates scatter plots for each period showing the detected peak temperatures. Horizontal lines
    represent the mean temperature and standard deviations (1σ, 2σ, 3σ) for each period.

    The function saves the generated plots as PNG images in a specified directory.

    Args:
        temperature_df (pd.DataFrame): A DataFrame containing the temperature data with columns 'subject_id', 
                                       'sensor_status' (temperature readings), 'ts_datetime' (timestamp),
                                       and 'peaks' (indicator for detected peaks).
        save_path (str): The path to the directory where the generated plots will be saved.

    Returns:
        None: The function saves the plots to the specified directory but does not return any values.

    Notes:
        - The winter period is defined as November to February, and the summer period is defined as March to October.
        - The function calculates the average temperature and standard deviation for both winter and summer periods.
        - Detected peaks are plotted as red points on the scatter plots for each season.
        - Horizontal lines are drawn to represent the mean temperature and the standard deviations (1σ, 2σ, 3σ) for both seasons.
        - X-axis labels (dates) are adjusted to avoid overcrowding by selecting every 5th peak for winter and every 10th peak for summer.
        - The plots are saved in a folder named after the subject, with the overall plot filename indicating the subject and the temperature data.
    """    
    # Extract the subject name from the subject_id column
    subject_id = temperature_df['subject_id'].iloc[0]
    subject_name = f"Subject_{subject_id}"
    
    # Ensure save_path exists and create a folder for the subject's peak values
    save_folder = save_path#os.path.join(save_path, f"{subject_name}_peak_values")
    os.makedirs(save_folder, exist_ok=True)
    
    # Step 0: Detect peaks on the entire dataset
    # sensor_values = temperature_df['sensor_status'].values
    # peaks, _ = find_peaks(sensor_values, prominence=1.5)
    # temperature_df['peaks'] = 0
    # temperature_df.loc[peaks, 'peaks'] = 1  # Mark peaks as '1' in the 'peaks' column
        
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

    # Winter subif plot (November to February)
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
    """
    Plots a heatmap of device usage during temperature peaks in the kitchen for a given subject.

    This function takes monthly device usage data and generates a heatmap showing the percentage of times a device
    was used during temperature peaks in the kitchen for each month. The heatmap is saved as a PNG file in a subject-specific
    folder.

    Args:
        monthly_usage (dict or pd.DataFrame): A dictionary or DataFrame where each key/column represents a device, and the 
                                              corresponding values represent the percentage of times that device was used 
                                              during temperature peaks in the kitchen for each month.
        subject (str): The name or identifier of the subject whose device usage is being plotted.
        save_path (str): The path to the directory where the generated heatmap image will be saved.

    Returns:
        None: The function saves the heatmap to a subject-specific directory but does not return any values.

    Notes:
        - The `monthly_usage` data should be structured such that each device's usage percentages are available for each month.
        - The function formats the index of the input data as "YYYY-MM" for easy identification of months.
        - The heatmap uses the "YlGnBu" color map and annotates each cell with the value formatted to one decimal place.
        - The plot is saved in a folder named after the subject and includes the subject's name in the file name.
    """    
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
    """
    Plots daily activity of kitchen devices for a given subject, focusing on cooking activities and temperature peaks.

    This function generates time series plots for kitchen devices used during cooking activities and temperature peaks
    on specific days. Each plot visualizes the usage of devices in the kitchen, with activity periods marked by time ranges.
    The plot is saved in a subject-specific folder as a PNG image.

    Args:
        date_peak_times (dict): A dictionary where the keys are dates (as strings or datetime objects), and the values are
                                 DataFrames containing peak times for temperature data. Each DataFrame should have columns
                                 like 'ts_on' and 'ts_off' marking the start and end times of temperature peaks.
        date_cooking_devices (dict): A dictionary where the keys are dates, and the values are DataFrames containing device 
                                      usage information. Each DataFrame should include columns such as 'sensor_id', 'ts_on', 
                                      and 'ts_off'.
        subject (str): The name or identifier of the subject whose kitchen device usage is being plotted.
        subject_kitchen_devices (list): A list of device IDs that are located in the kitchen and used during cooking activities.
        save_path (str): The path to the directory where the generated plot images will be saved.

    Returns:
        None: The function generates and saves plots but does not return any values.

    Notes:
        - The plot for each date shows the usage of kitchen devices over time. Devices are represented on the y-axis, and time is shown on the x-axis.
        - The function will create a folder for the subject if it doesn't already exist.
        - If no device data is available for a specific date, a plot with default time settings is generated.
        - Each plot is saved as a PNG image with the date as part of the filename.
        - The time axis is formatted for the Europe/Rome timezone and includes ticks every 15 minutes.
        - The `sensor_id` column should be present in the device usage DataFrame.
    """    
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
    """
    Computes daily temperature statistics for a given dataset.
    
    This function processes the input DataFrame to compute the daily median temperature (`daily_avg_temp`) and 
    the daily standard deviation of temperature (`daily_avg_std`) based on the 'sensor_status' column. The computed
    statistics are then merged back into the original DataFrame.
    
    Args:
        temp (pd.DataFrame): A DataFrame containing temperature data with at least two columns:
                              - 'ts_datetime': A datetime column representing the timestamp of each temperature reading.
                              - 'sensor_status': A column representing the temperature readings (e.g., sensor values).
    
    Returns:
        pd.DataFrame: The original DataFrame with additional columns for the computed daily statistics:
                      - 'daily_avg_temp': The median temperature for each day.
                      - 'daily_avg_std': The standard deviation of temperature for each day.
    
    Notes:
        - The 'ts_datetime' column should be in datetime format for correct extraction of the date.
        - The 'sensor_status' column is expected to contain numeric temperature values.
        - The function assumes that the temperature data is timestamped and can be grouped by date.
    """
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
## result, category_name, df_stove_temperature, data_subject, subject, pth,  threshold_peak_duration = result, category_name, df_stove_temperature, data_subject.copy(), subject,pth, 100
def plot_peak_and_kitchen_devices(result, category_name, df_stove_temperature, data_subject, subject, pth,  threshold_peak_duration = 100):
    """
    Removes peaks from the list that are below the threshold defined by the daily average temperature 
    and its standard deviation.
    
    This function filters out peaks from the given list if the temperature at the peak is less than or
    equal to the sum of the daily average temperature and the daily standard deviation. This helps remove
    temperature peaks that do not significantly exceed the daily temperature variation.
    
    Args:
        temperature_df (pd.DataFrame): A DataFrame containing temperature data with at least the following columns:
                                        - 'sensor_status': The temperature reading for each timestamp.
                                        - 'daily_avg_temp': The daily average temperature for each day.
                                        - 'daily_avg_std': The daily standard deviation of the temperature.
        peaks (list): A list of indices representing the temperature peaks to be filtered. These indices correspond
                      to rows in `temperature_df`.
    
    Returns:
        list: A filtered list of peaks where each peak's temperature exceeds the sum of the daily average temperature 
              and standard deviation.
    
    Notes:
        - The `temperature_df` must contain the 'sensor_status', 'daily_avg_temp', and 'daily_avg_std' columns.
        - The `peaks` list should contain valid indices that can be used to access rows in the `temperature_df`.
    """
    df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
    
    daily_stats = df_stove_temperature.groupby('date')['sensor_status'].agg(
        daily_avg_temperature='mean',
        daily_median_temperature='median',
        daily_std_temperature='std'
    ).reset_index()

    df_stove_temperature = pd.merge(df_stove_temperature, daily_stats, on='date', how='left')
    seasons = natsorted(list(result.keys()))

    is_active_in_kitchen = []
    for sson in range(len(seasons)):
        season = seasons[sson]
        peak_info = result[season]['categories'][category_name]
        temp_list = []
        ## For each peak in that duration
        for pi in range(len(peak_info)):
            peak_duration_minutes = peak_info[pi]['duration_minutes']
            if 4!=5:#peak_duration_minutes > threshold_peak_duration:
                # Define the time window for the peak
                st = peak_info[pi]['left_time']
                et = peak_info[pi]['right_time']
                pt = peak_info[pi]['peak_time']  
                peak_index = peak_info[pi]['peak_index']
                # break
                backward_time = st- pd.Timedelta(hours=2)
                forward_time = et + pd.Timedelta(hours=2)
                temperature_data_in_peak = df_stove_temperature[(df_stove_temperature['ts_datetime'] >= st- pd.Timedelta(hours=2)) & 
                                                                (df_stove_temperature['ts_datetime'] <= et+ pd.Timedelta(hours=2))]
                
                data_dict = {}                    
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                
                # Part 1: Plot the raw temperature curve in the upper subplot
                ax1.plot(temperature_data_in_peak['ts_datetime'], 
                          temperature_data_in_peak['sensor_status'], 
                          color='red', label='Temperature Reading', linewidth=2)
                
                ax1.scatter(temperature_data_in_peak['ts_datetime'], 
                          temperature_data_in_peak['sensor_status'], 
                          color='blue')
                
                
                ax1.plot(temperature_data_in_peak['ts_datetime'], 
                          temperature_data_in_peak['daily_median_temperature'], 
                          label='Median temp. of the day', color='orange')
                
                ax1.plot(temperature_data_in_peak['ts_datetime'], 
                          temperature_data_in_peak['daily_median_temperature'] + temperature_data_in_peak['daily_std_temperature'], 
                          label='Std. dev. temperature', color='green')


                ax1.axvline(st, color='purple', linestyle='--', linewidth=1.5)
                ax1.axvline(et, color='purple', linestyle='--', linewidth=1.5)
                
                # Annotate st and et times near the vertical lines
                ax1.text(st, ax1.get_ylim()[1], st.strftime("%H:%M:%S"), 
                          color='purple', ha='right', va='bottom', fontsize=10, rotation=90)
                ax1.text(et, ax1.get_ylim()[1], et.strftime("%H:%M:%S"), 
                          color='purple', ha='right', va='bottom', fontsize=10, rotation=90)
                
                # ax1.set_title(subject+ "( "+ season + "- Temperature peaks", fontsize=14)
                ax1.set_ylabel("Temperature", fontsize=12)
                ax1.legend(loc='best')
                ax1.grid(True, linestyle='--', alpha=0.6)
                
                subject_kitchen_devices = sorted(subject_devices[subject])
                for key in natsorted(data_subject.keys()):
                # for key in natsorted(subject_kitchen_devices):
                    # print(key)
                    if "Stove" not in key and "Shower" not in key:
                        
                        temp = data_subject[key][1].copy()
                        # temp['ts_on'] = temp['ts_on'].apply(lambda x: max(x, st))  
                        # temp['ts_off'] = temp['ts_off'].apply(lambda x: min(x, et)) 
                        # device_data = temp[(temp['ts_on'] >= st) & (temp['ts_on'] <= et)]
                        device_data = temp[(temp['ts_on'] <= et) & (temp['ts_off'] >= st)]
                        
                        
                        if len(device_data) > 0:
                            print(key)
                            data_dict[key] = device_data
                            for idx, row in device_data.iterrows():
                                ax2.plot(
                                    [row['ts_on'], row['ts_off']],
                                    [key, key],
                                    linewidth=8,
                                    label=key if idx == 0 else "",  # Label only first occurrence in legend
                                    alpha=0.8,
                                    color=color_mapping.get(key, 'grey')
                                )
                        else:
                            data_dict[key] = pd.DataFrame()
                            ax2.plot(
                                [st, st], [key, key],
                                color='grey', linewidth=1, alpha=0.5
                            )

                # Plot the temperature peak line
                ax2.plot([st, et], ['Temperature Peak', 'Temperature Peak'],
                          linewidth=8, label='Temperature Peak', alpha=0.8, color='black')
                # Mark the peak time with a red star
                ax2.scatter(pt, 'Temperature Peak', color='red', s=200, marker='*')
                ax2.axvline(pt)
                # Customize the lower subplot
                ax2.set_title(subject + "( "+ season + " ) - Devices used during temperature peak", fontsize=14)
                ax2.set_xlabel("Time", fontsize=12)
                ax2.set_ylabel("Devices/ Sensors", fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.6)
                # ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title="Devices")
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz='Europe/Rome'))
                    ax.tick_params(axis='x', rotation=90)  # Rotate the x-axis labels for better readability
                # Set the common x-axis label
                fig.tight_layout()
                # Save the combined plot with two subplots
                timestamp_str = st.strftime('%Y-%m-%d_%H-%M')  # Format: '2024-08-01_19-13-42'
                save_path = os.path.join(pth, f"{subject}_{pi}_{peak_duration_minutes:1f}_{season}_{category_name}.png")
                plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure after saving
                
                # is_active_in_kitchen.append([subject, season, st,et, pt, len(device_data)])

    # subject_is_active_in_kitchen.append(is_active_in_kitchen)  
# 
def plot_temperature_peak_stats(monthly_peak_counts, monthly_avg_durations, monthly_days_with_peaks, save_path, subject):
    """
    Plots temperature peak statistics (peak counts, average peak duration, and days with peaks) for each month 
    and saves the plot as a PNG image.

    This function generates a bar chart with three subplots to visualize the following metrics:
    1. The number of temperature peaks detected for each month.
    2. The average duration of temperature peaks (in minutes) for each month.
    3. The number of days with temperature peaks for each month.

    The plots are saved in a folder named after the subject under the specified `save_path`.

    Args:
        monthly_peak_counts (dict): A dictionary where keys are month-year strings (e.g., '2023-01') 
                                    and values are the count of temperature peaks detected in that month.
        monthly_avg_durations (dict): A dictionary where keys are month-year strings and values are 
                                      `timedelta` objects representing the average duration of temperature peaks 
                                      in that month.
        monthly_days_with_peaks (dict): A dictionary where keys are month-year strings and values are 
                                        the number of days in that month that had temperature peaks.
        save_path (str): The path where the plot images will be saved.
        subject (str): The name of the subject to be used for labeling and naming the saved plot.

    Returns:
        None

    Notes:
        - The function creates a folder under `save_path` named `<subject>_peak_stats` to store the plot.
        - The function generates three bar charts: one for peak counts, one for average duration, and one for 
          the number of days with peaks.
        - The bar heights are annotated with their respective values for better readability.
    """    
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
    """
    Calculates monthly statistics for temperature peaks, including the number of peaks, 
    average peak duration, and the number of days with temperature peaks.

    This function processes a dictionary containing temperature peak data for each day of each month. 
    It calculates the following statistics for each month:
    1. Total number of temperature peaks.
    2. Average duration of temperature peaks.
    3. Number of unique days on which temperature peaks were observed.

    Args:
        monthly_temp_peaks (dict): A dictionary where keys are month-timestamp values (e.g., '2024-06') and 
                                   values are dictionaries containing daily data for that month. Each daily data 
                                   is a DataFrame with temperature peaks, where each row represents a peak with 
                                   'ts_on' (start time) and 'ts_off' (end time) columns.

    Returns:
        tuple: A tuple containing three dictionaries:
            - monthly_peak_counts (dict): The total number of temperature peaks for each month.
            - monthly_avg_durations (dict): The average duration of temperature peaks (in timedelta format) for each month.
            - monthly_days_with_peaks (dict): The number of unique days with temperature peaks for each month.

    Notes:
        - The function uses `pd.Timedelta` to calculate durations and averages.
        - If no peaks are observed in a month, the average duration is set to 0.
        - The function processes the data by iterating through each day of each month and calculating the statistics.
    """    
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
    """
    Calculates the percentage of device usage based on the total number of temperature peaks.

    This function takes in the counts of how many times each device was used during temperature peaks, 
    and calculates the percentage of usage for each device relative to the total number of temperature peaks.

    Args:
        usage_counts (dict): A dictionary where keys are device names and values are the count of times the device was used during temperature peaks.
        total_temp_peak_count (int): The total number of temperature peaks observed.

    Returns:
        dict: A dictionary where keys are device names and values are the percentage of usage for each device.

    Notes:
        - If there are no temperature peaks (i.e., total_temp_peak_count is 0), the usage percentage for all devices will be set to 0.
        - The percentage is calculated as (device usage count / total temperature peaks) * 100.

    Example:
        usage_counts = {'device_1': 15, 'device_2': 25}
        total_temp_peak_count = 50
        result = calculate_device_usage_percentage(usage_counts, total_temp_peak_count)
        # result would be {'device_1': 30.0, 'device_2': 50.0}
    """    
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
    """
    Calculates the percentage usage of kitchen devices based on their overlap with temperature peaks.

    This function calculates how often each device from a specified list was used during temperature peaks
    based on data from specific dates. It checks for overlaps between temperature peak periods and the activation
    times of kitchen devices, then calculates the percentage usage of each device.

    Args:
        date_peak_times (dict): A dictionary where keys are dates (as datetime objects) and values are DataFrames
                                containing the temperature peak times for that date. The DataFrames should have
                                'ts_on' and 'ts_off' columns representing the start and end times of the peaks.
        date_cooking_devices (dict): A dictionary where keys are dates (as datetime objects) and values are DataFrames
                                      containing the activation times of kitchen devices on those dates. The DataFrames
                                      should have 'sensor_id', 'ts_on', and 'ts_off' columns.
        subject_kitchen_devices2 (list): A list of kitchen device names (sensor_ids) to track the usage of.

    Returns:
        dict: A dictionary where keys are device names (sensor_ids) and values are the percentage of time the device was
              used during temperature peaks, calculated based on the total number of temperature peaks for each date.

    Notes:
        - If a device is active during a temperature peak (i.e., the peak's start time is before the device's end time,
          and the peak's end time is after the device's start time), it is counted as used during that peak.
        - The usage percentage is computed relative to the total number of temperature peaks across all dates.

    Example:
        date_peak_times = {date1: df1, date2: df2}  # DataFrames with temperature peak times
        date_cooking_devices = {date1: cooking_df1, date2: cooking_df2}  # DataFrames with cooking device activations
        subject_kitchen_devices2 = ['device_1', 'device_2']
        
        result = calculate_device_usage_count(date_peak_times, date_cooking_devices, subject_kitchen_devices2)
        # result will be a dictionary with the usage percentage of each device during temperature peaks
    """    
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
    """
    Splits a DataFrame by day based on the 'ts_on' column and stores the resulting data in a dictionary.
    
    This function takes a DataFrame with a timestamp column ('ts_on'), extracts the date (day, month, year) from the timestamp, 
    and splits the data into separate DataFrames for each day. Each day’s data is stored in a dictionary, where the keys are the 
    formatted date strings ('dd_mm_yyyy'), and the values are the corresponding DataFrames containing the data for that day.
    
    Args:
        result_df (pd.DataFrame): A DataFrame containing a 'ts_on' column with timestamps, and any other columns representing 
                                  the data that should be split by day.
    
    Returns:
        dict: A dictionary where keys are date strings in the format 'dd_mm_yyyy', and values are DataFrames containing the 
              data for each day, with the 'date' column removed.
    
    Notes:
        - The 'ts_on' column is expected to be in a format that can be parsed by pandas to a datetime object. 
        - The 'date' column is added temporarily to group the data, but it is removed from each group before storing it in the dictionary.
    
    Example:
        result_df = pd.DataFrame({
            'sensor_id': ['sensor1', 'sensor2', 'sensor1'],
            'ts_on': ['2024-12-01 08:00:00', '2024-12-01 09:00:00', '2024-12-02 10:00:00'],
            'value': [23, 25, 22]
        })
        
        daily_data_dict = split_by_day(result_df)
        # daily_data_dict will contain:
        # {
        #     '01_12_2024': DataFrame with data for Dec 1, 2024
        #     '02_12_2024': DataFrame with data for Dec 2, 2024
        # }
    """
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
    """
    Combines and sorts data from two DataFrames (stove/shower and environmental devices) by timestamp, 
    then splits the resulting DataFrame by day.

    This function takes two DataFrames containing sensor data for stove/shower devices and environmental devices. 
    It concatenates these DataFrames, sorts the combined data by the 'ts_on' timestamp, 
    and then splits the sorted data into separate DataFrames for each day using the `split_by_day` function.

    Args:
        stove_shower_df_t1 (pd.DataFrame): A DataFrame containing sensor data for stove and shower devices, 
                                             with a 'ts_on' column containing timestamps.
        environmentals_devices_df_t1 (pd.DataFrame): A DataFrame containing sensor data for environmental devices, 
                                                     also with a 'ts_on' column containing timestamps.

    Returns:
        dict: A dictionary where keys are date strings (formatted as 'dd_mm_yyyy') and values are DataFrames 
              containing the combined and sorted data for each day. Each day’s data includes both stove/shower and 
              environmental devices, with the 'date' column removed.

    Notes:
        - The 'ts_on' column is expected to be in a datetime-compatible format, as it is used for sorting.
        - If either of the input DataFrames is empty, the function will return an empty dictionary.
        - The function uses the `split_by_day` function to split the combined DataFrame by day.

    Example:
        stove_shower_df_t1 = pd.DataFrame({
            'sensor_id': ['stove1', 'shower1'],
            'ts_on': ['2024-12-01 08:00:00', '2024-12-01 09:00:00'],
            'value': [100, 75]
        })
        
        environmentals_devices_df_t1 = pd.DataFrame({
            'sensor_id': ['sensor1', 'sensor2'],
            'ts_on': ['2024-12-01 08:30:00', '2024-12-02 10:00:00'],
            'value': [20, 22]
        })
        
        daily_data = combine_and_sort_df(stove_shower_df_t1, environmentals_devices_df_t1)
        # daily_data will contain:
        # {
        #     '01_12_2024': DataFrame with combined data for Dec 1, 2024
        #     '02_12_2024': DataFrame with data for Dec 2, 2024
        # }
    """    
    daily_data_all_devices_dict = {}
    if len(stove_shower_df_t1) > 0 or len(environmentals_devices_df_t1) > 0:
        combined_df_t1 = pd.concat([stove_shower_df_t1, environmentals_devices_df_t1], ignore_index=True) # combined stove, shower, and environmental sensors data
        sorted_df_t1 = combined_df_t1.sort_values(by='ts_on')
        daily_data_all_devices_dict = split_by_day(sorted_df_t1)    
    return daily_data_all_devices_dict

def compute_stove_usage_duration(daily_data_dict):
    """
    Computes the maximum stove usage duration for each day from the provided daily data.

    This function calculates the longest continuous usage of the stove (sensor_id 'Stove_Temp') for each day. 
    For each date, the stove usage is identified, and the duration between the 'ts_on' and 'ts_off' timestamps is 
    computed. The maximum duration for each day is returned.

    Args:
        daily_data_dict (dict): A dictionary where the keys are dates (in 'dd_mm_yyyy' format) and the values are 
                                 DataFrames containing sensor data. Each DataFrame contains sensor usage information, 
                                 including the 'sensor_id', 'ts_on' (timestamp when the device is turned on), and 'ts_off' 
                                 (timestamp when the device is turned off).

    Returns:
        dict: A dictionary where the keys are dates (in 'dd_mm_yyyy' format) and the values are the maximum stove usage 
              duration for that day, represented as a Timedelta object.

    Notes:
        - Only the stove data (sensor_id == 'Stove_Temp') is considered for duration calculation.
        - If no stove usage is recorded for a particular day, the function will return a zero Timedelta (no stove usage).
        - The result contains the maximum stove usage duration for each day found in the input data.

    Example:
        daily_data_dict = {
            '01_12_2024': pd.DataFrame({
                'sensor_id': ['Stove_Temp', 'Stove_Temp'],
                'ts_on': ['2024-12-01 08:00:00', '2024-12-01 09:30:00'],
                'ts_off': ['2024-12-01 08:30:00', '2024-12-01 10:00:00']
            }),
            '02_12_2024': pd.DataFrame({
                'sensor_id': ['Stove_Temp'],
                'ts_on': ['2024-12-02 07:00:00'],
                'ts_off': ['2024-12-02 08:00:00']
            })
        }

        max_stove_usage = compute_stove_usage_duration(daily_data_dict)
        # max_stove_usage will be:
        # {
        #     '01_12_2024': Timedelta('0 days 00:30:00'),
        #     '02_12_2024': Timedelta('0 days 01:00:00')
        # }
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