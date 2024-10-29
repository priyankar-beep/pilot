#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:58:48 2024

@author: hubble
"""

from main_utility import *

def find_peak_duration_v1(daily_data, peak_index, max_time_diff= pd.Timedelta(minutes=10)):
    start = peak_index
    while start > 0 and \
          daily_data['sensor_status'].iloc[start] >= daily_data['sensor_status'].iloc[start - 1] and \
          (daily_data['ts_datetime'].iloc[start] - daily_data['ts_datetime'].iloc[start - 1]) < max_time_diff:
        start -= 1

    end = peak_index
    while end < len(daily_data) - 1 and \
          daily_data['sensor_status'].iloc[end] >= daily_data['sensor_status'].iloc[end + 1] and \
          (daily_data['ts_datetime'].iloc[end + 1] - daily_data['ts_datetime'].iloc[end]) < max_time_diff:
        end += 1
    
    duration = end - start
    return duration, start, end

# def find_peak_duration_v3(daily_data, peak_index, daily_avg_temperature):
#     left_index = peak_index
#     # Traverse left until sensor_status drops below daily_avg_temperature or trend is not decreasing
#     while left_index > 0 and \
#           daily_data['sensor_status'].iloc[left_index] >= daily_avg_temperature and \
#           daily_data['sensor_status'].iloc[left_index] >= daily_data['sensor_status'].iloc[left_index - 1]:
#         # Uncomment the following line to add max time difference condition
#         # and (daily_data['ts_datetime'].iloc[left_index] - daily_data['ts_datetime'].iloc[left_index - 1]) < max_time_diff:
#         left_index -= 1
        
#     right_index = peak_index
#     # Traverse right until sensor_status drops below daily_avg_temperature or trend is not decreasing
#     while right_index < len(daily_data) - 1 and \
#           daily_data['sensor_status'].iloc[right_index] >= daily_avg_temperature and \
#           daily_data['sensor_status'].iloc[right_index] >= daily_data['sensor_status'].iloc[right_index + 1]:
#         # Uncomment the following line to add max time difference condition
#         # and (daily_data['ts_datetime'].iloc[right_index + 1] - daily_data['ts_datetime'].iloc[right_index]) < max_time_diff:
#         right_index += 1

#     # Calculate duration in terms of index difference
#     duration = right_index - left_index
#     return duration, left_index, right_index


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




def find_peak_duration_v2(data, peak_index,room_temperature,tolerance,patience):

    
    if peak_index < 0 or peak_index >= len(data):
        raise ValueError("Peak index must be within the bounds of the data list.")
    
    peak_value = data[peak_index]
    tolerance_range = (room_temperature - tolerance, room_temperature)

    left_index = find_left_index(data, peak_index, peak_value, tolerance_range, patience)
    right_index = find_right_index(data, peak_index, peak_value, tolerance_range, patience)

    # Calculate the duration of the peak
    duration = right_index - left_index
    return duration, left_index, right_index


def find_left_index(data, peak_index, peak_value, tolerance_range, patience):
    left_index = peak_index
    left_patience_count = 0
    
    while left_index > 0:
        left_index -= 1  # Move left

        if data[left_index] < peak_value:
            if tolerance_range[0] <= data[left_index] <= tolerance_range[1]:
                left_patience_count += 1
                if left_patience_count >= patience:
                    break
            else:
                left_patience_count = 0

    return left_index


def find_right_index(data, peak_index, peak_value, tolerance_range, patience):
    right_index = peak_index
    right_patience_count = 0

    while right_index < len(data) - 1:
        right_index += 1  # Move right

        if data[right_index] < peak_value:
            if tolerance_range[0] <= data[right_index] <= tolerance_range[1]:
                right_patience_count += 1
                if right_patience_count >= patience:
                    break
            else:
                right_patience_count = 0

    return right_index


if __name__ == "__main__":
    
    # define colors to show start and end of the peak
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    num_colors = len(colors)
    
    ## Read the data
    with open('/home/hubble/work/serenade/data/subject_data_sept_2024.pkl', 'rb') as file:
        data = pickle.load(file)
    
    ## List the names of the subject
    subjects = natsorted(list(data.keys()))
    
    ## Query interval is whole day
    QUERY_INTERVAL_START = '00:00:00'  # Start of the day
    QUERY_INTERVAL_END = '23:59:59'    # End of the day
    
    ## 
    specific_devices = ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']
    tolerance = 1
    patience = 1
    
    sub = 3
    subject = subjects[sub] # Name of the subject
    data_subject = data[subject] # Load the whole data
    df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0] ## Load subject's temperature data in the kitchen
    
    ## For safety convert the time to date time format with Europe/Rome Timezone
    df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc= True)  
    df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
    
    # start and end date of temperature data collection
    data_start_date = df_stove_temperature['ts_datetime'].min()
    data_end_date = df_stove_temperature['ts_datetime'].max()
    
    ## Specify the date where we have to pick the data
    start_date = pd.to_datetime('2024-03-01').tz_localize('Europe/Rome')
    end_date = pd.to_datetime('2024-03-05').tz_localize('Europe/Rome')
    
    temperature_df = df_stove_temperature[
    (df_stove_temperature['ts_datetime'] >= start_date) &
    (df_stove_temperature['ts_datetime'] <= end_date)].copy()
    
    ## Compute the peak
    peaks, properties = find_peaks(temperature_df['sensor_status'].values, prominence=1.5)
    
    def compute_daily_temperature(temp):
        ### Compute daily average temperature
        temp.loc[:, 'date'] = temp['ts_datetime'].dt.date
        daily_avg = temp.groupby('date')['sensor_status'].mean().reset_index()
        daily_avg.rename(columns={'sensor_status': 'daily_avg_temp'}, inplace=True)
        temp = pd.merge(temp, daily_avg, on='date', how='left')
        return temp
    
    
    
    def remove_peaks_below_daily_avg(temperature_df, peaks):
        filtered_peaks = [
            peak for peak in peaks
            if temperature_df.loc[peak, 'sensor_status'] >= temperature_df.loc[peak, 'daily_avg_temp']
        ]
        
        return filtered_peaks
    
    
    temperature_df = compute_daily_temperature(temperature_df.copy())
    peaks = remove_peaks_below_daily_avg(temperature_df, peaks)
  
    ## Make the figure
    plt.figure(figsize=(10, 5))
    
    # Plot temperature data and mark peaks
    plt.plot(temperature_df['ts_datetime'], temperature_df['sensor_status'], label='Temperature', color='blue')
    plt.plot(temperature_df['ts_datetime'].iloc[peaks], temperature_df['sensor_status'].iloc[peaks], 'ro', label='Peaks')
    plt.plot(temperature_df['ts_datetime'], temperature_df['daily_avg_temp'], label='Temperature', color='magenta')
    
    # Generate a complete date range with Europe/Rome timezone
    date_range = pd.date_range(
        start=temperature_df['ts_datetime'].min().date(), 
        end=temperature_df['ts_datetime'].max().date(),
        tz="Europe/Rome"
    )
    
    # Draw a vertical dashed line for each date in the complete range
    for date in date_range:
        plt.axvline(x=pd.Timestamp(date), color='gray', linestyle='--', linewidth=0.7)
    for p in range(len(peaks)):
        peak_index = peaks[p] # indiex of the peak
        peak_temperature = temperature_df.iloc[peak_index]['sensor_status'] # Peak Temperature
        room_temperature = temperature_df.iloc[peak_index]['daily_avg_temp'] # Room Temperature
        peak_time = temperature_df['ts_datetime'].iloc[peak_index] # When did the peak occur
        duration, left, right = find_peak_duration_v3(temperature_df.copy(), peak_index,room_temperature)

        left_time = temperature_df['ts_datetime'].iloc[left]
        right_time = temperature_df['ts_datetime'].iloc[right]
        
        print(p, '\t',peak_index, '\t', left_time, '\t', right_time,'\t', peak_time)
        
        # Select color for the current peak
        color = colors[p % num_colors]  # Cycle through colors
        
        # Plot vertical lines for left and right times of each peak with the same color
        plt.axvline(x=left_time, color=color, linestyle='--', linewidth=1.5, label=f'Peak {p+1} Start' if p == 0 else "")
        plt.axvline(x=right_time, color=color, linestyle='-', linewidth=1.5, label=f'Peak {p+1} End' if p == 0 else "")
        plt.annotate(f'{peak_index}', xy=(peak_time, room_temperature + 2), 
             ha='center', color=color, fontsize=8, fontweight='bold')
        
        
    # Set x-ticks and rotate for readability
    # plt.xticks(date_range, rotation=90)
    plt.title('Temperature Readings with Peaks')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    
    # Adjust legend to avoid duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

