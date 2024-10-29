#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:28:01 2024

@author: hubble
"""
from main_utility import *
from subject_kitchen_device_mapping import *

#%% Read the data


#%%
cooking_devices = ['CoffeMachine', 'Cookware', 'Dishes', 'Dishes_Glasses', 'Dishes_Silverware', 'FoodStorage',\
                   'FoodStorageKitchen', 'FoodStorageLivingRoom', 'Freezer', 'Glasses', 'Microwave',\
                       'MotionKitchen', 'PresenceKitchen', 'PresenceKitchen_Stove', 'PresenceKitchen_Table',\
                           'Refrigerator', 'Silverware', 'Stove']
    
#%% 
## The following commented code runs only one time
# path_data = '/home/hubble/work/serenade/data/subject_data_sept_2024/FINAL_DATA'
# subject_dfs = read_csv_files_v2(path_data)
# with open('subject_data_sept_2024.pkl', 'wb') as file:
#     pickle.dump(subject_dfs, file)


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

with open('/home/hubble/work/serenade/data/subject_data_sept_2024.pkl', 'rb') as file:
    data = pickle.load(file)
subjects = natsorted(list(data.keys()))

#%%
QUERY_INTERVAL_START = '00:00:00'  # Start of the day
QUERY_INTERVAL_END = '23:59:59'    # End of the day
specific_devices = ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']


# Define the devices where ts_datetime should be used instead of ts_on
specific_devices = ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']

# First detect the temperature peak
for sub in range(3,4):#len(subjects)):
    subject = subjects[sub]
    subject_kitchen_devices = sorted(subject_devices[subject])
    subject_kitchen_devices2 = sorted(subject_devices[subject])
    subject_kitchen_devices.append('Temperature Peak')
    data_subject = data[subject]
    df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0]
    ## For safety convert the time to date time format with Europe/Rome Timezone
    df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc= True)  
    df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
    

    start_date = df_stove_temperature['ts_datetime'].min()
    # end_date = (start_date + pd.offsets.MonthEnd(0)).normalize()
    end_date = df_stove_temperature['ts_datetime'].max()
    monthly_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    for month_end in monthly_range:
        month_start = month_end - pd.offsets.MonthEnd(1) + pd.offsets.Day(1)
        
        month_value = month_start.month
        year_value = month_start.year
        
        temperature_df = df_stove_temperature[
            (df_stove_temperature['ts_datetime'] >= month_start) & 
            (df_stove_temperature['ts_datetime'] <= month_end)
        ].copy()

        peaks, properties = find_peaks(temperature_df['sensor_status'].values, prominence = 1.5)
        environmentals_devices_df_t1 = non_shower_stove_data(data_subject, specific_devices, QUERY_INTERVAL_START, QUERY_INTERVAL_END, year_value, month_value,)
        temperature_df = compute_daily_temperature(temperature_df.copy())
        peaks = remove_peaks_below_daily_avg(temperature_df, peaks)
        peak_dates = temperature_df.loc[peaks, 'ts_datetime'].dt.date
        
        date_index_pairs = {}
        for index in peaks:
            date = temperature_df.loc[index, 'ts_datetime'].date()  # Extract the date
            if date not in date_index_pairs:
                date_index_pairs[date] = []  # Initialize a list for this date
            date_index_pairs[date].append(index) 
        
        ## Only iterate over the dates which have peaks    
        dates = list(date_index_pairs.keys())
        indices_list = list(date_index_pairs.values())
        date_peak_times = {}
        date_cooking_devices = {}
        for i in range(len(dates)):
            date = dates[i]
            indices = indices_list[i]
            daily_environmental_data = environmentals_devices_df_t1[
                environmentals_devices_df_t1['ts_on'].dt.date == date
            ]
            
            peak_start_end_times = []
            cooking_devices_list = []
            for j in range(len(indices)):
                peak_index = indices[j]
                peak_temperature = temperature_df.iloc[peak_index]['sensor_status'] # Peak Temperature
                room_temperature = temperature_df.iloc[peak_index]['daily_avg_temp'] # Room Temperature
                peak_time = temperature_df['ts_datetime'].iloc[peak_index] # When did the peak occur
                duration, left, right = find_peak_duration_v3(temperature_df.copy(), peak_index,room_temperature)
                left_time = temperature_df['ts_datetime'].iloc[left]
                right_time = temperature_df['ts_datetime'].iloc[right]
                
                ## Devices used during the peak
                filtered_data = daily_environmental_data[
                    (daily_environmental_data['ts_on'] >= left_time) & 
                    (daily_environmental_data['ts_on'] <= right_time)
                ].copy()
                cooking_devices = filtered_data[filtered_data['sensor_id'].isin(subject_kitchen_devices)]
                peak_start_end_times.append({'sensor_id' : 'Temperature Peak', 'subject_id': int(subject.split('_')[1]), 'ts_on': left_time, 'ts_off':right_time, 'ts_peak':peak_time})
                cooking_devices_list.append(cooking_devices)
            df_peak_start_end_times = pd.DataFrame(peak_start_end_times)
            date_peak_times[date] = df_peak_start_end_times
            date_cooking_devices[date] = pd.concat(cooking_devices_list, ignore_index=True)
        plot_daily_activity(date_peak_times, date_cooking_devices, subject, subject_kitchen_devices, '/home/hubble')
                

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
    return usage_counts

# Example Usage:
# Assuming date_peak_times and date_cooking_devices are defined
# date_peak_times = {
#     '2024-03-01': temperature_peak_df,  # DataFrame of temperature peaks
#     # ... other dates
# }
# date_cooking_devices = {
#     '2024-03-01': cooking_devices_df,  # DataFrame of device usage
#     # ... other dates
# }
subject_kitchen_devices = [
    'Cookware',
    'Dishes_Glasses',
    'Freezer',
    'Microwave',
    'MotionKitchen',
    'Refrigerator',
    'Silverware'
]

device_usage_counts = calculate_device_usage_count(date_peak_times, date_cooking_devices, subject_kitchen_devices)

# Print the results
for device, count in device_usage_counts.items():
    print(f"Device: {device}, Count of activations during Temperature Peak: {count}")


# Example Usage:
# Assuming date_peak_times and date_cooking_devices are defined
# date_peak_times = {
#     '2024-03-01': temperature_peak_df,  # DataFrame of temperature peaks
#     # ... other dates
# }
# date_cooking_devices = {
#     '2024-03-01': cooking_devices_df,  # DataFrame of device usage
#     # ... other dates
# }

device_usage_percentages = calculate_device_usage_percentage(date_peak_times, date_cooking_devices)

# Print the results
for date, percentage in device_usage_percentages.items():
    print(f"Date: {date}, Percentage of devices used during Temperature Peak: {percentage:.2f}%")









        
