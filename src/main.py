#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:28:01 2024

@author: hubble
"""
from main_utility import *
from subject_kitchen_device_mapping import *

#%% Read the data
with open('/home/hubble/work/serenade/data/subject_data_sept_2024.pkl', 'rb') as file:
    data = pickle.load(file)
subjects = natsorted(list(data.keys()))

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





#%%
QUERY_INTERVAL_START = '00:00:00'  # Start of the day
QUERY_INTERVAL_END = '23:59:59'    # End of the day
specific_devices = ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']


# Define the devices where ts_datetime should be used instead of ts_on
specific_devices = ['Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']
all_peaks_timings = []
# First detect the temperature peak
for sub in range(len(subjects)):
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
    monthly_usage = {}
    monthly_temp_peaks = {}
    
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
                std = temperature_df.iloc[peak_index]['daily_avg_std']
                peak_time = temperature_df['ts_datetime'].iloc[peak_index] # When did the peak occur
                duration, left, right = find_peak_duration_v3(temperature_df.copy(), peak_index,room_temperature, 3, std)
                left_time = temperature_df['ts_datetime'].iloc[left]
                right_time = temperature_df['ts_datetime'].iloc[right]
                
                if left_time.date() != right_time.date():
                    pass
                
                all_peaks_timings.append({'subject_id':subject,'peak_start':left_time ,'peak_end':right_time})
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
        percentage_usage = calculate_device_usage_count(date_peak_times, date_cooking_devices, subject_kitchen_devices2)
        monthly_usage[month_start] = percentage_usage
        monthly_temp_peaks[month_start] = date_peak_times
        # plot_daily_activity(date_peak_times, date_cooking_devices, subject, subject_kitchen_devices, '/home/hubble/cooking')
    # plot_device_usage_heatmap(monthly_usage, subject, "/home/hubble/cooking")
    monthly_peak_counts, monthly_avg_durations, monthly_days_with_peaks = calculate_monthly_peak_stats(monthly_temp_peaks)
    # plot_temperature_peak_stats(monthly_peak_counts, monthly_avg_durations, monthly_days_with_peaks, "/home/hubble/cooking", subject)

#%%
# Convert to DataFrame
df_peaks = pd.DataFrame(all_peaks_timings)
# Calculate duration in minutes
df_peaks['duration'] = (df_peaks['peak_end'] - df_peaks['peak_start']).dt.total_seconds() / 60
# Sort by ascending order of duration
df_peaks_sorted = df_peaks.sort_values(by='duration').reset_index(drop=True)

# Sort the DataFrame and exclude the last four durations
df_peaks_sorted = df_peaks.sort_values(by='duration').reset_index(drop=True)
durations = df_peaks_sorted['duration'][:-4]  # Exclude the last four entries

# Calculate the CDF for the modified durations
cdf = np.arange(1, len(durations) + 1) / len(durations)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(durations, cdf, marker='o', linestyle='-', color='b')

# Set x-axis ticks at intervals of 30 minutes up to the maximum duration
plt.xticks(np.arange(0, durations.max() + 30, 30),rotation=90)

# Set y-axis ticks at finer intervals (e.g., every 10%)
plt.yticks(np.arange(0, 1.1, 0.1))

# Labeling axes and title
plt.xlabel('Temperature Peak Duration in minutes', fontsize=12)
plt.ylabel('Cumulative Percentage of Points (value * 100)', fontsize=12)
plt.title('Cumulative Distribution Function of Duration', fontsize=12)

# Show the plot
plt.grid()
plt.show()

