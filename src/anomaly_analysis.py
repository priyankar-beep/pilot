#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:32:57 2024

@author: hubble
"""

from utility_anomaly_analysis import *
subject_id = 'subject_3'
print(data[subject_id].keys())
df = data[subject_id]['Stove_Hum_Temp_temp'][0].copy()
specific_devices = ['Shower_Hum_Temp', 'Stove_Hum_Temp','Stove_Hum_Temp_temp', 'Stove_Hum_Temp_humidity', 'Shower_Hum_Temp_temp', 'Shower_Hum_Temp_humidity']

with open('/home/hubble/work/serenade/data/data_matteo_upto_september_25_2024_corrected.pkl', 'rb') as file:
    data = pickle.load(file)
# subject_id = 'subject_2'
# print(data[subject_name].keys())

# arranged_np_data, complete_days, binarized_arranged_np_data = arrange_data_by_day_numpy(df)
# df = data[subject_id]['MotionLivingroom'][1].copy()
# arranged_np_data = arrange_data_by_day_numpy_environmentals(df)
# filtered_df = df[~((df['ts_datetime'].dt.month.isin([7, 8])) & (df['ts_datetime'].dt.year == 2023))]

dta = []
for sd in range(len(specific_devices)):
    print(specific_devices[sd])
    df = data[subject_id][specific_devices[sd]][0].copy()
    arranged_np_data, complete_days, binarized_arranged_np_data, datewise_usage = arrange_data_by_day_numpy(df)
    filtered_df = filter_rows_by_time_and_date(datewise_usage, '11:00:00', '14:59:00', 2024, 1)
    if specific_devices[sd] == 'Stove_Hum_Temp_temp':        
        sensor_id_mapping = {'Stove_Hum_Temp': 'Stove_Temp' }
        
    elif specific_devices[sd] == 'Stove_Hum_Temp_humidity':
        sensor_id_mapping = {'Stove_Hum_Temp': 'Stove_Humidity' }
        
    elif specific_devices[sd] == 'Shower_Hum_Temp_temp':
        sensor_id_mapping = {'Shower_Hum_Temp': 'Shower_Temp' }    
        
    elif specific_devices[sd] == 'Shower_Hum_Temp_humidity':
        sensor_id_mapping = {'Shower_Hum_Temp': 'Shower_Humitidy' }    
    
    filtered_df.loc[:, 'sensor_id'] = filtered_df['sensor_id'].replace(sensor_id_mapping)
    dta.append(filtered_df)
   

hum_temp_df = pd.concat(dta, axis=0, ignore_index=True)
device_dict = data[subject_id]
environmentals_df = filter_and_merge_data(device_dict, '11:00:00', '14:59:00', 2024, 1)
combined_df = pd.concat([hum_temp_df, environmentals_df], ignore_index=True)
sorted_df = combined_df.sort_values(by='ts_on')
daily_data_dict = split_by_day(sorted_df)

#%%
## Analyze the specific months and plot a specific date
df = daily_data_dict['23_02_2024'].copy()
plt.figure(figsize=(14, 6))
colors = plt.cm.viridis(np.linspace(0, 1, df['sensor_id'].nunique()))
color_mapping = dict(zip(df['sensor_id'].unique(), colors))
sensor_to_num = {sensor: idx for idx, sensor in enumerate(sorted(df['sensor_id'].unique()))}
for sensor in df['sensor_id'].unique():
    # Filter DataFrame for the current sensor
    sensor_df = df[df['sensor_id'] == sensor]
    
    # Draw solid lines for activation periods
    for _, row in sensor_df.iterrows():
        plt.hlines(y=sensor_to_num[sensor], xmin=row['ts_on'], xmax=row['ts_off'],
                    colors=color_mapping[sensor], linewidth=6)

# Set x-ticks to show full datetime from min ts_on to max ts_off
plt.xticks(rotation=90)
plt.xlim(df['ts_on'].min(), df['ts_off'].max())
plt.xlabel('Time of Day')
plt.title('Sensor Activations Over Time')
plt.grid()
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=15))  # Change interval to 15 minutes
plt.yticks(ticks=list(sensor_to_num.values()), labels=list(sensor_to_num.keys()))
plt.tight_layout()
plt.show()

#%%



def get_items_used_around_stove(df, stove_temp_id='Stove_Temp', stove_humidity_id='Stove_Humidity', time_window=20):   
    stove_temp = df[df['sensor_id'] == stove_temp_id]
    stove_humidity = df[df['sensor_id'] == stove_humidity_id]

    if not stove_temp.empty:
        stove_activations = stove_temp[['ts_on', 'ts_off']]
    elif not stove_humidity.empty:
        stove_activations = stove_humidity[['ts_on', 'ts_off']]
    else:
        return pd.DataFrame()  # No stove activation found

    stove_activations = stove_activations.drop_duplicates().sort_values(by='ts_on')

    
    combined_items = pd.DataFrame()  # Initialize an empty DataFrame to collect results

    for index, stove_event in stove_activations.iterrows():
        stove_activation_time = stove_event['ts_on']
        stove_deactivation_time = stove_event['ts_off']

        # Define the time window for items used before stove activation
        time_window_start = stove_activation_time - pd.Timedelta(minutes=time_window)

        # Filter for items used before stove activation
        items_used_before_stove = df[(df['ts_off'] >= time_window_start) & (df['ts_on'] < stove_activation_time)]

        # Filter for items used during stove activation
        items_used_during_stove = df[(df['ts_on'] < stove_deactivation_time) & (df['ts_off'] > stove_activation_time)]

        # Combine the two DataFrames for this stove event
        combined_items_event = pd.concat([
            items_used_before_stove[['sensor_id', 'ts_on', 'ts_off']],
            items_used_during_stove[['sensor_id', 'ts_on', 'ts_off']]
        ], ignore_index=True)

        # Add the combined items for this event to the overall combined_items DataFrame
        combined_items = pd.concat([combined_items, combined_items_event], ignore_index=True)

    combined_items = combined_items.drop_duplicates()
    combined_items = combined_items.sort_values(by='ts_on')

    return combined_items

# def get_items_used_before_stove(df, stove_temp_id='Stove_Temp', stove_humidity_id='Stove_Humidity', time_window=20):
#     # Identify stove activation time
#     stove_temp = df[df['sensor_id'] == stove_temp_id]
#     stove_humidity = df[df['sensor_id'] == stove_humidity_id]
#     stove_sensor = None
#     if not stove_temp.empty:
#         stove_activation_time = stove_temp['ts_on'].min()
#         stove_sensor = stove_temp.loc[stove_temp['ts_on'].idxmin()] 
#     elif not stove_humidity.empty:
#         stove_activation_time = stove_humidity['ts_on'].min()
#         stove_sensor = stove_humidity.loc[stove_humidity['ts_on'].idxmin()] 
#     else:
#         # If no stove activation found, compute the first activated sensor's time and the last off time
#         first_activated_sensor = df['ts_on'].min()
#         last_off_sensor = df['ts_off'].max()

#         if first_activated_sensor is not None and last_off_sensor is not None:
#             # Filter for sensors used between the first activation and the last deactivation
#             items_used_before_stove = df[(df['ts_on'] >= first_activated_sensor) & (df['ts_off'] <= last_off_sensor)]
#             items_used_before_stove = items_used_before_stove.sort_values(by='ts_on')
#             return items_used_before_stove[['sensor_id', 'ts_on', 'ts_off']]
#         else:
#             return pd.DataFrame()
    
#     if stove_activation_time is not None:
#         time_window_start = stove_activation_time - pd.Timedelta(minutes=time_window)
#         items_used_before_stove = df[(df['ts_off'] >= time_window_start) & (df['ts_on'] < stove_activation_time)]
#         items_used_before_stove = items_used_before_stove.sort_values(by='ts_on')
#         if stove_sensor is not None and stove_sensor['sensor_id'] not in items_used_before_stove['sensor_id'].values:
#             stove_sensor_df = pd.DataFrame([stove_sensor])  
#             items_used_before_stove = pd.concat([items_used_before_stove, stove_sensor_df], ignore_index=True)
         
#         return items_used_before_stove[['sensor_id', 'ts_on', 'ts_off']]
#     else:
#         print("No stove activation found.")
#         return pd.DataFrame()
    
def get_items_used_before_and_during_stove(df, stove_temp_id='Stove_Temp', stove_humidity_id='Stove_Humidity', time_window=10):
    # Identify stove activation time
    stove_temp = df[df['sensor_id'] == stove_temp_id]
    stove_humidity = df[df['sensor_id'] == stove_humidity_id]
    stove_sensor = None
    
    # Find the earliest stove activation time and deactivation time
    if not stove_temp.empty:
        stove_activation_time = stove_temp['ts_on'].min()
        stove_deactivation_time = stove_temp['ts_off'].max()
        stove_sensor = stove_temp.loc[stove_temp['ts_on'].idxmin()] 
    elif not stove_humidity.empty:
        stove_activation_time = stove_humidity['ts_on'].min()
        stove_deactivation_time = stove_humidity['ts_off'].max()
        stove_sensor = stove_humidity.loc[stove_humidity['ts_on'].idxmin()] 
    else:
        # If no stove activation found, compute the first activated sensor's time and the last off time
        first_activated_sensor = df['ts_on'].min()
        last_off_sensor = df['ts_off'].max()

        if first_activated_sensor is not None and last_off_sensor is not None:
            # Filter for sensors used between the first activation and the last deactivation
            items_used = df[(df['ts_on'] >= first_activated_sensor) & (df['ts_off'] <= last_off_sensor)]
            items_used = items_used.sort_values(by='ts_on')
            return items_used[['sensor_id', 'ts_on', 'ts_off']]
        else:
            return pd.DataFrame()
    
    # Ensure we have a stove activation time
    if stove_activation_time is not None and stove_deactivation_time is not None:
        time_window_start = stove_activation_time - pd.Timedelta(minutes=time_window)
        
        # Filter for sensors used before the stove activation
        items_used_before_stove = df[(df['ts_off'] >= time_window_start) & (df['ts_on'] < stove_activation_time)]
        
        # Filter for sensors used during the stove activation using overlap
        items_used_during_stove = df[(df['ts_on'] < stove_deactivation_time) & 
                                      (df['ts_off'] > stove_activation_time)]
        
        # Combine both filtered DataFrames
        combined_items = pd.concat([items_used_before_stove, items_used_during_stove], ignore_index=True)
        combined_items = combined_items.drop_duplicates()
        combined_items = combined_items.sort_values(by='ts_on')

        
        # # Ensure stove sensor is included as the last element if it's not already in the list
        # if stove_sensor is not None and stove_sensor['sensor_id'] not in combined_items['sensor_id'].values:
        #     stove_sensor_df = pd.DataFrame([stove_sensor])  
        #     combined_items = pd.concat([combined_items, stove_sensor_df], ignore_index=True)

        # Sort combined items by timestamp
        combined_items = combined_items.sort_values(by='ts_on')
        
        return combined_items[['sensor_id', 'ts_on', 'ts_off']]
    else:
        print("No stove activation found.")
        return pd.DataFrame()    
    
dates = list(daily_data_dict.keys())
device_sequence = {}

# Step 1: Process each day and collect device usage
for d in range(len(dates)):
    df = daily_data_dict[dates[d]].copy()
    items_used_before_stove = get_items_used_before_and_during_stove(df)
    merged_df = merge_continuous_sensor_occurrences(items_used_before_stove)
    if not merged_df.empty:
        merged_df = merged_df.sort_values(by='ts_on')
    
    if len(merged_df) == 0:
        device_sequence[dates[d]] = []
    else:
        print(dates[d])
        device_sequence[dates[d]] = merged_df['sensor_id'].tolist()

# Step 2: Create a report of device usage frequency for each day
daily_device_counts = {}

for date, devices in device_sequence.items():
    device_counts = Counter(devices)  # Count occurrences of each device for the day
    sorted_device_counts = dict(device_counts.most_common())  # Sort by frequency in decreasing order
    daily_device_counts[date] = sorted_device_counts

# Step 3: Print or use the daily device counts report
for date, counts in daily_device_counts.items():
    print(f"Date: {date}, Device Counts: {counts}")

    
from collections import Counter

