#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:14:38 2024

@author: hubble
"""
#%%
from utility import *
# data_path = '/home/hubble/Downloads/pilot_data_download/code/DATA'
data_path = '/home/hubble/work/serenade/data'
#%%
# subject_data_june2024_august2024
# subject_data_sepetember2023_may2024

## One time run only 
# subject_dfs = read_csv_files(data_path)
# with open('subject_data_sepetember2023_may2024_2hours.pkl', 'wb') as file:
#     pickle.dump(subject_dfs, file)

# with open('results/subject_data_june2024_august2024.pkl', 'rb') as file:
#     data = pickle.load(file)


with open('/home/hubble/work/serenade/src/subject_data_sepetember2023_may2024_2hours.pkl', 'rb') as file:
    data = pickle.load(file)
#%%
subjects = list(data.keys())
years_months = {
    2023: [9,10,11,12],
    2024: [1,2,3,4,5,6,7,8]
}

st = pd.to_datetime('18:00:00').time()
et = pd.to_datetime('23:59:00').time()

st = None
et = None

years = list(years_months.keys())
subjectwise_count = {}
subjectwise_temperature_data = {}
subjectwise_humidity_data = {}

for sn in range(len(subjects)):
    subject_name = subjects[sn]#'subject_2'
    data_subject_1 = data[subject_name]
    environmental_sensors = list(data_subject_1.keys())
    house_entrance = pd.DataFrame()
    if 'HouseEntrance.csv' in environmental_sensors:
        house_entrance = data[subject_name]['HouseEntrance.csv'][1]
        
    all_out_event_counts = {}
    all_out_event_timings = {}
    all_temperature_peak_counts = {}
    all_humidity_peak_counts = {}
    temperature_data = {}
    humidity_data = {}
    
    for year in years:
        months = years_months[year]
        all_out_event_counts[year] = {}
        all_out_event_timings[year] = {}
        all_temperature_peak_counts[year] = {}
        all_humidity_peak_counts[year] = {}
        temperature_data[year] = {}
        humidity_data[year] = {}
        
        for month in months:
            if len(house_entrance) > 0:
                
                out_counts, out_timings, __ = count_virtual_out_events(house_entrance, data_subject_1, year = year, month = month, st=st, et=et)
                all_out_event_counts[year][month] = out_counts
                all_out_event_timings[year][month] = out_timings
                
            else:
                out_counts = 0
                out_timings = pd.to_datetime('2023-01-01 10:00:00')
                
            peak_count_temperature, peaks_temperature, df_with_temperature_peaks = count_peaks( data_subject_1, sensor_file = 'Stove_Hum_Temp_temp.csv', year = year, month = month,  prominence=1.5,st=st,et=et)
            peak_count_humidity, peaks_humidity, df_with_humidity_peaks = count_peaks(data_subject_1, sensor_file = 'Stove_Hum_Temp_humidity.csv', year = year, month = month, prominence=15,st=st,et=et)
            all_temperature_peak_counts[year][month] = peak_count_temperature
            all_humidity_peak_counts[year][month] = peak_count_humidity
            
            # if len(df_with_humidity_peaks) > 0 and len(df_with_temperature_peaks) > 0:
            #     plot_signal_with_peaks(df_with_humidity_peaks, peaks_humidity, df_with_temperature_peaks, peaks_temperature, 'temperature peaks', 'humidity peaks', subject_name, year, month)
            
    plot_event_and_peak_counts(all_out_event_counts, all_temperature_peak_counts, all_humidity_peak_counts, subject_name, strng = 'whole_day')
    subjectwise_count[subject_name] = [all_temperature_peak_counts,all_humidity_peak_counts,all_out_event_counts]


# with open('withouttiny_dinner_count_sepetember2023_may2024.pkl', 'wb') as pickle_file:
#     pickle.dump(subjectwise_count, pickle_file)

# with open('withouttiny_breakfast_count_june2024_august2024.pkl', 'wb') as pickle_file:
#     pickle.dump(subjectwise_count, pickle_file)
    
with open('results/withouttiny_dinner_count_june2024_august2024.pkl', 'rb') as file:
    data1 = pickle.load(file)
    
with open('results/withouttiny_dinner_count_sepetember2023_may2024.pkl', 'rb') as file:
    data2 = pickle.load(file)    
    
    
def merge_data_dicts(data1, data2, temp):
    merged_data = {}

    # Get all unique keys from both data1 and data2
    all_keys = set(data1.keys()).union(set(data2.keys()))

    for key in all_keys:
        if key in data1 and key in data2:  # If the key is present in both dictionaries
            merged_dict = {}

            dict1 = data1[key][temp]
            dict2 = data2[key][temp]  

            # Sum the values for each year and month where both dict1 and dict2 have values
            for year, months_dict in dict1.items():
                merged_dict[year] = {}
                for month, value in months_dict.items():
                    merged_value = value + dict2.get(year, {}).get(month, 0)
                    merged_dict[year][month] = merged_value

            # Place the merged_dict in the first position of the list, keeping the rest from data1
            merged_data[key] = merged_dict

        elif key in data1 and key not in data2:  # If the key is only in data1, include it as is
            merged_data[key] = data1[key][temp]

        elif key in data2 and key not in data1:  # If the key is only in data2, include it as is
            merged_data[key] = data2[key][temp]

    return merged_data

# Example usage:
# Assuming data1 and data2 are already defined dictionaries
merged_temp_peak = merge_data_dicts(data1, data2, 0)
merged_humid_peak = merge_data_dicts(data1, data2, 1)
merged_out_peak = merge_data_dicts(data1, data2, 2)


for subject_name in merged_temp_peak.keys():
    # Extract the peak counts and out event counts for the current subject
    all_temperature_peak_counts = merged_temp_peak[subject_name]
    all_humidity_peak_counts = merged_humid_peak[subject_name]
    all_out_event_counts = merged_out_peak[subject_name]
    
    # Call the plotting function for the current subject
    plot_event_and_peak_counts(
        all_out_event_counts,
        all_temperature_peak_counts,
        all_humidity_peak_counts,
        subject_name,  strng = 'dinner_withouttiny'
    )






years = [2023, 2024]
results = {}

for year in years:
    results[year] = {}
    data_year = all_out_event_timings[year]  # Corrected from 2024 to year

    months = list(data_year.keys())
    for month in months:
        tiny_count = 0
        short_count = 0
        medium_count = 0
        long_count = 0

        month_data = data_year[month]
        if len(month_data) > 0:
            for md in range(len(month_data)):
                duration = ((month_data[md][2] - month_data[md][1]).total_seconds()) / 60
                if duration < 10:
                    tiny_count += 1
                elif duration < 60:
                    short_count += 1
                elif duration < 240:
                    medium_count += 1
                else:
                    long_count += 1
        
        # Store the counts for each month under the corresponding year
        results[year][month] = {
            "TINY": tiny_count,
            "SHORT": short_count,
            "MEDIUM": medium_count,
            "LONG": long_count
        }

# Now results will have a structure like:
# {
#     2023: {
#         1: {"TINY": ..., "SHORT": ..., "MEDIUM": ..., "LONG": ...},
#         2: {"TINY": ..., "SHORT": ..., "MEDIUM": ..., "LONG": ...},
#         ...
#     },
#     2024: {
#         1: {"TINY": ..., "SHORT": ..., "MEDIUM": ..., "LONG": ...},
#         2: {"TINY": ..., "SHORT": ..., "MEDIUM": ..., "LONG": ...},
#         ...
#     }
# }





years = [2023, 2024]
results = {}
tiny_count = 0
short_count = 0
medium_count = 0
long_count = 0
for year in years:
    results[year] = {}
    data_year = all_out_event_timings[2024]

    months = list(data_year.keys())
    for month in months:
        month_data = data_year[month]
        if len(month_data) > 0:
            for md in range(len(month_data)):
                duration = ((month_data[md][2] - month_data[md][1]).total_seconds())/60
                if duration < 10:
                    tiny_count += 1
                elif duration < 60:
                    short_count += 1
                elif duration < 240:
                    medium_count += 1
                else:
                    long_count += 1
        results[year] = {
            "TINY": tiny_count,
            "SHORT": short_count,
            "MEDIUM": medium_count,
            "LONG": long_count
        }
       





years = [2023, 2024]
results = {}
tiny_count = 0
short_count = 0
medium_count = 0
long_count = 0
for year in years:
    results[year] = {}
    data_year = all_out_event_timings[2024]

    months = list(data_year.keys())
    for month in months:
        month_data = data_year[month]
        if len(month_data) > 0:
            for md in range(len(month_data)):
                duration = ((month_data[md][2] - month_data[md][1]).total_seconds())/60
                if duration < 10:
                    tiny_count += 1
                elif duration < 60:
                    short_count += 1
                elif duration < 240:
                    medium_count += 1
                else:
                    long_count += 1
    results[year] = {
        "TINY": tiny_count,
        "SHORT": short_count,
        "MEDIUM": medium_count,
        "LONG": long_count
    }
                    
    

data_subject_2 = data['subject_2']
environmental_sensors = list(data_subject_2.keys())
concatenated_df = pd.DataFrame()


def plot_sensor_active_periods(data_subject_2, environmental_sensors, st, et):
    # Define your start and end dates using pd.to_datetime()
    start_date = st#pd.to_datetime('2023-07-21 00:00:00')
    end_date = et#pd.to_datetime('2023-07-26 00:00:00')
    concatenated_df_1 = pd.DataFrame()
    df_combined_non_temp_humidity = []
    df_combined_temp_humidity = {}
    for sens in range(len(environmental_sensors)):
        sensor_name = environmental_sensors[sens]
        if sensor_name in ['Shower_Hum_Temp_humidity.csv','Shower_Hum_Temp_temp.csv','Stove_Hum_Temp_humidity.csv','Stove_Hum_Temp_temp.csv']:
            df_sensor = data_subject_2[sensor_name][0]
            df_sensor['ts_datetime'] = pd.to_datetime(df_sensor['ts_datetime'])
            filtered_df = df_sensor[(df_sensor['ts_datetime'] >= start_date) & (df_sensor['ts_datetime'] <= end_date)]
            sub_df_sensor = filtered_df[['sensor_id', 'ts_datetime', 'sensor_status']]
            df_combined_temp_humidity[sensor_name] = sub_df_sensor
            sub_df_sensor = pd.DataFrame()
        else:
            df_sensor = data_subject_2[sensor_name][1]
            df_sensor['ts_on'] = pd.to_datetime(df_sensor['ts_on'])
            df_sensor['ts_off'] = pd.to_datetime(df_sensor['ts_off'])
            filtered_df = df_sensor[(df_sensor['ts_on'] >= start_date) & (df_sensor['ts_off'] <= end_date)]
            sub_df_sensor = filtered_df[['sensor_id', 'ts_on', 'ts_off']]
            df_combined_non_temp_humidity.append(sub_df_sensor)
            sub_df_sensor = pd.DataFrame()
    df_combined_non_temp_humidity = pd.concat(df_combined_non_temp_humidity, ignore_index=True)
    # df_combined_temp_humidity = pd.concat(df_combined_temp_humidity, ignore_index=True)
       

    # Create a timeline plot
    fig = px.timeline(df_combined_non_temp_humidity, x_start='ts_on', x_end='ts_off', y='sensor_id', color='sensor_id',
                      labels={'sensor_id': 'Sensor ID'})
    
    # Customize the layout
    fig.update_layout(
        title="Sensor Active Periods Over Time",
        xaxis_title="Time",
        yaxis_title="Sensor ID",
    )
    
    temp_humdity_sensor_names = list(df_combined_temp_humidity.keys())
    for sensor in temp_humdity_sensor_names:
        sensor_df = df_combined_temp_humidity[sensor]
        
        fig.add_trace(
            go.Scatter(
                x=sensor_df['ts_datetime'],  # Timestamp for x-axis
                y=[sensor] * len(sensor_df),  # Repeat the sensor name for y-axis to align with the existing timeline
                mode='markers+lines',
                name=f'{sensor}'[:-4],
                marker=dict(size=8),  # You can customize the size of the marker
                text=sensor_df['sensor_status'],  # Use sensor_status for hover text
                hoverinfo='text',  # Show only the text in hover
                line=dict(dash='dash')  # Dashed line for distinction
            )
        )

    # Show the combined plot
    fig.show()



st = pd.to_datetime('2023-07-21 00:00:00')
et = pd.to_datetime('2023-07-26 00:00:00')

plot_sensor_active_periods(data_subject_2, environmental_sensors, st, et)

 
# Example data for two sensors with start and end times
data2 = {
    'sensor_name': ['Sensor 1'] * 10 + ['Sensor 2'] * 10,  # Sensor names
    't1': pd.date_range(start='2023-09-01 00:00:00', periods=10, freq='H').tolist() +
          pd.date_range(start='2023-09-02 00:00:00', periods=10, freq='H').tolist(),  # Start times
    't2': pd.date_range(start='2023-09-01 01:00:00', periods=10, freq='H').tolist() +
          pd.date_range(start='2023-09-02 01:00:00', periods=10, freq='H').tolist(),  # End times
}

# Create a DataFrame
df = pd.DataFrame(data2)

# Create a timeline plot
fig = px.timeline(df, x_start='t1', x_end='t2', y='sensor_name', color='sensor_name',
                  labels={'sensor_name': 'Sensor Name'})

# Customize the layout
fig.update_layout(
    title="Sensor Active Periods Over Time",
    xaxis_title="Time",
    yaxis_title="Sensor Name",
)

# Show the plot
fig.show()