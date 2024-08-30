#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:14:38 2024

@author: hubble
"""
#%%
import numpy as np, pandas as pd, os, pickle
from natsort import natsorted
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
#%%
# data_path = '/home/hubble/Downloads/pilot_data_download/code/DATA'
data_path = '/home/hubble/work/serenade/data'
subjects_threshold_dict = {
    'subject_1': {
        'PlugTvHall.csv': 30,
        'CoffeMachine.csv': None,
        'Microwave.csv': 1500,
        'WashingMachine.csv': 1300,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_2': {
        'PlugTvHall.csv': 60,
        'CoffeMachine.csv': None,
        'Microwave.csv': None,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_3': {
        'PlugTvHall.csv': 15,
        'CoffeMachine.csv': 800,
        'Microwave.csv': 1000,
        'WashingMachine.csv': None,
        'Printer.csv': 250,
        'PlugTvKitchen.csv': None
    },
    'subject_4': {
        'PlugTvHall.csv': None,
        'CoffeMachine.csv': None,
        'Microwave.csv': 1000,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_5': {
        'PlugTvHall.csv': 45,
        'CoffeMachine.csv': None,
        'Microwave.csv': 1200,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_7': {
        'PlugTvHall.csv': 15,
        'CoffeMachine.csv': None,
        'Microwave.csv': None,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': 15
    },
    'subject_8': {
        'PlugTvHall.csv': 30,
        'CoffeMachine.csv': 400,
        'Microwave.csv': None,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_9': {
        'PlugTvHall.csv': 60,
        'CoffeMachine.csv': 800,
        'Microwave.csv': None,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_11': {
        'PlugTvHall.csv': 30,
        'CoffeMachine.csv': None,
        'Microwave.csv': None,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': 15
    },
    'subject_12': {
        'PlugTvHall.csv': 20,
        'CoffeMachine.csv': None,
        'Microwave.csv': None,
        'WashingMachine.csv': None,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_13': {
        'PlugTvHall.csv': None,
        'CoffeMachine.csv': None,
        'Microwave.csv': 1000,
        'WashingMachine.csv': 1500,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
    },
    'subject_14': {
        'PlugTvHall.csv': 30,
        'CoffeMachine.csv': None,
        'Microwave.csv': None,
        'WashingMachine.csv': 1300,
        'Printer.csv': None,
        'PlugTvKitchen.csv': None
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
    
def read_csv_files(data_path):
    subjects = natsorted(os.listdir(data_path))
    
    subject_dfs = {}
    for s in range(len(subjects)):
        print('-*'*40)
        subject_s = subjects[s]
        path_subject = os.path.join(data_path, subject_s, 'environmentals')
        environmentals_sensors = natsorted(os.listdir(path_subject))
        
        dfs_dict = {}
        for es in range(len(environmentals_sensors)):
            filename = environmentals_sensors[es]
            if '_events' in filename:
                continue
            
            if filename.endswith('.csv') and not filename.startswith('.~lock'):
                print(os.path.join(path_subject, filename))
                df = pd.read_csv(os.path.join(path_subject, filename))
                if len(df) > 0:
                    df = df[~df['sensor_status'].isin(['unavailable', 'unknown'])]               
                    df, isEnvironmentalSensor = remove_continuous_on_off(df)
                    df['ts_datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    
                    if not isEnvironmentalSensor:
                        if filename in ['Shower_Hum_Temp_temp.csv', 'Shower_Hum_Temp_humidity.csv', 'Stove_Hum_Temp_temp.csv', 'Stove_Hum_Temp_humidity.csv']:
                            df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
                            dfs_dict[filename] = [df, pd.DataFrame()]
                        else:   
                            threshold_value = get_threshold_value(subjects_threshold_dict, subject_s, filename)
                            df['sensor_status'] = pd.to_numeric(df['sensor_status'], errors='coerce')
                            df = convert_real_valued_to_on_off(threshold_value, df)  
                            df_processed = process_sensor_data(df)
                            dfs_dict[filename] = [df, df_processed]
                    else:
                        df_processed = process_sensor_data(df)
                        dfs_dict[filename] = [df, df_processed]
        subject_dfs[subject_s] = dfs_dict
    return subject_dfs

def count_virtual_out_events(house_entrance, data_subject_1, year = None, month = None, st = None, et = None):
    if year != None and month != None:
        # print('+-'*20)
        house_entrance = house_entrance[(house_entrance['ts_on'].dt.year == year) & (house_entrance['ts_on'].dt.month == month)]
        if st != None and et != None:
            # print('xxxxxxxxxxxxxx')
            house_entrance = house_entrance[
                (house_entrance['ts_on'].dt.time >= st) &
                (house_entrance['ts_on'].dt.time < et)]   
        # print(house_entrance)
    out_event_timimg = []
    out_event_count = 0
    for he in range(1, len(house_entrance)):
        # print('-------------------------------')
        ws = house_entrance.iloc[he-1]['ts_off']
        we = house_entrance.iloc[he]['ts_on']
        # print(he, ws, we)
        is_out_event = True
        
        for j in range(len(environmental_sensors)):
            sensor_name = environmental_sensors[j]
            # print(j ,sensor_name)
            if sensor_name not in ['HouseEntrance.csv','Hum_Temp_Bath_humidity.csv', 'Hum_Temp_Bath_temp.csv', 'Hum_Temp_Stove_humidity.csv', 'Hum_Temp_Stove_temp.csv']: 
                sensor_df_raw = data_subject_1[sensor_name][0]
                sensor_df_processed = data_subject_1[sensor_name][1]
                
                ## I can not work on the sensors which provide real values because they are being utilized or not,
                ## they will always provide some real values. Instead I have to utilize on those sensors which provide values
                ## like on and off
                
                if len(sensor_df_processed) > 0:
                    temp_df = sensor_df_processed[(sensor_df_processed['ts_on'] >= ws) & (sensor_df_processed['ts_on'] <= we)]
                    # print(sensor_name, len(temp_df))
                    if not temp_df.empty:
                        is_out_event = False
                        # print('----------')
                        break
                    
        if is_out_event == True:
            out_event_count = out_event_count + 1
            out_event_timimg.append((he,ws,we))
            
    return out_event_count, out_event_timimg
      

def detect_peaks(df, value_col, prominence=1.5):
    if value_col not in df.columns:
        return [], df 
    df = df.copy()
    peaks, properties = find_peaks(df[value_col], prominence=prominence)
    peak_indicator = [0] * len(df)
    for peak in peaks:
        if peak < len(peak_indicator):
            peak_indicator[peak] = 1
    df['peak_detected'] = peak_indicator
    return peaks,df

def count_peaks(data_subject_1, sensor_file, year = None, month = None, prominence = 1.5, st=None, et = None):
    if sensor_file in data_subject_1:
        temperatue_data = data_subject_1[sensor_file][0]    
        if year != None and month != None:
            temperatue_data = temperatue_data[(temperatue_data['ts_datetime'].dt.year == year) & (temperatue_data['ts_datetime'].dt.month == month)]
            if st != None and et !=None:
                print('xxxxxxxxxxxx')
                temperatue_data = temperatue_data[
                    (temperatue_data['ts_datetime'].dt.time >= st) &
                    (temperatue_data['ts_datetime'].dt.time < et)] 
            # print(temperatue_data)
            
        peaks, df_temperature_data= detect_peaks(temperatue_data, 'sensor_status', prominence=prominence)        
        peak_count = len(peaks)
    else:
        peak_count = 0
        peaks = []
        df_temperature_data = pd.DataFrame()
    return peak_count, peaks, df_temperature_data
    
    
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
    plt.savefig(f'{subject_name}_{year}_{month}.png')
    # Display the plot
    plt.show()

def plot_event_and_peak_counts(all_out_event_counts, all_temperature_peak_counts, all_humidity_peak_counts, subject_name, strng):
    months_years = []
    out_event_counts = []
    temperature_peak_counts = []
    humidity_peak_counts = []

    # Iterate over years and months, combining all counts
    for year in sorted(set(all_out_event_counts.keys()).union(all_temperature_peak_counts.keys()).union(all_humidity_peak_counts.keys())):
        for month in sorted(set(all_out_event_counts.get(year, {}).keys())
                            .union(all_temperature_peak_counts.get(year, {}).keys())
                            .union(all_humidity_peak_counts.get(year, {}).keys())):
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
    bars3 = ax.bar(x + width, humidity_peak_counts, width, label='Humidity Peaks', color='green')

    # Customize the plot
    ax.set_xlabel('Month-Year', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(f'Monthly Probable Out Events, Temperature Peaks, and Humidity Peaks for {subject_name}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(months_years, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{subject_name}_{strng}_events_peaks.png')
    plt.show()


#%%
# subject_data_june2024_august2024
# subject_data_sepetember2023_may2024

## One time run only 
# subject_dfs = read_csv_files(data_path)
# with open('subject_data_sepetember2023_may2024.pkl', 'wb') as file:
#     pickle.dump(subject_dfs, file)

# with open('subject_data_june2024_august2024.pkl', 'rb') as file:
#     data = pickle.load(file)


with open('subject_data_june2024_august2024.pkl', 'rb') as file:
    data = pickle.load(file)
#%%
subjects = list(data.keys())
years_months = {
    2023: [9,10,11,12],
    2024: [1,2,3,4,5,6,7,8]
}

st = None#pd.to_datetime('18:00:00').time()
et = None#pd.to_datetime('23:59:00').time()

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
                out_counts, out_timings = count_virtual_out_events(house_entrance, data_subject_1, year = year, month = month, st=st, et=et)
                all_out_event_counts[year][month] = out_counts
                all_out_event_timings[year][month] = out_timings
                
            else:
                out_counts = 0
                out_timings = pd.to_datetime('2023-01-01 10:00:00')
                
            peak_count_temperature, peaks_temperature, df_with_temperature_peaks = count_peaks( data_subject_1, sensor_file = 'Stove_Hum_Temp_temp.csv', year = year, month = month,  prominence=1.5,st=st,et=et)
            peak_count_humidity, peaks_humidity, df_with_humidity_peaks = count_peaks(data_subject_1, sensor_file = 'Stove_Hum_Temp_humidity.csv', year = year, month = month, prominence=15,st=st,et=et)
            all_temperature_peak_counts[year][month] = peak_count_temperature
            all_humidity_peak_counts[year][month] = peak_count_humidity
            
            if len(df_with_humidity_peaks) > 0 and len(df_with_temperature_peaks) > 0:
                plot_signal_with_peaks(df_with_humidity_peaks, peaks_humidity, df_with_temperature_peaks, peaks_temperature, 'temperature peaks', 'humidity peaks', subject_name, year, month)
            
    # plot_event_and_peak_counts(all_out_event_counts, all_temperature_peak_counts, all_humidity_peak_counts, subject_name, strng = 'whole_day')
    # subjectwise_count[subject_name] = [all_temperature_peak_counts,all_humidity_peak_counts,all_out_event_counts]


# with open('dinner_count_june2024_august2024.pkl', 'wb') as pickle_file:
#     pickle.dump(subjectwise_count, pickle_file)
    
# with open('count_june2024_august2024.pkl', 'rb') as file:
#     data1 = pickle.load(file)
    
# with open('count_sepetember2023_may2024.pkl', 'rb') as file:
#     data2 = pickle.load(file)    
    
    
# def merge_data_dicts(data1, data2, temp):
#     merged_data = {}

#     # Get all unique keys from both data1 and data2
#     all_keys = set(data1.keys()).union(set(data2.keys()))

#     for key in all_keys:
#         if key in data1 and key in data2:  # If the key is present in both dictionaries
#             merged_dict = {}

#             dict1 = data1[key][temp]
#             dict2 = data2[key][temp]  

#             # Sum the values for each year and month where both dict1 and dict2 have values
#             for year, months_dict in dict1.items():
#                 merged_dict[year] = {}
#                 for month, value in months_dict.items():
#                     merged_value = value + dict2.get(year, {}).get(month, 0)
#                     merged_dict[year][month] = merged_value

#             # Place the merged_dict in the first position of the list, keeping the rest from data1
#             merged_data[key] = merged_dict

#         elif key in data1 and key not in data2:  # If the key is only in data1, include it as is
#             merged_data[key] = data1[key][temp]

#         elif key in data2 and key not in data1:  # If the key is only in data2, include it as is
#             merged_data[key] = data2[key][temp]

#     return merged_data

# # Example usage:
# # Assuming data1 and data2 are already defined dictionaries
# merged_temp_peak = merge_data_dicts(data1, data2, 0)
# merged_humid_peak = merge_data_dicts(data1, data2, 1)
# merged_out_peak = merge_data_dicts(data1, data2, 2)


# for subject_name in merged_temp_peak.keys():
#     # Extract the peak counts and out event counts for the current subject
#     all_temperature_peak_counts = merged_temp_peak[subject_name]
#     all_humidity_peak_counts = merged_humid_peak[subject_name]
#     all_out_event_counts = merged_out_peak[subject_name]
    
#     # Call the plotting function for the current subject
#     plot_event_and_peak_counts(
#         all_out_event_counts,
#         all_temperature_peak_counts,
#         all_humidity_peak_counts,
#         subject_name,  strng = 'wholeday'
#     )



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
                    
          