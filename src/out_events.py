#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:14:38 2024

@author: hubble
"""
#%%
from utility import *
data_path = '/home/hubble/Downloads/Data_upTo_25Sep/DATA2'
#%%
# subject_dfs = read_csv_files_v2(data_path)
# with open('data_matteo_up_to_september_25_2024.pkl', 'wb') as file:
#     pickle.dump(subject_dfs, file)

with open('/home/hubble/work/serenade/src/data_matteo_up_to_september_25_2024.pkl', 'rb') as file:
    data = pickle.load(file)

#%%
subjects = list(data.keys())
years_months = {
    2023: [9,10,11,12],
    2024: [1,2,3,4,5,6,7,8,9]
}

years = list(years_months.keys())
subjectwise_count = {}
subjectwise_temperature_data = {}
subjectwise_humidity_data = {}

for sn in range(len(subjects)):
    subject_name = subjects[sn]#'subject_2'
    print(subject_name)
    data_subject_1 = data[subject_name]
    environmental_sensors = list(data_subject_1.keys())
    house_entrance = pd.DataFrame()
    if 'HouseEntrance' in environmental_sensors:
        house_entrance = data[subject_name]['HouseEntrance'][1]
        
    all_out_event_timings = {}
    all_temperature_peak_counts = {}
    humidity_data = {}
    
    for year in years:
        months = years_months[year]
        all_out_event_timings[year] = {}
        all_temperature_peak_counts[year] = {}
        
        for month in months:
            if len(house_entrance) > 0:
                out_counts, out_df, _ = count_virtual_out_events(house_entrance, data_subject_1, environmental_sensors, year = year, month = month, st=None, et=None)
                all_out_event_timings[year][month] = out_df
            else:
                out_counts = -99
                out_df = pd.to_datetime('1999-01-01 00:00:00')
                
            peak_count_temperature, _,_ ,df_with_temperature_peaks = detect_and_filter_peaks_by_time( data_subject_1, sensor_file= 'Stove_Hum_Temp_temp', prominence=1.5,  year=year, month=month,  st=None,  et=None )
            all_temperature_peak_counts[year][month] = df_with_temperature_peaks
            
    subjectwise_count[subject_name] = [all_temperature_peak_counts, all_out_event_timings]


# with open('outcome_matteos_data.pkl', 'wb') as file:
#     pickle.dump(subjectwise_count, file)

#%%

st = pd.to_datetime('06:00:00').time()
et = pd.to_datetime('09:59:00').time()

st = pd.to_datetime('11:00:00').time()
et = pd.to_datetime('14:59:00').time()

st = pd.to_datetime('18:00:00').time()
et = pd.to_datetime('23:59:00').time()

st = None
et = None

#%% 
# Compute count_out and temperature peaks for each subject
plot_df_all = {}

# Loop over each subject in subjectwise_count
for subject in list(subjectwise_count.keys()):
    print(f"Processing subject: {subject}")
    
    temp_subject = subjectwise_count[subject]
    outs = temp_subject[1]
    peaks_temperature = temp_subject[0]

    years = list(outs.keys())

    # Loop over years and months for each subject
    data_to_plot = []
    for year in years:
        outs_year = outs[year]
        peaks_year = peaks_temperature[year]
        months = list(outs_year.keys())
        
        for month in months:
            out_month = outs_year[month]
            peaks_month = peaks_year[month]

            # Process outs
            count_outs, outs_df = process_out_month(out_month, year, month, st=st, et=et, filter_duration=False)

            # Process peaks_temperature
            count_temperature_peaks, temperature_df = process_peaks_month(peaks_month, year, month, st=st, et=et)

            # Append the results to the final list for all subjects
            data_to_plot.append({
                'subject': subject,
                'year': year,
                'month': month,
                'count_outs': count_outs,
                'count_temperature_peaks': count_temperature_peaks
            })

    # Convert the collected data into a DataFrame
    df_to_plot = pd.DataFrame(data_to_plot)
    plot_df_all[subject] = df_to_plot

#%% Plot the graph
for subject in list(plot_df_all.keys()):
    if subject == 'subject_8':
        continue
    subject_data = plot_df_all[subject]
    plot_bar_for_subject(subject_data, subject)

#%%