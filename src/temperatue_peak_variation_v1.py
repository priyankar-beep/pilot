#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:42:04 2024

@author: hubble
"""

from main_utility import *
from subject_kitchen_device_mapping import *

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

    subject_index_mapping = {
        'subject_1': 0,
        'subject_2': 1,
        'subject_3': 2,
        'subject_4': 3,
        'subject_5': 4,
        'subject_7': 5,
        'subject_8': 6,
        'subject_9': 7,
        'subject_10': 8,
        'subject_11': 9,
        'subject_12': 10,
        'subject_13': 11,
        'subject_14': 12,
    }
    
    # categories = {
    #     "within_1_sigma": [],
    #     "between_1_and_2_sigma": [],
    #     "between_2_and_3_sigma": [],
    #     "above_3_sigma": []
    # }
    
    all_devices = []
    for sub in range(len(subjects)):
        subject = subjects[sub]
        subject_keys = list(data[subject].keys())  
        all_devices.extend(subject_keys)

    pth = '/home/hubble/temp/between_2_and_3_sigma'
    subject_results = {}
    for sub in range(len(subjects)):
        subject = subjects[sub]  # Name of the subject
        print(subject)
        data_subject = data[subject]  # Load the whole data    
        df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0]  # Load subject's temperature data in the kitchen
        # Convert the time to datetime format with Europe/Rome Timezone
        df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc=True)  
        df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
        
       
        # Start and end date of temperature data collection
        data_start_date = df_stove_temperature['ts_datetime'].min()  # Not useful right now
        data_end_date = df_stove_temperature['ts_datetime'].max()  # Not useful right now
        result = analyze_seasonal_peaks_with_duration(df_stove_temperature)
        plot_peaks_and_mean_temperature(df_stove_temperature, '/home/hubble/temp')
        plot_duration_peaks(result, '/home/hubble/temp', subject)
        df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
        daily_stats = df_stove_temperature.groupby('date')['sensor_status'].agg(
            daily_avg_temperature='mean',
            daily_median_temperature='median',
            daily_std_temperature='std'
        ).reset_index()
        df_stove_temperature = pd.merge(df_stove_temperature, daily_stats, on='date', how='left')
        seasons = sorted(list(result.keys()))
        threshold_peak_duration = 100  # Threshold for peak duration
        for sson in range(len(seasons)):
            season = seasons[sson]
            peak_info = result[season]['categories']['between_2_and_3_sigma']
            temp_list = []
            for pi in range(len(peak_info)):
                peak_duration_minutes = peak_info[pi]['duration_minutes']
                if peak_duration_minutes >= threshold_peak_duration:
                    # Define the time window for the peak
                    st = peak_info[pi]['left_time']
                    et = peak_info[pi]['right_time']
                    pt = peak_info[pi]['peak_time']
                    # print(pi, st, et, pt)
                    # print('****')
                    
                    temperature_data_in_peak = df_stove_temperature[(df_stove_temperature['ts_datetime'] >= st) & 
                                                                    (df_stove_temperature['ts_datetime'] <= et)]
                    
                    data_dict = {}                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    # Part 1: Plot the raw temperature curve in the upper subplot
                    ax1.plot(temperature_data_in_peak['ts_datetime'], 
                             temperature_data_in_peak['sensor_status'], 
                             color='red', label='Temperature Reading', linewidth=2)
                    ax1.scatter(temperature_data_in_peak['ts_datetime'], 
                             temperature_data_in_peak['sensor_status'], 
                             color='blue')
                    
                    # ax1.plot(temperature_data_in_peak['ts_datetime'], 
                    #          temperature_data_in_peak['daily_avg_temperature'], 
                    #          label='Daily Avg Temperature', color='green', linestyle='-', linewidth=2)
                    
                    ax1.plot(temperature_data_in_peak['ts_datetime'], 
                             temperature_data_in_peak['daily_median_temperature'], 
                             label='Daily Median Temperature', color='orange')
                    ax1.plot(temperature_data_in_peak['ts_datetime'], 
                             temperature_data_in_peak['daily_median_temperature'] + temperature_data_in_peak['daily_std_temperature'], 
                             label='Daily Std Temperature', color='green')

                    # ax1.axvline(pt)
                    ax1.set_title(subject + " - Temperature Curve", fontsize=14)
                    ax1.set_ylabel("Temperature (Sensor Status)", fontsize=12)
                    # ax1.legend(loc='lower right')
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    
                    # Part 2: Plot the device activities and temperature peaks in the lower subplot
                    subject_kitchen_devices = sorted(subject_devices[subject])
                    # for key in natsorted(data_subject.keys()):
                    for key in natsorted(subject_kitchen_devices):
                        print(key)
                        if "Stove" not in key and "Shower" not in key:
                            
                            temp = data_subject[key][1]
                            device_data = temp[(temp['ts_on'] >= st) & (temp['ts_on'] <= et)]
                            if len(device_data) > 0:
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
                    ax2.set_title(subject + "( "+ season + " ) - Active devices during long temperature peak", fontsize=14)
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
                    save_path = os.path.join(pth, f"{subject}_{season}_combined_{timestamp_str}.png")
                    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
                    # plt.close()  # Close the figure after saving
                    
                    temp_list.append(data_dict)
    
            subject_results[subject] = result

    # pth = '/home/hubble/temp/long_peaks_details_above_3sigma'
    # subject_results = {}
    # for sub in range(len(subjects)):
    #     subject = subjects[sub] # Name of the subject
    #     print(subject)
    #     data_subject = data[subject] # Load the whole data    
    #     df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0] ## Load subject's temperature data in the kitchen
    #     ## For safety convert the time to date time format with Europe/Rome Timezone
    #     df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc= True)  
    #     df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
        
    #     # start and end date of temperature data collection
    #     data_start_date = df_stove_temperature['ts_datetime'].min() # of no use right now
    #     data_end_date = df_stove_temperature['ts_datetime'].max() # of no use right now
    #     result = analyze_seasonal_peaks_with_duration(df_stove_temperature)
        
    #     # plot_peaks_and_mean_temperature(df_stove_temperature, '/home/hubble/temp')
    #     # plot_duration_peaks(result, '/home/hubble/temp', subject)
        
    #     seasons = sorted(list(result.keys()))
    #     threshold_peak_duration = 100
    #     for sson in range(len(seasons)):
    #         season = seasons[sson]
    #         peak_info = result[season]['categories']['above_3_sigma']
    #         temp_list = []
    #         for pi in range(len(peak_info)):
    #             peak_duration_minutes = peak_info[pi]['duration_minutes']
    #             if peak_duration_minutes >= threshold_peak_duration:

    #                 # Plot the raw temperature curve

    #                 st = peak_info[pi]['left_time']
    #                 et = peak_info[pi]['right_time']
    #                 pt = peak_info[pi]['peak_time']
                    
    #                 temperature_data_in_peak = df_stove_temperature[(df_stove_temperature['ts_datetime'] >= st) & 
    #                                             (df_stove_temperature['ts_datetime'] <= et)]

    #                 print('------------------')
    #                 data_dict = {}
    #                 print('***************************')
    #                 print(st,et)
    #                 # break
    #                 plt.figure(figsize=(12, 8))

    #                 for key in natsorted(data_subject.keys()):
    #                     print(key)
    #                     if "Stove" not in key and "Shower" not in key:
    #                         temp = data_subject[key][1]
    #                         device_data = temp[(temp['ts_on'] >= st) & (temp['ts_on'] <= et)]
    #                         if len(device_data) > 0:
    #                             data_dict[key] = device_data
    #                             for idx, row in device_data.iterrows():
    #                                 plt.plot(
    #                                     [row['ts_on'], row['ts_off']],
    #                                     [key, key],
    #                                     linewidth=8,
    #                                     label=key if idx == 0 else "",  # Label only first occurrence in legend
    #                                     alpha=0.8,
    #                                     color=color_mapping.get(key, 'grey')
    #                                 )                                    
    #                         else:
    #                             data_dict[key] = pd.DataFrame()
    #                             plt.plot(
    #                                 [st, st], [key, key],
    #                                 color='grey', linewidth=1, alpha=0.5
    #                             )
                                
    #                 plt.plot(
    #                     [st, et],
    #                     ['Temperature Peak', 'Temperature Peak'],
    #                     linewidth=8,
    #                     label= 'Temperature Peak',
    #                     alpha=0.8,
    #                     color=color_mapping.get('Stove', 'black')
    #                 )
                    
    #                 plt.plot(temperature_data_in_peak['ts_datetime'], temperature_data_in_peak['sensor_status'], 
    #                          color='blue', label='Temperature Reading', linewidth=2)
                    
    #                 plt.scatter(pt,'Temperature Peak', color='red', s=200, marker = '*')
    #                 plt.axvline(pt)
                    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz='Europe/Rome'))
                    # plt.xticks(rotation=90, ha="right")
    #                 plt.title(subject +"-Active devices in the house during long temperature teak", fontsize=14)
    #                 plt.xlabel("Time", fontsize=12)
    #                 plt.ylabel("Devices/ Sensors", fontsize=12)
    #                 #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title="Devices")
    #                 plt.grid(True, linestyle='--', alpha=0.6)
    #                 plt.tight_layout()
    #                 timestamp_str = st.strftime('%Y-%m-%d_%H-%M')  # e.g., '2024-08-01_19-13-42'
    #                 save_path = os.path.join(pth,f"{subject}_{season}_{timestamp_str}.png")
    #                 plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')  # Save with high quality
    #                 plt.close()
                    
    #                 temp_list.append(data_dict)

    #     subject_results[subject]= result    