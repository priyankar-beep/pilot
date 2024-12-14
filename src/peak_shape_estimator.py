#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:14:04 2024

@author: hubble
"""
from main_utility import *
    
if __name__ == "__main__":        
    ## Read the data
    with open('/home/hubble/work/serenade/data/subject_data_sept_2024.pkl', 'rb') as file:
        data = pickle.load(file)
    
    deltas =     {
        'subject_1': 5.67, 'subject_2': 2.12, 'subject_3': 4.14,
        'subject_4': 5.28, 'subject_5': 4.75, 'subject_7': 5.59,
        'subject_8': 5.06, 'subject_9': 6.08, 'subject_10': 5.06,
        'subject_11': 9.09, 'subject_12': 3.11, 'subject_13': 1.13,
        'subject_14': 5.90
    }
    
    subjects = natsorted(list(data.keys()))
    for sub in range(len(subjects)):
        subject_inactive_in_kitchen = []
        subject = subjects[sub]  # Name of the subject
        print(subject)
        data_subject = data[subject]  # Load the whole data    
        df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0]  # Load subject's temperature data in the kitchen
        df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc=True)  
        df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
        
        df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
        ## datewise mean, median, standard deviation, maximum temperature of the day
        daily_stats = df_stove_temperature.groupby('date')['sensor_status'].agg(
            daily_avg_temperature='mean',
            daily_median_temperature='median',
            daily_std_temperature='std',
            daily_max_temperature='max',
        ).reset_index()
        
        df_stove_temperature = pd.merge(df_stove_temperature, daily_stats, on='date', how='left')
        peaks_info = analyze_seasonal_peaks_with_duration_v2(df_stove_temperature)
        signal = df_stove_temperature['sensor_status'].values # Get all the temperature readings in an array
        peaks, _ = find_peaks(signal, prominence = 1.5)
        peak_times = df_stove_temperature['ts_datetime'].iloc[peaks].tolist()
        
        df_stove_temperature['peaks'] = 0
        df_stove_temperature.loc[peaks, 'peaks'] = 1  # Mark peaks as 1
        
        # fig, ax1 = plt.subplots(figsize=(12, 6))  # Create a single Axes object
        # ax1.plot(df_stove_temperature['ts_datetime'], 
        #          df_stove_temperature['sensor_status'], 
        #          color='blue', linewidth=2)
        
        # ax1.scatter(df_stove_temperature['ts_datetime'], 
        #          df_stove_temperature['sensor_status'], 
        #          color='pink')
        
        delta_T = deltas[subject]
        for pi in range(20):#len(peaks)):
            peak_index = peaks[pi]
            
            peak_time = df_stove_temperature['ts_datetime'].iloc[peak_index]
            median_temperature = df_stove_temperature['daily_median_temperature'].iloc[peak_index]
            peak_height = signal[peak_index]
            backward_time, forward_time, time_difference, diff_peak_backward, diff_peak_forward, stop_temp = find_time_interval_v2(df_stove_temperature, peak_index, delta_T, useMedian = False)
            backward_index = df_stove_temperature[df_stove_temperature['ts_datetime'] == backward_time].index[0]
            forward_index = df_stove_temperature[df_stove_temperature['ts_datetime'] == forward_time].index[0]
            fwhm = (peak_time - backward_time).total_seconds() / 60
            angle = compute_angle_between_peak_and_previous_to_peak(df_stove_temperature,peak_index,signal)
            
            left_time =  peaks_info[pi]['left_time']
            right_time = peak_time + (peak_time - left_time)
            
            if fwhm <= 30:
                peak_color = 'green'
            elif 30 < fwhm <= 60:
                peak_color = 'orange'
                # if angle < 45:
                #     peak_color = 'pink'
            elif 60 < fwhm <= 90:
                peak_color = 'red'
                # if angle < 45:
                #     peak_color = 'pink'
            else:
                peak_color = 'black'
                
            plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, '', left_time, right_time, peak_time, pi,'/home/hubble/figures/',fwhm,delta_T)

            
        #     ax1.plot([backward_time, forward_time ], [stop_temp, stop_temp], color='red', linestyle='-', linewidth=2)

        #     ax1.scatter(peak_time, peak_height, color=peak_color, s=200, marker='o')
        #     # ax1.text(peak_time, peak_height + 0.5, f"{fwhm:.0f} min", color=peak_color, fontsize=10, ha='center')
        #     ax1.text(
        #         peak_time,
        #         peak_height + 3.5,
        #         f"({fwhm:.0f} min,{angle:.1f}°,{peak_time.strftime('%H:%M')})",
        #         color=peak_color,
        #         fontsize=10,
        #         ha='center'
        #     )
        
        # # Customize plot
        # ax1.set_title("Temperature peaks in kitchen for "+subject +" Delta_T = "+str(delta_T), fontsize=14)
        # ax1.set_xlabel("Timestamp")
        # ax1.set_ylabel("Temperature")
        # ax1.grid(True, linestyle='--', alpha=0.6)
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz='Europe/Rome'))
        # ax1.tick_params(axis='x', rotation=90)
        # # Display the plot
        # plt.tight_layout()
        # plt.show()
                    
    
    

