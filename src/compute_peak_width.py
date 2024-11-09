#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:58:48 2024

@author: hubble
"""
from main_utility import *
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
    
    sub = 0
    subject = subjects[sub] # Name of the subject
    data_subject = data[subject] # Load the whole data
    ## Specify the date where we have to pick the data
    start_date = pd.to_datetime('2024-01-01').tz_localize('Europe/Rome')
    end_date = pd.to_datetime('2024-01-31').tz_localize('Europe/Rome')
    
    df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0] ## Load subject's temperature data in the kitchen
    
    ## For safety convert the time to date time format with Europe/Rome Timezone
    df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc= True)  
    df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
    
    # start and end date of temperature data collection
    data_start_date = df_stove_temperature['ts_datetime'].min()
    data_end_date = df_stove_temperature['ts_datetime'].max()
    
    temperature_df = df_stove_temperature[
    (df_stove_temperature['ts_datetime'] >= start_date) &
    (df_stove_temperature['ts_datetime'] <= end_date)].copy()
    
    ## Compute the peak
    peaks, properties = find_peaks(temperature_df['sensor_status'].values, prominence=3)
    temperature_df = compute_daily_temperature(temperature_df.copy())
    peaks = remove_peaks_below_daily_avg(temperature_df, peaks)
  
    ## Make the figure
    plt.figure(figsize=(10, 5))
    
    # Plot temperature data and mark peaks
    plt.plot(temperature_df['ts_datetime'], temperature_df['sensor_status'], label='Temperature', color='red')
    plt.scatter(temperature_df['ts_datetime'], temperature_df['sensor_status'], color='blue')
    plt.plot(temperature_df['ts_datetime'].iloc[peaks], temperature_df['sensor_status'].iloc[peaks], 'ro', label='Peaks')
    plt.plot(temperature_df['ts_datetime'], temperature_df['daily_avg_temp'], label='Room Temperature', color='magenta')
    plt.plot(temperature_df['ts_datetime'], temperature_df['daily_avg_temp'] + temperature_df['daily_avg_std'], 
         label='Standard deviation', color='magenta', linestyle=':')

    
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
        std = temperature_df.iloc[peak_index]['daily_avg_std'] # Room Temperature
        print(peak_temperature, room_temperature,std)

        peak_time = temperature_df['ts_datetime'].iloc[peak_index] # When did the peak occur
        duration, left, right = find_peak_duration_v3(temperature_df.copy(), peak_index, 3, 'median')

        left_time = temperature_df['ts_datetime'].iloc[left]
        right_time = temperature_df['ts_datetime'].iloc[right]
        
        print(p, '\t',peak_index, '\t', left_time, '\t', right_time,'\t', peak_time)
        
        # Select color for the current peak
        color = colors[p % num_colors]  # Cycle through colors
        
        # Plot vertical lines for left and right times of each peak with the same color
        plt.axvline(x=left_time, color=color, linestyle='--', linewidth=1.5, label=f'Peak {p+1} Start' if p == 0 else "")
        plt.axvline(x=right_time, color=color, linestyle='-', linewidth=1.5, label=f'Peak {p+1} End' if p == 0 else "")
        plt.annotate(f'{peak_index}', xy=(peak_time, peak_temperature + 0.25), 
             ha='center', color=color, fontsize=8, fontweight='bold')
        # # Annotate start and end times
        # plt.annotate(f'Start: {left_time.strftime("%Y-%m-%d %H:%M:%S")}', xy=(left_time, peak_temperature + 0.5),
        #               ha='center', color=color, fontsize=8, fontweight='bold', rotation=45)
        # plt.annotate(f'End: {right_time.strftime("%Y-%m-%d %H:%M:%S")}', xy=(right_time, peak_temperature + 0.5),
        #               ha='center', color=color, fontsize=8, fontweight='bold', rotation=45)
         
        
    # Set x-ticks and rotate for readability
    # plt.xticks(date_range, rotation=90)
    plt.title(subject + ' Start date ='+ str(start_date) +' End date = '+ str(end_date))
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    
    # Adjust legend to avoid duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# # Step 1: Create a new column for the daily average temperature
df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
df_stove_temperature['daily_avg_temp'] = df_stove_temperature.groupby('date')['sensor_status'].transform('mean')

# Step 2: Calculate the standard deviation of the sensor readings for each day
df_stove_temperature['daily_avg_std'] = df_stove_temperature.groupby('date')['sensor_status'].transform('std')

# Step 3: Find peaks in the 'sensor_status' data using find_peaks
peaks, _ = find_peaks(df_stove_temperature['sensor_status'], prominence=1.5)

# Step 4: Mark all detected peaks as 1 in the 'peaks' column
df_stove_temperature['peaks'] = 0  # Reset the peaks column to 0
df_stove_temperature.loc[peaks, 'peaks'] = 1  # Mark detected peaks

# Step 5: Define a function to remove peaks below the daily average + std threshold
def remove_peaks_below_daily_avg(temperature_df, peaks):
    filtered_peaks = [
        peak for peak in peaks
        if temperature_df.loc[peak, 'sensor_status'] > (
            temperature_df.loc[peak, 'daily_avg_temp'] + temperature_df.loc[peak, 'daily_avg_std']
        )
    ]
    return filtered_peaks

# Step 6: Apply the function to filter out peaks
filtered_peaks = remove_peaks_below_daily_avg(df_stove_temperature, peaks)

# Step 7: Mark only the valid peaks as 1 in the 'peaks' column
df_stove_temperature['peaks'] = 0  # Reset the peaks column to 0
df_stove_temperature.loc[filtered_peaks, 'peaks'] = 1  # Mark the valid peaks

# Display the final DataFrame with valid peaks
print(df_stove_temperature)