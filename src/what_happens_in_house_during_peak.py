#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:42:04 2024

@author: hubble
"""

from main_utility import *


def plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, stop_temp, st , et, pt, pi, pth, fwhm):
    """
    Plots device usage during a temperature peak event.
    
    Description:
    This function generates a plot showing the stove temperature over time, with vertical lines indicating key time points. It also visualizes device usage during the temperature peak, highlighting the devices used and their usage intervals.
    
    Input:
    - subject (str): The subject identifier (e.g., 'subject_1').
    - subject_devices (dict): Dictionary with subject identifier as the key and a list of device names used for cooking.
    - data_subject (dict): Dictionary with subject names as keys and corresponding data.
    - df_stove_temperature (pd.DataFrame): DataFrame containing stove temperature data for subject in question
    - peak_index (int): Index of the temperature peak in the temperature data.
    - backward_time (pd.Timestamp): Start time of the observation window (before the peak).
    - forward_time (pd.Timestamp): End time of the observation window (after the peak).
    - stop_temp (float): Temperature threshold for stopping conditions.
    - st (pd.Timestamp): Start time of the device usage window.
    - et (pd.Timestamp): End time of the device usage window.
    - pt (pd.Timestamp): Peak time of the temperature event.
    - pth (str): File path where the plot should be saved.
    - delta_T (float, optional, default=5.0): Temperature difference for visual reference.
    - season (str, optional, default='Season_X'): Season during the event.
    - category_name (str, optional, default='Category_A'): Category of the event.
    - pi (str, optional, default='01'): Identifier for the individual or system.
    - color_mapping (dict, optional): Mapping of device names to colors in the plot.
    
    Output:
    - None: The function saves the plot as a PNG file to the specified path (`pth`).
    """
    temperature_data_in_peak = df_stove_temperature[(df_stove_temperature['ts_datetime'] >= (backward_time - pd.Timedelta(hours=2))) & 
                                                    (df_stove_temperature['ts_datetime'] <= (forward_time + pd.Timedelta(hours=2)))]
                            
    
    ## average temperature of the day
    peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
    avg_temp = df_stove_temperature.iloc[peak_index]['daily_avg_temperature']
    median_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
    std_temp = df_stove_temperature.iloc[peak_index]['daily_std_temperature']
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Part 1: Plot the raw temperature curve in the upper subplot
    ax1.plot(temperature_data_in_peak['ts_datetime'], 
              temperature_data_in_peak['sensor_status'], 
              color='red', linewidth=2)
    
    ax1.scatter(temperature_data_in_peak['ts_datetime'], 
              temperature_data_in_peak['sensor_status'], 
              color='blue')
    
    ax1.text(pt, peak_temp - 0.5, f"{fwhm:.0f} min", color=peak_color, fontsize=10, ha='center')
    

    # ax1.axvline(pt,linestyle=':')
    ax1.axhline(y=avg_temp, color='gray', linestyle='-', linewidth=1.5, label=f'Mean ({avg_temp:.2f})')
    ax1.axhline(y=median_temp, color='magenta', linestyle='-', linewidth=1.5, label=f'Median ({median_temp:.2f})')
    # ax1.axhline(y=std_temp, color='magenta', linestyle='-', linewidth=1.5, label='Std.')
    ax1.axhline(y=median_temp + delta_T, color='blue', linestyle='-', linewidth=1.5, label=f'Median ({median_temp:.2f}) + Delta ({delta_T:.2f})')
    # ax1.axhline(y=stop_temp, color='green', linestyle='-', linewidth=1.5, label=f'Peak ({peak_temp:.2f}) - Delta/2 ({delta_T/2})')

    ax1.axvline(backward_time, color='blue', linestyle='-', linewidth=1.5, label=f'Observation Start ({backward_time.strftime("%H:%M:%S")})')
    ax1.axvline(forward_time, color='blue', linestyle='-', linewidth=1.5, label = f'Observation End ({forward_time.strftime("%H:%M:%S")}')
    
    # Annotate st and et times near the vertical lines
    ax1.text(backward_time, ax1.get_ylim()[1], backward_time.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    ax1.text(forward_time, ax1.get_ylim()[1], forward_time.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    ax1.text(pt, ax1.get_ylim()[1], pt.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
        
    ax1.axvline(st, color='black', linestyle='--', linewidth=1.5)
    ax1.axvline(et, color='black', linestyle='--', linewidth=1.5)
    
    # Annotate st and et times near the vertical lines
    ax1.text(st, ax1.get_ylim()[1], st.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    ax1.text(et, ax1.get_ylim()[1], et.strftime("%H:%M:%S"), 
              color='black', ha='right', va='bottom', fontsize=10, rotation=90)
    
    # ax1.set_title(subject+ "("+ season + ")", fontsize=14)
    ax1.set_ylabel("Temperature", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    subject_kitchen_devices = sorted(subject_devices[subject])
    count_var = 0
    for key in natsorted(data_subject.keys()):
    # for key in natsorted(subject_kitchen_devices):
        if "Stove" not in key and "Shower" not in key:
            temp = data_subject[key][1].copy()

            # device_data = temp[(temp['ts_on'] <= et) & (temp['ts_off'] >= st)].copy()
            device_data = temp[(temp['ts_on'] <= forward_time) & (temp['ts_off'] >= backward_time)].copy()
            # device_data = temp[(temp['ts_on'] <= forward_time+ pd.Timedelta(hours=2)) & (temp['ts_off'] >= backward_time- pd.Timedelta(hours=2))]
            device_data['ts_on'] = device_data['ts_on'].apply(lambda x: max(x, st))
            device_data['ts_off'] = device_data['ts_off'].apply(lambda x: min(x, et))
            if len(device_data) > 0:
                count_var = count_var + 1
                usage_count = len(device_data)
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
                    
                ax2.text(row['ts_off'] + pd.Timedelta(seconds=180), key, f"Count: {usage_count}", verticalalignment='center', color='black', fontsize=10)
          
            else:
                data_dict[key] = pd.DataFrame()
                ax2.plot(
                    [backward_time, backward_time], [key, key],
                    color='grey', linewidth=1, alpha=0.5
                )

    # Plot the temperature peak line
    ax2.plot([backward_time, forward_time], ['Temperature Peak', 'Temperature Peak'],
              linewidth=8, label='Temperature Peak', alpha=0.8, color='black')
    # Mark the peak time with a red star
    ax2.scatter(pt, 'Temperature Peak', color='red', s=200, marker='*')
    ax2.axvline(pt)
    # Customize the lower subplot
    ax2.set_title(subject  + " ) - Devices used during temperature peak", fontsize=14)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Devices/ Sensors", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title="Devices")
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz='Europe/Rome'))
        ax.tick_params(axis='x', rotation=90)  # Rotate the x-axis labels for better readability
    # Set the common x-axis label
    fig.tight_layout()
    
    timestamp_str = backward_time.strftime('%Y-%m-%d_%H-%M')  # Format: '2024-08-01_19-13-42'
    save_path = os.path.join(pth, f"{pi}_{subject}_{timestamp_str}.png")
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure after saving

def find_time_interval_v2(df_stove_temperature, peak_index, delta_T, useMedian):
    # Get peak temperature and timestamp at the peak_index
    peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
    peak_time = df_stove_temperature.iloc[peak_index]['ts_datetime']

    # Determine the stopping temperature
    if useMedian:
        stop_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
    else:
        stop_temp = peak_temp - (delta_T / 2)

    # Initialize times to None
    backward_time, forward_time = None, None

    # Backward loop: Check temperature values before the peak
    for i in range(peak_index - 1, -1, -1):  # Loop backwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp or (i > 0 and df_stove_temperature.iloc[i - 1]['sensor_status'] < temp):
            # Stop if temperature is <= stop_temp or an increasing trend is observed
            backward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break
    
    # If no stopping point is found, use the earliest time in the DataFrame
    if backward_time is None:
        backward_time = df_stove_temperature.iloc[0]['ts_datetime']

    # Forward loop: Check temperature values after the peak
    for i in range(peak_index + 1, len(df_stove_temperature)):  # Loop forwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp or (i < len(df_stove_temperature) - 1 and df_stove_temperature.iloc[i + 1]['sensor_status'] > temp):
            # Stop if temperature is <= stop_temp or an increasing trend is observed
            forward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break

    # If no stopping point is found, use the latest time in the DataFrame
    if forward_time is None:
        forward_time = df_stove_temperature.iloc[-1]['ts_datetime']

    # Calculate the time differences
    if backward_time and forward_time:
        time_difference = forward_time - backward_time
    else:
        time_difference = None  # Assign None if either time is not found

    return (
        backward_time,
        forward_time,
        time_difference,
        peak_time - backward_time,
        forward_time - peak_time,
        stop_temp
    )


def find_time_interval(df_stove_temperature, peak_index, delta_T, useMedian):
    """
    Finds the time interval before and after a temperature peak where the temperature falls below a threshold.
    
    Description:
    This function identifies the times when the stove temperature drops below a threshold (`stop_temp`) both before and after the temperature peak, based on a given delta (temperature difference). It calculates the time interval between these two times and the duration between the peak time and the backward/forward times.
    
    Input:
    - df_stove_temperature (pd.DataFrame): DataFrame containing stove temperature data with columns such as 'sensor_status' (temperature) and 'ts_datetime' (timestamp).
    - peak_index (int): Index of the peak temperature in the DataFrame.
    - delta_T (float, optional, default=10): The temperature difference used to define the threshold for stop temperature (`stop_temp`).
    
    Output:
    - backward_time (pd.Timestamp or None): The time when the temperature first falls below `stop_temp` before the peak.
    - forward_time (pd.Timestamp or None): The time when the temperature first falls below `stop_temp` after the peak.
    - time_difference (pd.Timedelta or None): The time difference between `forward_time` and `backward_time`.
    - peak_time_diff_backward (pd.Timedelta): The time difference between the peak time and `backward_time`.
    - peak_time_diff_forward (pd.Timedelta): The time difference between the peak time and `forward_time`.
    - stop_temp (float): The calculated stop temperature threshold based on the peak temperature and `delta_T`.
    """
    # Get peak temperature at the peak_index
    peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
    peak_time = df_stove_temperature.iloc[peak_index]['ts_datetime']
    # print(delta_T)
    
    if useMedian == True:
        # print('-----------')
        stop_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
        # print(stop_temp)
    else:
        # print('********')
        stop_temp = peak_temp - (delta_T / 2)

    # Initialize times to None
    backward_time, forward_time = None, None

    # Backward loop: Check temperature values before the peak
    for i in range(peak_index - 1, -1, -1):  # Loop backwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp:
            backward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break  # Stop once condition is met
            
    if backward_time is None:
        backward_time = df_stove_temperature.iloc[0]['ts_datetime']

    # Forward loop: Check temperature values after the peak
    for i in range(peak_index + 1, len(df_stove_temperature)):  # Loop forwards from peak_index
        temp = df_stove_temperature.iloc[i]['sensor_status']
        if temp <= stop_temp:
            forward_time = df_stove_temperature.iloc[i]['ts_datetime']
            break  # Stop once condition is met
            
    if forward_time is None:
        forward_time = df_stove_temperature.iloc[-1]['ts_datetime']

    # Calculate the time difference
    if backward_time and forward_time:
        time_difference = forward_time - backward_time
    else:
        time_difference = None  # Assign None if either time is not found

    return backward_time, forward_time, time_difference, peak_time - backward_time, forward_time-peak_time, stop_temp

def is_cooking_peak_priyankar(num_points, temper, alpha, beta, gamma=40 , N=2, theta=30):
    is_cooking_peak = False
    if num_points >= N:
        is_cooking_peak = True
    else:
        if temper >= theta:
            is_cooking_peak = True
        else:
            if alpha >= gamma and np.abs(beta) >= gamma:
                is_cooking_peak = True
            else:
                is_cooking_peak = False
    return is_cooking_peak

def compute_angle(df_temp, point_a, point_b):
    """
    Computes the angle between two temperature readings based on their time difference and temperature difference.
    
    Description:
    This function calculates the angle of the line connecting two temperature readings, where the temperature difference is divided by the time difference between the two readings. The angle is returned in degrees, representing the slope between the two points on a temperature-time graph.
    
    Input:
    - df_temp (pd.DataFrame): DataFrame containing temperature data with columns such as 'sensor_status' (temperature) and 'ts_datetime' (timestamp).
    - point_a (int): Index of the first point in the DataFrame to calculate the angle from.
    - point_b (int): Index of the second point in the DataFrame to calculate the angle to.
    
    Output:
    - angle (float): The angle in degrees representing the slope between the two points, calculated using the formula `angle = atan(slope)`, where `slope = (temp_b - temp_a) / time_diff_hours`.
    
    Notes:
    - The angle is calculated using the arctangent of the slope of the temperature change over time.
    - If the time difference is zero, the function will raise a division by zero error.
    """
    temp_a = df_temp.loc[point_a, 'sensor_status']
    temp_b = df_temp.loc[point_b, 'sensor_status']
    time_diff_hours = (df_temp.loc[point_b, 'ts_datetime'] - df_temp.loc[point_a, 'ts_datetime']).total_seconds() / 3600.0
    slope = (temp_b - temp_a) / time_diff_hours
    angle = math.atan(slope) * (180 / math.pi)
    return angle

def if_kitchen_devices_used(data_subject, subject_devices, subject,st,et):
    """
    Checks if any kitchen devices were used during a specified time interval and returns relevant data.

    Description:
    This function evaluates the usage of kitchen devices within a given time window (from start time `st` to end time `et`) for a specific subject. It checks devices associated with the subject, filters the device data based on the time window, and returns a dictionary with the device data, a count of the devices used, and a list of the devices used.

    Input:
    - data_subject (dict): Dictionary containing data for each device. The key is the device name, and the value is a list with the first element being the device name and the second element being a DataFrame containing device data (including 'ts_on' and 'ts_off' columns).
    - subject_devices (dict): Dictionary mapping subjects to their devices. The key is the subject, and the value is a list of devices associated with the subject.
    - subject (str): The subject ID to check for device usage.
    - st (datetime): The start time of the time interval to check.
    - et (datetime): The end time of the time interval to check.

    Output:
    - data_dict (dict): A dictionary where the keys are the names of kitchen devices, and the values are DataFrames containing the device data filtered for the time interval.
    - count_var (int): The number of kitchen devices that were used during the specified time interval.
    - cooking_device (list): A list of the names of the kitchen devices that were used during the specified time interval.

    Notes:
    - The function excludes devices with names containing "Stove" or "Shower" from the analysis.
    - Device data is filtered by checking whether the 'ts_on' time is before the end time (`et`) and the 'ts_off' time is after the start time (`st`).
    """    
    data_dict = {}          
    count_var = 0
    cooking_device = []
    subject_kitchen_devices = sorted(subject_devices[subject])
    
    # for key in natsorted(data_subject.keys()):
    for key in natsorted(subject_kitchen_devices):
        if "Stove" not in key and "Shower" not in key:
            temp = data_subject[key][1].copy()
            # device_data = temp[(temp['ts_on'] >= st) & (temp['ts_on'] <= et)] ## any device which is used between st and et
            device_data = temp[(temp['ts_on'] <= et) & (temp['ts_off'] >= st)] ## any device usage which overlaps with st and et
            if len(device_data) > 0:
                count_var = count_var + 1
                cooking_device.append(key)
                data_dict[key] = device_data
            else:
                data_dict[key] = pd.DataFrame()
    return data_dict, count_var, cooking_device

def create_box_plot(df_stove_temperature, subject, savepath):
    """
    Creates and saves box plots with scatter plots to visualize stove temperature data for a specific subject.

    Description:
    This function generates two box plots:
    - The first plot shows the distribution of stove temperature sensor data ('sensor_status') with a scatter plot overlay.
    - The second plot displays the daily temperature standard deviation with a scatter plot overlay for each day.
    Both plots are created side-by-side for comparison. The plots are saved as a PNG file in the specified directory with the subject's name.

    Input:
    - df_stove_temperature (DataFrame): A pandas DataFrame containing stove temperature data with columns such as 'sensor_status' and 'sensor_datetime'.
    - subject (str): The subject name or ID to be used as the title of the plot and for naming the saved file.
    - savepath (str): The directory path where the plot image will be saved.

    Output:
    - Saves a PNG image with the box plots in the specified `savepath` directory with the subject's name.

    Notes:
    - The first plot visualizes the distribution of stove temperature readings ('sensor_status') with scatter points overlaid.
    - The second plot shows the standard deviation of daily stove temperatures, with scatter points added for better visibility.
    - The plots use different colors for visual differentiation: light blue for temperature distribution and light coral for the standard deviation.
    - Gridlines are added to the y-axis for clarity, and the plots are adjusted to fit titles and labels.

    Example:
    create_box_plot(df_stove_temperature, "Subject_1", "/path/to/save")
    """    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(subject, fontsize=16)
    
    # Subplot 1: Box plot for 'sensor_status' with scatter
    sns.boxplot(
        data=df_stove_temperature,
        y="sensor_status",
        color="skyblue",
        ax=axes[0]
    )
    sns.stripplot(
        data=df_stove_temperature,
        y="sensor_status",
        color="darkblue",
        alpha=0.2,  # Make scatter points semi-transparent
        size=2,  # Set marker size
        jitter=True,  # Add jitter to scatter points for visibility
        ax=axes[0]
    )
    axes[0].set_title("Temperature Distribution", fontsize=14)
    axes[0].set_ylabel("Sensor Status", fontsize=12)
    axes[0].set_xlabel("Sensor", fontsize=12)
    
    # Subplot 2: Box plot for standard deviation with scatter
    sns.boxplot(
        data=std_per_day,
        y="sensor_status",
        color="lightcoral",
        ax=axes[1]
    )
    sns.stripplot(
        data=std_per_day,
        y="sensor_status",
        color="darkred",
        alpha=0.2,
        size=3,
        jitter=True,
        ax=axes[1]
    )
    axes[1].set_title("Temperature Standard Deviation (Daily Temperature)", fontsize=14)
    axes[1].set_ylabel("Standard Deviation", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    
    # Add gridlines for better visualization
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit titles
    filename = subject+'.png'
    plt.savefig(os.path.join(savepath,filename), dpi=300)
    plt.show()
    
def classify_time_of_day(time):
    """
    Classifies the given time into a meal category based on predefined time ranges.
    
    Description:
    This function takes a `time` as input and classifies it into one of the following categories:
    - 'breakfast': If the time falls within the predefined breakfast time range.
    - 'lunch': If the time falls within the predefined lunch time range.
    - 'dinner': If the time falls within the predefined dinner time range.
    - 'other': If the time does not fall within any of the predefined meal time ranges.
    
    Input:
    - time (datetime or time object): The specific time to classify. It is expected to be a `datetime` or `time` object.
    
    Output:
    - (str): A string representing the meal category ('breakfast', 'lunch', 'dinner', or 'other') based on the input `time`.
    
    Assumptions:
    - `breakfast_start`, `breakfast_end`, `lunch_start`, `lunch_end`, `dinner_start`, and `dinner_end` are predefined time objects representing the start and end times for breakfast, lunch, and dinner, respectively.
    - breakfast_time = ("06:00:00", "10:00:00")
    - lunch_time = ("12:00:00", "14:00:00")
    - dinner_time = ("18:00:00", "21:00:00")
    - breakfast_start, breakfast_end = pd.to_timedelta(breakfast_time[0]), pd.to_timedelta(breakfast_time[1])
    - lunch_start, lunch_end = pd.to_timedelta(lunch_time[0]), pd.to_timedelta(lunch_time[1])
    - dinner_start, dinner_end = pd.to_timedelta(dinner_time[0]), pd.to_timedelta(dinner_time[1])
    
    Example:
    classify_time_of_day(datetime.time(7, 30))  # Returns 'breakfast'
    classify_time_of_day(datetime.time(13, 0))  # Returns 'lunch'
    """
    if breakfast_start <= time <= breakfast_end:
        return 'breakfast'
    elif lunch_start <= time <= lunch_end:
        return 'lunch'
    elif dinner_start <= time <= dinner_end:
        return 'dinner'
    else:
        return 'other'
    
if __name__ == "__main__":        
    ## Read the data
    with open('/home/hubble/work/serenade/data/subject_data_sept_2024.pkl', 'rb') as file:
        data = pickle.load(file)
    
    breakfast_time = ("06:00:00", "10:00:00")
    lunch_time = ("11:00:00", "15:00:00")
    dinner_time = ("18:00:00", "23:59:00")
    breakfast_start, breakfast_end = pd.to_timedelta(breakfast_time[0]), pd.to_timedelta(breakfast_time[1])
    lunch_start, lunch_end = pd.to_timedelta(lunch_time[0]), pd.to_timedelta(lunch_time[1])
    dinner_start, dinner_end = pd.to_timedelta(dinner_time[0]), pd.to_timedelta(dinner_time[1])
    percentile_threshold = 0.75 
    subjects = natsorted(list(data.keys()))   
    pth = '/home/hubble/temp/prof_claudio/start_end/' 
    peaks_info = {}
    conf_mat = []
    preparint_food = {}
    meal_info = {}
    for sub in range(len(subjects)):
        subject_inactive_in_kitchen = []
        subject = subjects[sub]  # Name of the subject
        print(subject)
        # delta_T = deltas[subject]
        data_subject = data[subject]  # Load the whole data    
        df_stove_temperature = data_subject['Stove_Hum_Temp_temp'][0]  # Load subject's temperature data in the kitchen
        ## Plot the box plot to understand the distribution of the temperature
        # create_box_plot(df_stove_temperature, subject, '/home/hubble/')
       
        # Convert the time to datetime format with Europe/Rome Timezone for safety
        df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime'], utc=True)  
        df_stove_temperature['ts_datetime'] = pd.to_datetime(df_stove_temperature['ts_datetime']).dt.tz_convert('Europe/Rome')
        
        # Start and end date of temperature data collection
        data_start_date = df_stove_temperature['ts_datetime'].min()  # Not useful right now
        data_end_date = df_stove_temperature['ts_datetime'].max()  # Not useful right now
        
        # sensor_values = df_stove_temperature['sensor_status'].values # Get all the temperature readings in an array
        # preliminary_peaks, properties = find_peaks(sensor_values, prominence=(None,None))  # Adjust prominence for initial detection
        # prominences = properties["prominences"]

        # average_prominence = 1.5#np.mean(properties["prominences"])
        # average prominence to find the actual peaks
        # actual_peaks, _ = find_peaks(sensor_values, prominence=average_prominence)
        # print(subject, average_prominence, len(actual_peaks))
        
        # Compute peak duration, and divide peak in categories like 1_sigma, 2_sigma, etc.
        peaks_info = analyze_seasonal_peaks_with_duration_v2(df_stove_temperature)
        # plot_peaks_and_mean_temperature(df_stove_temperature, '/home/hubble/temp') # where are the peaks in graph
        # plot_duration_peaks(result, '/home/hubble/temp', subject)
        # threshold_peak_duration = 100 
        # plot_peak_and_kitchen_devices(result, category_name, df_stove_temperature, data_subject, subject,pth, 100)
        
        
        df_stove_temperature['date'] = df_stove_temperature['ts_datetime'].dt.date
        ## datewise mean, median, standard deviation, maximum temperature of the day
        daily_stats = df_stove_temperature.groupby('date')['sensor_status'].agg(
            daily_avg_temperature='mean',
            daily_median_temperature='median',
            daily_std_temperature='std',
            daily_max_temperature='max',
        ).reset_index()
        
        df_stove_temperature = pd.merge(df_stove_temperature, daily_stats, on='date', how='left')
        
        # peaks_info = analyze_seasonal_peaks_with_duration_v2(df_stove_temperature)

        
        # df_stove_temperature['rounded_temperature'] = df_stove_temperature['sensor_status'].round()
        # df_stove_temperature['sigma_threshold'] = df_stove_temperature['daily_avg_temperature'] + 2 * df_stove_temperature['daily_std_temperature']
        # df_stove_temperature['above_3sigma'] = (df_stove_temperature['sensor_status'] > df_stove_temperature['sigma_threshold']).astype(int)
        # df_stove_temperature['daily_max_minus_median'] = df_stove_temperature['daily_max_temperature'] - df_stove_temperature['daily_median_temperature']
        # df_daily_analysis = df_stove_temperature.groupby('date').agg(
        #     daily_median_temperature=('daily_median_temperature', 'median'),
        #     daily_max_temperature=('daily_max_temperature', 'max')
        # ).reset_index()
        # # Add a new column for the difference
        # df_daily_analysis['daily_max_minus_median'] = (
        #     df_daily_analysis['daily_max_temperature'] - df_daily_analysis['daily_median_temperature']
        # )
        # df_daily_analysis['daily_max_minus_median_rounded'] = df_daily_analysis['daily_max_minus_median'].round().astype(int)
        
        std_per_day = pd.DataFrame(df_stove_temperature.groupby('date')['sensor_status'].std())
        
       
        breakfast_peaks = 0
        lunch_peaks = 0
        dinner_peaks = 0
        other_peaks = 0
        unique_dates = set()
               
        
        df_stove_temperature['time_of_day'] = pd.to_timedelta(df_stove_temperature['ts_datetime'].dt.time.astype(str))
        df_stove_temperature['time_category'] = df_stove_temperature['time_of_day'].apply(classify_time_of_day)
        
        threshold = df_stove_temperature['daily_std_temperature'].quantile(percentile_threshold)
        filtered_data = df_stove_temperature[df_stove_temperature['daily_std_temperature'] > threshold]
        # Step 4: Group by time_category and calculate standard deviation
        grouped_std = filtered_data.groupby('time_category')['daily_std_temperature'].mean()
        
        # # Step 5: Compute weighted average
        # weights = filtered_data['time_category'].value_counts()
        # weighted_avg_std = (grouped_std * weights).sum() / weights.sum()
        
        # # print("Grouped Standard Deviations by Time of Day:\n", grouped_std)
        # print("Weighted Average of Standard Deviations:", weighted_avg_std)
        # delta_T = weighted_avg_std
        
        std_per_day = std_per_day.dropna()
        percentile_values = std_per_day['sensor_status'].quantile(percentile_threshold)
        values_above_percentile = std_per_day[std_per_day['sensor_status'] > percentile_values]['sensor_status']
        average_above_percentile = values_above_percentile.mean()
        delta_T = average_above_percentile
        print(delta_T)
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        cnt = 0
        meal_data = {}
        meal_timing = {}

        for pi in range(len(peaks_info)):
            pt = peaks_info[pi]['peak_time']  
            peak_time_str = pt.strftime("%H:%M:%S")
            peak_date = pt.date()
            peak_index = peaks_info[pi]['peak_index']
            # peak_temp = peaks_info[pi]['peak_temperature']
            peak_temp = df_stove_temperature.iloc[peak_index]['sensor_status']
            
            left_time =  peaks_info[pi]['left_time']
            right_time = pt + (pt - left_time)
            
            if peak_date not in meal_data:
                # print('--------------------')
                meal_data[peak_date] = {'breakfast': 0, 'lunch': 0, 'dinner': 0, 'other_times': 0}
                meal_timing[peak_date] = {'breakfast': '0', 'lunch': '0', 'dinner': '0', 'other_times': '0'}



            if breakfast_time[0] <= peak_time_str <= breakfast_time[1]:
                breakfast_peaks += 1
                meal_data[peak_date]['breakfast'] = 1
                meal_timing[peak_date]['breakfast'] = peak_time_str
                
            elif lunch_time[0] <= peak_time_str <= lunch_time[1]:
                lunch_peaks += 1
                meal_data[peak_date]['lunch'] = 1
                meal_timing[peak_date]['lunch'] = peak_time_str
                
            elif dinner_time[0] <= peak_time_str <= dinner_time[1]:
                dinner_peaks += 1
                meal_data[peak_date]['dinner'] = 1
                meal_timing[peak_date]['dinner'] = peak_time_str
            else:
                other_peaks += 1
                meal_data[peak_date]['other_times'] = 1   
                meal_timing[peak_date]['other_times'] = peak_time_str
                
            unique_dates.add(peak_date)
  
            avg_temp = df_stove_temperature.iloc[peak_index]['daily_avg_temperature']
            median_temp = df_stove_temperature.iloc[peak_index]['daily_median_temperature']
            std_temp = df_stove_temperature.iloc[peak_index]['daily_std_temperature']
                
                
            backward_time, forward_time, time_difference, diff_peak_backward, diff_peak_forward, stop_temp = find_time_interval(df_stove_temperature, peak_index, delta_T, useMedian = False)
            backward_index = df_stove_temperature[df_stove_temperature['ts_datetime'] == backward_time].index[0]
            forward_index = df_stove_temperature[df_stove_temperature['ts_datetime'] == forward_time].index[0]
            fwhm = (pt - backward_time).total_seconds() / 60
            
            if fwhm <= 30:
                peak_color = 'green'
            elif 30 < fwhm <= 60:
                peak_color = 'orange'
            elif 60 < fwhm <= 90:
                peak_color = 'red'
            else:
                peak_color = 'black'
                
                
            # if forward_time.date() != backward_time.date():
            #     print(backward_time, forward_time)
            # backward_time = pt - timedelta(minutes=15)
            # forward_time = pt + timedelta(minutes=15)
            
            data_dict, count_var, cooking_devices = if_kitchen_devices_used(data_subject, subject_devices, subject, backward_time, forward_time)
            
            if count_var > 0:
                cnt = cnt + 1
                
            if peak_temp >= median_temp+ delta_T:
                if count_var > 0:
                    tp = tp + 1
                    # plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, '', left_time, right_time, pt, pi,'/home/hubble/',fwhm)
                else:
                    fp = fp + 1
                    # plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, '', left_time, right_time, pt, pi, '/home/hubble/temp/fp')
                    
            else:
                if count_var > 0:
                    fn = fn + 1
                    # plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, '', left_time, right_time, pt, pi, '/home/hubble/temp/fn')
                else:
                    tn = tn + 1
                    # plot_device_usage_prof_claudio(subject, subject_devices, data_subject, df_stove_temperature, peak_index, backward_time, forward_time, '', left_time, right_time, pt, pi, '/home/hubble/temp/tn')
        
        conf_mat.append(
            {'subject':subject,
             'total_peaks':len(peaks_info),
             'tp':tp,
             'fp':fp,
             'fn':fn,
             'tn':tn
             }
            )

        preparint_food[subject] = (cnt,len(peaks_info))  
        print(cnt,len(peaks_info))  
        meal_data = pd.DataFrame.from_dict(meal_data, orient='index').reset_index()
        meal_timing = pd.DataFrame.from_dict(meal_timing, orient='index').reset_index()
        meal_info[subject] = (meal_data, meal_timing)
    conf_mat_ = pd.DataFrame(conf_mat)  
    column_sums = conf_mat_.iloc[:, 1:].sum()
    conf_mat_.loc['Total'] = [''] + column_sums.tolist()
    
    cnt = 0

                    
                    












                            

# if count_var != -99:
#     ## Approach 1
#     time_window = timedelta(minutes=3)
#     start_window = pt - time_window
#     end_window = pt + time_window
    
  
#     points_around_peak = df_stove_temperature[(df_stove_temperature['ts_datetime'] >= start_window) &
#                       (df_stove_temperature['ts_datetime'] <= end_window) &
#                       (df_stove_temperature['ts_datetime'] != pt)]

#     before_peak_idx = peak_index - 1
#     after_peak_idx = peak_index + 1
    
#     angle_before = compute_angle(df_stove_temperature, before_peak_idx, peak_index)
#     angle_after = compute_angle(df_stove_temperature, peak_index, after_peak_idx)
    
#     # Count the number of data points in this window
#     num_points = len(points_around_peak)
#     print(st, pt, num_points, count_var, angle_before, angle_after)
    

#     peak_reading = df_stove_temperature.iloc[peak_index]['sensor_status']
#     icp = is_cooking_peak_priyankar(num_points, peak_reading, angle_before, angle_after)
    

    
#     subject_inactive_in_kitchen.append(
#         {'subject': subject,
#           'season':season,
#           'peak start': st,
#           'peak end': et,
#           'peak at': pt,
#           'icp' : icp,
#           'num_points': num_points,
#           'cooking devices': count_var,
#           'temperature': peak_reading ,
#           'peak_index':peak_index,
#           'angle_before':angle_before,
#           'angle_after':angle_after})
    
# dff = pd.DataFrame(subject_inactive_in_kitchen)
# k.append(dff)


        
       
