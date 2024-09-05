import pandas as pd, numpy as np, os
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"

from datetime import datetime
from scipy.signal import find_peaks
from sklearn.manifold import TSNE
print('...')
# subject_index_mapping = {0: 'Subject-1', 1:'Subject-2',2:'Subject-3',3:'Subject-4',4:'Subject-5',5:'Subject-5',6:'Subject-7',7:'Subject-8'}
subject_index_mapping = {0: 'Subject-1', 1:'Subject-2',2:'Subject-3',3:'Subject-7',4:'Subject-8',5:'Subject-9',6:'Subject-11',7:'Subject-12'}

sensor_files = [
    "CoffeMachine.csv",
    "CoffeMachine_events.csv",
    "Cookware.csv",
    "Dishes.csv",
    "Dishes_Glasses.csv",
    "Dishes_Silverware.csv",
    "FoodStorage.csv",
    "FoodStorageKitchen.csv",
    "FoodStorageLivingRoom.csv",
    "Freezer.csv",
    "HouseEntrance.csv",
    "Hum_Temp_Bath_humidity.csv",
    "Hum_Temp_Bath_temp.csv",
    "Stove_Hum_Temp_humidity.csv",
    "Stove_Hum_Temp_temp.csv",
    # "Hum_Temp_Stove_humidity.csv",
    # "Hum_Temp_Stove_temp.csv",
    "Medicines.csv",
    "Microwave.csv",
    "Microwave_events.csv",
    "MotionBathroom.csv",
    "MotionBedroom.csv",
    "MotionDiningTable.csv",
    "MotionGuestRoom.csv",
    "MotionKitchen.csv",
    "MotionLivingRoomSofa.csv",
    "MotionLivingRoomTablet.csv",
    "MotionLivingroom.csv",
    "MotionOffice.csv",
    "MotionOtherRoom.csv",
    "MotionOtherroom.csv",
    "MotionPrimaryBathroom.csv",
    "MotionSecondaryBathroom.csv",
    "PlugTvHall.csv",
    "PlugTvHall_events.csv",
    "PlugTvKitchen.csv",
    "Printer.csv",
    "Refrigerator.csv",
    "Silverware.csv",
    "WashingMachine.csv",
    "printer_events.csv",
    "washingMachine_events.csv"
]
#data_path, subjects, unwanted_values, skip_substrings= data_path, subjects, unwanted_values, skip_substring
def load_subjectwise_data(data_path, subjects, unwanted_values, skip_substrings):
    subject_data = []
    for sub in range(10):
        subject_sub = subjects[sub]
        subject_data_path = os.path.join(data_path, subject_sub, 'environmentals')
        print(subject_data_path)
    
        cleaned_data = []
        c = 0
        for sen in range(len(sensor_files)):
            sensor_file = sensor_files[sen]
            
            ## Do not pick file with word event in it
            if any(substring in sensor_file for substring in skip_substrings):
                continue
    
            sensor_file_path = os.path.join(subject_data_path, sensor_file) # This sensor may not be installed so use try-catch
            try:
                print('=====>',sen, sensor_file)
                sensor_df = pd.read_csv(sensor_file_path)
                sensor_df = sensor_df[~sensor_df['sensor_status'].isin(['unavailable', 'unknown'])]
                unique_values = set(sensor_df['sensor_status'].unique())
                ## Idea: if on off is present it is a magnetic sensor, if no on-off present then it is real-valued sensor
                contains_on = 'on' in unique_values
                contains_off = 'off' in unique_values
    
                if contains_on or contains_off:
                    sensor_df, _ = preprocess_dataframe(sensor_df) # This function will remove continuous on  or continuous off values
                else:
                    sensor_df['sensor_status'] = pd.to_numeric(sensor_df['sensor_status'], errors='coerce')
                    # sensor_df.to_csv(sensor_file)
    
                data_type = sensor_df['sensor_status'].dtype
    
                sdf = sensor_df.copy()
                if data_type == 'float64':
                    ## Make sure sensor readings are real values
                    sdf['sensor_status'] = pd.to_numeric(sdf['sensor_status'], errors='coerce')
                    ## Detect the peak
                    peaks, sdf = detect_peaks(sdf, 'sensor_status', prominence = 1)
                    sdf = sdf.rename(columns={'sensor_status': 'sensor_values', 'peak_detected': 'sensor_status'})
                    sdf = sdf.reset_index(drop=True)
                    filtered_df = []
                    # filtered_df = filter_consecutive_on_and_off_in_real_valued_sensor(sdf) ## Only for real-valued sensors
                    # filtered_df = filtered_df.reset_index(drop=True)
                    # filtered_df = post_process_sensor_data(filtered_df)
                    cleaned_data.append({'sensor':sensor_file, 'filtered_df':filtered_df, 'sensor_df':sdf})
                else:
                    filtered_df = post_process_sensor_data(sdf)
                    # filtered_df.to_csv(sensor_file+'.csv')
                    cleaned_data.append({'sensor':sensor_file, 'filtered_df':filtered_df, 'sensor_df':sdf})
    
            except FileNotFoundError:
                sensor_df = pd.DataFrame()
                cleaned_data.append({'sensor':sensor_file, 'filtered_df':pd.DataFrame(), 'sensor_df':pd.DataFrame()})
            except Exception as e:
                sensor_df = pd.DataFrame()
                cleaned_data.append({'sensor':sensor_file, 'filtered_df':pd.DataFrame(), 'sensor_df':pd.DataFrame()})
        subject_data.append(cleaned_data)
    return subject_data

def remove_unknown_unavailable(data_path):
    df = pd.read_csv(data_path)
    unwanted_values = ['unavailable', 'unknown']
    df.replace(unwanted_values, np.nan, inplace=True)
    df_cleaned = df.dropna().copy()
    
    for i in reversed(list(df_cleaned.index)[1:]):
        # Compare the current row's sensor status with the previous row's sensor status
        if df_cleaned.loc[i].sensor_status == df_cleaned.iloc[df_cleaned.index.get_loc(i)-1].sensor_status:
            # Drop the current row if the statuses match
            df_cleaned.drop([i], inplace=True)
            
    np_array = np.array(df_cleaned, dtype='object')
    return df_cleaned, np_array

def get_available_months_years(np_array, column_number_with_time = 2):
    timestamps = np_array[:, column_number_with_time].astype(int)
    datetime_values = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]
    dates = pd.to_datetime(datetime_values)
    periods = dates.to_period('M')
    unique_periods = np.unique(periods)
    return unique_periods

def extract_monthly_data(np_array, target_month = '2024-03'):
    df = pd.DataFrame(np_array, columns=['sensor_id', 'value', 'timestamp', 'subject_id'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    target_period = pd.Period(target_month, freq='M')
    df.set_index('timestamp', inplace=True)
    filtered_df = df[df.index.to_period('M') == target_period]
    indices = filtered_df.index
    filtered_np_array = filtered_df.reset_index().values
    return filtered_df, indices, filtered_np_array

def filter_consecutive_on_and_off_in_real_valued_sensor(df):
    valid_indices = []
    for i in range(len(df) - 1):
        if df.loc[i, 'sensor_status'] == 'on' and df.loc[i + 1, 'sensor_status'] == 'off':
            valid_indices.append(i)
            valid_indices.append(i + 1)
    filtered_df = df.iloc[valid_indices]
    return filtered_df

def time_current_off_and_next_on(df):
    results = []
    for i in range(len(df)-1):
        ts_off = df.loc[i, 'ts_off']
        next_ts_on = df.loc[i + 1, 'ts_on']
        results.append({'ts_off': ts_off, 'next_ts_on': next_ts_on, 'difference': np.abs(next_ts_on - ts_off)})
    return results

def plot_data_with_peaks(df, time_col, value_col, peaks):
    df[time_col] = pd.to_datetime(df[time_col], unit='ms')
    df_peaks = df.iloc[peaks]    
    fig = px.line(df, x=time_col, y=value_col, title='Data with Peaks')
    fig.add_trace(
        go.Scatter(
            x=df_peaks[time_col],
            y=df_peaks[value_col],
            mode='markers',
            name='Peaks',
            marker=dict(color='red', size=10, symbol='x')
        )
    )
    fig.update_layout(
        xaxis_title='Timestamp',
        yaxis_title='Value',
        legend_title='Legend',
        template='plotly_white'
    )
    fig.update_yaxes(range=[df[value_col].min() - 1, df[value_col].max() + 1])
    fig.show()


def filter_sensors_by_window(df, start_time, end_time):
    # Convert 'ts_on' and 'ts_off' to datetime if they aren't already
    df['ts_on'] = pd.to_datetime(df['ts_on'])
    df['ts_off'] = pd.to_datetime(df['ts_off'])
    
    # Convert start_time and end_time to datetime
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # Filter rows where the sensor was active in the window
    active_sensors = df[((df['ts_on'] <= end_time) & (df['ts_off'] >= start_time))]
    
    return active_sensors

def filter_data_for_date(df, date_str):
    date = pd.to_datetime(date_str)
    filtered_df = df[(df['ts_on'].dt.date == date.date()) | (df['ts_off'].dt.date == date.date())]
    # filtered_df = df.loc[(df['ts_on'].dt.date == date) | (df['ts_off'].dt.date == date)]

    if filtered_df.empty:
        # print(f"No data found for the date: {date_str}")
        return filtered_df
    else:
        return filtered_df
    
def prepare_plot_data(date_specific_data):
    plot_data = []
    
    for entry in date_specific_data:
        sensor_name = entry['sensor']
        sensor_data = entry['data']
        
        if not sensor_data.empty:
            # Extracting the active periods
            for index, row in sensor_data.iterrows():
                plot_data.append({
                    'sensor': sensor_name,
                    'start_time': row['ts_on'],
                    'end_time': row['ts_off']
                })
    
    return pd.DataFrame(plot_data)

def plot_tsne(sample_size, data):
    # Aggregate data into samples
    aggregated_data = []
    for i in range(0, len(data), sample_size):
        sample = data[i:i+sample_size]
        if len(sample) == sample_size:
            aggregated_data.append(sample['sensor_status'].values)
    aggregated_data = np.array(aggregated_data)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=0)
    tsne_results = tsne.fit_transform(aggregated_data)
    
    # Plot the t-SNE results
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Sensor Status')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title(f't-SNE Plot with Sample Size {sample_size}')
    plt.show()

def remove_unknown_unavailable_v2(df):
    unwanted_values = ['unavailable', 'unknown']
    df.replace(unwanted_values, np.nan, inplace=True)
    df_cleaned = df.dropna().copy()
    
    for i in reversed(list(df_cleaned.index)[1:]):
        # Compare the current row's sensor status with the previous row's sensor status
        if df_cleaned.loc[i].sensor_status == df_cleaned.iloc[df_cleaned.index.get_loc(i)-1].sensor_status:
            # Drop the current row if the statuses match
            df_cleaned.drop([i], inplace=True)
            
    np_array = np.array(df_cleaned, dtype='object')
    return df_cleaned, np_array

    
def detect_peaks(df, value_col, prominence_value):
    peaks, properties = find_peaks(df[value_col], prominence=prominence_value)
    peak_indicator = ['off'] * len(df)
    for peak in peaks:
        if peak < len(peak_indicator):
            peak_indicator[peak] = 'on'
    df['peak_detected'] = peak_indicator
    return peaks,df

def plot_duration_distribution(on_off_diff):
    # Convert the list to a DataFrame
    df = pd.DataFrame(on_off_diff, columns=['end_time', 'start_time', 'duration'])
    
    # Ensure the time columns are in datetime format
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Calculate the duration in seconds and minutes
    df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['duration_minutes'] = df['duration_seconds'] / 60
    
    # Create bins of five minutes up to 120 minutes, and include one bin for > 2 hours
    bins = list(range(0, 120, 5)) + [float('inf')]
    labels = [f"{i}-{i+5} min" for i in range(0, 115, 5)] + [">120 min"]
    
    # Bin the data
    df['binned_duration'] = pd.cut(df['duration_minutes'], bins=bins, labels=labels, right=False)
    
    # Check for NaNs in the binned_duration
    if df['binned_duration'].isna().any():
        print("Some durations couldn't be binned. Please check the data.")
    
    # Count the number of occurrences in each bin
    bin_counts = df['binned_duration'].value_counts(sort=False).reset_index()
    bin_counts.columns = ['binned_duration', 'count']
    
    # Plot using plotly
    fig = px.bar(bin_counts, x='binned_duration', y='count', title='Distribution of Durations',
                 labels={'binned_duration': 'Duration Bins', 'count': 'Count'})
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()


def post_process_sensor_data(df):
    # Check if the first row's sensor_status is 'off', and remove it if true because I want to start my analysis from ON
    if df.iloc[0]['sensor_status'] == 'off':
        df = df.iloc[1:].reset_index(drop=True)

    # print(np.unique(df['sensor_status']))
    # Create masks for filtering 'on' and 'off' statuses
    mask_on = df['sensor_status'] == 'on'
    mask_off = df['sensor_status'] == 'off'

    # Separate DataFrames for 'on' and 'off' statuses
    df_on = df[mask_on].copy()
    df_off = df[mask_off].copy()

    # # Rename timestamp columns
    df_on.rename(columns={'ts': 'ts_on_ms'}, inplace=True)  # Keep original milliseconds
    df_off.rename(columns={'ts': 'ts_off_ms'}, inplace=True)  # Keep original milliseconds
            
    df_on['ts_on'] = pd.to_datetime(df_on['ts_on_ms'], unit='ms')
    df_off['ts_off'] = pd.to_datetime(df_off['ts_off_ms'], unit='ms')

    # # Reset index for merging
    df_on.reset_index(drop=True, inplace=True)
    df_off.reset_index(drop=True, inplace=True)

    df_conc = pd.merge(df_on, df_off[['ts_off_ms', 'ts_off']], left_index=True, right_index=True)    

    # # Calculate duration
    df_conc['duration_readable'] = df_conc['ts_off'] - df_conc['ts_on']
    df_conc['duration_ms'] = df_conc['ts_off_ms'] - df_conc['ts_on_ms']
    
    new_order = ['sensor_id', 'subject_id', 'ts_on', 'ts_off', 'duration_readable', 'ts_on_ms', 'ts_off_ms', 'duration_ms', 'sensor_status']
    df_conc = df_conc[new_order]    
    return df_conc
    
def preprocess_dataframe(df):
    unwanted_values = ['unavailable', 'unknown']
    df.replace(unwanted_values, np.nan, inplace=True)
    df_cleaned = df.dropna().copy()
    
    for i in reversed(list(df_cleaned.index)[1:]):
        # Compare the current row's sensor status with the previous row's sensor status
        if df_cleaned.loc[i].sensor_status == df_cleaned.iloc[df_cleaned.index.get_loc(i)-1].sensor_status:
            # Drop the current row if the statuses match
            df_cleaned.drop([i], inplace=True)
            
    np_array = np.array(df_cleaned, dtype='object')
    return df_cleaned, np_array    


def filter_data_for_month(df, month_str):
    
    # Convert the month string to a datetime object
    month = pd.to_datetime(month_str, format='%Y-%m')

    # Filter the DataFrame based on the month and year
    filtered_df = df[(df['ts_on'].dt.year == month.year) & (df['ts_on'].dt.month == month.month) |
                     (df['ts_off'].dt.year == month.year) & (df['ts_off'].dt.month == month.month)]
    filtered_df = filtered_df.reset_index(drop=True)
    # print(filtered_df)
    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        return filtered_df  # Return the empty DataFrame if no data is found
    else:
        return filtered_df  # Return the filtered DataFrame
    
def remove_continuous_on_off(df_cleaned):    
    for i in reversed(list(df_cleaned.index)[1:]):
            # Compare the current row's sensor status with the previous row's sensor status
            if df_cleaned.loc[i].sensor_status == df_cleaned.iloc[df_cleaned.index.get_loc(i)-1].sensor_status:
                # Drop the current row if the statuses match
                df_cleaned.drop([i], inplace=True)
    return df_cleaned

def plot_duration_distribution2(on_off_diff, bin_duration=30, next_on_time=None, current_off_time=None):
    # Convert the list to a DataFrame
    df = pd.DataFrame(on_off_diff, columns=['end_time', 'start_time', 'duration'])
    
    # Ensure the time columns are in datetime format
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Calculate the duration in seconds and minutes
    df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['duration_minutes'] = df['duration_seconds'] / 60
    
    # Create bins based on the bin_duration and add a bin for durations greater than the max bin
    max_bin = (120 // bin_duration) * bin_duration
    bins = list(range(0, max_bin + bin_duration, bin_duration)) + [float('inf')]
    labels = [f"{i}-{i+bin_duration} min" for i in range(0, max_bin, bin_duration)] + [f">{max_bin} min"]
    
    # Bin the data
    df['binned_duration'] = pd.cut(df['duration_minutes'], bins=bins, labels=labels, right=False)
    
    # Check for NaNs in the binned_duration
    if df['binned_duration'].isna().any():
        print("Some durations couldn't be binned. Please check the data.")
    
    # Count the number of occurrences in each bin
    bin_counts = df['binned_duration'].value_counts(sort=False).reset_index()
    bin_counts.columns = ['binned_duration', 'count']
    
    # Plot using plotly
    fig = px.bar(bin_counts, x='binned_duration', y='count', title='Distribution of Durations',
                 labels={'binned_duration': 'Duration Bins', 'count': 'Count'})
    
    # Add annotations for next_on_time and current_off_time
    annotations = []
    
    if next_on_time is not None:
        annotations.append(
            go.layout.Annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.2,
                text=f"Next On Time: {next_on_time}",
                showarrow=False,
                font=dict(size=12, color="black")
            )
        )
    
    if current_off_time is not None:
        annotations.append(
            go.layout.Annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.3,
                text=f"Current Off Time: {current_off_time}",
                showarrow=False,
                font=dict(size=12, color="black")
            )
        )
    
    fig.update_layout(xaxis_tickangle=-45, annotations=annotations)
    
    fig.show()
    
#%%
def timedelta_to_seconds(td):
    return td.total_seconds()

def create_combined_df(out_event_counts, peak_counts, all_dates):
    combined_data = {'Date': all_dates}
    combined_data['Out Events'] = get_values_for_dates({date: count for month_data in out_event_counts.values() for date, count in month_data.items()}, all_dates)
    combined_data['Peaks'] = get_values_for_dates({date: count for month_data in peak_counts.values() for date, count in month_data.items()}, all_dates)
    return pd.DataFrame(combined_data)

# Function to get values for dates
def get_values_for_dates(data_dict, all_dates):
    return [data_dict.get(date, 0) for date in all_dates]

# Plot the data
def plot_out_event_and_peak_counts(combined_df, month, sn, kk):
    if kk == 13:
        temp_str = 'Temperature Peaks'
    elif kk ==12:
        temp_str = 'Humidity Peaks'
        
    plt.figure(figsize=(14, 7))

    x = np.arange(len(combined_df['Date']))  # X locations for the groups
    bar_width = 0.4  # Width of the bars

    # Plot bars
    plt.bar(x - bar_width / 2, combined_df['Out Events'], bar_width, label='Probable Out Events', alpha=0.7)
    plt.bar(x + bar_width / 2, combined_df['Peaks'], bar_width, label=temp_str, alpha=0.7)

    # Format x-axis
    plt.xticks(x, combined_df['Date'], rotation=90)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    
    # Add titles and labels
    plt.title(f'Probable Out-Events and Peak Counts for {month} and for '+str(subject_index_mapping[sn]))
    plt.xlabel('Date')
    plt.ylabel('Count')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig('out_events_and_peak_count'+str(month)+'_'+str(subject_index_mapping[sn])+'.png')

    plt.show()


def background_subtraction_denoise(df, background_method='median', window_size=5):
    df = df.copy()

    # Ensure that sensor_value is numeric
    df['sensor_values'] = pd.to_numeric(df['sensor_values'], errors='coerce')

    # Sort the DataFrame by timestamp for consistent rolling
    df = df.sort_values(by='ts')
    df['ts'] = pd.to_datetime(df['ts'], unit = 'ms',errors='coerce')


    # Calculate the background based on the chosen method
    if background_method == 'mean':
        df['background'] = df['sensor_values'].rolling(window=window_size, center=True, min_periods=1).mean()
    elif background_method == 'median':
        df['background'] = df['sensor_values'].rolling(window=window_size, center=True, min_periods=1).median()
    else:
        raise ValueError("background_method must be either 'mean' or 'median'")

    # Subtract the background from the original sensor value to get the denoised signal
    df['denoised_sensor_value'] = df['sensor_values'] - df['background']
    df['denoised_sensor_value'] = df['denoised_sensor_value'].apply(lambda x: max(x, 0))
    return df

#dff, peaks = df_with_peaks, peaks
def plot_raw_sensor_values(dff,peaks=None): 
    import matplotlib.dates as mdates
    plt.figure(figsize=(14, 7))
    dff['ts'] = pd.to_datetime(dff['ts'], unit='ms')
    plt.plot(dff['ts'], dff['sensor_values'], label='Sensor Values', color='blue')
    
    
    if peaks is not None:
        plt.scatter(dff.iloc[peaks]['ts'], dff.iloc[peaks]['sensor_values'], color='red', label='Peaks', zorder=5)

    unique_dates = dff['datetime'].dt.date.unique()
    # print(unique_dates)
    for date in unique_dates:
        start_of_day = pd.Timestamp(date)
        end_of_day = pd.Timestamp(date) + pd.Timedelta(days=1)
        
        plt.axvline(x=start_of_day, color='gray', linestyle='--', alpha=0.6)
        plt.axvline(x=end_of_day, color='gray', linestyle='--', alpha=0.6)
    
    
    # Add titles and labels
    plt.title('Sensor Values Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Value')
    
    # Format x-axis to show each date properly
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Rotate x-axis labels by 90 degrees for better space management
    plt.xticks(rotation=90, ha='center')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_denoised_sensor_values(dff,peaks=None): 
    import matplotlib.dates as mdates

    plt.figure(figsize=(14, 7))

    # Plot the denoised sensor values
    plt.plot(dff['ts'], dff['denoised_sensor_value'], label='Denoised Sensor Value', color='blue')
    
    if peaks is not None:
        plt.scatter(dff.iloc[peaks]['ts'], dff.iloc[peaks]['denoised_sensor_value'], color='red', label='Peaks', zorder=5)


    # Add vertical lines at the start and end of each day
    unique_dates = dff['ts'].dt.date.unique()
    # print(unique_dates)
    for date in unique_dates:
        start_of_day = pd.Timestamp(date)
        end_of_day = pd.Timestamp(date) + pd.Timedelta(days=1)
        
        plt.axvline(x=start_of_day, color='gray', linestyle='--', alpha=0.6)
        plt.axvline(x=end_of_day, color='gray', linestyle='--', alpha=0.6)
    
    
    # Add titles and labels
    plt.title('Denoised Sensor Values Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Denoised Sensor Value')
    
    # Format x-axis to show each date properly
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Rotate x-axis labels by 90 degrees for better space management
    plt.xticks(rotation=90, ha='center')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
   

def detect_peaks(df, column_name, prominence=1):
    peaks, _ = find_peaks(df[column_name], prominence=prominence)
    df['peak_detected'] = 0
    df.loc[peaks, 'peak_detected'] = 1
    return peaks, df

def compute_probable_out_events(house_entrance_df, month, year):
    house_entrance_df_specific_month = house_entrance_df[
        (house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

    out_event_count = {}
    out_event_times = {}  
    
    for i in range(1, len(house_entrance_df_specific_month)):
        ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
        we = house_entrance_df_specific_month.iloc[i]['ts_on']
        time_difference = we - ws
        
        is_out_event = True
        for j in range(len(df)):
            # J = 9 belongs to the magnetic sensor of the door
            if j == 9:
                continue
            
            temp_sensor = df[j]['sensor'] # sensor name
            temp_sensor_df = df[j]['filtered_df'] # 
            
            if len(temp_sensor_df) > 0:
                # Check if any sensor was activated during the ws-we interval
                temp_df = temp_sensor_df[(temp_sensor_df['ts_on'] >= ws) & (temp_sensor_df['ts_on'] <= we)]
        
                if not temp_df.empty:
                    is_out_event = False
                    break
        
        if is_out_event:
            date_str = ws.date().strftime('%Y-%m-%d')  # Convert the date to a string for counting
            out_event_info = (ws, we)  # Tuple of start and end times of the out-event
            
            if date_str in out_event_count:
                out_event_count[date_str] += 1
                out_event_times[date_str].append(out_event_info)
            else:
                out_event_count[date_str] = 1
                out_event_times[date_str] = [out_event_info]
                
    return out_event_count, out_event_times

def compute_probable_monthly_out_events(house_entrance_df,df, month, year):
    # Filter data for the specific month and year
    house_entrance_df_specific_month = house_entrance_df[
        (house_entrance_df['ts_on'].dt.year == year) & 
        (house_entrance_df['ts_on'].dt.month == month)
    ]

    # Initialize counters
    out_event_count = 0  # Count for the entire month
    out_event_times = []  # List to store all out-event times for the month

    # Loop through each possible out-event in the filtered DataFrame
    for i in range(1, len(house_entrance_df_specific_month)):
        ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
        we = house_entrance_df_specific_month.iloc[i]['ts_on']
        time_difference = we - ws
        
        is_out_event = True
        
        # Check for activity from other sensors within the ws-we interval
        for j in range(len(df)):
            if j == 9:  # Skip the magnetic sensor of the door
                continue
            
            temp_sensor_df = df[j]['filtered_df']
            
            if len(temp_sensor_df) > 0:
                # Check if any sensor was activated during the ws-we interval
                temp_df = temp_sensor_df[(temp_sensor_df['ts_on'] >= ws) & (temp_sensor_df['ts_on'] <= we)]
        
                if not temp_df.empty:
                    is_out_event = False
                    break
        
        # If no sensor activity is detected, count it as an out event
        if is_out_event:
            out_event_count += 1  # Increment the monthly count
            out_event_info = (ws, we)  # Tuple of start and end times of the out-event
            out_event_times.append(out_event_info)  # Store the out-event times
    
    return out_event_count, out_event_times



def create_all_out_event_peaks(df, year, month):

    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

    if 'date' not in df.columns:
        df['date'] = df['ts'].dt.date    

    # Filter the DataFrame for the specific year and month
    df_month = df[(df['ts'].dt.year == year) & (df['ts'].dt.month == month)]
    peaks_per_day = {}
    
    # Count peaks per day
    for date, group in df_month.groupby('date'):
        peaks_count = group['peak_detected'].sum()  # Count of peaks on this date
        date_str = date.strftime('%Y-%m-%d')
        peaks_per_day[date_str] = peaks_count
    
    return peaks_per_day



def plot_sensor_values_with_peaks(stove_humidity_raw_data, year, month, detect_peaks):
    # Convert 'ts' from milliseconds to datetime
    stove_humidity_raw_data['datetime'] = pd.to_datetime(stove_humidity_raw_data['ts'], unit='ms')
    
    # Filter DataFrame by year and month
    filtered_df_raw_stove = stove_humidity_raw_data[
        (stove_humidity_raw_data['datetime'].dt.year == year) &
        (stove_humidity_raw_data['datetime'].dt.month == month)
    ]
    
    # Reset index after filtering
    filtered_df_raw_stove = filtered_df_raw_stove.reset_index(drop=True)
    
        
    # Detect peaks
    peaks_raw, df_with_peaks_raw = detect_peaks(filtered_df_raw_stove, 'sensor_values', prominence=2)
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    
    # Plot sensor values
    plt.plot(filtered_df_raw_stove['datetime'], filtered_df_raw_stove['sensor_values'], linestyle='-', color='b', label='Sensor Values')
    
    # Plot peaks
    plt.plot(filtered_df_raw_stove['datetime'].iloc[peaks_raw], filtered_df_raw_stove['sensor_values'].iloc[peaks_raw], 'ro', label='Detected Peaks')
    
    unique_dates = filtered_df_raw_stove['datetime'].dt.date.unique()
    print(unique_dates)
    for date in unique_dates:
        start_of_day = pd.Timestamp(date)
        end_of_day = pd.Timestamp(date) + pd.Timedelta(days=1)
        
        plt.axvline(x=start_of_day, color='gray', linestyle='--', alpha=0.6)
        plt.axvline(x=end_of_day, color='gray', linestyle='--', alpha=0.6)
        
        
    # Formatting the plot
    plt.title(f'Sensor Values with Detected Peaks for {year}-{month:02d}')
    plt.xlabel('Time (Date and Time)')
    plt.ylabel('Sensor Values')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    
    # Show the legend
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    
def plot_sensor_values_with_peaks22(dff,year, month, detect_peaks): 
    import matplotlib.dates as mdates
    dff['datetime'] = pd.to_datetime(dff['ts'], unit='ms')
    dff = dff[
        (dff['datetime'].dt.year == year) &
        (dff['datetime'].dt.month == month)
    ]
    peaks_raw, df_with_peaks_raw = detect_peaks(dff, 'sensor_values', prominence=2)
    plt.figure(figsize=(14, 7))

    # Plot the denoised sensor values
    plt.plot(dff['datetime'], dff['sensor_values'], label='sensor_values Sensor Value', color='blue')
    
    if peaks_raw is not None:
        plt.scatter(dff.iloc[peaks_raw]['datetime'], dff.iloc[peaks_raw]['sensor_values'], color='red', label='Peaks', zorder=5)


    # Add vertical lines at the start and end of each day
    unique_dates = dff['datetime'].dt.date.unique()
    print(unique_dates)
    for date in unique_dates:
        # Convert date to timestamp for plotting
        start_of_day = pd.Timestamp(date)
        end_of_day = start_of_day + pd.Timedelta(days=1)
        
        # Plot vertical lines for the start and end of the day
        plt.axvline(x=start_of_day, color='gray', linestyle='--', alpha=0.6)
        plt.axvline(x=end_of_day, color='gray', linestyle='--', alpha=0.6)
        
    
    # Add titles and labels
    plt.title('Denoised Sensor Values Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Denoised Sensor Value')
    
    # Format x-axis to show each date properly
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Rotate x-axis labels by 90 degrees for better space management
    plt.xticks(rotation=90, ha='center')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
    
