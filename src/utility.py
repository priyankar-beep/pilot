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
    "Hum_Temp_Stove_humidity.csv",
    "Hum_Temp_Stove_temp.csv",
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

def load_subjectwise_data(data_path, subjects, unwanted_values, skip_substrings):
    subject_data = []
    for sub in range(3):#len(subjects)):
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
                    peaks, sdf = detect_peaks(sdf, 'sensor_status', prominence_value = 1)
                    sdf = sdf.rename(columns={'sensor_status': 'sensor_values', 'peak_detected': 'sensor_status'})
                    sdf = sdf.reset_index(drop=True)
                    filtered_df = filter_consecutive_on_and_off_in_real_valued_sensor(sdf) ## Only for real-valued sensors
                    filtered_df = filtered_df.reset_index(drop=True)
                    filtered_df = post_process_sensor_data(filtered_df)
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