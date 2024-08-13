import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from utility import *

data_path = '/home/hubble/work/serenade/data'
subjects = ['subject_1','subject_2','subject_3','subject_4','subject_5','subject_7','subject_8']
unwanted_values = ['unavailable', 'unknown']
skip_substrings = ['event', 'HousejkjlEntrance']

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




def plot_denoised_sensor_values(dff,peaks=None): 
    import matplotlib.dates as mdates

    plt.figure(figsize=(14, 7))

    # Plot the denoised sensor values
    plt.plot(dff['ts'], dff['denoised_sensor_value'], label='Denoised Sensor Value', color='blue')
    
    if peaks is not None:
        plt.scatter(dff.iloc[peaks]['ts'], dff.iloc[peaks]['denoised_sensor_value'], color='red', label='Peaks', zorder=5)


    # Add vertical lines at the start and end of each day
    unique_dates = dff['ts'].dt.date.unique()
    print(unique_dates)
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
        (house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)
    ]

    out_event_count = {}
    out_event_times = {}  # Dictionary to store out-event times
    
    # Loop through the dataset to find out-events
    for i in range(1, len(house_entrance_df_specific_month)):
        ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
        we = house_entrance_df_specific_month.iloc[i]['ts_on']
        time_difference = we - ws
        
        is_out_event = True
        for j in range(len(df)):
            if j == 9:
                continue
            temp_sensor = df[j]['sensor']
            temp_sensor_df = df[j]['filtered_df']
            
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

def create_all_out_event_peaks(df, year, month):

    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')

    # Extract 'date' from 'ts' if not already present
    if 'date' not in df.columns:
        df['date'] = df['ts'].dt.date    

    # Filter the DataFrame for the specific year and month
    df_month = df[(df['ts'].dt.year == year) & (df['ts'].dt.month == month)]
    print(df_month.head())
    # Initialize dictionary to store the number of peaks per day
    peaks_per_day = {}
    
    # Count peaks per day
    for date, group in df_month.groupby('date'):
        peaks_count = group['peak_detected'].sum()  # Count of peaks on this date
        date_str = date.strftime('%Y-%m-%d')
        peaks_per_day[date_str] = peaks_count
    
    return peaks_per_day


subject_data = load_subjectwise_data(data_path, subjects, unwanted_values, skip_substrings)

df = subject_data[2].copy()
house_entrance_df = df[9]['filtered_df'].copy()
house_entrance_df['ts_on'] = pd.to_datetime(house_entrance_df['ts_on'])

months_of_interest = {
    # 2023: [11, 12],
    2024: [3, 4, 5],
}
years = list(months_of_interest.keys())

# Function to convert timedelta to seconds
def timedelta_to_seconds(td):
    return td.total_seconds()


all_out_event_counts = {}
all_out_event_times = {}
all_out_event_peaks = {}


for year in years:
    all_out_event_counts[year] = {}
    all_out_event_times[year] = {}
    all_out_event_peaks[year] = {}
    
    months = months_of_interest[year]
    for month in months:
        out_event_count, out_event_times = compute_probable_out_events(house_entrance_df, month, year)

        all_out_event_counts[year][month] = out_event_count
        all_out_event_times[year][month] = out_event_times
        
        k = 13
        stove_sensor_name = df[k]['sensor']
        stove_humidity_df = df[k]['filtered_df']
        stove_humidity_raw_data = df[k]['sensor_df']
        
        if len(stove_humidity_raw_data) > 0:
            # Check if any sensor was activated during the ws-we interval
            denoised_df = background_subtraction_denoise(stove_humidity_raw_data, background_method='mean', window_size=5)
            denoised_df_month = denoised_df[
                (denoised_df['ts'].dt.year == year) & (denoised_df['ts'].dt.month == month)
            ]
            denoised_df_month = denoised_df_month.reset_index(drop = True)
            peaks, df_with_peaks = detect_peaks(denoised_df_month, 'denoised_sensor_value', prominence=0.3)
            peaks_per_day = create_all_out_event_peaks(df_with_peaks, year, month)

            # plot_denoised_sensor_values(df_with_peaks, peaks)
            all_out_event_peaks[year][month] = peaks_per_day

def create_combined_df(out_event_counts, peak_counts, all_dates):
    combined_data = {'Date': all_dates}
    combined_data['Out Events'] = get_values_for_dates({date: count for month_data in out_event_counts.values() for date, count in month_data.items()}, all_dates)
    combined_data['Peaks'] = get_values_for_dates({date: count for month_data in peak_counts.values() for date, count in month_data.items()}, all_dates)
    return pd.DataFrame(combined_data)

# Function to get values for dates
def get_values_for_dates(data_dict, all_dates):
    return [data_dict.get(date, 0) for date in all_dates]

# Plot the data
def plot_out_event_and_peak_counts(combined_df, month):
    plt.figure(figsize=(14, 7))

    x = np.arange(len(combined_df['Date']))  # X locations for the groups
    bar_width = 0.4  # Width of the bars

    # Plot bars
    plt.bar(x - bar_width / 2, combined_df['Out Events'], bar_width, label='Out Events', alpha=0.7)
    plt.bar(x + bar_width / 2, combined_df['Peaks'], bar_width, label='Peaks', alpha=0.7)

    # Format x-axis
    plt.xticks(x, combined_df['Date'], rotation=90)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    
    # Add titles and labels
    plt.title(f'Out-Events and Peak Counts for {month}')
    plt.xlabel('Date')
    plt.ylabel('Count')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

for year in all_out_event_counts.keys():
    for month in all_out_event_counts[year].keys():
        out_event_counts = all_out_event_counts[year][month]
        peak_counts = all_out_event_peaks[year][month]

        # Extract dates from both out_event_counts and peak_counts
        out_event_dates = set(out_event_counts.keys())
        peak_dates = set(peak_counts.keys())
        
        # Combine and sort all dates
        all_dates = sorted(out_event_dates.union(peak_dates))
        
        # Create combined DataFrame
        combined_df = create_combined_df({month: out_event_counts}, {month: peak_counts}, all_dates)
        
        # Plot
        plot_out_event_and_peak_counts(combined_df, f'{year}-{month:02d}')
# for year in all_out_event_counts.keys():
#     for month in all_out_event_counts[year].keys():
#         out_event_counts = all_out_event_counts[year]
#         peak_counts = all_out_event_peaks[year]
#         # Extract dates from both out_event_counts and peak_counts
#         out_event_dates = {date for month_data in out_event_counts.values() for date in month_data.keys()}
#         peak_dates = {date for month_data in peak_counts.values() for date in month_data.keys()}
        
#         # Combine and sort all dates
#         all_dates = sorted(out_event_dates.union(peak_dates))
        
#         # Prepare data for plotting
#         def get_values_for_dates(data_dict, all_dates):
#             return [data_dict.get(date, 0) for date in all_dates]
        
#         # Create a DataFrame for plotting
#         def create_combined_df(out_event_counts, peak_counts):
#             combined_data = {'Date': all_dates}
#             combined_data['Out Events'] = get_values_for_dates({date: count for month_data in out_event_counts.values() for date, count in month_data.items()}, all_dates)
#             combined_data['Peaks'] = get_values_for_dates({date: count for month_data in peak_counts.values() for date, count in month_data.items()}, all_dates)
#             return pd.DataFrame(combined_data)
        
#         combined_df = create_combined_df(out_event_counts, peak_counts)
#         plot_out_event_and_peak_counts(combined_df)
        

# def plot_out_event_and_peak_counts(combined_df):
#     plt.figure(figsize=(14, 7))

#     x = np.arange(len(combined_df['Date']))  # X locations for the groups
#     bar_width = 0.4  # Width of the bars

#     # Plot bars
#     plt.bar(x - bar_width / 2, combined_df['Out Events'], bar_width, label='Out Events', alpha=0.7)
#     plt.bar(x + bar_width / 2, combined_df['Peaks'], bar_width, label='Peaks', alpha=0.7)

#     # Format x-axis
#     plt.xticks(x, combined_df['Date'], rotation=90)
#     plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    
#     # Add titles and labels
#     plt.title('Out-Events and Peak Counts by Date')
#     plt.xlabel('Date')
#     plt.ylabel('Count')

#     # Add grid for better readability
#     plt.grid(axis='y', linestyle='--', alpha=0.7)

#     # Show legend
#     plt.legend()

#     # Display the plot
#     plt.tight_layout()
#     plt.show()