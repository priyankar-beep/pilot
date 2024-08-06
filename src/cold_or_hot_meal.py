import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from scipy.signal import find_peaks

def remove_unknown_unavailable(data_path):
    df = pd.read_csv(data_path)
    unwanted_values = ['unavailable', 'unknown']
    df.replace(unwanted_values, np.nan, inplace=True)
    df_cleaned = df.dropna()
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

###
data_path = '/home/hubble/work/serenade/data/subject_3/environmentals/Stove_Hum_Temp_temp.csv'
cleaned_df, cleaned_array = remove_unknown_unavailable(data_path)
available_months = get_available_months_years(cleaned_array, column_number_with_time = 2)
target_month = '2024-03'
df_month,indices, array_month = extract_monthly_data(cleaned_array, target_month)


timestamps = array_month[:, 0]
values = array_month[:, 2].astype(float)

# # Plotting using Matplotlib
# plt.figure(figsize=(12, 6))

# # Seaborn lineplot with NumPy array
# sns.lineplot(x=timestamps, y=values, label='Values')

# # Adding titles and labels
# plt.title(target_month)
# plt.xlabel('Timestamp')
# plt.ylabel('Values')

# plt.xticks(timestamps, [timestamp.strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps], rotation=45)


# # Formatting x-axis to avoid clutter
# # plt.xticks(rotation=45)
# plt.tight_layout()

# # Show plot
# plt.legend()
# plt.show()


# Convert to DataFrame for Plotly
df = pd.DataFrame(array_month, columns=['timestamp', 'sensor_id', 'value', 'subject_id'])

# Plotting using Plotly
fig = px.line(df, x='timestamp', y='value', title='Plot of Values from Second Column',
              labels={'timestamp': 'Timestamp', 'value': 'Values'},
              markers=True)

# Customize hover data
fig.update_traces(mode='lines+markers', hovertemplate='%{x}<br>Value: %{y}')

# Customize layout
fig.update_layout(
    xaxis=dict(
        tickformat='%Y-%m-%d %H:%M:%S',
        tickangle=45,
        nticks=10,
        tickmode='linear'
    ),
    yaxis_title='Values'
)

# Show plot
fig.show()