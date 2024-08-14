import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from utility import *

data_path = '/home/hubble/work/serenade/data'
subjects = ['subject_1','subject_2','subject_3','subject_4','subject_5','subject_7','subject_8']
unwanted_values = ['unavailable', 'unknown']
skip_substrings = ['event', 'HousejkjlEntrance']

subject_index_mapping = {0: 'subject-1', 1:'subject-2', 2:'subject-3', 3:'subject-4', 4: 'subject-5', 5:'subject-7', 6:'subject-8'}
subject_data = load_subjectwise_data(data_path, subjects, unwanted_values, skip_substrings)


# Function to format y-axis ticks as time
def format_time(x, pos):
    return str(timedelta(seconds=int(x)))

# Function to convert timedelta to seconds
def timedelta_to_seconds(td):
    return td.total_seconds()

# Function to get week number of the month
def week_of_month(date):
    return (date.day - 1) // 7 + 1
#%%
sn = 2
# Replace these with your actual subject_data and data extraction logic
df = subject_data[sn].copy()
house_entrance_df = df[9]['filtered_df'].copy()
house_entrance_df['ts_on'] = pd.to_datetime(house_entrance_df['ts_on'])
"""
Date Specific Trends
"""
months_of_interest = {
    2023: [10,11,12],
    2024: [1,2,3,4,5]
}
years = list(months_of_interest.keys())


# Iterate over each year
for year in years:
    months = months_of_interest[year]
    for month in months:
        house_entrance_df_specific_month = house_entrance_df[(house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

        out_event_count = {}
        out_event_durations = {}
        
        # Loop through the dataset to find out-events
        for i in range(1, len(house_entrance_df_specific_month)):
            ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
            we = house_entrance_df_specific_month.iloc[i]['ts_on']
            time_difference = we - ws
            
            # Assume the subject was outside
            is_out_event = True
            
            # Check each sensor to see if it was activated during the ws-we interval
            for j in range(len(df)):
                if j == 9:
                    continue
                
                temp_sensor = df[j]['sensor']
                temp_sensor_df = df[j]['filtered_df']
                
                if len(temp_sensor_df) > 0:
                    # Filter the data to check if any sensor was activated between ws and we
                    temp_df = temp_sensor_df[(temp_sensor_df['ts_on'] >= ws) & (temp_sensor_df['ts_on'] <= we)]
                    
                    # If any sensor was activated, it's not an out-event
                    if not temp_df.empty:
                        is_out_event = False
                        break
            
            # If no sensors were activated, it was an out-event
            if is_out_event:
                date_str = ws.date().strftime('%Y-%m-%d')  # Convert the date to a string for counting
                if date_str in out_event_count:
                    out_event_count[date_str] += 1
                else:
                    out_event_count[date_str] = 1
                
                # Collect durations and dates for scatter plot
                if date_str not in out_event_durations:
                    out_event_durations[date_str] = []
                out_event_durations[date_str].append(timedelta_to_seconds(time_difference))
        
        # Prepare data for plotting
        date_list = list(out_event_count.keys())
        count_list = list(out_event_count.values())
        
        # Prepare data for scatter plot
        scatter_dates = []
        scatter_durations = []
        
        for date in date_list:
            if date in out_event_durations:
                scatter_dates.extend([date] * len(out_event_durations[date]))
                scatter_durations.extend(out_event_durations[date])
        
        # Create a plot
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Bar plot for number of out-events per day
        ax1.bar(date_list, count_list, color='skyblue', label='Number of Probable Out-Events', alpha=0.6)
        
        # Create a second y-axis for the duration plot
        ax2 = ax1.twinx()
        scatter = ax2.scatter(scatter_dates, scatter_durations, color='red', label='Duration of Problable Out-Events', alpha=0.7, s=50)
        
        # Format x-axis to show all dates of the month
        ax1.set_xticks(date_list)
        ax1.set_xticklabels(date_list, rotation=90)

        # Format y-axis for counts
        ax1.set_ylabel('Number of Probable Out-Events', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')

        # Format y-axis for durations
        ax2.set_ylabel('Duration of Probable Out-Events (Seconds)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Format y-axis to show time of day
        # ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_time))

        # Add titles and labels
        plt.title(f'Probable Out-Events and Durations for {year}-{month:02d} for {subject_index_mapping[sn]}')
        ax1.set_xlabel('Date')

        # Display the plot
        plt.tight_layout()  # Adjust plot to fit everything
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.savefig('out_events_plot_'+str(subject_index_mapping[sn])+'_'+str(year)+'_'+str(month)+'.png')
        plt.show()


# Sample data for demonstration purposes
# Replace these with your actual subject_data and data extraction logic
# df = subject_data[0].copy()
# house_entrance_df = df[9]['filtered_df'].copy()
# house_entrance_df['ts_on'] = pd.to_datetime(house_entrance_df['ts_on'])
#%%
"""
Monthly trends
"""

years = list(months_of_interest.keys())

# Initialize dictionaries to store counts and durations of out-events per month
monthly_event_count = {}
monthly_event_durations = {}


# Iterate over each year
for year in years:
    months = months_of_interest[year]
    for month in months:
        house_entrance_df_specific_month = house_entrance_df[(house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

        # Initialize count and durations for the current month
        out_event_count = 0
        durations = []
        
        # Loop through the dataset to find out-events
        for i in range(1, len(house_entrance_df_specific_month)):
            ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
            we = house_entrance_df_specific_month.iloc[i]['ts_on']
            time_difference = we - ws
            
            # Assume the subject was outside
            is_out_event = True
            
            # Check each sensor to see if it was activated during the ws-we interval
            for j in range(len(df)):
                if j == 9:
                    continue
                
                temp_sensor = df[j]['sensor']
                temp_sensor_df = df[j]['filtered_df']
                
                if len(temp_sensor_df) > 0:
                    # Filter the data to check if any sensor was activated between ws and we
                    temp_df = temp_sensor_df[(temp_sensor_df['ts_on'] >= ws) & (temp_sensor_df['ts_on'] <= we)]
                    
                    # If any sensor was activated, it's not an out-event
                    if not temp_df.empty:
                        is_out_event = False
                        break
            
            # If no sensors were activated, it was an out-event
            if is_out_event:
                out_event_count += 1
                durations.append(timedelta_to_seconds(time_difference))
        
        # Store the count of out-events and durations for the current month
        month_key = f"{year}-{month:02d}"
        monthly_event_count[month_key] = out_event_count
        monthly_event_durations[month_key] = durations

# Convert dictionaries to lists for plotting
months = list(monthly_event_count.keys())
counts = list(monthly_event_count.values())
durations = [item for sublist in monthly_event_durations.values() for item in sublist]
durations_months = [month_key for month_key, dur_list in monthly_event_durations.items() for _ in dur_list]

# Create a bar plot for monthly out-events
fig, ax1 = plt.subplots(figsize=(14, 7))

# Bar plot for number of out-events per month
bars = ax1.bar(months, counts, color='skyblue', edgecolor='black', alpha=0.7)

# Create a secondary y-axis for the duration plot
ax2 = ax1.twinx()
scatter = ax2.scatter(durations_months, durations, color='red', label='Duration of Probable Out-Events', alpha=0.7, s=50)

# Add titles and labels
ax1.set_title('Number of Probable Out-Events and Durations per Month ' +str(subject_index_mapping[sn]))
ax1.set_xlabel('Month')
ax1.set_ylabel('Number of Probable Out-Events')
ax2.set_ylabel('Duration of Probable Out-Events (Seconds)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# Add grid for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.7)
# Add legend
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

# Show the plot
plt.tight_layout()  # Adjust plot to ensure everything fits without overlap
plt.savefig('out_events_plot_montly_trend_'+str(subject_index_mapping[sn])+'.png')

plt.show()

# %%

years = list(months_of_interest.keys())
# Iterate over each year
for year in years:
    months = months_of_interest[year]
    for month in months:
        house_entrance_df_specific_month = house_entrance_df[(house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

        # Initialize dictionaries to store counts and durations of out-events per week
        week_event_count = {}
        week_event_durations = {}

        # Loop through the dataset to find out-events
        for i in range(1, len(house_entrance_df_specific_month)):
            ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
            we = house_entrance_df_specific_month.iloc[i]['ts_on']
            time_difference = we - ws

            # Assume the subject was outside
            is_out_event = True
            
            # Check each sensor to see if it was activated during the ws-we interval
            for j in range(len(df)):
                if j == 9:
                    continue
                
                temp_sensor = df[j]['sensor']
                temp_sensor_df = df[j]['filtered_df']
                
                if len(temp_sensor_df) > 0:
                    # Filter the data to check if any sensor was activated between ws and we
                    temp_df = temp_sensor_df[(temp_sensor_df['ts_on'] >= ws) & (temp_sensor_df['ts_on'] <= we)]
                    
                    # If any sensor was activated, it's not an out-event
                    if not temp_df.empty:
                        is_out_event = False
                        break

            # If no sensors were activated, it was an out-event
            if is_out_event:
                week = week_of_month(ws)
                
                # Update count of out-events per week
                if week in week_event_count:
                    week_event_count[week] += 1
                else:
                    week_event_count[week] = 1
                
                # Collect durations for scatter plot
                if week not in week_event_durations:
                    week_event_durations[week] = []
                week_event_durations[week].append(timedelta_to_seconds(time_difference))
        
        # Prepare data for plotting
        weeks = sorted(week_event_count.keys())
        counts = [week_event_count[week] for week in weeks]

        # Prepare data for scatter plot
        scatter_weeks = []
        scatter_durations = []
        
        for week in weeks:
            if week in week_event_durations:
                scatter_weeks.extend([week] * len(week_event_durations[week]))
                scatter_durations.extend(week_event_durations[week])
        
        # Create a bar plot for weekly out-events
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar plot for number of out-events per week
        # bars = ax1.bar(weeks, counts, color='skyblue', edgecolor='black', alpha=0.7)
        bars = ax1.bar(weeks, counts, color='skyblue', edgecolor='black', alpha=0.7, label='Number of Probable Out-Events')


        # Create a secondary y-axis for the duration plot
        ax2 = ax1.twinx()
        scatter = ax2.scatter(scatter_weeks, scatter_durations, color='red', label='Duration of Out-Events', alpha=0.7, s=50)

        # Add titles and labels
        ax1.set_title(f'Number of Probable Out-Events and Durations per Week for month {year}-{month:02d} for '+str(subject_index_mapping[sn]))
        ax1.set_xlabel('Week of Month')
        ax1.set_ylabel('Number of Probable Out-Events')
        ax2.set_ylabel('Duration of Out-Events (Seconds)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add grid for better readability
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend
        fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        # Show the plot
        plt.tight_layout()  # Adjust plot to ensure everything fits without overlap
        plt.savefig('out_events_plot_weekly_trend_'+str(year)+'_'+str(month)+'_'+str(subject_index_mapping[sn])+'.png')

        plt.show()
