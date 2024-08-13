#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:32:26 2024

@author: hubble
"""

from utility import *
data_path = '/home/hubble/work/serenade/data'
subjects = ['subject_1','subject_2','subject_3','subject_4','subject_5','subject_7','subject_8']
unwanted_values = ['unavailable', 'unknown']
skip_substrings = ['event', 'HousejkjlEntrance']

# %% Monthly analysis
for subject_index in range(len(subject_data)):
    df = subject_data[subject_index].copy()
    total_activation_counts = {}

    # Iterate over each sensor to find its activation count within the month
    for j in range(len(df)):
        if j == 9:
            continue  # Skip the sensor with index 9

        sensor = df[j]['sensor']
        current_data = df[j]['filtered_df']
        
        if len(current_data) > 0:
            # Filter the data for the specific month
            filtered_data = filter_data_for_month(current_data, '2023-12')
            activation_count = len(filtered_data)

            # Accumulate the activation count for each sensor
            if sensor in total_activation_counts:
                total_activation_counts[sensor] += activation_count
            else:
                total_activation_counts[sensor] = activation_count

    # Prepare data for plotting
    sensors = list(total_activation_counts.keys())
    activation_counts = list(total_activation_counts.values())

    # Plotting the total activation counts for each sensor in the month
    plt.figure(figsize=(10, 6))
    plt.bar(sensors, activation_counts, color='skyblue')

    # Add titles and labels
    plt.title(f'Total Sensor Activation Counts in November 2023 - Subject {subject_index+1}')
    plt.xlabel('Sensors')
    plt.ylabel('Total Activation Count')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=-90)

    # Display the plot
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
    plt.show()


# %% Loop over different months
df = subject_data[0].copy()
house_entrance_df = df[9]['filtered_df'].copy()
house_entrance_df['ts_on'] = pd.to_datetime(house_entrance_df['ts_on'])

months_of_interest = {
    2023: [12],

}
years = list(months_of_interest.keys())

# Iterate over each year
for y in range(len(years)):
    year = years[y]
    months = months_of_interest[year]
    for m in range(len(months)):
        month = months[m]
        house_entrance_df_specific_month = house_entrance_df[(house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

        out_event_count = {}
        out_event_duration = {}
        
        # Loop through the dataset to find out-events
        for i in range(1, len(house_entrance_df_specific_month)):
            ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
            we = house_entrance_df_specific_month.iloc[i]['ts_on']
            time_difference = we - ws
            print(we, ws, time_difference)
            
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
                    
                if date_str not in out_event_durations:
                    out_event_durations[date_str] = []
                out_event_durations[date_str].append(timedelta_to_seconds(time_difference))
            
                    
        # Print the results
        for date, count in out_event_count.items():
            print(f"Date: {date}, Out-events: {count}")
        
                    
        
        # Assuming out_event_count dictionary is already populated
        # Example: out_event_count = {'2023-11-01': 3, '2023-11-02': 5, ...}
        
        # Convert dictionary to lists for plotting
        dates = list(out_event_count.keys())
        counts = list(out_event_count.values())
        
        # Sort by date for better readability
        sorted_indices = sorted(range(len(dates)), key=lambda k: dates[k])
        dates = [dates[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(dates, counts, color='skyblue')
        
        # Add titles and labels
        plt.title('Number of Out-Events per Date')
        plt.xlabel('Date')
        plt.ylabel('Number of Out-Events')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Display the plot
        plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
        plt.show()



df = subject_data[0].copy()
house_entrance_df = df[9]['filtered_df'].copy()
house_entrance_df['ts_on'] = pd.to_datetime(house_entrance_df['ts_on'])

months_of_interest = {
    2023: [12],
    
}
years = list(months_of_interest.keys())

# Function to get week number of the month
def week_of_month(date):
    return (date.day - 1) // 7 + 1

# Iterate over each year
for year in years:
    months = months_of_interest[year]
    for month in months:
        house_entrance_df_specific_month = house_entrance_df[(house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

        # Initialize dictionary to store counts of out-events per week
        week_event_count = {}

        # Loop through the dataset to find out-events
        for i in range(1, len(house_entrance_df_specific_month)):
            ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
            we = house_entrance_df_specific_month.iloc[i]['ts_on']
            
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
                if week in week_event_count:
                    week_event_count[week] += 1
                else:
                    week_event_count[week] = 1
        
        # Convert dictionary to lists for plotting
        weeks = sorted(week_event_count.keys())
        counts = [week_event_count[week] for week in weeks]
        
        # Create a line plot for the weekly out-events
        plt.figure(figsize=(12, 6))
        plt.bar(weeks, counts, color='skyblue', edgecolor='black')

        # Add titles and labels
        plt.title(f'Number of Problable Out-Events per Week for {year}-{month:02d}')
        plt.xlabel('Week of Month')
        plt.ylabel('Number of Out-Events')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Show the plot
        plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
        plt.show()

        
#%%

df = subject_data[0].copy()
house_entrance_df = df[9]['filtered_df'].copy()
house_entrance_df['ts_on'] = pd.to_datetime(house_entrance_df['ts_on'])

months_of_interest = {
    2023: [9, 10, 11, 12],
    2024: [1, 2, 3, 4, 5]
}
years = list(months_of_interest.keys())

# Initialize dictionary to store counts of out-events per month
monthly_event_count = {}

# Iterate over each year
for year in years:
    months = months_of_interest[year]
    for month in months:
        house_entrance_df_specific_month = house_entrance_df[(house_entrance_df['ts_on'].dt.year == year) & (house_entrance_df['ts_on'].dt.month == month)]

        # Initialize count for the current month
        out_event_count = 0
        
        # Loop through the dataset to find out-events
        for i in range(1, len(house_entrance_df_specific_month)):
            ws = house_entrance_df_specific_month.iloc[i-1]['ts_off']
            we = house_entrance_df_specific_month.iloc[i]['ts_on']
            
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
        
        # Store the count of out-events for the current month
        month_key = f"{year}-{month:02d}"
        monthly_event_count[month_key] = out_event_count

# Convert dictionary to lists for plotting
months = list(monthly_event_count.keys())
counts = list(monthly_event_count.values())

# Create a bar plot for monthly out-events
plt.figure(figsize=(14, 7))
plt.bar(months, counts, color='skyblue', edgecolor='black')

# Add titles and labels
plt.title('Number of Out-Events per Month')
plt.xlabel('Month')
plt.ylabel('Number of Out-Events')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()  # Adjust plot to ensure everything fits without overlap
plt.show()



