#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:50:05 2024

@author: hubble
"""
#%%
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
#%%
def update_training_data(training_df, new_data):
    return pd.concat([training_df, new_data], ignore_index=True)

#%%
df = denoise_df_yearwise[0]
df['datetime'] = pd.to_datetime(df['datetime'])
df['week'] = df['datetime'].dt.isocalendar().week
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

groups = df.groupby(['year', 'month', 'week'])
weekly_dfs_list = [group.reset_index(drop=True) for (year, month, week), group in groups]

historical_df = pd.concat((weekly_dfs_list[0], weekly_dfs_list[1])).reset_index(drop=True)
scaler = StandardScaler()
X_train = historical_df[['denoised_sensor_value']].values
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the GMM on the historical data
gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
gmm.fit(X_train_scaled)

# List to store labels for all test data
all_test_data = []

# Loop through each test week starting from the third week
for tst in range(2, len(weekly_dfs_list)):
    # Prepare test data
    temp_test = weekly_dfs_list[tst]
    X_test = temp_test[['denoised_sensor_value']].values
    X_test_scaled = scaler.transform(X_test)

    # Compute anomaly scores
    anomaly_scores = -gmm.score_samples(X_test_scaled)  # Higher score indicates anomaly
    threshold = np.percentile(anomaly_scores, 95)
    lbls = np.where(anomaly_scores > threshold, 'anomalous', 'normal')
    temp_test['anomaly_score'] = anomaly_scores
    temp_test['label'] = lbls

    # Append to the list of all test data
    all_test_data.append(temp_test)

    # Update training data
    historical_df = pd.concat((historical_df, temp_test)).reset_index(drop=True)
    X_train_updated = historical_df[['denoised_sensor_value']].values
    X_train_scaled_updated = scaler.fit_transform(X_train_updated)
    
    # Retrain the GMM with the updated training data
    gmm.fit(X_train_scaled_updated)

# Concatenate all test data into a single DataFrame
all_test_df = pd.concat(all_test_data).reset_index(drop=True)

# Plot results
plt.figure(figsize=(14, 7))

# Plot denoised_sensor_value
plt.plot(all_test_df['datetime'], all_test_df['denoised_sensor_value'], color='gray', alpha=0.3, label='Denoised Sensor Value')

# Plot anomalies only
anomalous_df = all_test_df[all_test_df['label'] == 'anomalous']
plt.scatter(anomalous_df['datetime'], anomalous_df['denoised_sensor_value'],
            color='red', label='Anomalous Data', edgecolor='k')

# Show plot details
plt.title('Anomaly Detection - Test Data')
plt.xlabel('Date and Time')
plt.ylabel('Denoised Sensor Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
#%%%

# Initial training on historical data (first week)
historical_df = weekly_dfs_list[0]
scaler = StandardScaler()
X_train = historical_df[['denoised_sensor_value']].values
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the GMM on the historical data
gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
gmm.fit(X_train_scaled)

# Compute anomaly scores for historical data
historical_anomaly_scores = -gmm.score_samples(X_train_scaled)
historical_threshold = np.percentile(historical_anomaly_scores, 95)
historical_labels = np.where(historical_anomaly_scores > historical_threshold, 'anomalous', 'normal')

# Determine the proportion of anomalies in the historical data
historical_anomalous_count = (historical_labels == 'anomalous').sum()
historical_total_count = len(historical_labels)

# Set the anomalous threshold ratio based on the training data
anomalous_threshold_ratio = historical_anomalous_count / historical_total_count


# List to store labels for all test data
all_test_data = []

# Loop through each test week starting from the second week
for tst in range(1, len(weekly_dfs_list)):
    # Prepare test data
    temp_test = weekly_dfs_list[tst]
    X_test = temp_test[['denoised_sensor_value']].values
    X_test_scaled = scaler.transform(X_test)

    # Compute anomaly scores
    anomaly_scores = -gmm.score_samples(X_test_scaled)  # Higher score indicates anomaly
    threshold = np.percentile(anomaly_scores, 95)
    lbls = np.where(anomaly_scores > threshold, 'anomalous', 'normal')
    temp_test['anomaly_score'] = anomaly_scores
    temp_test['label'] = lbls

    # Count the number of anomalous and normal labels
    anomalous_count = (lbls == 'anomalous').sum()
    normal_count = (lbls == 'normal').sum()

    # Determine if the week is considered anomalous based on the count
    if anomalous_count / len(lbls) > anomalous_threshold_ratio:
        temp_test['week_label'] = 'anomalous'
    else:
        temp_test['week_label'] = 'normal'

    # Append to the list of all test data
    all_test_data.append(temp_test)

    # Update training data
    historical_df = pd.concat((historical_df, temp_test)).reset_index(drop=True)
    X_train_updated = historical_df[['denoised_sensor_value']].values
    X_train_scaled_updated = scaler.fit_transform(X_train_updated)
    
    # Retrain the GMM with the updated training data
    gmm.fit(X_train_scaled_updated)

# Concatenate all test data into a single DataFrame
# all_test_df = pd.concat(all_test_data).reset_index(drop=True)
all_test_df = all_test_df.dropna(subset=['datetime']).reset_index(drop=True)

# Plot results
plt.figure(figsize=(14, 7))

# Plot denoised_sensor_value
plt.plot(all_test_df['datetime'], all_test_df['denoised_sensor_value'], color='gray', alpha=0.3, label='Denoised Sensor Value')

# Plot anomalies only
anomalous_df = all_test_df[all_test_df['label'] == 'anomalous']
plt.scatter(anomalous_df['datetime'], anomalous_df['denoised_sensor_value'],
            color='red', label='Anomalous Data', edgecolor='k')

# Draw vertical lines and annotate weeks
weeks = all_test_df['datetime'].dt.to_period('W').unique()

for week in weeks:
    week_data = all_test_df[all_test_df['datetime'].dt.to_period('W') == week]
    start_date = week_data['datetime'].min()
    end_date = week_data['datetime'].max()

    # Determine if the week is anomalous based on labels
    is_anomalous = week_data['week_label'].iloc[0]  # Assuming 'week_label' column exists

    # Draw vertical lines
    plt.axvline(start_date, color='blue', linestyle='--', linewidth=1)
    plt.axvline(end_date, color='blue', linestyle='--', linewidth=1)

    # Add text annotation between the lines
    plt.text(start_date + (end_date - start_date) / 2, plt.ylim()[1] * 0.95,
             'Anomalous' if is_anomalous == 'anomalous' else 'Normal',
             color='black', ha='center', va='top', fontsize=10,
             bbox=dict(facecolor='yellow' if is_anomalous == 'anomalous' else 'green', alpha=0.5))

# Show plot details
plt.title('Anomaly Detection - Test Data')
plt.xlabel('Date and Time')
plt.ylabel('Denoised Sensor Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#%%%


# Prepare historical data
historical_df = weekly_dfs_list[0]  # Training data
scaler = StandardScaler()
X_train = historical_df[['denoised_sensor_value']].values
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the GMM on the historical data
gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
gmm.fit(X_train_scaled)

anomaly_scores_train = -gmm.score_samples(X_train_scaled)  # Higher score indicates anomaly

mean_anomaly_score_train = np.mean(anomaly_scores_train)
std_anomaly_score_train = np.std(anomaly_scores_train)

# Define the threshold range using ± standard deviations
lower_threshold = mean_anomaly_score_train - std_anomaly_score_train
upper_threshold = mean_anomaly_score_train + std_anomaly_score_train

# anomalous_count = (historical_df['label'] == 'anomalous').sum()
# total_count = len(historical_df)
# anomalous_threshold_ratio = anomalous_count / total_count
# List to store labels for all test data
all_test_data = []

# Loop through each test week
for tst in range(1, len(weekly_dfs_list)):
    # Prepare test data
    temp_test = weekly_dfs_list[tst]
    X_test = temp_test[['denoised_sensor_value']].values
    X_test_scaled = scaler.transform(X_test)

    # Compute anomaly scores for the test data
    anomaly_scores_test = -gmm.score_samples(X_test_scaled)  # Higher score indicates anomaly

    # Label test data based on the threshold range
    lbls = np.where((anomaly_scores_test < lower_threshold) | (anomaly_scores_test > upper_threshold), 'anomalous', 'normal')
    temp_test['anomaly_score'] = anomaly_scores_test
    temp_test['label'] = lbls

    # Count anomalies and determine the week label based on the threshold range
    anomalous_count = np.sum(lbls == 'anomalous')
    total_count = len(lbls)
    
    # Define a ratio of anomalies that determines if a week is anomalous
    anomalous_ratio = anomalous_count / total_count
    if anomalous_ratio > anomalous_threshold_ratio:
        temp_test['week_label'] = 'anomalous'
    else:
        temp_test['week_label'] = 'normal'

    # Append to the list of all test data
    all_test_data.append(temp_test)

    # Update training data with the new test data
    historical_df = pd.concat((historical_df, temp_test)).reset_index(drop=True)
    X_train_updated = historical_df[['denoised_sensor_value']].values
    X_train_scaled_updated = scaler.fit_transform(X_train_updated)
    
    # Retrain the GMM with the updated training data
    gmm.fit(X_train_scaled_updated)

# Concatenate all test data into a single DataFrame
all_test_df = pd.concat(all_test_data).reset_index(drop=True)

plt.figure(figsize=(14, 7))

# Plot denoised_sensor_value
plt.plot(all_test_df['datetime'], all_test_df['denoised_sensor_value'], color='gray', alpha=0.3, label='Denoised Sensor Value')

# Plot anomalies only
anomalous_df = all_test_df[all_test_df['label'] == 'anomalous']
plt.scatter(anomalous_df['datetime'], anomalous_df['denoised_sensor_value'],
            color='red', label='Anomalous Data', edgecolor='k')

# Draw vertical lines and annotate weeks
weeks = all_test_df['datetime'].dt.to_period('W').unique()

for week in weeks:
    week_data = all_test_df[all_test_df['datetime'].dt.to_period('W') == week]
    start_date = week_data['datetime'].min()
    end_date = week_data['datetime'].max()

    # Determine if the week is anomalous based on labels
    is_anomalous = week_data['week_label'].iloc[0]  # Assuming 'week_label' column exists

    # Draw vertical lines
    plt.axvline(start_date, color='blue', linestyle='--', linewidth=1)
    plt.axvline(end_date, color='blue', linestyle='--', linewidth=1)

    # Add text annotation between the lines
    plt.text(start_date + (end_date - start_date) / 2, plt.ylim()[1] * 0.95,
             'Anomalous' if is_anomalous == 'anomalous' else 'Normal',
             color='black', ha='center', va='top', fontsize=10,
             bbox=dict(facecolor='yellow' if is_anomalous == 'anomalous' else 'green', alpha=0.5))

# Show plot details
plt.title('Anomaly Detection - Test Data')
plt.xlabel('Date and Time')
plt.ylabel('Denoised Sensor Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# Load training data
historical_df = weekly_dfs_list[0]  # Training data

# Feature scaling
scaler = StandardScaler()
X_train = historical_df[['denoised_sensor_value']].values
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the GMM on the historical data
gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
gmm.fit(X_train_scaled)

# Compute anomaly scores for training data
anomaly_scores_train = -gmm.score_samples(X_train_scaled)  # Higher score indicates anomaly

# Compute mean and standard deviation of anomaly scores
mean_anomaly_score_train = np.mean(anomaly_scores_train)
std_anomaly_score_train = np.std(anomaly_scores_train)

# Define the threshold range using ± standard deviations
lower_threshold = mean_anomaly_score_train - std_anomaly_score_train
upper_threshold = mean_anomaly_score_train + std_anomaly_score_train

# Classify training data based on the thresholds
historical_df['anomaly_score'] = anomaly_scores_train
historical_df['label'] = np.where(
    (anomaly_scores_train < lower_threshold) | (anomaly_scores_train > upper_threshold),
    'anomalous',
    'normal'
)

# Compute the ratio of anomalous events in the training data
anomalous_count = (historical_df['label'] == 'anomalous').sum()
total_count = len(historical_df)
anomalous_threshold_ratio = anomalous_count / total_count

# Print computed thresholds and anomalous ratio
print(f'Lower Threshold: {lower_threshold}')
print(f'Upper Threshold: {upper_threshold}')
print(f'Anomalous Ratio in Training Data: {anomalous_threshold_ratio}')



all_test_data = []

# Loop through each test week starting from the second week
for tst in range(1, len(weekly_dfs_list)):
    # Prepare test data
    temp_test = weekly_dfs_list[tst]
    X_test = temp_test[['denoised_sensor_value']].values
    X_test_scaled = scaler.transform(X_test)

    # Compute anomaly scores for the test data
    anomaly_scores_test = -gmm.score_samples(X_test_scaled)  # Higher score indicates anomaly

    # Label test data based on the threshold range
    lbls = np.where((anomaly_scores_test < lower_threshold) | (anomaly_scores_test > upper_threshold), 'anomalous', 'normal')
    temp_test['anomaly_score'] = anomaly_scores_test
    temp_test['label'] = lbls

    # Count anomalies and determine the week label based on the threshold range
    anomalous_count = np.sum(lbls == 'anomalous')
    total_count = len(lbls)
    
    # Calculate the ratio of anomalies
    anomalous_ratio = anomalous_count / total_count

    # Determine if the week is anomalous based on the ratio and threshold
    if anomalous_ratio > anomalous_threshold_ratio:
        temp_test['week_label'] = 'anomalous'
    else:
        temp_test['week_label'] = 'normal'

    # Append to the list of all test data
    all_test_data.append(temp_test)

    # Update training data with the new test data
    historical_df = pd.concat((historical_df, temp_test)).reset_index(drop=True)
    X_train_updated = historical_df[['denoised_sensor_value']].values
    X_train_scaled_updated = scaler.fit_transform(X_train_updated)
    
    # Retrain the GMM with the updated training data
    gmm.fit(X_train_scaled_updated)

# Concatenate all test data into a single DataFrame
all_test_df = pd.concat(all_test_data).reset_index(drop=True)

# Plot results
plt.figure(figsize=(14, 7))

# Plot denoised_sensor_value
plt.plot(all_test_df['datetime'], all_test_df['denoised_sensor_value'], color='gray', alpha=0.3, label='Denoised Sensor Value')

# Plot anomalies only
anomalous_df = all_test_df[all_test_df['label'] == 'anomalous']
plt.scatter(anomalous_df['datetime'], anomalous_df['denoised_sensor_value'],
            color='red', label='Anomalous Data', edgecolor='k')

# Draw vertical lines and annotate weeks
weeks = all_test_df['datetime'].dt.to_period('W').unique()

for week in weeks:
    week_data = all_test_df[all_test_df['datetime'].dt.to_period('W') == week]
    start_date = week_data['datetime'].min()
    end_date = week_data['datetime'].max()

    # Determine if the week is anomalous based on labels
    is_anomalous = week_data['week_label'].iloc[0]  # Assuming 'week_label' column exists

    # Draw vertical lines
    plt.axvline(start_date, color='black', linestyle='--', linewidth=1)
    plt.axvline(end_date, color='black', linestyle='--', linewidth=1)

    # Add text annotation between the lines
    plt.text(start_date + (end_date - start_date) / 2, plt.ylim()[1] * 0.95,
             'Anomalous' if is_anomalous == 'anomalous' else 'Normal',
             color='black', ha='center', va='top', fontsize=10,
             bbox=dict(facecolor='yellow' if is_anomalous == 'anomalous' else 'green', alpha=0.5))

# Show plot details
plt.title('Anomaly Detection - Test Data')
plt.xlabel('Date and Time')
plt.ylabel('Denoised Sensor Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()