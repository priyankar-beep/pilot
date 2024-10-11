#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:16:30 2024

@author: hubble
"""

import os
import pandas as pd
import logging

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("empty_csv_files.log"),  # Log to a file named 'empty_csv_files.log'
        logging.StreamHandler()  # Also output to console
    ]
)

def find_empty_csv_files(data_path):
    """
    Traverses the directory structure to find empty CSV files for each subject and device.
    
    Parameters:
    - data_path (str): Path to the main data directory containing subject folders.

    Returns:
    - List of dictionaries containing 'subject', 'device', and 'file' for each empty CSV file.
    """
    empty_csv_files = []

    # Check if the main data path exists
    if not os.path.exists(data_path):
        logging.error(f"Path not found: {data_path}")
        return empty_csv_files

    # Traverse each subject folder
    for subject_folder in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject_folder)

        if os.path.isdir(subject_path):
            environmentals_folder = os.path.join(subject_path, 'environmentals')

            if not os.path.exists(environmentals_folder):
                logging.warning(f"Environmentals folder not found for subject: {subject_folder}")
                continue

            # Traverse device folders within each subject's 'environmentals' folder
            for device_folder in os.listdir(environmentals_folder):
                device_path = os.path.join(environmentals_folder, device_folder)

                if os.path.isdir(device_path):
                    # Traverse CSV files within each device folder
                    for csv_file in os.listdir(device_path):
                        if csv_file.endswith('.csv'):
                            csv_file_path = os.path.join(device_path, csv_file)
                            if is_csv_empty(csv_file_path):
                                empty_csv_files.append({
                                    'subject': subject_folder,
                                    'device': device_folder,
                                    'file': csv_file
                                })
                                logging.info(f"Empty CSV: Subject = {subject_folder}, Device = {device_folder}, File = {csv_file}")
    return empty_csv_files


def is_csv_empty(file_path):
    """
    Checks if a CSV file is empty.
    
    Parameters:
    - file_path (str): The full path to the CSV file.

    Returns:
    - bool: True if the file is empty, False otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        return df.empty
    except Exception as e:
        logging.error(f"Failed to read CSV {file_path}: {e}")
        return False


if __name__ == "__main__":
    # Define the main path to the data
    data_path = '/home/hubble/Downloads/Data_upTo_25Sep(1)/DATA2/'

    # Find and log empty CSV files
    empty_csv_files = find_empty_csv_files(data_path)

    # Summary of empty CSVs
    if empty_csv_files:
        logging.info(f"Found {len(empty_csv_files)} empty CSV files.")
    else:
        logging.info("No empty CSV files found.")

