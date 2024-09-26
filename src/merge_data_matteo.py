import pandas as pd, pickle, os
from natsort import natsorted
 

# # Load the pickle file
# with open('/home/hubble/Downloads/gpt_zeroshot_subject_a.pkl', 'rb') as file:
#     data = pickle.load(file)




# Path to the 'DATA' folder
data_path = '/home/hubble/Downloads/Data_upTo_25Sep/DATA2/'

# Function to sort the files based on month and year
def sort_files_by_date(file_list):
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    
    def extract_month_year(file_name):
        month, year = file_name.split('.')[0].split('_')
        return int(year), months[month]
    
    return sorted(file_list, key=extract_month_year)

# Loop through each subject and sensor folder
for subject in natsorted(os.listdir(data_path)):
    subject_path = os.path.join(data_path, subject, 'environmentals')
    
    # Check if the folder exists
    if os.path.exists(subject_path):
        for sensor_folder in os.listdir(subject_path):
            sensor_path = os.path.join(subject_path, sensor_folder)
            
            # Get all CSV files in the sensor folder
            csv_files = [f for f in os.listdir(sensor_path) if f.endswith('.csv')]
            
            # Sort the files by month and year
            sorted_files = sort_files_by_date(csv_files)
            
            # Initialize an empty DataFrame to hold the merged data
            merged_df = pd.DataFrame()
            
            # Read and merge the CSV files
            for csv_file in sorted_files:
                file_path = os.path.join(sensor_path, csv_file)
                print(file_path)
                df = pd.read_csv(file_path)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            
            # Save the merged CSV file with a descriptive name
            merged_file_name = f"{subject}_merged.csv"
            merged_file_path = os.path.join(sensor_path, merged_file_name)
            merged_df.to_csv(merged_file_path, index=False)

            print(f"Merged file saved as {merged_file_name} for {subject} - {sensor_folder}")
