import numpy as np, pandas as pd, yaml, os, pytz, re # This is sample comment
from datetime import datetime
from collections import defaultdict


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.read_csv_files()

    def read_csv_files(self):
        print('hiii')


    # def _read_config(self):
    #     with open(self.config_path, 'r') as file:
    #         config = yaml.safe_load(file)
    #         self.path = config['dataset']['path']
    #         self.sensor_details = config['sensors']
    #         # self.sensors_present_at_a_location = config['sensors']

    #     with open(self.config_path_sensors_properties, 'r') as file:
    #         config_sensor_properties = yaml.safe_load(file)    

    #     return config, config_sensor_properties
    