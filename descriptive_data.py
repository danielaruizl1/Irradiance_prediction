import pandas as pd 
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Upload the data
df = pd.read_excel('dataset.xlsx')

# Convert datetime.time to minutes
def time_to_minutes(time_val):
    # Convert the string to a datetime object
    formato = "%H:%M:%S"
    if type(time_val) == str:
        time_val = datetime.strptime(time_val, formato).time()
    # Convert the datetime object to minutes
    return time_val.hour * 60 + time_val.minute

numerical = pd.DataFrame({'hour': df['Hour'], 'temperature': df['temperature'], 'humidity': df['humidity'], 'ghi1':df['ghi1']})
numerical['hour'] = numerical['hour'].apply(time_to_minutes)
print(numerical.describe())

# Normalize the data with min-max scaler
scaler = MinMaxScaler()
numerical[numerical.columns] = scaler.fit_transform(numerical)
print(numerical.describe())

# Standardize the data with standard scaler
scaler = StandardScaler()
numerical[numerical.columns] = scaler.fit_transform(numerical)
print(numerical.describe())
