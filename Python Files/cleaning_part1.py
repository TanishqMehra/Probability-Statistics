import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime

df = pd.read_csv('12.csv', delimiter=',', parse_dates=True)
# print(df.columns)
for column in df.columns:
  if (column!="Date"):
    # print(column)
    value_array = df[column].to_numpy()
    # print(len(value_array))
    initial_value = value_array[0]
    value_array = np.diff(value_array)
    # print(len(value_array))
    value_array = np.insert(value_array, 0, initial_value)
    df[column] = value_array

print(df)  
df.to_csv('12_processed.csv', index=False)
