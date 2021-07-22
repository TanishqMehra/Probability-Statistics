import pandas as pd
import numpy as np
import os
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import geom
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt

#This is the file that calculates daily cases from the cumulative cases
DATA = pd.read_csv('12_processed.csv',index_col = 0)

print(DATA)

test_data = DATA
#We replace the negative values with nan
test_data[test_data < 0] = np.nan

# test_data.head()

# test_data['MN confirmed'].index[test_data['MN confirmed'].apply(np.isnan)]

#print(test_data.loc[267:270])

# test_data_new = test_data.where(~(test_data == np.nan), other=(test_data.fillna(method='ffill')+test_data.fillna(method='bfill')/2))

#We replace the nan values with the average of nearest neighbours (above and below)
test_data_new = (test_data.fillna(method='ffill') + test_data.fillna(method='bfill'))//2

#print(DATA.Date)
#test_data_new.loc[267:270]

test_data_new.reset_index()

#Saving the cleaned data
test_data_new.to_csv('12_final_processed.csv')
