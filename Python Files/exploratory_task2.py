import pandas as pd
import numpy as np
import os
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import geom
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt

ExpData = pd.read_csv('US_confirmed.csv',index_col=0)

DailyConf = ExpData - ExpData.shift(axis=1).fillna(0)

DailyConf = DailyConf.transpose()
DailyConf.index = pd.to_datetime(DailyConf.index)

testData = DailyConf['CA']

LA_AI = pd.read_csv('LA2020-21.csv', index_col=0)

LA_AI.index = pd.to_datetime(LA_AI.index)

def KS2Sample(data1_, data2_):

  uniq_val_state1, freq_val_state1 = np.unique(data1_, return_counts=True)
  cdf_state1 = np.cumsum(freq_val_state1)
  cdf_state1 = cdf_state1/cdf_state1[-1]

  uniq_val_state2, freq_val_state2 = np.unique(data2_, return_counts=True)
  cdf_state2 = np.cumsum(freq_val_state2)
  cdf_state2 = cdf_state2/cdf_state2[-1]

  X2, Y2, X1, Y1 = uniq_val_state2, cdf_state2, uniq_val_state1, cdf_state1
  max_diff = -1

  '''
  To find eCDF value for values that aren't in second distribution but are present
  in the first distribution.
  The eCDF of the previous value present in the second distribution is given
  '''
  predict_ecdf_for_X2 = interp1d(X2, Y2, kind='previous', bounds_error=False, fill_value=(0.0,1.0)) 

  for i, value_ in enumerate(X1):
    left_cdf_X1, right_cdf_X1 = 0, 0
    if i == 0: 
      left_cdf_X1 = 0
      right_cdf_X1 = Y1[i]
    elif i == len(X1)-1:
      left_cdf_X1 = Y1[i-1]
      right_cdf_X1 = 1
    else:
      left_cdf_X1 = Y1[i-1]
      right_cdf_X1 = Y1[i]
    # print("For i %d age %d left:%f right:%f" %(i, age, left_cdf, right_cdf)) 

    if value_ in X2:
      j = list(X2).index(value_)
      if j == 0: 
        left_cdf_X2 = 0
        right_cdf_X2 = Y2[j]
      elif j == len(X2)-1:
        left_cdf_X2 = Y2[j-1]
        right_cdf_X2 = 1
      else:
        left_cdf_X2 = Y2[j-1]
        right_cdf_X2 = Y2[j]
    else:
      interp_cdf = predict_ecdf_for_X2(value_)
      left_cdf_X2 = interp_cdf
      right_cdf_X2 = interp_cdf
      # print("Value evaluated from interp at value %f cdf %f" %(value_, interp_cdf))
    
    max_val = max(abs(left_cdf_X1 - left_cdf_X2), abs(right_cdf_X1 - right_cdf_X2))
    if max_diff < max_val:
      max_diff = max_val

  if max_diff > 0.05:
    print("Since Max distance(%f) > 0.05, we reject Null Hypothesis (The Los Angeles Ozone AQI Value has same distribution pre and during COVID)\n\n" % max_diff)
  else:
    print("Since Max distance(%f) <= 0.05, we accept Null Hypothesis (The Los Angeles Ozone AQI Value has same distribution pre and during COVID)\n\n" % max_diff)

  #Use only for plotting
  # plt.figure(figsize=(8,8))
  # plt.step(X1, Y1, label="eCDF of Pre-COVID")
  # plt.step(X2, Y2, label="eCDF of During-COVID")
  # plt.xlabel('Ozone Air Quality Index Value')
  # plt.ylabel('eCDF')
  # title=('K-S Test')
  # plt.title(title)
  # plt.legend(loc="upper left")
  # plt.grid()
  # # plt.show()

  # return max_diff

data_pre_covid = np.array(LA_AI[LA_AI.index.to_series().between('2020-04-15','2020-11-01')]['Ozone AQI Value'])
data_post_covid = np.array(LA_AI[LA_AI.index.to_series().between('2020-11-01','2021-03-01')]['Ozone AQI Value'])

KS2Sample(data_pre_covid, data_post_covid)

#Use only for plotting
# plt.figure(figsize=(20,8))
# plt.plot(testData, label='Calfornia Daily Confirmed Cases')
# plt.plot(LA_AI['Ozone AQI Value']*200, label='Ozone AQI Value * 200')
# plt.xlabel('Time')
# plt.ylabel('Values')
# title=('Comparison Plot')
# plt.title(title)
# plt.legend(loc="upper left")
# plt.grid()