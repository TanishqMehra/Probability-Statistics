import pandas as pd
import numpy as np
import os
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import geom
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt

DATA = pd.read_csv('12_final_processed.csv', index_col=0)

from_date = pd.to_datetime('2020-10-01', format='%Y-%m')
to_date = pd.to_datetime('2020-12-31', format='%Y-%m')
date_vector = pd.to_datetime(DATA.Date,  format='%Y-%m-%d')
DATA = DATA[(date_vector >= from_date) & (date_vector <= to_date)]

def sampleMean(data_):
  return np.mean(data_)

def sampleVar(data_):
  return np.var(data_)

def MMEPoisson(data_):
  return [sampleMean(data_)]

def MMEGeometric(data_):
  return [1/sampleMean(data_)]

def MMEBinomial(data_):
  mean_ = sampleMean(data_)
  var_ = sampleVar(data_)

  # mme_n = (mean_ ** 2)/(mean_ - var_)
  # mme_p = mean_ / mme_n

  mme_p = 1 - ((np.sum(data_ ** 2)/len(data_) - mean_**2)/mean_)
  mme_n = mean_/mme_p

  return [mme_n, mme_p]

# 1-sample K-S test
def KS1Sample(data_, true_distro_):
  state1_ = data_.columns[0]
  state2_ = data_.columns[1]

  MME = []
  if true_distro_ == 'poisson':
    MME = MMEPoisson(data_[state1_])
  elif true_distro_ == 'geometric':
    MME = MMEGeometric(data_[state1_])
  elif true_distro_ == 'binomial':
    MME = MMEBinomial(data_[state1_])
  else: return None

  print("The MME values : ", MME)

  uniq_val_state2, freq_val_state2 = np.unique(data_[state2_], return_counts=True)
  cdf_state2 = np.cumsum(freq_val_state2)
  cdf_state2 = cdf_state2/cdf_state2[-1]

  # print(uniq_val_state2)
  # print(freq_val_state2)
  # print(cdf_state2)

  max_diff = -1
  for i, value_ in enumerate(uniq_val_state2):
    left_ecdf, right_ecdf = 0, 0
    if i == 0: 
      left_ecdf = 0
      right_ecdf = cdf_state2[i]
    elif i == len(uniq_val_state2)-1:
      left_ecdf = cdf_state2[i-1]
      right_ecdf = 1
    else:
      left_ecdf = cdf_state2[i-1]
      right_ecdf = cdf_state2[i]

    true_cdf = 0
    if true_distro_ == 'poisson':
      true_cdf = poisson.cdf(value_, MME[0])
    elif true_distro_ == 'geometric':
      true_cdf = geom.cdf(value_, MME[0])
    elif true_distro_ == 'binomial':
      true_cdf = binom.cdf(value_, MME[0], MME[1])

    
    diff_ = max(abs(left_ecdf - true_cdf), abs(right_ecdf - true_cdf))
    if diff_ > max_diff:
      max_diff = diff_
    
    # print(value_, left_ecdf, right_ecdf, true_cdf, diff_)

  if max_diff > 0.05:
    print("Since Max distance(%f) > 0.05, we reject Null Hypothesis (%s has same distribution as %s having true distribution of %s)\n\n" % (max_diff, state2_, state1_, true_distro_))
  else:
    print("Since Max distance(%f) <= 0.05, we accept Null Hypothesis (%s has same distribution as %s having true distribution of %s)\n\n" % max_diff)

  # return max_diff

KS1Sample(DATA[['MI confirmed', 'MN confirmed']], 'poisson')
KS1Sample(DATA[['MI confirmed', 'MN confirmed']], 'geometric')
KS1Sample(DATA[['MI confirmed', 'MN confirmed']], 'binomial')

KS1Sample(DATA[['MI deaths', 'MN deaths']], 'poisson')
KS1Sample(DATA[['MI deaths', 'MN deaths']], 'geometric')
KS1Sample(DATA[['MI deaths', 'MN deaths']], 'binomial')

# 2-sample K-S test
def KS2Sample(data_):
  state1_ = data_.columns[0]
  state2_ = data_.columns[1]

  uniq_val_state1, freq_val_state1 = np.unique(data_[state1_], return_counts=True)
  cdf_state1 = np.cumsum(freq_val_state1)
  cdf_state1 = cdf_state1/cdf_state1[-1]

  uniq_val_state2, freq_val_state2 = np.unique(data_[state2_], return_counts=True)
  cdf_state2 = np.cumsum(freq_val_state2)
  cdf_state2 = cdf_state2/cdf_state2[-1]

  X1, X2, Y1, Y2 = None, None, None, None

  if len(uniq_val_state1) > len(uniq_val_state2):
    X1, Y1, X2, Y2 = uniq_val_state2, cdf_state2, uniq_val_state1, cdf_state1
  else:
    X2, Y2, X1, Y1 = uniq_val_state2, cdf_state2, uniq_val_state1, cdf_state1

  # print(X1, Y1, X2, Y2, sep='\n')
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
    print("Since Max distance(%f) > 0.05, we reject Null Hypothesis (%s has same distribution as %s)\n\n" % (max_diff, state2_, state1_))
  else:
    print("Since Max distance(%f) <= 0.05, we accept Null Hypothesis (%s has same distribution as %s)\n\n" % max_diff)

  # return max_diff

KS2Sample(DATA[['MI confirmed', 'MN confirmed']])
KS2Sample(DATA[['MI deaths', 'MN deaths']])

# P-test
def PTest(data_, points_ = 1000):

  state1_ = data_.columns[0]
  state2_ = data_.columns[1]

  d1_mean_ = np.mean(data_[state1_])
  d2_mean_ = np.mean(data_[state2_])
  t_observed = abs(d1_mean_ - d2_mean_)

  # print(data_[state1_].shape, data_[state2_].shape)
  mixed_bag = list(np.concatenate((data_[state1_], data_[state2_])))
  # print(len(mixed_bag))
  # print(t_observed)
  permutations = []
  while len(permutations) < points_:
      permutations.append(random.sample(mixed_bag, len(mixed_bag)))
  permutations = np.array(permutations)
  # print(permutations.shape)
  d1_perm = permutations[:, 0 : data_[state1_].shape[0]]
  d2_perm = permutations[:, data_[state2_].shape[0] : ]
  # print(d1_perm.shape, d2_perm.shape)
  d1_perm_mean_ = np.mean(d1_perm, axis=1)
  d2_perm_mean_ = np.mean(d2_perm, axis=1)
  t_perm_ = np.abs(d1_perm_mean_ - d2_perm_mean_)
  p_value = np.sum(t_perm_ > t_observed)/(t_perm_.shape[0])
  print("The p-value is ", p_value)
  if p_value <= 0.05:
    print("Since the p-value <= 0.05, we reject the Null Hypothesis (%s has same distribution as %s)\n\n" % (state1_, state2_))
  else:
    print("Since the p-value > 0.05, we accept the Null Hypothesis (%s has same distribution as %s)\n\n" % (state1_, state2_))

PTest(DATA[['MI confirmed', 'MN confirmed']])

PTest(DATA[['MI deaths', 'MN deaths']])