import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime


df = pd.read_csv('12_final_processed.csv', delimiter=',', parse_dates=True)
df["Date"] = pd.to_datetime(df["Date"])

def processDateStringToDate(date_str):
  return datetime.datetime.strptime(date_str, '%Y-%m-%d')

def processDataInDateRange(df, start_date_str, end_date_str):
  start_date = processDateStringToDate(start_date_str)
  end_date = processDateStringToDate(end_date_str)

  mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
  df_subdata = df.loc[mask] 
  df_subdata["Total deaths"] = df_subdata["MN deaths"] + df_subdata["MI deaths"]
  df_subdata["Total confirmed"] = df_subdata["MN confirmed"] + df_subdata["MI confirmed"]
  df_subdata = df_subdata.reset_index()
  return df_subdata



start_date_str = '2020-06-01'
end_date_str = '2020-07-26'
data = processDataInDateRange(df, start_date_str, end_date_str)
data.to_csv('BI_2D_data.csv')

import matplotlib.pyplot as plt
from scipy.stats import gamma

deaths_data = np.array(data["Total deaths"])
lambda_mme = np.sum(deaths_data[:28])/len(deaths_data[:28])

plt.figure(figsize=(12,8))

def plot_posterior_distributions(alpha, beta, label):
  #Reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html 
  x_values = np.linspace(gamma.ppf(0.01, alpha, scale=1/beta),
                      gamma.ppf(0.99, alpha, scale=1/beta), 100)
  y_values = gamma.pdf(x_values, alpha, scale=1/beta)
  plt.title("Question 2D")

  #Finding MAP by finding the peak of the graph and finding its corresponding x value
  map_index = np.argmax(y_values)
  map_value = x_values[map_index]
  label= "Posterior distro " + label + " with MAP: " + str(round(map_value,3))
  
  plt.xlabel("Total deaths")
  plt.ylabel("Posterior distribution(Gamma) PDF")
  
  plt.plot(x_values,y_values , label=label)
  plt.legend()
 


#First posterior distribution is a Gamma distribution 
first_posterior_data = deaths_data[28:35]

#The first posterior distribution is a conjugate prior of the second posterior and hence taking all the data from fifth week
second_posterior_data = deaths_data[28:42]
#Similarly....
third_posterior_data = deaths_data[28:49]
fourth_posterior_data = deaths_data[28:56]

# print(first_posterior_data)
# print(second_posterior_data)
# print(third_posterior_data)
# print(fourth_posterior_data)

plot_posterior_distributions(np.sum(first_posterior_data) +1, len(first_posterior_data)+ (1/lambda_mme), "after 5th Week")
plot_posterior_distributions(np.sum(second_posterior_data) +1, len(second_posterior_data)+ (1/lambda_mme), "after 6th week")
plot_posterior_distributions(np.sum(third_posterior_data) +1, len(third_posterior_data)+ (1/lambda_mme), "after 7th Week")
plot_posterior_distributions(np.sum(fourth_posterior_data) +1, len(fourth_posterior_data)+ (1/lambda_mme), "after 8th Week")


plt.show()

