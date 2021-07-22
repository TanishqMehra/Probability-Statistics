# Probability-Statistics
My team's project for course CSE 544: Probability &amp; Statistics under Prof. Anshul Gandhi at Stony Brook University.

Analysis of COVID-19 Dataset 

Developers: 
Aayushi Nirmal
Drushti Mewada
Karan Dipesh Gada
Tanishq Sandeep Mehra


Technology: **Python**

Libraries: **pandas, numpy, sklearn, scipy**

Tasks Performed:
1. Dataset Cleaning:
  - Missing/Negative values
  - **Tukey's Outlier Detection**
2. Used the COVID19 dataset to predict the COVID19 fatality and cases for the fourth week in August 2020 using data from the first three weeks of August 2020. 
  Prediction Techniques used: 
   -  **Exponentially Weighted Moving Average(EWMA) with alpha 0.5**
   -  **Exponentially Weighted Moving Average(EWMA) with alpha 0.8**
   -  **Auto Regression AR(3)**
   -  **Auto Regression AR(5)**
3. Applied the **Wald’s test**, **Z-test**, and **T-test** (assume all are applicable) to check whether the mean of COVID19 deaths and #cases are different for Feb’21 and March’21 in the two states
4. Tested equality of COVID cases distributions in the two states using **K-S test**(both 1-sample and 2-sample tests) and **Permutation test** 
5. Applied **Bayesian Inference** to obtain posteriors using the fifth week to eigth week data after calculating MME of the first four weeks. 
6. Exploratory Task: Applied **Pearson’s correlation** to confirm that US weekly product supplied of kerosene-Type jet fuel had a significant dip from the month
of January 2020 to August 2020 due to COVID-19 cases.
7. Used **K-S Test** to test whether the distribution of AQI Value for Los Angeles just before prime of COVID19 (April, 2020 to November, 2020) and during prime of COVID (November, 2020 to March, 2021) is different or same.
8. Applied **Multiple Linear Regression** to infer that Type jet fuel has a significant dip from the month of January 2020 to August 2020 due to the surge in COVID-19 cases.
