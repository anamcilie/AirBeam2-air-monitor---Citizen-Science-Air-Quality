"""
@author: Ana MC Ilie

Citation:  Ilie, A.M.C., McCarthy, N., Velasquez, L. et al. Air pollution exposure assessment at schools and playgrounds
in Williamsburg Brooklyn NYC, with a view to developing a set of policy solutions. J Environ Stud Sci (2022). https://doi.org/10.1007/s13412-022-00777-7


"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pandas import Series        # To work on series
import statsmodels
import warnings                   # To ignore the warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set_style("white")
plt.style.use("seaborn")
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(14, 8)})

# sns.set() # setting to default settings
# plt.rcParams # set default matplotlib settings

# finding the current directory
abs_path = os.getcwd()
abs_path

# change to desired folder where .csv file is present - Use forward backslash
path = r'C:\Users\Desktop\02152019 Data Analysis'
data = pd.read_csv(path + '/bcche_stationary_network_5_72659__20190208-7952-17ali0n.csv')

# Exploratory Data Analysis

data.dtypes # find the datatypes of all variables
data.columns # list of all column names - original list
data.shape # Dimensions of original dataset (rows, columns)
print(data.head(10)) # first 10 rows of dataset
data.index # index - number of rows and columns
data.info() # information on datatypes and number of elements

# rename columns from default values
data.columns = ["Object ID", "Session_Name", "Timestamp", "Lat", "Long",
                "Temperature", "PM-1", "PM-10", "PM-2.5", "RH"]

data = data.iloc[8:] # removing the first 7 blank rows and header
data = data.reset_index() # resetting the index to start from 0

# Convert the datatype of certain columns to float type 
data[['Temperature', 'PM-1', 'PM-10', 'PM-2.5','RH']] = data[['Temperature', 'PM-1', 'PM-10', 'PM-2.5','RH']].apply(pd.to_numeric)

# Convert the timestamp to right format and datetime object
data['Timestamp'] = data['Timestamp'].str.replace('T',' ').str.replace(' ',' ')
# Timestamp needs to be converted to datetime object (for time series manipulations)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# store descriptive statistics in a different dataframe and .csv file
desc_stat = data.describe()
desc_stat.to_csv(path+'\Descriptive_Stats.csv')

# see the results of desired column
print (desc_stat['PM-2.5'])
print (desc_stat['RH'])

# Missing Value Analysis
# list number of missing values for each column
print (data.isnull().sum())
null_counts = data.isnull().sum()

# Run following 6 lines together for plot
plt.figure(figsize=(14,8))
plt.xticks(np.arange(len(null_counts)), null_counts.index, rotation='vertical')
plt.xlabel('Columns')
plt.ylabel('Number of Attributes with Missing Data')
plt.bar(np.arange(len(null_counts)), null_counts)
plt.title('Missing Value Analysis')

# replacing all the zeros with NaN
data = data[data.columns].replace(0, np.nan)
# dropping all the rows with NaN in the columns 
data.dropna(inplace=True)
# recheck the shape of dataframe if values changed
data.shape

# save the cleaned data file to another .csv, in the same folder
data.to_csv(path + "\Cleaned_Data_File.csv")


# Continous Variables in the dataset
data_cont = data.iloc[:, 6:11].reset_index() 
data_cont = data_cont.drop(['index'], axis = 1)
data_cont.shape
data_cont.columns

# Boxplot Analysis to see presence of outliers
sns.boxplot(y=data_cont["Temperature"], data=data)
sns.boxplot(y=data_cont["PM-1"], data=data)
sns.boxplot(y=data_cont["PM-10"], data=data)
sns.boxplot(y=data_cont["PM-2.5"], data=data)
sns.boxplot(y=data_cont["RH"], data=data)

# Function to remove outliers using IQR (Inter Quartile Range)
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 # Interquartile range
    # assume the factor of 2.0 for the ranges
    low_lim  = q1 - 2*iqr
    high_lim = q3 + 2*iqr
    df_out = df_in.loc[(df_in[col_name] > low_lim) & (df_in[col_name] < high_lim)]
    return df_out

# Executing the function to remove outliers
data_cont = remove_outlier(data_cont, 'Temperature')
data_cont = remove_outlier(data_cont, 'PM-1')
data_cont = remove_outlier(data_cont, 'PM-10')
data_cont = remove_outlier(data_cont, 'PM-2.5')
data_cont = remove_outlier(data_cont, 'RH')


# Variable Correlation - Heatmap
corr = data_cont.corr() # Calculating the correlations between variables
# Correlation
df_num_corr = data_cont.corr()['PM-2.5'] 
features_list = df_num_corr[abs(df_num_corr) > 0.1].sort_values(ascending=False)
print("These are {} correlation values of variables with PM-2.5:\n{}".format(len(features_list), features_list))
# plot the heatmap for correlations
plt.figure(figsize=(20,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            annot = True, annot_kws={"size": 11})

# Basic Plots

data_cont.boxplot()

# Pair Plot
sns.pairplot(data_cont)

# Line Plot - continous variables
pm_conc = ['PM-1','PM-10','PM-2.5']
plt.plot(data_cont[pm_conc])
plt.xlabel('Timestamp')
plt.ylabel('PM Concentration')
plt.show()


# Histogram
data_cont.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# Distribution Plot
plt.figure(figsize=(20,10))
sns.distplot(data_cont['PM-1'], label='PM-1')
sns.distplot(data_cont['PM-10'], label='PM-10')
sns.distplot(data_cont['PM-2.5'], label='PM-2.5')
plt.xlabel('Variable Estimate')
plt.title('Distribution Plot - Kernel Density Estimate')
plt.legend()


# Time Series Analysis
data2 = data.set_index('Timestamp')
data2.head(3)

data2['Month'] = data2.index.month
data2['Year'] = data2.index.year
data2['Weekday Name'] = data2.index.weekday_name
data2.sample(5, random_state = 0)
# time slicing
data_Jan = data2.loc['2019-01-01':'2019-01-31'] # January Data
data_Jan2 = data2.loc['2019-01']  # January Data

data_Jan['PM-2.5'].plot(linewidth = 0.5)

cols_plot = ['PM-1', 'PM-10', 'PM-2.5']
axes = data_Jan[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('PM Concentration')

# slicing and plotting at the same time
data2.loc['2019-01-31 15:00:00':'2019-01-31 20:30:00', 'PM-2.5'].plot()

# number of observations per timestamp
data2.groupby(level=0).count()

# mean values on a daily basis
data2.resample('D').mean()
