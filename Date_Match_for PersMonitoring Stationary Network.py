"""
@author: Ana MC Ilie

Citation: Ilie, A.M.C., McCarthy, N., Velasquez, L. et al. Air pollution exposure assessment at schools and playgrounds
in Williamsburg Brooklyn NYC, with a view to developing a set of policy solutions. J Environ Stud Sci (2022). https://doi.org/10.1007/s13412-022-00777-7
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import warnings                   # To ignore the warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set_style("white")
plt.style.use("seaborn")
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(14, 8)})
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
# sns.set() # setting to default settings
# plt.rcParams # set default matplotlib settings

# finding the current directory
abs_path = os.getcwd()
abs_path

# change to desired folder where .csv file is present - Use forward backslash
path = r'G:\booklet\Data Analysis\Personal Monitoring\PM stops'
data_FEM = pd.read_csv(path + '\W4 for persMonit_Cleaned_Data_File.csv')

######         EP GROUPS         ##############
data_Williamsburg = pd.read_csv(path + '\EP4 09222019 min avg.csv')

# data_FEM stands for the Stationary network AirBeam2
# data_Williamsburg personal monitoring data

data_FEM.info()
data_Williamsburg.info()

# rename columns
data_FEM = data_FEM.rename(columns={'Timestamp': 'DateTime'})
data_Williamsburg = data_Williamsburg.rename(columns={'Minute of Timestamp':'DateTime'})

# Timestamp needs to be converted to datetime object (for time series manipulations)
data_FEM['DateTime'] = pd.to_datetime(data_FEM['DateTime'])
data_Williamsburg['DateTime'] = pd.to_datetime(data_Williamsburg['DateTime'])

# Time Series Analysis
data_FEM = data_FEM.set_index('DateTime')
data_FEM['time'] = data_FEM.index.strftime('%H:%M:%S')
data_FEM.head(3)

data_Williamsburg = data_Williamsburg.set_index('DateTime')
data_Williamsburg['time'] = data_Williamsburg.index.strftime('%H:%M:%S')
data_Williamsburg.head(3)


# time slicing
data_FEM= data_FEM.loc['2019-09-22 13:07:00':'2019-09-22 13:19:00'] #  Data
data_Williamsburg= data_Williamsburg.loc['2019-09-22 13:07:00':'2019-09-22 13:19:00'] #  Data

# averages
data_FEM_avg = [np.mean(data_FEM['W PM-2.5'])]*len(data_FEM['W PM-2.5'])
print('Average of FEM Data:', data_FEM_avg[1])

data_Williamsburg_avg = [np.mean(data_Williamsburg['Avg. Pm-2.5'])]*len(data_Williamsburg['Avg. Pm-2.5'])
print('Average of EP Data:', data_Williamsburg_avg[1])

data_merged = pd.merge(data_FEM, data_Williamsburg, how='inner', on='DateTime')
data_merged.info()


data_merged.to_csv(path + "\Output W4_EP4 09222019_Merged_Clean_File.csv")
path
# store descriptive statistics in a different dataframe and .csv file
desc_stat = data_merged.describe()
desc_stat.to_csv(path+'\Statistics_W4_EP4 09222019_Descriptive_Stats.csv')

# Line Plot - continous variables
plt.plot(data_FEM['time'], data_FEM['W PM-2.5'], label='W', c='b', linewidth = 2 )
plt.plot(data_FEM_avg, label='Avg_W', c='black', linestyle='--')
plt.plot(data_Williamsburg['time'], data_Williamsburg['Avg. Pm-2.5'], label='EP',  c='r', linewidth = 2)
plt.plot(data_Williamsburg_avg, label='Avg_EP', c='green', linestyle='--')
plt.xlabel('Time')
plt.ylabel('PM 2.5 values')
plt.title('Data Comparison - Same Time Period')
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig(path+'\Plot W4_EP4 09222019.png', dpi=300, bbox_inches='tight')



