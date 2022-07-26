
"""
@author: Ana MC Ilie

Citation:  Ilie, A.M.C., McCarthy, N., Velasquez, L. et al. Air pollution exposure assessment at schools and playgrounds
in Williamsburg Brooklyn NYC, with a view to developing a set of policy solutions. J Environ Stud Sci (2022). https://doi.org/10.1007/s13412-022-00777-7


"""

import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import os
os.path
plt.style.use('default')
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


os.getcwd() # Get the default working directory
# Set the file path where .csv files are stored
path = r'C:\Personal Monitoring activities\Data Analysis June 10th\analysis python code'
os.chdir(path)

# Read the .csv file. If your file is .xls, save it as .csv first and then proceed
data = pd.read_csv('EP1 MARCH APRIL MAY JUNE one column no temp and humidity.csv')


data.dtypes # find the datatypes of all variables
data.columns # list of all column names - original list
data.shape # Dimensions of original dataset (rows, columns)
print(data.head(10)) # first 10 rows of dataset
data.index # index - number of rows and columns
data.info() # information on datatypes and number of elements

df = data.set_index(pd.DatetimeIndex(data['Date_Time']))
df = df.drop(['Date_Time'],axis = 1)


# analyzing only march data
data_march_25 = df['2019-03-25']
data_march_27 = df['2019-03-27']
data_march = df['2019-03']
df['2019-03-11':] # get all the data after March 11th
df['2019-03-10':'2019-03-30'] # time slicing, between certain periods

#Time series MARCH
data_march_27.plot(figsize=(20,10), linewidth=5, fontsize=20) # simple line chart

#data_march.boxplot() NO temp and NO humidity file 
data_march_27.boxplot()
plt.ylabel('Measured Value')
plt.xticks(rotation=90)
plt.show()

data_march.hist()

# analyzing only april data
data_april_1 = df['2019-04-1']
data_april_3 = df['2019-04-3']
data_april_10 = df['2019-04-10']
data_april_15 = df['2019-04-15']
data_april_17 = df['2019-04-17']
data_april = df['2019-04']
df['2019-04-01':] # get all the data after April 1st
df['2019-04-01':'2019-04-30'] # time slicing, between certain periods
#Time series APRIL
data_april_1.plot(figsize=(20,10), linewidth=5, fontsize=20) # simple line chart

#data_march.boxplot() NO temp and NO humidity file 
data_april_17.boxplot()
plt.ylabel('Measured Value')
plt.xticks(rotation=90)
plt.show()

data_april.hist()

# analyzing only MAY data
data_may_6 = df['2019-05-6']
data_may_15 = df['2019-05-15']
data_may_20 = df['2019-05-20']
data_may_22 = df['2019-05-22']
data_may_29 = df['2019-05-29']
data_may = df['2019-05']
df['2019-05-01':] # get all the data after April 1st
df['2019-05-01':'2019-05-31'] # time slicing, between certain periods
#Time series APRIL
data_may_6.plot(figsize=(20,10), linewidth=5, fontsize=20) # simple line chart

#data_march.boxplot() NO temp and NO humidity file 
data_may_29.boxplot()
plt.ylabel('Measured Value')
plt.xticks(rotation=90)
plt.show()

data_may.hist()


# analyzing only JUNE data
data_june_3 = df['2019-06-3']
data_june = df['2019-06']
df['2019-06-01':] # get all the data after April 1st
df['2019-06-01':'2019-06-30'] # time slicing, between certain periods
#Time series APRIL
data_june_3.plot(figsize=(20,10), linewidth=5, fontsize=20) # simple line chart


#data_march.boxplot() NO temp and NO humidity file 
data_june_3.boxplot()
plt.ylabel('Measured Value')
plt.xticks(rotation=90)
plt.show()

data_june.hist()


from datetime import datetime
# format yyyy, mm, dd
start_date = datetime(2019, 3, 11)
end_date = datetime(2019, 3, 27)
data_march[(start_date<=data_march.index) & (data_march.index<= end_date)].plot(grid=True)

# Correlation between values in same time period
x = data_march['F']
y = data_march['PM2.5']
plt.scatter(x, y)
import numpy as np
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))



# Time series with all variables
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xlabel('Time')
ax1.set_ylabel('PM Concentration', color='g')
ax2.set_ylabel('R and F', color='b')

ax1.plot(data['2019-03'], data['PM1'])
ax1.plot(data['2019-03'], data['PM10'])
ax1.plot(data['2019-03'], data['PM2.5'])

ax2.plot(data['2019-03'], data['F'], linestyle = '--' )
ax2.plot(data['2019-03'], data['RH'], linestyle = '--')

ax1.tick_params(axis ='x', rotation = 90)

#plt.xticks(rotation=90)
fig.legend()
plt.show()


# Saving the cleaned file into the same folder
#data.to_csv('EP1 may 06th 5pm csv_Cleaned.csv', index=False)

       
