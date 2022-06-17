
"""
@author: Ana MC Ilie

Citation: Ilie A.M.C., 2022. Processing citizen-generated air quality data using Python programming language for a data science high school curriculum.  
"""

import pandas as pd
from pandas.compat import StringIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
from pandas import Series        # To work on series
import statsmodels
import statsmodels.api as sm 
from pylab import figure, axes, pie, title, show

import warnings                   # To ignore the warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("white")
plt.style.use("seaborn")
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(14, 8)})


# finding the current directory
abs_path = os.getcwd()
abs_path

# change to desired folder where .csv file is present - Use forward backslash
path = r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\cleaned data'
data = pd.read_csv(path + '/all airbeams PM2.5 CLEANED file.csv')

# Exploratory Data Analysis

data.dtypes # find the datatypes of all variables
data.columns # list of all column names - original list
data.shape # Dimensions of original dataset (rows, columns)
print(data.head(10)) # first 10 rows of dataset
data.index # index - number of rows and columns
data.info() # information on datatypes and number of elements

# Convert the datatype of certain columns to float type 
data[['PM-2.5_W1','PM-2.5_W2A','PM-2.5_W2b','PM-2.5_W3','PM-2.5_W4','PM-2.5_W5','PM-2.5_W6a','PM-2.5_W6b','PM-2.5_W7','PM-2.5_W8','PM-2.5_W9','PM-2.5_W10a','PM-2.5_W10b','PM-2.5_W11a']] = data[['PM-2.5_W1','PM-2.5_W2A','PM-2.5_W2b','PM-2.5_W3','PM-2.5_W4','PM-2.5_W5','PM-2.5_W6a','PM-2.5_W6b','PM-2.5_W7','PM-2.5_W8','PM-2.5_W9','PM-2.5_W10a','PM-2.5_W10b','PM-2.5_W11a']].apply(pd.to_numeric)
data[['Timestamp_W1','Timestamp_W2a','Timestamp_W2b','Timestamp_W3','Timestamp_W4','Timestamp_W5','Timestamp_W6a','Timestamp_W6b','Timestamp_W7','Timestamp_W8','Timestamp_W9','Timestamp_W10a','Timestamp_W10b','Timestamp_W11a']] = data[['Timestamp_W1','Timestamp_W2a','Timestamp_W2b','Timestamp_W3','Timestamp_W4','Timestamp_W5','Timestamp_W6a','Timestamp_W6b','Timestamp_W7','Timestamp_W8','Timestamp_W9','Timestamp_W10a','Timestamp_W10b','Timestamp_W11a']].apply(pd.to_numeric)

# Timestamp needs to be converted to datetime object (for time series manipulations)
data['Timestamp_W1'] = pd.to_datetime(data['Timestamp_W1'])
data['Timestamp_W2a'] = pd.to_datetime(data['Timestamp_W2a'])
data['Timestamp_W2b'] = pd.to_datetime(data['Timestamp_W2b'])
data['Timestamp_W3'] = pd.to_datetime(data['Timestamp_W3'])
data['Timestamp_W4'] = pd.to_datetime(data['Timestamp_W4'])
data['Timestamp_W5'] = pd.to_datetime(data['Timestamp_W5'])
data['Timestamp_W6a'] = pd.to_datetime(data['Timestamp_W6a'])
data['Timestamp_W6b'] = pd.to_datetime(data['Timestamp_W6b'])
data['Timestamp_W7'] = pd.to_datetime(data['Timestamp_W7'])
data['Timestamp_W8'] = pd.to_datetime(data['Timestamp_W8'])
data['Timestamp_W9'] = pd.to_datetime(data['Timestamp_W9'])
data['Timestamp_W10a'] = pd.to_datetime(data['Timestamp_W10a'])
data['Timestamp_W10b'] = pd.to_datetime(data['Timestamp_W10b'])
data['Timestamp_W11a'] = pd.to_datetime(data['Timestamp_W11a'])



# Timestamp needs to be converted to datetime object (for time series manipulations)
data[['Timestamp_W1','Timestamp_W2a','Timestamp_W2b','Timestamp_W3','Timestamp_W4','Timestamp_W5','Timestamp_W6a','Timestamp_W6b','Timestamp_W7','Timestamp_W8','Timestamp_W9','Timestamp_W10a','Timestamp_W10b','Timestamp_W11a']] = data[['Timestamp_W1','Timestamp_W2a','Timestamp_W2b','Timestamp_W3','Timestamp_W4','Timestamp_W5','Timestamp_W6a','Timestamp_W6b','Timestamp_W7','Timestamp_W8','Timestamp_W9','Timestamp_W10a','Timestamp_W10b','Timestamp_W11a']].apply(pd.to_numeric)



# store descriptive statistics in a different dataframe and .csv file
desc_stat = data.describe()
desc_stat.to_csv(path+'\Descriptive_Stats.csv')

# see the results of desired column
print (desc_stat['PM-2.5_W1'])
print (desc_stat['PM-2.5_W2A'])



# Missing Value Analysis
# list number of missing values for each column
print (data.isnull().sum())
null_counts = data.isnull().sum()
null_counts.to_csv(path+ '\List number of missing values for each column.csv')


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
data.to_csv(path + "\Cleaned_Data_File_allairbeams.csv")



# Continous Variables in the dataset

data_cont = data.iloc[:, 0:42].reset_index() 
data_cont = data_cont.drop(['index'], axis = 1)
data_cont.shape
data_cont.columns


# Boxplot Analysis to see presence of outliers and save the file.png
fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W1"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W1.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W2A"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W2A.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W2b"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W2b.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W3"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W3.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W4"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W4.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W5"], data=data)
fig.savefig(r'C:\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W5.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W6a"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W6a.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W6b"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W6b.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W7"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W7.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W8"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W8.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W9"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W9.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W10a"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W10a.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W10b"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W10b.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.boxplot(y=data_cont["PM-2.5_W11a"], data=data)
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplot_W11a.png') # save the figure to file
plt.close(fig) # close the figure


# Basic Plots boxplots all variables
fig = plt.figure()
data_cont.boxplot()
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\boxplots all variables.png') # save the figure to file
plt.close(fig) # close the figure

# Pair Plot
sns.pairplot(data_cont)


# Line Plot - continous variables
fig = plt.figure()
pm_conc = ['PM-2.5_W1','PM-2.5_W2A','PM-2.5_W2b','PM-2.5_W3','PM-2.5_W4','PM-2.5_W5','PM-2.5_W6a','PM-2.5_W6b','PM-2.5_W7','PM-2.5_W8','PM-2.5_W9','PM-2.5_W10a','PM-2.5_W10b','PM-2.5_W11a']
plt.plot(data_cont[pm_conc])
plt.xlabel('Timestamp')
plt.ylabel('PM Concentration')
plt.show()
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\lineplot.png') # save the figure to file
plt.close(fig) # close the figure

# Histogram
data_cont.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# Distribution Plot - Kernel Density Estimate
plt.figure(figsize=(20,10))
sns.distplot(data_cont['PM-2.5_W1'], label='PM-2.5_W1')
sns.distplot(data_cont['PM-2.5_W2A'], label='PM-2.5_W2A')
sns.distplot(data_cont['PM-2.5_W2b'], label='PM-2.5_W2b')
sns.distplot(data_cont['PM-2.5_W3'], label='PM-2.5_W3')
sns.distplot(data_cont['PM-2.5_W4'], label='PM-2.5_W4')
sns.distplot(data_cont['PM-2.5_W5'], label='PM-2.5_W5')
sns.distplot(data_cont['PM-2.5_W6a'], label='PM-2.5_W6a')
sns.distplot(data_cont['PM-2.5_W6b'], label='PM-2.5_W6b')
sns.distplot(data_cont['PM-2.5_W7'], label='PM-2.5_W7')
sns.distplot(data_cont['PM-2.5_W8'], label='PM-2.5_W8')
sns.distplot(data_cont['PM-2.5_W9'], label='PM-2.5_W9')
sns.distplot(data_cont['PM-2.5_W10a'], label='PM-2.5_W10a')
sns.distplot(data_cont['PM-2.5_W10b'], label='PM-2.5_W10b')
sns.distplot(data_cont['PM-2.5_W11a'], label='PM-2.5_W11a')
plt.xlabel('Variable Estimate')
plt.title('Distribution Plot - Kernel Density Estimate')
plt.legend()



# Linear regression models
fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W1", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W1.png') # save the figure to file
plt.close(fig) # close the figure

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W2b", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W2b.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W3", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W3.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W4", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W4.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W5", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W5.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W6a", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W6a.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W6b", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W6b.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W7", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W7.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W8", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W8.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W9", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W9.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W10a", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W10a.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W10b", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W10b.png') # save the figure to file
plt.close(fig)

fig = plt.figure()
sns.regplot(x="PM-2.5_W2A", y="PM-2.5_W11a", data=data);
fig.savefig(r'C:\Users\Data Analysis\Intercomparison\Fixed DEC site\March 18th 9am\linearreg_W11a.png') # save the figure to file
plt.close(fig)



# Conditioning on other variables
#sns.lmplot(x="PM-2.5_W2A", y="PM-2.5_W1", col="Timestamp_W1", data=data);


# Plotting a regression in other contexts 
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W1", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W2b", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W3", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W4", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W5", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W6a", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W6b", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W7", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W8", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W9", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W10a", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W10b", data=data, kind="reg");
sns.jointplot(x="PM-2.5_W2A", y="PM-2.5_W11a", data=data, kind="reg");


# other pairplot

sns.pairplot(data, x_vars=['PM-2.5_W1','PM-2.5_W2b','PM-2.5_W3','PM-2.5_W4','PM-2.5_W5','PM-2.5_W6a','PM-2.5_W6b','PM-2.5_W7','PM-2.5_W8','PM-2.5_W9','PM-2.5_W10a','PM-2.5_W10b','PM-2.5_W11a'], y_vars=["PM-2.5_W2A"],
             height=5, aspect=.8, kind="reg");

             
# CREATE LINEAR REGRESSION
regr = linear_model.LinearRegression()

## X usually means our input variables (or independent variables)
## Y usually means our output/dependent variable

X = data["PM-2.5_W1"]
y = data["PM-2.5_W2A"]
## let's add an intercept (beta_0) to our model 
X = sm.add_constant(X)             
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
# Print out the statistics
model.summary()  
           
# to_csv(path + "\linear regression.csv")


# Time Series Analysis
data2 = data.set_index('Timestamp_W3')
data2.head(3)

data2['Month'] = data2.index.month
data2['Year'] = data2.index.year
data2['Weekday Name'] = data2.index.weekday_name
data2.sample(5, random_state = 0)

# time slicing
data_March = data2.loc['2019-03-12':'2019-03-17'] #  Data
data_March2 = data2.loc['2019-03']  # Data
data_March['PM-2.5_W1'].plot(linewidth = 0.5)
cols_plot = ['PM-2.5_W1']
axes = data_March[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('PM Concentration')

# slicing and plotting at the same time
data2.loc['2019-03-7 15:00:00':'2019-03-17 20:30:00', 'PM-2.5_W1'].plot()







