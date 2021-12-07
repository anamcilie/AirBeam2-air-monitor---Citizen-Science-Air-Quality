"""
@author: Ana MC Ilie

Citation: Ilie et. al, 2022. Air Pollution Exposure Assessment at Schools and Playgrounds in Williamsburg Brooklyn NYC, with a view to developing a set of policy solutions. 
Journal of Environmental Studies and Sciences, Springer. 
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
path = r'D:\Data Analysis\Stationary Network\AirBeams\Williamsburg\Williamsburg Pilot Study airbeams\colocation W'
data_FEM = pd.read_csv(path + '\W10a for colocation_Cleaned_Data_File.csv')

######         EP GROUPS         ##############
data_Williamsburg = pd.read_csv(path + '\W10b for colocation_Cleaned_Data_File.csv')

# data_FEM stands for the 1st AirBeam2 in the pelicanbox
# data_Williamsburg stands for the 2nd AirBeam2 in the same pelicanbox

data_FEM.info()
data_Williamsburg.info()

# rename columns
data_FEM = data_FEM.rename(columns={'Timestamp': 'DateTime'})
data_Williamsburg = data_Williamsburg.rename(columns={'Timestamp':'DateTime'})

# Timestamp needs to be converted to datetime object (for time series manipulations)
data_FEM['DateTime'] = pd.to_datetime(data_FEM['DateTime'])
data_Williamsburg['DateTime'] = pd.to_datetime(data_Williamsburg['DateTime'])

data_merged = pd.merge(data_FEM, data_Williamsburg, how='inner', on='DateTime')
data_merged.info()


# Linear Regression 
from sklearn.linear_model import LinearRegression
X = data_merged['PM2.5'].values
Y = data_merged['AirBeam2-PM2.5'].values
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

lin_regressor = LinearRegression()  # create object for the class
lin_regressor.fit(X, Y)  # perform linear regression
Y_pred = lin_regressor.predict(X)  # make predictions
r_sq = lin_regressor.score(X,Y) # R-square
coeff = lin_regressor.coef_
intercept = lin_regressor.intercept_

plt.scatter(X, Y, color = 'blue')
plt.plot(X, Y_pred, color='red')
plt.xlim(0)
plt.ylim(0)
plt.xlabel('W10a PM2.5')
plt.ylabel('W10b PM2.5')
plt.title('Linear Regression - W10a vs W10b')
plt.text(X.min(),Y.max(), 'Equation: y = %0.2f x + (%0.2f)'% (coeff, intercept), bbox=dict(facecolor='white', alpha=0.8))
plt.text(X.min(),Y.max()-2,'R-squared = %0.2f' % r_sq, bbox=dict(facecolor='white', alpha=0.8))
plt.show()
plt.savefig(path+'\Plot W10a-W10b - Linear Regression.png', dpi=300, bbox_inches='tight')

data_merged.to_csv(path + "\Output W10a_W10b_Merged_Clean_File.csv")
path













































