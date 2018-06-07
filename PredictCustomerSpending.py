'''
500 samples of customer . 

Features include Average Area Income, Average Area House Age, Average Area No of Rooms, Area Population

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Load dataset from csv file
df=pd.read_csv('Ecommerce Customers')
# get basic information about the dataset
df.head()
df.describe()
df.info()

# get dataset columns
df.columns

# explore data - relationship between different variables

# jointplot
sns.jointplot(x=df['Time on Website'],y=df['Yearly Amount Spent'])

sns.pairplot(df)

sns.lmplot('Length of Membership','Yearly Amount Spent',df)


# get standard distribution for target (Price)

sns.distplot(USAhousing['Price'])  # price is normally distributed 

sns.heatmap(USAhousing.corr())  # get headmap to get correlation with different variables

#Training a Linear Regression Mode

# define inputs (ignore address)
X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

#define output file
y=df['Yearly Amount Spent']

#Create train and est sets using Train Test Split, 40% test data and 60% training data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# Model Evaluation
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) # use pandas data frame to organize and display coeff

print(coeff_df)

# Perform predictions

predictions = lm.predict(X_test)

# Model Evaluation - visual
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50)

# Model Evaluation - Metrics MSE-RMSE-MAE
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
