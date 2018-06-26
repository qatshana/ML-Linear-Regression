
'''
Machine Learning Program to Predict Survival in Titanic Ship
Program domestrate how to train  Logistical Regression to predict outcomes 
Data        ---  5000 samples of housing data 
Features    ---   Average Area Income, Average Area House Age, Average Area No of Rooms, Area Population
Target      ---  Price

'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

USAhousing = pd.read_csv('USA_Housing.csv')
# get basic information about the dataset
USAhousing.head()
USAhousing.describe()
USAhousing.info()

# get dataset columns
USAhousing.columns

# explore data - relationship between different variables
sns.pairplot(USAhousing)

# get standard distribution for target (Price)

sns.distplot(USAhousing['Price'])  # price is normally distributed 

sns.heatmap(USAhousing.corr())  # get headmap to get correlation with different variables

#Training a Linear Regression Mode

# define inputs (ignore address)
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

#define output file
y = USAhousing['Price']

#Create train and est sets using Train Test Split, 40% test data and 60% training data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# Train Model

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
sns.distplot((y_test-predictions),bins=50);

# Model Evaluation - Metrics MSE-RMSE-MAE

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
