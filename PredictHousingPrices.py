
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

def load_training_data():
	'''
    Explore data - relationship between different variables

    '''
	USAhousing = pd.read_csv('USA_Housing.csv')
	return USAhousing

def explore_data(USAhousing):
	'''
    Explore data - relationship between different variables

    '''
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


def process_data(train):
	'''
    Process data, nothing to process
    '''
	return train


def split_feature_target(USAhousing):
	X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
	               'Avg. Area Number of Bedrooms', 'Area Population']] # define inputs (ignore address)
	y = USAhousing['Price'] #define output file
	return X,y 

if __name__=='__main__':

    train=load_training_data() # load data
    train=process_data(train)  # process/clean data

    X,y=split_feature_target(train)

    #Create train and est sets using Train Test Split, 40% test data and 60% training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    lm = LinearRegression()
    lm.fit(X_train,y_train)
    # Model Evaluation
    print("Intercept point = %3.2f "%lm.intercept_)
    coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']) # use pandas data frame to organize and display coeff
    print(coeff_df)

    # Perform predictions
    predictions = lm.predict(X_test)
	
	# Model Evaluation - visual
    plt.scatter(y_test,predictions)
    sns.distplot((y_test-predictions),bins=50);
    
    # Model Evaluation - Error
    print("\n\nError Calculations")
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
