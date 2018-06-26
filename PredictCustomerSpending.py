'''
Machine Learning Program to Predict Yearly Customer Spending
Program domestrate how to train  Linear Regression to predict outcomes 
Data        ---  500 samples of customers data 
Features    ---  Time on website, Time on App, length of membershipt and others
Target      ---  Annual Yearly Spending

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



def load_training_data():
    '''
    Load training data from csv file
    output -- training set in dataframe format

    '''
    train = pd.read_csv('Ecommerce Customers')
    return train

def explore_data(df):
	'''
    Explore data - relationship between different variables

    '''
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

def process_data(df):
	'''
    Process data, nothing to process
    '''
	return df

def split_feature_target(df):
	# define inputs (ignore address)
	X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

	#define output file
	y=df['Yearly Amount Spent']
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
