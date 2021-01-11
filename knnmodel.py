# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:24:06 2020

@author: Alexander
"""
# Import modules to process data and create the model
# Import numpy
import numpy as np
# Import pandas
import pandas as pd
# Import from sklearn KNN and train-test-split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
# Import matplotlip to plot results
import matplotlib.pyplot as plt
# Customized float (4 after the dot) formatting in a pandas DataFrame
pd.options.display.float_format = '{:.4f}'.format

# Load input data
History_data = pd.read_excel(r'C:\Users\Alexander\Desktop\history.xlsx',
                             sheet_name='history_data')
#print(History_data)
X = History_data.drop(["Property_ID", "Price"], axis=1)
#print(X)
Y = History_data.drop(["Property_ID", "#rooms", "Type", "Location"],
                      axis=1)
#print(Y)

# Partition training and testing data
test_size_ratio = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=test_size_ratio, 
                                                    random_state= 50)

# Convert to numpy arrays for easier manipulation
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
Y_train = Y_train.astype(np.float)
Y_test = Y_test.astype(np.float)


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))

# Plot the results of ther results  
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# Initialize learning model
knn = KNeighborsRegressor(n_neighbors = 3)

# Train the model
knn.fit(X_train, Y_train)

# Test model's performance using Median Relative Absolute Error
Y_pred = knn.predict(X_test)
accuracy = np.median(np.abs(Y_pred - Y_test) / Y_test)
#print(accuracy)

Property_data = pd.read_excel(r'C:\Users\Alexander\Desktop\property.xlsx', 
                              sheet_name='Sheet1')
#print(Property_data)
# Drop the columns list ID and Price to make the predictions
Property = Property_data.drop(['List_ID', 'Price'], axis=1)
#print(Property)
# Use the knn model to predict the price on the given excel file
Y_pred = knn.predict(Property)
#print(Y_pred)
predictions = pd.DataFrame({'Predictions': Y_pred[:, 0]})
#print(predictions)
# Concatonate the two dataframes (Property_data and predictions) to one
df = pd.concat([Property_data, predictions], axis= 1)
#print(df)
df = df.rename(columns = {'#room': 'rooms'})
df

def find_properties(rooms, typeprop, location, price):
    """ Find the properties with the specified characteristics"""
    if price == 0:
        found_properties = (df[(df['rooms'] == rooms) & (df['Type'] == typeprop) &
                               (df['Location'] == location)])
    else:
        # If price is given then only properties that cost less or are
        # equal to the provided price are displayed
        found_properties=(df[(df['rooms'] == rooms) & (df['Type'] == typeprop) &
                             (df['Location'] == location) & (df['Price'] <= price)])
    return found_properties
