# https://medium.com/@contactsunny/linear-regression-in-python-using-scikit-learn-f0f7b125a204

import numpy
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pandas.read_csv('salaryData.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into the training set and test set
# We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,
# and 10 rows will go into the testing set.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# create our object

linearRegressor = LinearRegression()

#Next, we have to fit this model to our data, in other words, we have to make it “learn” using our training data. For that, its just one other line of code:
linearRegressor.fit(xTrain, yTrain)

# You can now start testing the model with the testing dataset you have.
yPrediction = linearRegressor.predict(xTest)

# The next step is to see how well your prediction is working. 
plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
plot.title('Salary vs Experience (Training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

# Now let’s look at the plot for the test set, and the code for that is here:
plot.scatter(xTest, yTest, color = 'red')
plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
plot.title('Salary vs Experience (Test set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()
