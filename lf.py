import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(pd.read_csv('meterial\\Salary_dataset.csv'))

df.head(5)

df.shape

y = df['YearsExperience'].values.reshape(-1, 1)
x = df['Salary'].values.reshape(-1, 1)

x

y

plt.scatter(x, y)
plt.xlabel('House Age')
plt.ylabel('Med Inc')
plt.title("Data Plot")

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.20, random_state= 3)

xtrain.shape

xtest.shape

ytest.shape

xtest

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(xtrain, ytrain)

lr.predict(xtest)

ytest

y_pred = lr.predict(x)

plt.scatter(xtrain, ytrain, label='Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.xlabel('Independent Variable (x)')
plt.ylabel('Dependent Variable (y)')
plt.title('Data and Regression Line')
plt.legend()
plt.show()


ytest

lr.predict([[45300]])

print(lr.coef_)

print(lr.intercept_)
