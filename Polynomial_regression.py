"""Source duoc tham khao tu https://codeload.github.com/palVikram/Machine-Learning-using-Python/zip/master"""

# Step 1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2 Importing the dataset
dataset = pd.read_csv('Position_Salaries2.csv')

X = dataset.iloc[:, 1:2].values
print(X)
y = dataset.iloc[:, 2].values
print(y)
# Step3: Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X, y)

# Step 4: Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
poly_reg.fit(x_poly, y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly, y)

# Step 5: Visualising the Linear Regression results
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Linear Regression")
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()



x_grid=np.arange(min(X), max(X),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
# Step6: Visualising the Polynomial Regression results
plt.scatter(X,y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()
print('Bias: \n', lin_reg2.intercept_)
print('Thetas: \n', lin_reg2.coef_)

#Step 7
# Predicting a new result with linear regression
print(lin_reg.predict([[6]]))
#Predicting a new result with Polynomial resgression
print(lin_reg2.predict(poly_reg.fit_transform([[6]])))
def predict_salary(level):
    return  (890.15151515 * (level**4)) + (-15463.28671331 * (level**3)) + (94765.44289063 * (level**2)) + (-211002.33100292 * level) + 184166.66666719597
print(predict_salary(6))






