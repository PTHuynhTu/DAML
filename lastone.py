"""Source được tham khao tu:
https://github.com/ybenzaki/multivariate_linear_regression_python
https://datatofish.com/multiple-linear-regression-python/"""

#Step 1 Import thu vien can thiet
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#Step 2 Import dataset (dataframe)
dataframe = pd.read_csv('data.csv')
X = dataframe[['kich_thuoc_theo_feet_vuong','so_phong']]
Y = dataframe['gia']
print(X)
print(Y)
# Step 3 with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Bias: \n', regr.intercept_)
print('Thetas: \n', regr.coef_)
# Step 4 prediction with sklearn
kichthuocmoi = 4500
sophongmoi = 5
print ('Gia nha du doan thuc te voi 4500 feet vuong va 5 phong: \n', regr.predict([[kichthuocmoi ,sophongmoi]]))
# Step 5 Ve ra du lieu with statsmodels
X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(dataframe["kich_thuoc_theo_feet_vuong"], dataframe["so_phong"], dataframe["gia"], c='r', marker='^')
ax.set_xlabel('kich thuoc theo feet vuong')
ax.set_ylabel('so_phong')
ax.set_zlabel('gia theo $')

#Step 6 Ve ra model
def predict_price_of_house(dientich, so_phong):
    return 138.21140719 * dientich + 632.27498599 * so_phong + 57635.97842858877 # not scaled
    #return 1.094e+05 * taille_maison + (6578.3549 * nb_chambre) # scaled

def predict_all(lst_sizes, lst_nb_chmbres):
    predicted_prices = []
    for ii in range(0, len(Y)):
        predicted_prices.append(predict_price_of_house(lst_sizes[ii], lst_nb_chmbres[ii]))
    return predicted_prices
ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.plot_trisurf(dataframe["kich_thuoc_theo_feet_vuong"], dataframe["so_phong"], predict_all(dataframe["kich_thuoc_theo_feet_vuong"], dataframe["so_phong"]))
plt.show()


