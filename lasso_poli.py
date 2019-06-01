#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#from IPython import get_ipython
#ipy = get_ipython()
#if ipy is not None:
 #   ipy.run_line_magic('matplotlib', 'inline')
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 12, 10

#đọc dữ liệu từ Position_Salaries.csv
dataset = pd.read_csv('Position_Salaries.csv')
#set biến x là dữ liệu vị trí
x = dataset.iloc[:, 0].values
#set bien y la tien luong tuong ung voi vi tri
y = dataset.iloc[:, 1].values
# vẽ đồ thị các x,y tương ứng
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')
plt.show()
#tính các biến x từ bậc 2 tới bậc 15 tương ứng mỗi vòng lặp x tăng một bậc
for i in range(2,16):  #bậc 1 tồn tại
    colname = 'x_%d'%i      #biến x mới của bậc mới
    data[colname] = data['x']**i
print(data.head())


#Import Linear Regression model from scikit-learn.
from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    #khởi tạo predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)]) #nếu bậc lớn hơn 2 thì biến x được mở rộng ra(2 feature)
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    #vẽ sơ đồ polinomial tương ứng
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Degree = %d'%power)
    #trả về kết quả của res
    rss = ((sum((y_pred-data['y'])**2))/(2*len(x)))  #cost function
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret
#Khởi tạo dataframe để chứa các theta
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Khai báo degree cho model
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

#Lặp lại các bước tính tăng dần theo power(degree) và cho ra kết quả
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
plt.show()

#lasso regression
from sklearn.linear_model import Lasso
def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    model = lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    print(model.score(data[predictors],data['y']))
    print('Bias: \n', lassoreg.intercept_)
    print('Thetas: \n', lassoreg.coef_)
    #Vẽ sơ đồ theo lasso
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred) #vẽ model dự đoán
        plt.plot(data['x'],data['y'],'.') # vẽ dữ liệu thật
        plt.title('Lamda = %.3g'%alpha) # tên của model

    #trả về kết quả của ret
    rss = ((sum((y_pred-data['y'])**2))/(2*len(x))) #cost function
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret
#Khởi tạo dữ liệu dự đoán cho 15 bậc của x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Khai báo alpha(lamda) để test
alpha_lasso = [1e-50, 1e-10, 1e-8, 1e-5,1, 10, 100, 1000, 10000, 1000000]

#khởi tạo dataframe để chứa các theta
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Chọn ra 6 trong số 10 dữ liệu alpha(lamda) để cho vào đồ thị
models_to_plot = {1e-50:231, 10:232,100:233, 1000:234, 10000:235, 1000000:236}

#Lặp lại 10 giá trị của alpha(lamda) để so sánh kết quả.
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)
plt.show()
z = []
z = coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)
print(z)
#source code from https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/?fbclid=IwAR2aouDiAbjaW5J3fBHe3fhqpQGCqLGyu-LJ5STYgb-Pxv_El4yHrssyMiQ
