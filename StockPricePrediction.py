import pandas as pd
import numpy as np
import math 
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import pandas_datareader.data as web

from pandas.plotting import scatter_matrix
from pandas import Series, DataFrame
from matplotlib import style

from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv('AAPL.csv')
#df = web.DataReader("AAPL", 'yahoo')

print("Data frame size", df.count())
print("Data frame head",df.head())

dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High']-df['Close'])/df['Close'] * 100.0 
dfreg['PCT_change'] = (df['Close'] - df['Open'])/df['Open'] * 100.0


#Data Preprocessing and cleaning
# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 10)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

#Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)

#Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)


#Linear regression Prediction
yclfreg_predict = clfreg.predict(X_test)

# Quadratic Regression 2 Prediction
yclfpoly2_predict = clfpoly2.predict(X_test)

# Quadratic Regression 3 Prediction
yclfpoly3_predict = clfpoly3.predict(X_test)

# KNN Regression Prediction
yclfknn_predict = clfknn.predict(X_test)

# Lasso Regression Prediction
ylasso_predict = lasso.predict(X_test)

# Ridge Regression Prediction
yridge_predict = ridge.predict(X_test)

#Linear regression Model Evaluation
linerRegressionAccuracy = r2_score(y_test, yclfreg_predict)

#Quadratic Regression 2 Model Evaluation
quadraticRegression2Accuracy = r2_score(y_test, yclfpoly2_predict)

#Quadratic Regression 3 Model Evaluation
quadraticRegression3Accuracy = r2_score(y_test, yclfpoly3_predict)

#KNN Regression Model Evaluation
knnAccuracy = r2_score(y_test, yclfknn_predict)

# Lasso Regression Model Evaluation
lassoRegressionAccuracy = r2_score(y_test, ylasso_predict)

# Ridge Regression Model Evaluation
ridgeRegressionAccuracy = r2_score(y_test, yridge_predict)

print("Linear regression accuracy = ", linerRegressionAccuracy*100)
print("Quadratic Regression 2 accuracy = ", quadraticRegression2Accuracy*100)
print("Quadratic Regression 3 accuracy = ", quadraticRegression3Accuracy*100)
print("KNN Regression accuracy = ", knnAccuracy*100)
print("Lasso Regression accuracy = ", lassoRegressionAccuracy*100)
print("Ridge Regression accuracy = ", ridgeRegressionAccuracy*100)


predictions = [linerRegressionAccuracy, quadraticRegression2Accuracy, quadraticRegression3Accuracy, knnAccuracy, lassoRegressionAccuracy, linerRegressionAccuracy]
bestPrediction = max(predictions)

if (bestPrediction == linerRegressionAccuracy):
	print("Best prediction is through LinearRegression with accuracy", linerRegressionAccuracy*100)
if(bestPrediction == quadraticRegression2Accuracy):
	print("Best prediction is through quadraticRegression2 with accuracy", quadraticRegression2Accuracy*100)
if(bestPrediction == quadraticRegression3Accuracy):
	print("Best prediction is through quadraticRegression3 with accuracy", quadraticRegression3Accuracy*100)
if(bestPrediction == knnAccuracy):
	print("Best prediction is through KNN regression with accuracy", knnAccuracy*100)
if(bestPrediction == lassoRegressionAccuracy):
	print("Best prediction is through Lasso regression with accuracy", lassoRegressionAccuracy*100)
if(bestPrediction == ridgeRegressionAccuracy):
	print("Best prediction is through Ridge with accuracy", ridgeRegressionAccuracy*100)


# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(y_test, yclfreg_predict)
#plt.show()

# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(y_test, yclfpoly2_predict)
#plt.show()


# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(y_test, yclfpoly3_predict)
#plt.show()


# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(y_test, yclfknn_predict)
#plt.show()


# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(y_test, ylasso_predict)
#plt.show()


# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(y_test, yridge_predict)
plt.show()