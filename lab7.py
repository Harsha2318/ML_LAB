import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Linear Regression: California Housing
d = fetch_california_housing(as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(d.data[['AveRooms']], d.target)
m = LinearRegression().fit(Xtr, ytr)
plt.scatter(Xte, yte,alpha =0.5)
plt.plot(Xte, m.predict(Xte), color='red')
plt.title("Linear Regression")
plt.show()

# Polynomial Regression: Auto MPG (seaborn)
df = sns.load_dataset('mpg').dropna()
Xtr, Xte, ytr, yte = train_test_split(df[['displacement']], df['mpg'])
poly = PolynomialFeatures(2)
Xtr2 = poly.fit_transform(Xtr)
Xte2 = poly.transform(Xte)
m2 = LinearRegression().fit(Xtr2, ytr)
plt.scatter(Xte, yte, alpha=0.5)
plt.scatter(Xte, m2.predict(Xte2), color='red')
plt.title("Polynomial Regression")
plt.show()
