import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

def data():
	n = 10
	df = pd.DataFrame()
	df['x'] = np.linspace(0,10,num=n)
	df['y'] = df['x'] + np.random.normal(0,1,size=n)

	X, y = df.drop('y',axis=1), df['y']
	return df, X, y

df = pd.read_csv("ols.csv")
X, y = df.drop('y', axis=1), df['y']

lm = LinearRegression()
lm.fit(X, y)
beta0 = lm.intercept_
beta1 = lm.coef_[0]

fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
ax.scatter(df['x'], df['y'], s=95)
ax.plot(df['x'], df['x']*lm.coef_[0] + lm.intercept_, c='orange')
ax.set_xlabel("x", fontsize=12, labelpad=-5)
ax.set_ylabel("y", fontsize=12, labelpad=-5)
ax.set_title(f"OLS\n$\\beta_1$ = {beta1:.3f}, $\\beta_0$ = {lm.intercept_:.2f}", fontsize=14)
plt.tight_layout()
plt.savefig(f"../images/ols.svg", bbox_inches=0, pad_inches=0)
plt.show()

y.iloc[-1] = 100 # add outlier
lm = LinearRegression()
lm.fit(X, y)
beta0 = lm.intercept_
beta1 = lm.coef_[0]

fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
ax.scatter(df['x'], df['y'], s=95)
ax.plot(df['x'], df['x']*lm.coef_[0] + lm.intercept_, c='orange')
ax.set_xlabel("x", fontsize=12, labelpad=-5)
ax.set_ylabel("y", fontsize=12, labelpad=-5)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(f"OLS with outlier\n$\\beta_1$ = {beta1:.3f}, $\\beta_0$ = {lm.intercept_:.2f}", fontsize=14)
plt.tight_layout()
plt.savefig(f"../images/ols_outlier.svg", bbox_inches=0, pad_inches=0)

lm = Lasso(alpha=45)
lm.fit(X, y)
beta0 = lm.intercept_
beta1 = lm.coef_[0]

fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
ax.scatter(df['x'], df['y'], s=95)
ax.plot(df['x'], df['x']*lm.coef_[0] + lm.intercept_, c='orange')
ax.set_xlabel("x", fontsize=12, labelpad=-5)
ax.set_ylabel("y", fontsize=12, labelpad=-5)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(f"L1 Lasso with outlier\n$\\beta_1$ = {beta1:.3f}, $\\beta_0$ = {lm.intercept_:.2f}", fontsize=14)
plt.tight_layout()
plt.savefig(f"../images/lasso.svg", bbox_inches=0, pad_inches=0)
