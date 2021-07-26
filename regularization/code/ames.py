import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle, Ellipse
import mpl_toolkits.mplot3d.art3d as art3d
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import matplotlib as mpl
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pandas.api.types import is_numeric_dtype
def normalize(X): # well, I'm creating standard variables here (u-x)/sigma
    for colname in X.columns:
        if is_numeric_dtype(X[colname]):
            u = np.mean(X[colname])
            s = np.std(X[colname])
            X[colname] = (X[colname] - u) / s

df_ames = pd.read_csv("ames.csv")

#print(df_ames.info())
cols_with_missing = df_ames.columns[df_ames.isnull().any()]
cols = set(df_ames.columns) - set(cols_with_missing)

X = df_ames[cols]
X = X.drop('SalePrice', axis=1)

normalize(X)
X = pd.get_dummies(X)
y = df_ames['SalePrice']

print("mean house price", np.mean(y))


np.random.seed(1) # get stable result for article

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lm = LinearRegression()
lm.fit(X_train, y_train)
unreg_score = lm.score(X_test, y_test)
unreg_mae = mean_absolute_error(y_test, lm.predict(X_test))

coef = lm.coef_
coef = np.clip(coef,-1e8,1e8) #Clip so we can display

fig, ax = plt.subplots(1, 1, figsize=(5,4))
ax.bar(range(len(coef)),coef)
ax.set_xlabel("Regression coefficient $\\beta_i$", fontsize=12)
ax.set_ylabel("Regression coefficient value\nclipped to 1e8 magnitude", fontsize=12)
#ax.set_ylim(-1e8,1e8)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:8.1e}'))
ax.set_yticks([-1e8,0,1e8])
ax.set_title(f"AMES data OLS coefficients\nR^2 = {unreg_score:.1e}, MAE = {unreg_mae:.1e} dollars")
plt.tight_layout()
plt.savefig(f"../images/ames_ols.svg", bbox_inches=0, pad_inches=0)
plt.show()

# REGULARIZE using grid search

tuned_parameters = {'alpha': [.1, 1, 10, 50, 70, 80, 90, 100, 120, 140, 300, 500]}
grid = GridSearchCV(
    Ridge(), tuned_parameters, scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=0
)
grid.fit(X_test, y_test)
print("Ridge best:", grid.best_params_)
best_alpha = grid.best_params_['alpha']
# lm = grid.best_estimator_

lm = Ridge(alpha=best_alpha, tol=0.1)
lm.fit(X_train, y_train)
reg_score = lm.score(X_test, y_test)
reg_mae = mean_absolute_error(y_test, lm.predict(X_test))

coef = lm.coef_

fig, ax = plt.subplots(1, 1, figsize=(4.8,4))
ax.bar(range(len(coef)),coef)
ax.set_xlabel("Regression coefficient $\\beta_i$", fontsize=12)
ax.set_ylabel("Regression coefficient value", fontsize=12)
ax.set_ylim(-8000,20000)
ax.set_yticks([-5000,0,5000,10000,15_000,20_000])
# ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.1}'))
ax.set_title(f"AMES data Ridge(alpha={best_alpha}) coefficients\nR^2 = {reg_score:.2f}, MAE = {reg_mae:,.0f} dollars")
plt.tight_layout()
plt.savefig(f"../images/ames_L2.svg", bbox_inches=0, pad_inches=0)
plt.show()
