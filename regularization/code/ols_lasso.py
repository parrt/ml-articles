import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Circle, Ellipse
import mpl_toolkits.mplot3d.art3d as art3d
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

def data():
	n = 10
	df = pd.DataFrame()
	df['x'] = np.linspace(0,10,num=n)
	df['y'] = df['x'] + np.random.normal(0,1,size=n)

	X, y = df.drop('y',axis=1), df['y']
	return df, X, y


def loss(b0,b1,x,y):
	return np.mean((y - (b0 + b1 * x)) ** 2)


def olsfit(X, y):
	global opt_beta0, opt_beta1
	lm = LinearRegression()
	lm.fit(X, y)
	opt_beta0 = lm.intercept_
	opt_beta1 = lm.coef_[0]


def ols():
	X, y = df.drop('y', axis=1), df['y']
	olsfit(X,y)
	fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
	ax.scatter(df['x'], df['y'], s=95)
	ax.plot(df['x'], df['x']*opt_beta1 + opt_beta0, 's-', c='orange', markersize=6, alpha=.7)
	ax.set_xlabel("x", fontsize=12, labelpad=-5)
	ax.set_ylabel("y", fontsize=12, labelpad=-5)
	ax.set_title(f"OLS\n$\\beta_1$ = {opt_beta1:.3f}, $\\beta_0$ = {opt_beta0:.2f}", fontsize=14)
	plt.tight_layout()
	plt.savefig(f"../images/ols.svg", bbox_inches=0, pad_inches=0)
	plt.show()


def ols3D():
	X, y = df.drop('y', axis=1), df['y']
	olsfit(X,y)
	beta0 = np.linspace(-10,10,num=150)
	beta1 = np.linspace(-1,3,num=100)
	Z = np.zeros(shape=(len(beta0),len(beta1)))
	min_loss = (99999,0,0)
	for i,b0 in enumerate(beta0):
		for j,b1 in enumerate(beta1):
			l = loss(b0,b1,df['x'].values,y.values)
			print(b0,b1,'->',l)
			Z[i,j] = l
			if l < min_loss[0]:
				min_loss = (l, b0, b1)

	print("min loss", min_loss)

	# 3D

	fig = plt.figure(figsize=(3.9,3.7))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel("$\\beta_0$", labelpad=0)
	ax.set_ylabel("$\\beta_1$")
	B0, B1 = np.meshgrid(beta0, beta1)
	ax.plot_surface(B0, B1, Z.T, alpha=0.7, cmap='coolwarm')
	optcir = Ellipse(xy=(opt_beta0,opt_beta1), width=1.5, height=1.5/5, color='k')
	ax.add_patch(optcir)
	art3d.pathpatch_2d_to_3d(optcir, z=0)
	ax.contour(B0, B1, Z.T, levels=30, linewidths=.5, cmap='coolwarm',
					   zdir='z', offset=0)

	# ax.view_init(elev=47, azim=-112)
	ax.view_init(elev=50, azim=-103)
	plt.tight_layout()
	plt.savefig(f"../images/ols_loss_3D.svg", bbox_inches=0, pad_inches=0)
	plt.show()

def ols2D():
	X, y = df.drop('y', axis=1), df['y']
	olsfit(X,y)
	beta0 = np.linspace(-10, 10, num=150)
	beta1 = np.linspace(-1, 3, num=100)
	Z = np.zeros(shape=(len(beta0), len(beta1)))
	min_loss = (99999, 0, 0)
	for i, b0 in enumerate(beta0):
		for j, b1 in enumerate(beta1):
			l = loss(b0, b1, df['x'].values, y.values)
			print(b0, b1, '->', l)
			Z[i, j] = l
			if l < min_loss[0]:
				min_loss = (l, b0, b1)

	print("min loss", min_loss)
	fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
	ax.set_xlabel("$\\beta_0$")
	ax.set_ylabel("$\\beta_1$")
	B0, B1 = np.meshgrid(beta0, beta1)
	ax.contour(B0, B1, Z.T, levels=30, linewidths=1.0, cmap='coolwarm')
	ax.scatter([opt_beta0],[opt_beta1], s=90, c='k')
	plt.tight_layout()
	plt.savefig(f"../images/ols_loss_2D.svg", bbox_inches=0, pad_inches=0)
	plt.show()


def ols_outlier():
	X, y = df.drop('y', axis=1), df['y']
	y = y.copy()
	y.iloc[-1] = 100 # add outlier
	lm = LinearRegression()
	lm.fit(X, y)
	beta0 = lm.intercept_
	beta1 = lm.coef_[0]

	fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
	ax.scatter(X, y, s=95)
	ax.plot(df['x'], df['x']*beta1 + beta0, 's-', c='orange', markersize=6, alpha=.7)
	ax.set_xlabel("x", fontsize=12, labelpad=-5)
	ax.set_ylabel("y", fontsize=12, labelpad=-5)
	ax.set_xlabel("x", fontsize=12)
	ax.set_ylabel("y", fontsize=12)
	ax.set_title(f"OLS with outlier\n$\\beta_1$ = {beta1:.3f}, $\\beta_0$ = {beta0:.2f}", fontsize=14)
	plt.tight_layout()
	plt.savefig(f"../images/ols_outlier.svg", bbox_inches=0, pad_inches=0)
	plt.show()


def lasso():
	X, y = df.drop('y', axis=1), df['y']
	y = y.copy()
	y.iloc[-1] = 100 # add outlier
	lm = Lasso(alpha=45)
	lm.fit(X, y)
	beta0 = lm.intercept_
	beta1 = lm.coef_[0]

	fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
	ax.scatter(X, y, s=95)
	ax.plot(df['x'], df['x']*beta1 + beta0, 's-', c='orange', markersize=6, alpha=.7)
	ax.set_xlabel("x", fontsize=12, labelpad=-5)
	ax.set_ylabel("y", fontsize=12, labelpad=-5)
	ax.set_xlabel("x", fontsize=12)
	ax.set_ylabel("y", fontsize=12)
	ax.set_title(f"L1 Lasso with outlier\n$\\beta_1$ = {beta1:.3f}, $\\beta_0$ = {beta0:.2f}", fontsize=14)
	plt.tight_layout()
	plt.savefig(f"../images/lasso.svg", bbox_inches=0, pad_inches=0)
	plt.show()


def ridge():
	X, y = df.drop('y', axis=1), df['y']
	y = y.copy()
	y.iloc[-1] = 100 # add outlier
	lm = Ridge(alpha=300)
	lm.fit(X, y)
	beta0 = lm.intercept_
	beta1 = lm.coef_[0]

	fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
	ax.scatter(X, y, s=95)
	ax.plot(df['x'], df['x']*beta1 + beta0, 's-', c='orange', markersize=6, alpha=.7)
	ax.set_xlabel("x", fontsize=12, labelpad=-5)
	ax.set_ylabel("y", fontsize=12, labelpad=-5)
	ax.set_xlabel("x", fontsize=12)
	ax.set_ylabel("y", fontsize=12)
	ax.set_title(f"L2 Ridge with outlier\n$\\beta_1$ = {beta1:.3f}, $\\beta_0$ = {beta0:.2f}", fontsize=14)
	plt.tight_layout()
	plt.savefig(f"../images/ridge.svg", bbox_inches=0, pad_inches=0)
	plt.show()


def ols2D_outlier():
	X, y = df.drop('y', axis=1), df['y']
	y = y.copy()
	y.iloc[-1] = 100 # add outlier
	olsfit(X,y)
	beta0 = np.linspace(-20, 0, num=150)
	beta1 = np.linspace(0, 10, num=100)
	Z = np.zeros(shape=(len(beta0), len(beta1)))
	min_loss = (99999, 0, 0)
	for i, b0 in enumerate(beta0):
		for j, b1 in enumerate(beta1):
			l = loss(b0, b1, df['x'].values, y.values)
			print(b0, b1, '->', l)
			Z[i, j] = l
			if l < min_loss[0]:
				min_loss = (l, b0, b1)

	print("min loss", min_loss)
	fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
	ax.set_xlabel("$\\beta_0$")
	ax.set_ylabel("$\\beta_1$")
	B0, B1 = np.meshgrid(beta0, beta1)
	ax.contour(B0, B1, Z.T, levels=30, linewidths=1.0, cmap='coolwarm')
	ax.scatter([opt_beta0],[opt_beta1], s=90, c='k')
	plt.tight_layout()
	plt.savefig(f"../images/ols_loss_2D_outlier.svg", bbox_inches=0, pad_inches=0)
	plt.show()


df = pd.read_csv("ols.csv")
X, y = df.drop('y', axis=1), df['y']

ols()
ols_outlier()
ols2D()
ols2D_outlier()
ols3D()
lasso()
ridge()
