import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np
import collections

from sklearn import tree
from sklearn.externals.six import StringIO  
import pydotplus # must install
from IPython.display import Image  
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error, mean_absolute_error
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 

bookcolors = {'crimson': '#a50026', 'red': '#d73027', 'redorange': '#f46d43',
              'orange': '#fdae61', 'yellow': '#fee090', 'sky': '#e0f3f8', 
              'babyblue': '#abd9e9', 'lightblue': '#74add1', 'blue': '#4575b4',
              'purple': '#313695'}

def data():
#     df = pd.DataFrame(data={"sqfeet":[700, 750, 800, 850, 900,950,1000]})
#     df["rent"] = pd.Series([1160, 1160, 1175, 1200, 1280,1310,2000])
    df = pd.DataFrame(data={"sqfeet":[750, 800, 850, 900,950]})
    df["rent"] = pd.Series([1160, 1175, 1200, 1250,2000])
    df = df.sort_values('sqfeet')
    return df


def get_leaf_samples(t):
    samples = collections.defaultdict(list)
    dec_paths = t.decision_path(df.sqfeet.values.reshape(-1, 1))
    for d, dec in enumerate(dec_paths):
        for i in range(t.tree_.node_count):
            if dec.toarray()[0][i]  == 1:
                samples[i].append(d)
    return samples


class L2Stump:
    def __init__(self, X, residual):
        """
        We train on the (X,residual) but only to get
        the regions in the leaves with y_i. Then we grab mean
        of residual, y_i - F_{m-1}, in that region.
        Split determined using DecisionTreeRegressor to mirror L1Stump.
        """
        self.X, self.residual = X, residual
        self.tree_ = tree.DecisionTreeRegressor(max_depth=1)
        self.tree_.fit(X.values.reshape(-1, 1), residual)
        self.split = self.tree_.tree_.threshold[0]
        self.left = self.residual[self.X<self.split]
        self.right = self.residual[self.X>=self.split]
        
    def predict(self,x):
        if isinstance(x,np.ndarray) or isinstance(x,pd.Series):
            return np.array([self.predict(xi) for xi in x])
        lmean = np.mean(self.left)
        rmean = np.mean(self.right)
        return lmean if x < self.split else rmean
        
    
class L1Stump:
    def __init__(self, X, signs, residual):
        """
        We train on the (X,sign vector) but only to get
        the regions in the leaves with y_i. Then we grab median
        of residual, y_i - F_{m-1}, in that region.
        Split determined using DecisionTreeRegressor.
        """
        self.X, self.signs, self.residual = X, signs, residual
        self.tree_ = tree.DecisionTreeRegressor(max_depth=1)
        self.tree_.fit(X.values.reshape(-1, 1), signs)
        self.split = self.tree_.tree_.threshold[0]
        self.left = self.residual[self.X<self.split]
        self.right = self.residual[self.X>=self.split]
        
    def predict(self,x):
        if isinstance(x,np.ndarray) or isinstance(x,pd.Series):
            return np.array([self.predict(xi) for xi in x])
        lmed = np.median(self.left)
        rmed = np.median(self.right)
        return lmed if x < self.split else rmed

    
class GBM:
    def __init__(self, f0, stumps, eta):
        self.f0, self.stumps, self.eta = f0, stumps, eta
        
    def predict(self, x):
        delta = 0.0
        for t in self.stumps:
            delta += self.eta * t.predict(x)
        return self.f0 + delta
    
    def splits(self):
        s = []
        for t in self.stumps:
            s.append(t.split)
        return s


def l2boost(df, ycol, eta, M):
    f0 = df[ycol].mean()
    df['F0'] = f0

    stumps = []
    for s in range(1,M+1):
        df[f'res{s}'] = df[ycol] - df[f'F{s-1}']
        t = L2Stump(df.sqfeet, df[f'res{s}'])
        stumps.append(t)
        df[f'delta{s}'] = t.predict(df.sqfeet)
        df[f'F{s}'] = df[f'F{s-1}'] + eta * df[f'delta{s}']

    return GBM(f0, stumps, eta)


def l1boost(df, ycol, eta, M):
    f0 = df[ycol].median()
    df['F0'] = f0

    stumps = []
    for s in range(1,M+1):
        df[f'res{s}'] = df[ycol] - df[f'F{s-1}']
        df[f'sign{s}'] = np.sign(df[f'res{s}'])
        t = L1Stump(df.sqfeet, df[f'sign{s}'], df[f'res{s}'])
        stumps.append(t)
        df[f'delta{s}'] = t.predict(df.sqfeet)
        df[f'F{s}'] = df[f'F{s-1}'] + eta * df[f'delta{s}']

    return GBM(f0, stumps, eta)


def draw_vector(ax, x, y, dx, dy, yrange):
    ax.plot([x,x+dx], [y,y+dy], c='r', linewidth=.8)
    ay = y+dy
    yrange *= 0.03
    ad = -yrange if dy>=0 else yrange
    ax.plot([x+dx-4,x+dx], [ay+ad,ay], c='r', linewidth=.8)
    ax.plot([x+dx,x+dx+4], [ay,ay+ad], c='r', linewidth=.8)
    

def plot_composite(ax, df, gbm, stage, eta=1.0, legend=True):
    line1, = ax.plot(df.sqfeet,df.rent, 'o')

    sqfeet_range = np.arange(700-10,950+10,.2)
    y_pred = []
    for x in sqfeet_range:
        y_pred.append( gbm.predict(x) )
    line2, = ax.plot(sqfeet_range, y_pred, linewidth=.8, linestyle='--', c='k')
    labs = ['\Delta_1', '\Delta_2', '\Delta_3']
    if stage==1:
        label = r"$f_0 + \eta \Delta_1$"
    else:
        label=r"$f_0 + \eta("+'+'.join(labs[:stage])+")$"

    if legend:
        ax.legend(handles=[line2], fontsize=16,
                  loc='center left', 
                  labelspacing=.1,
                  handletextpad=.2,
                  handlelength=.7,
                  frameon=True,
                  labels=[label])
        
        
