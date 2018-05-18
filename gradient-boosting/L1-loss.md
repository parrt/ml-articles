# Gradient boosting: Heading in the right direction

\author{[Terence Parr](http://parrt.cs.usfca.edu) and [Jeremy Howard](http://www.fast.ai/about/#jeremy)}

In our previous article, <a href="L2-loss.html">Gradient boosting: Distance to target</a>, our model took steps towards the target $\vec y$ based upon the residual vector, $\vec y-F_{m-1}(X)$, which includes the magnitude not just the direction of $\vec y$ from our the previous composite model's prediction, $F_{m-1}(X)$. The residual vector makes composite models converge rapidly towards $\vec y$.  The negative, of course, is that using the magnitude makes the composite model chase outliers.   This occurs because mean computations are easily skewed by outliers and our regression tree stubs yield predictions using the mean of all target values in a leaf.  For noisy target variables, it makes more sense to step merely in the *direction* of $\vec y$ from $F_{m-1}$ rather than the magnitude and direction. 

This brings us to the second commonly-used vector with gradient boosting, which we can call the *sign vector*: $sign(y_i-F_{m-1}(\vec x_i))$. The sign vector elements are either -1, 0, or +1, one value for each observation $\vec x_i$.   No matter how distant the true target is from our current prediction, the vector used to take steps towards the target is just direction info without the magnitude.  This sign vector is a bit weird, though, because the vector elements are limited to -1, 0, or +1.  In two dimensions, such vectors can only point at multiples of 45 degrees (0, 45, 90, 135, ...) and so the sign vector will rarely point directly at the target. Just to be clear, the sign vector is not the unit vector in the direction of $y_i-F_{m-1}(\vec x_i)$.

If there are outliers in the target variable that we cannot remove, using just the direction is better than both direction and magnitude. We'll show in <a href="descent.html">Gradient boosting performs gradient descent</a> that using $sign(y-F_{m-1}(\vec x_i))$ as our step vector leads to a solution that optimizes the model according to the mean absolute value (MAE) or $L_1$  *loss function*: $\sum_{i=1}^{N} |y_i - F_M(\vec x_i)|$ for $N$ observations. 

Optimizing the MAE means we should start with the median, not the mean, as our initial model, $f_0$, since the median of $y$ minimizes the $L_1$ loss. (The median is the best single-value approximation of $L_1$ loss.)  Other than that, let's assume that the same recurrence relations will work for finding composite models based upon the sign of the difference, as we used for the residual vector version in the last article:

\latex{{
\begin{eqnarray*}
F_0(\vec x) &=& f_0(\vec x)\\
F_m(\vec x) &=& F_{m-1}(\vec x) + \eta \Delta_m(\vec x)\\
\end{eqnarray*}
}}

Let's assume $\eta = 1$ so that it drops out of the equation to simplify our discussion, but keep in mind that it's an important hyper-parameter you need to set in practice.  Recall that $F_m(\vec x)$ yields a predicted value, $y_i$, but $F_m(X)$ yields a predicted target vector, $\vec y$, one value for each $\vec x_i$ feature row-vector in matrix $X$. 

Here is the rental data again along with the initial $F_0$ model, the first residual, and the first sign vector:

\latex{{
{\small
\begin{tabular}[t]{rrrrr}
{\bf sqfeet} & {\bf rent} & $F_0$ & $\vec y$-$F_0$ & $sign(\vec y$-$F_0)$ \\
\hline
700 & 1125 & 1150 & -25 & -1\\
750 & 1150 & 1150 & 0 & 0\\
800 & 1135 & 1150 & -15 & -1\\
900 & 1300 & 1150 & 150 & 1\\
950 & 1350 & 1150 & 200 & 1\\
\end{tabular}
}
}}


<pyeval label="examples" hide=true>
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error, mean_absolute_error
#rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 

bookcolors = {'crimson': '#a50026', 'red': '#d73027', 'redorange': '#f46d43',
              'orange': '#fdae61', 'yellow': '#fee090', 'sky': '#e0f3f8', 
              'babyblue': '#abd9e9', 'lightblue': '#74add1', 'blue': '#4575b4',
              'purple': '#313695'}

def draw_vector(ax, x, y, dx, dy, yrange, c=bookcolors['red']):
    ax.plot([x,x+dx], [y,y+dy], c=c, linewidth=.8)
    ay = y+dy
    yrange *= 0.03
    ad = -yrange if dy>=0 else yrange
    ax.plot([x+dx-4,x+dx], [ay+ad,ay], c=c, linewidth=.8)
    ax.plot([x+dx,x+dx+4], [ay,ay+ad], c=c, linewidth=.8)
    
def data():
    df = pd.DataFrame(data={"sqfeet":[700,950,800,900,750]})
    df["rent"] = pd.Series([1125,1350,1135,1300,1150])
    df = df.sort_values('sqfeet')
    return df

class Stub:
    def __init__(self, X, residual, split):
        """
        We train on the residual or the sign vector but only to get
        the regions in the leaves with y_i. Then we grab mean/median
        of residual, y_i - F_{m-1}, in that region (L2/L1).
        """
        self.X, self.residual, self.split = X, residual, split
        self.left = self.residual[self.X<self.split]
        self.right = self.residual[self.X>=self.split]
        
    def l2predict(self,x):
        lmean = np.mean(self.left)
        rmean = np.mean(self.right)
        return lmean if x < self.split else rmean
        
    def l1predict(self,x):
        lmed = np.median(self.left)
        rmed = np.median(self.right)
        return lmed if x < self.split else rmed
    
class GBM:
    def __init__(self, f0, stubs, eta):
        self.f0, self.stubs, self.eta = f0, stubs, eta
        
    def l1predict(self, x):
        delta = 0.0
        for t in self.stubs:
            delta += eta * t.l1predict(x)
        return self.f0 + delta

def plot_composite(ax, gbm, stage, legend=True):
    line1, = ax.plot(df.sqfeet,df.rent, 'o')

    sqfeet_range = np.arange(700-10,950+10,.2)
    y_pred = []
    for x in sqfeet_range:
        delta = 0.0
        for t in gbm.stubs[0:stage]:
            delta += eta * t.l1predict(x)
        y_pred.append( f0 + delta )
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
				  		
df = data()

def boost(df, xcol, ycol, splits, eta, stages):
    """
    Update df to have direction_i, delta_i, F_i.
    Return MSE, MAE
    """
    f0 = df[ycol].median()
    df['F0'] = f0

    stubs = []
    for s in range(1,M+1):
        df[f'res{s}'] = df[ycol] - df[f'F{s-1}']
        df[f'sign{s}'] = np.sign(df[f'res{s}'])
        t = Stub(df.sqfeet, df[f'res{s}'], splits[s])
        stubs.append(t)
        df[f'delta{s}'] = [t.l1predict(x) for x in df[xcol]]
        df[f'F{s}'] = df[f'F{s-1}'] + eta * df[f'delta{s}']

    return GBM(f0, stubs, eta)

M = 3
eta = 1
splits = [None,850, 925, 725] # manually pick them
gbm = boost(df, 'sqfeet', 'rent', splits, eta, M)

mse = [mean_squared_error(df.rent, df['F'+str(s)]) for s in range(M+1)]
mae = [mean_absolute_error(df.rent, df['F'+str(s)]) for s in range(M+1)]
</pyeval>

Visually, we can see that the first sign vector has components pointing in the right direction of the true target from $f_0(X)$:
 
<pyfig label=examples hide=true width="32%">
f0 = df.rent.median()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), sharey=True)

ax = axes
line1, = ax.plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4, label="$y$")
# fake a line to get smaller red dot
line2, = ax.plot([0,0],[0,0], c='r', markersize=4, label=r"$sign(y-f_0({\bf x}))$", linewidth=.8)
ax.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c='k')
ax.set_xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
ax.set_ylim(df.rent.min()-10, df.rent.max()+20)
ax.text(815, f0+10, r"$f_0({\bf x})$", fontsize=18)

ax.set_ylabel(r"Rent ($y$)", fontsize=14)
ax.set_xlabel(r"SqFeet (${\bf x}$)", fontsize=14)

# draw arrows
for x,y,yhat in zip(df.sqfeet,df.rent,df.F0):
    draw_vector(ax, x, yhat, 0, np.sign(y-yhat)*2, df.rent.max()-df.rent.min())
    
ax.legend(handles=[line2], fontsize=16,
          loc='upper left', 
          labelspacing=.1,
          handletextpad=.2,
          handlelength=.7,
          frameon=True)

plt.tight_layout()
plt.show()
</pyfig>

<img style="float:right;margin:0px 0px 0px 0;" src="images/stubs-mae-delta1.svg" width="27%">  As we did in the first article, our goal is to create a series of nudges, $\Delta_m$, that gradually shift our initial approximation, $f_0(X)$, towards the true target rent vector, $\vec y$. The first stub, $\Delta_1$, should be trained on $sign(\vec y - F_0(X))$ and let's choose a split point of 850 because that groups the $y$ values into two similar (low variance) groups, lower and higher values. Because we are dealing with $L_1$ absolute difference and not $L_2$ squared difference, stubs should predict the median, not the mean, of the observations in each leaf. That means $\Delta_1$ would predict  -1 for $\vec x$<850 and 1 for $\vec x$>=850.

Without the distance to the target as part of our $\Delta_m$ nudges, however, the composite model $F_m(X)$ would step towards rent target vector $\vec y$ very slowly, one dollar at a time. We need to weight the $\Delta_m$ predictions so that the algorithm takes bigger steps. Unfortunately, we can't use a single weight per stage, like $w_m \Delta_m(\vec x)$, because it might force the composite model predictions to oscillate around, but never reach, an accurate prediction. A global weight per stage is just too coarse to allow tight convergence to $\vec y$ for all $y_i$ simultaneously. For example, if we set $w_1=100$ to get the fourth and fifth data points from 1150 to 1250 in one step, that would also push the other points very far below their true targets:

<pyfig label=examples hide=true width="32%">
f0 = df.rent.median()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), sharey=True)


ax = axes
line1, = ax.plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4, label="$y$")
# fake a line to get smaller red dot
line2, = ax.plot([0,0],[0,0], c='r', markersize=4, label=r"$100 \Delta_1({\bf x})$", linewidth=.8)
ax.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c='k')
ax.set_xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
ax.set_ylim(df.rent.min()-100, df.rent.max()+20)
ax.text(815, f0+10, r"$f_0({\bf x})$", fontsize=18)

ax.arrow(830,1050, -20, 0, linewidth=.8, head_width=6, head_length=4)
ax.text(834, 1050-8, "Oops!", fontsize=14)

ax.set_ylabel(r"Rent ($y$)", fontsize=14)
ax.set_xlabel(r"SqFeet (${\bf x}$)", fontsize=14)

# draw arrows
for x,y,yhat in zip(df.sqfeet,df.rent,df.F0):
    draw_vector(ax, x, yhat, 0, (np.sign(y-yhat) if y-yhat>0 else -1)*100, df.rent.max()-df.rent.min())
    
ax.legend(handles=[line2], fontsize=16,
          loc='upper left', 
          labelspacing=.1,
          handletextpad=.2,
          handlelength=.7,
          frameon=True)

plt.tight_layout()
plt.show()
</pyfig>

When training weak models on the residual vector, each element of the observation residual vector gets its own "weight," $y_i - F_{m-1}(\vec x_i)$, tailored to its distance to the target. That gives a hint that we should use multiple weights here too. We  shouldn't compute a weight for each observation, though, because then we're right back to the residual vector solution from the first article. So, let's try computing a weight for each group of similar feature vectors.  

Since we're using regression tree stabs for our weak models, we can give each stub leaf its own weight, but how do we compute those weights? The graph of rent versus sqfeet clearly shows that we need a small value for the left stub leaf and a much larger value for the right stub leaf. The goal should be to have the next $F_1$ model step into the middle of the $y$ rent values in each group (leaf) of observations, which means jumping to the median $y$ in each group. The weight we need for each leaf is the difference vector that gets us from $f_0$ to the median $y$ for each leaf group, which is just the median of $\vec y - f_0$ residual but restricted to the observations in the leaf. The signed vector already has the direction so let's use the absolute value of the median, meaning that our weights in this case are always positive. The sign vector elements will tell us which way to step. Equivalently, we can think of this process as having each stub leaf predict the median residual of the observations in that leaf (in which case we don't need the absolute value). This is a bit weird and an incredibly subtle point, so let's emphasize it in a callout:

<aside title="The difference between MSE and MAE GBM trees">

GBMs that optimize  MSE ($L_2$ loss) and MAE ($L_1$ loss) both train regression trees, $\Delta_m$, on direction vectors.  The first difference is that MSE trains trees on residual vectors and MAE trains trees on sign vectors. The goal of training the tree is to group similar observations into leaf nodes in both cases.  Because they are training on different data, the trees will group the observations in the training data differently. The actual training of the weak model trees always computes split points by trying to minimize the squared difference of target values within the two groups, even in the MAE case.  The second difference is that an MSE tree leaf predicts the average of the residuals, $y_i - F_{m-1}(\vec x_i)$, values for all $i$ observations in that leaf whereas an MAE tree leaf predicts the median of the residual. Both are predicting residuals. Weird, right?

Just to drive this home, MSE trains on residual vectors and the leaves predict the average residual. MAE trains on sign vectors, but the leaves predict residuals like MSE, albeit the median, not the average residual. It's weird because models don't typically train on one space (sign values) and predict values in a different space (residuals). It's perhaps easier to think of MAE as training on sign vectors and predicting sign values (-1, 0, +1) but then weighting that prediction by the absolute value of the median of the residual.

</aside>

The residual vector $\vec y - F_0$ has values -25, 0, -15 for the left leaf, which has a median of 15, so that is the weight for the left leaf. The right leaf has residuals 150 and 200, so the weight of the right leaf is 175.  The dashed line in the following graph illustrates composite model $F_1$.

<pyfig label=examples hide=true width="40%">
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,3.8))

plot_composite(ax, gbm, 1, legend=False)

ax.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle=':', c='k')

for x,d0,delta in zip(df.sqfeet,df[f'F0'],df[f'F1']):
    draw_vector(ax, x, d0, 0, delta-d0, df.rent.max()-df.rent.min())

ax.text(700-10, 1340, r"$F_0 = f_0 + \Delta_1({\bf x}; {\bf w}_1)$", fontsize=18)
ax.text(700-10, 1315, r"${\bf w}_1 = [15, 175]$", fontsize=16)
ax.text(900, f0-20, r"$f_0({\bf x})$", fontsize=18)

ax.set_ylabel(r"Rent ($y$)", fontsize=14)
ax.set_xlabel(r"SqFeet (${\bf x}$)", fontsize=14)

plt.tight_layout()

plt.show()
</pyfig>

To indicate that the weak models each have a vector of weights, the graph use notation $\Delta_m(\vec x; \vec w_m)$.

Let's eyeball two weights we need for the first weak model 

, $\vec w_1=[20,100]$, $\vec w_2=[5,30]$, and $\vec w_3=[5,20]$, then we get the following table of partial results (with $\eta=1$):

<!--
<pyeval label="examples" hide=true>
#print(df)
# manually print table in python
# for small phone, make 2 tables
o = ""
for i in range(len(df)):
    o += " & ".join([f"{v:.2f}" for v in df.iloc[i,:][['sqfeet','rent','F0','dir1']]]) + "\\" + "\n"

for i in range(len(df)):
    o += " & ".join([f"{v:.2f}" for v in df.iloc[i,4:15]]) + r"\\"+ "\n"

o = o.replace(".00", "")
print(o)
</pyeval>
-->

\latex{{
{\small
\setlength{\tabcolsep}{0.5em}
\begin{tabular}[t]{rrrrrrrrrrr}
&&& $sign$ &&&& $sign$\vspace{-1mm}\\  
$\Delta_1$ & $F_1$ & $\vec y$-$F_1$ & $\vec y$-$F_1$ & $\Delta_2$ & $F_2$ & $\vec y$-$F_2$ & $\vec y$-$F_2$ & $\Delta_3$ & $F_3$\\
\hline
-15 & 1135 & -10 & -1 & -5 & 1130 & -5 & -1 & -5 & 1125\\
-15 & 1135 & 15 & 1 & -5 & 1130 & 20 & 1 & 2.50 & 1132.50\\
-15 & 1135 & 0 & 0 & -5 & 1130 & 5 & 1 & 2.50 & 1132.50\\
175 & 1325 & -25 & -1 & -5 & 1320 & -20 & -1 & 2.50 & 1322.50\\
175 & 1325 & 25 & 1 & 25 & 1350 & 0 & 0 & 2.50 & 1352.50\\
\end{tabular}
}
}}

(with stub split points chosen manually as 850, 925, 725):


The blue dots are the sign vector elements used to train $\Delta_m$ weak models, the dashed lines are the predictions made by $\Delta_m$, and the dotted line is the origin at 0. The $\Delta_m$  models have a hard time predicting the sign vectors, as you can see, because the sign vector elements are dissimilar in one leaf of each stub. Here are the stubs that generate those dashed lines:

<img src="images/stubs-mae.svg" width="90%">

Despite the imprecision of the weak models, the weighted $\Delta_m$ predictions nudge $\hat{\vec y}$ closer and closer to the true $\vec y$. The following figure illustrates how using two different weights for the same model, one per stub leaf, allows more control over steps to the target than a single weight allows.

<pyfig label=examples hide=true width="90%">
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 4), sharey=True)

def draw_stage_residual(ax, df, stage):
    for x,d0,delta in zip(df.sqfeet,df[f'F{stage-1}'],df[f'F{stage}']):
        draw_vector(ax, x, d0, 0, delta-d0, df.rent.max()-df.rent.min())

# PLOT 1

ax = axes[0]
f0 = df.rent.median()
line1, = ax.plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4, label="$y$")
draw_stage_residual(ax, df, stage=1)
# fake a line to get smaller red dot
line2, = ax.plot(700,1000,linewidth=.8, c='r', markersize=4)
ax.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c='k')
ax.text(815, f0+10, r"$f_0({\bf x})$", fontsize=18)
ax.set_xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
ax.set_ylim(df.rent.min()-10, df.rent.max()+10)
ax.set_xlabel(r"SqFeet", fontsize=14)
ax.set_ylabel(r"Rent", fontsize=14)
ax.legend(handles=[line2], fontsize=15,
          loc='upper left', 
          labelspacing=.1,
          handletextpad=.2,
          handlelength=.7,
          frameon=True,
          labels=["$\Delta_1({\\bf x})$"])

# PLOT 2

ax = axes[1]
line1, = ax.plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4, label="$y$")
draw_stage_residual(ax, df, stage=2)
# fake a line to get smaller red dot
line2, = ax.plot(700,1000,linewidth=.8, c='r', markersize=4)
ax.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c='k')
ax.text(815, f0+10, r"$f_0({\bf x})$", fontsize=18)
ax.set_xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
ax.set_ylim(df.rent.min()-10, df.rent.max()+10)
ax.set_xlabel(r"SqFeet", fontsize=14)
ax.legend(handles=[line2], fontsize=15,
          loc='upper left', 
          labelspacing=.1,
          handletextpad=.2,
          handlelength=.7,
          frameon=True,
          labels=["$\Delta_2({\\bf x})$"])

# PLOT 3

ax = axes[2]
line1, = ax.plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4, label="$y$")
draw_stage_residual(ax, df, stage=3)
# fake a line to get smaller red dot
line2, = ax.plot(700,1000,linewidth=.8, c='r', markersize=4)
ax.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c='k')
ax.text(815, f0+10, r"$f_0({\bf x})$", fontsize=18)
ax.set_xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
ax.set_ylim(df.rent.min()-10, df.rent.max()+10)
ax.set_xlabel(r"SqFeet", fontsize=14)
ax.legend(handles=[line2], fontsize=15,
          loc='upper left', 
          labelspacing=.1,
          handletextpad=.2,
          handlelength=.7,
          frameon=True,
          labels=["$\Delta_3({\\bf x})$"])

plt.tight_layout()
plt.show()
</pyfig>

We've manually chosen these weights as nice round numbers so we can do the computations more easily.  In practice, a boosting algorithm must compute the optimal weights of a complicated expression you can find on page 6 of <a href="https://statweb.stanford.edu/~jhf/ftp/trebst.pdf">Friedman's paper</a>. As he points out, however, the sign vector case simplifies to the weight of a leaf being just the median of the true $y_i$ values for the observations in that leaf. Weight $w_l$ for leaf $l$ is $median(y_i - F_{m-1}(\vec x_i))$ for all $\vec x_i$ in leaf $l$. (See Algorithm 3 *LAD_TreeBoost* on page 7 of Friedman's paper.) In a sense, this weight computation is acting like the residual vector. Instead of a separate "weight" for each observation, however, we use a separate weight for all similar observations (those in the same leaf). This has the effect that the prediction for similar observations takes steps proportional to the median distance from the weak model's prediction to the true target values. As the weak model predictions get better, the algorithm takes smaller steps to zero in on the best prediction. 

The following sequence of diagrams shows the composite model predictions versus true $y_i$ values.

<!-- composite model -->

<pyfig label=examples hide=true width="90%">
df = data()
gbm = boost(df, 'sqfeet', 'rent', splits, eta, M)
  
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11.1, 3.5))

plot_composite(axes[0], gbm, 1)
plot_composite(axes[1], gbm, 2)
plot_composite(axes[2], gbm, 3)

plt.tight_layout()
plt.show()
</pyfig>

As we add more weak models, the composite model gets more and more accurate.

How accurate is it? As in the previous article, we can use a loss function $L(\vec y,\hat{\vec y})$, that computes the cost of predicting $\hat{\vec y}$ instead of $\vec y$.  Because we are worried about outliers, it's appropriate to use the mean absolute error (MAE) as our loss function:

\[
L(\vec y,F_M(X)) = \frac{1}{N} \sum_{i=1}^{N} |y_i - F_M(\vec x_i)|
\]

(Using vector operations, that summation is the $L_1$ norm: $||\vec y-F_M(X)||_1$).

As you can see, using the sign vector instead of the residual factor does not change the mechanism behind gradient boosting. The only difference is that the sequence of composite model predictions, $F_m(X)$, sweep through different points in $N$-space and the algorithm converges on different points. ($\vec y$ and $F_m(X)$ are in $N$-space because there are $N$ observations.)

In the next and final article, [Gradient boosting performs gradient descent](descent.html) we show how gradient boosting leads to the minimization of different loss functions, depending on the direction vecto r used in the algorithm.

## New notes

We train on the residual or the sign vector but only to get the regions in the leaves with y_i. Then we grab mean/median of y_i - F_{m-1} in that region.
