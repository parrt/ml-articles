# How to explain gradient boosting

[Gradient boosting machines](https://en.wikipedia.org/wiki/Gradient_boosting) (GBMs) are currently very popular and so it's a good idea for machine learning practitioners to understand how GBMs work. The problem is that understanding all of the mathematical machinery is tricky and, unfortunately, these details are needed to tune the hyper parameters. (Tuning the hyper parameters is needed to getting a decent GBM model unlike, say, Random Forests.)  

To get started, those with very strong mathematical backgrounds can go directly to the super-tasty 1999 paper by Jerome Friedman called [Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf).  To get the practical and implementation perspective, though, I recommend Ben Gorman's excellent blog [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/) and Prince Grover's [Gradient Boosting from scratch](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d) (written by a [data science](https://www.usfca.edu/arts-sciences/graduate-programs/data-science) graduate student at the University of San Francisco).

Figuring out the details enough to effectively use GBM takes a bit of time but clicks after a few days of reading. Explaining it to somebody else is vastly more difficult.  Students ask the most natural but hard-to-answer questions:

<ol>
	<li>Why is it called *gradient* boosting?		
	<li>What is "function space"?
	<li>Why is it called "gradient descent in function space"?
</ol>

The third question is the hardest to explain. As Gorman points out, "*This is the part that gets butchered by a lot of gradient boosting explanations*." (His blog post does a good job of explaining it, but I thought I would give my own perspective here.)

My goal in this article is not to explain how GBM works *per se*, but rather how to explain the tricky bits to students or other programmers.  I will assume readers are programmers, have a basic grasp of boosting already, can remember high school algebra, and have some minimal understanding of function derivatives (as one might find in the first semester of calculus).[review]

\sidenote[review]{To brush up on your vectors and derivatives, you can check out [The Matrix Calculus You Need For Deep Learning](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html) written with [Jeremy Howard](http://www.fast.ai/about/#jeremy).}

GBM is based upon the notion of boosting so let's start by getting a feel for how assembling a bunch of weak learners can lead to a strong model.

## A review of boosted regression

[Boosting](https://en.wikipedia.org/wiki/Boosting_\(meta-algorithm\)) is a loosely-defined strategy for combining the efforts of multiple weak models into a single, strong meta-model or composite model.   Mathematicians represent both the weak and composite models as functions. Given feature vector $\vec x$ and target value $y$, we can express a meta-model that predicts $\hat y$ as the addition of $M$ weak models, $f_i(\vec x)$:

\[
\hat y = F_M(\vec x) = f_1(\vec x) + ...  + f_M(\vec x) = \sum_{i=1}^M f_i(\vec x)
\]

The form of $f_i(\vec x)$ functions we use for the weak models is often the same and the terms of $F_M(\vec x)$ differ only by a scale factor so it's convenient to add a scale term to the equation:

\[
\hat y = F_M(\vec x) = w_1 f_1(\vec x) + ...  + w_M f_M(\vec x) = \sum_{i=1}^M w_i f_i(\vec x)
\]

Mathematicians call this "additive modeling" and electrical engineers use it for decomposing signals into a collection of sine waves representing the frequency components (insert terrifying flashback to Fourier analysis here.) 

It's often the case that an additive model can build the individual $f_i(\vec x)$ terms independently and in parallel, but that's not the case for boosting. Boosting constructs and adds weak models in a stage-wise fashion, one after the other, each one chosen to improve the overall model performance. The boosting strategy is greedy in the sense that choosing $f_i(\vec x)$ and $w_i$ never alters previous weights and functions. We can stop adding weak models when $F_M(\vec x)$'s performance is good enough or when $w_i f_i(\vec x)$ doesn't add anything.  Because greedy strategies choose one component model at a time, you will often see boosting models expressed using this equivalent, recursive formulation:

\[
F_i(\vec x) = F_{i-1} + w_i f_i(\vec x)
\]

Boosting itself does not specify how to choose the weights, the weak learners, or the number of models, $M$.   Boosting does not even specify the form of the $f_i(\vec x)$ models, which can be placeholders for linear regression models, regression trees, or any other model we want. 

Let's see if we can design a strategy for picking weak models to create our own boosting algorithm. 

## The intuition behind gradient boosting

To construct a boosted regression model, let's start by creating a crappy model, $f_0(\vec x)$, that predicts an initial approximation of $y$ given feature vector $\vec x$. Then, let's gradually nudge the overall $F_M(\vec x)$ model towards the known target value $y$ by adding one or more scaled tweaks, $w_i \Delta_i(\vec x)$:

\latex{{
\begin{eqnarray*}
\hat y & = & f_0(\vec x) + w_1 \Delta_1(\vec x) + w_2 \Delta_2(\vec x) + ...  + w_M \Delta_M(\vec x) \\
 & = & f_0(\vec x) + \sum_{i=1}^M w_i \Delta_i(\vec x)\\
 & = & F_M(\vec x)\\
\end{eqnarray*}
}}

Or, using a recurrence relation, let:

\latex{{
\begin{eqnarray*}
F_0(\vec x) &=& f_0(\vec x)\\
F_i(\vec x) &=& F_{i-1}(\vec x) + w_i \Delta_i(\vec x)\\
\end{eqnarray*}
}}

It might be helpful to think of this boosting approach as a golfer initially whacking a golf ball towards the hole at $y$ but only getting as far as $f_0(\vec x)$. The golfer[clipart] then repeatedly taps the ball more softly, working the ball towards the hole after reassessing at each stage.

\sidenote[clipart]{Golfer clipart from http://etc.usf.edu/clipart/}

<img src="images/golf-dir-vector.png" width="80%">

If we choose as weak models $\Delta_i (\vec x) = sign(y - \hat y) = sign(y - F_{i-1}(\vec x))$, then boosting moves the overall $F_i(\vec x)$ approximation closer and closer to $y$ because $\hat y$ moves in the direction of $y$.  If the initial whack, $f_0(\vec x)$, shoots past $y$, then $\Delta_1 (\vec x)$ would be -1 to indicate boosting needs to move in the negative direction. Each tweak is scaled by a $w_i$ to increase the jump size. (An overall learning rate variable is also typically used to speed up or slow down the overall approach of $\hat y$ to $y$.) As an example, let $f_0(\vec x)=70$, $w_i=10$, and $\Delta_i (\vec x)=1$, then $\hat y = F_3(\vec x) = 70 + 30 + 30 + 30 = 100$.

For real problems, the weak models can't learn the direction vectors perfectly and so $w_i \Delta_i (\vec x)$ gives a noisy jump towards target $y$. That's why the diagram simulates more than one jump, instead of just choosing $w_1=30$ and $\Delta_1 (\vec x)=1$ to get $\hat y = F_1(\vec x) = 70 + 30$.

We're not limited to training the $\Delta_i (\vec x)$ weak models on $sign(y - F_{i-1}(\vec x))$. We can train on any direction vector that points towards $y$ from $F_{i-1}(\vec x)$.  So, we could also train on the model error, $\Delta_i (\vec x) = y - F_{i-1}(\vec x)$, which would tell us not only the direction but also how far away we are from $y$.  We'd have to change how we pick the weights, but both $sign(y - \hat y)$ and $y - \hat y$ point us in the right direction.  If the golfer model chose $\Delta_1 (\vec x)=100-70=30$, then it could choose $w_1=1$ to get $\hat y = F_1(\vec x) = 70 + 30$. (Assuming $\Delta_1(\vec x)$ could reproduce direction vector 30 perfectly.)

We reached our first signpost because, without explaining anything further, we're ready to answer the first question posed at the beginning of this article.

<aside title="Q. Why is it called >gradient< boosting?">

**A**. We're boosting gradients because the weak models learn direction vectors, and the other common term for "direction vector" is, drumroll please, *gradient*. Yup, simple as that.
</aside>

There are several things to reinforce before moving on:

<ul>
	<li>The weak models learn direction **vectors** (gradients), not just magnitudes.
	<li>The initial model $f_0(\vec x)$ is trying to learn $y$ given $\vec x$, but the $\Delta_i (\vec x)$ tweaks are trying to learn gradients given $\vec x$.
	<li>All weak models, $f_0(\vec x)$ and $\Delta_i(\vec x)$, train on the original feature vector $\vec x$.
	<li>Two common gradient choices are $sign(y-F_{i-1}(\vec x))$ and $y-F_{i-1}(\vec x)$.
</ul>

This boosting approach makes sense and our intuition says that it should work.  It would be nice, however, to show that this boosting yields a correct model and that it converges towards a better model as we add weak learners. We'll get there, but first, let's look at a concrete example to see what gradient boosting on more than one data point (golf ball).
 
### Boosting example 

Imagine that we have square footage data on five apartments and their rent prices in dollars per month as our training data:

<pyeval label="examples" hide=true>
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np
#rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 

bookcolors = {'crimson': '#a50026', 'red': '#d73027', 'redorange': '#f46d43',
              'orange': '#fdae61', 'yellow': '#fee090', 'sky': '#e0f3f8', 
              'babyblue': '#abd9e9', 'lightblue': '#74add1', 'blue': '#4575b4',
              'purple': '#313695'}
# can't end with quote
</pyeval>

<pyeval label="examples" output="df" hide=true>
df = pd.DataFrame(data={"sqfeet":[700,950,800,900,750]})
df["rent"] = pd.Series([1050,1350,1100,1300,1200])
df = df.sort_values('sqfeet')
</pyeval>

where row $i$ is an observation with feature vector $\vec x_i$ and target value $y_i$. 

From this data, we'd like to build a GBM to predict rent price given square footage. Let's use the median of the rent prices as the initial model: $F_0(\vec x)$ = $f_0(\vec x)$ = 1200.

\latex{{
\begin{tabular}[t]{rrrr}
700 & 1050 & 1200 \\
\end{tabular}
}}

<pyeval label="examples" output="df" hide=true>
f0 = df.rent.median()
df['f0'] = f0
df['delta1'] = np.sign(df.rent - df.f0)
df['scaled1'] = df.delta1 * 50
df['delta2'] = np.sign(df.rent - (df.f0 + df.scaled1))
df['scaled2'] = df.delta2 * 50
df['delta3'] = np.sign(df.rent - (df.f0 + df.scaled1 + df.scaled2))
df['scaled3'] = df.delta3 * 50
df['delta4'] = np.sign(df.rent - (df.f0 + df.scaled1 + df.scaled2 + df.scaled3))
df = df.astype(int)
</pyeval>

<pyfig label=examples hide=true width="50%">
plt.plot(df.sqfeet,df.rent,'-o', linewidth=.8, markersize=3.5)
f0 = df.rent.median()
plt.plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c='k')
plt.xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
plt.text(df.sqfeet.min()+5, f0+15, r"$f_0({\bf x})$",
         fontsize=20)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
dir = np.sign(df.rent - f0) * 50
stage = 1
y_hat = np.array
for x,y_hat in zip(df.sqfeet,df.scaled1):
    if y_hat!=0:
        plt.arrow(x, f0, 0, y_hat, fc='r', ec='r',
                  linewidth=0.8,
                 head_width=4, head_length=8,
                  length_includes_head=True)

for x,y_hat in zip(df.sqfeet,df.scaled1+df.scaled2):
    if y_hat!=0:
        plt.arrow(x, f0, 0, y_hat, fc='r', ec='r',
                  linewidth=.8,
                 head_width=4, head_length=8,
                  length_includes_head=True)

for x,y_hat in zip(df.sqfeet,df.scaled1+df.scaled2+df.scaled3):
    if y_hat!=0:
        plt.arrow(x, f0, 0, y_hat, fc='r', ec='r',
                  linewidth=.8,
                 head_width=4, head_length=8,
                  length_includes_head=True)
plt.show()
</pyfig>

<pyfig label=examples hide=true width="80%">
f0 = df.rent.median()

red = bookcolors['redorange']
blue = bookcolors['blue']

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 2.5), sharey=True)
#plt.tight_layout()

#axes[0][0].plot(df.sqfeet,df.rent,'-o', linewidth=.8, markersize=4)

axes[0].plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4)
axes[0].plot([df.sqfeet.min()-10,df.sqfeet.max()+10], [f0,f0],
         linewidth=.8, linestyle='--', c=red)

# plt.xlim(df.sqfeet.min()-10,df.sqfeet.max()+10)
# plt.text(df.sqfeet.min()+5, f0+15, r"$f_0({\bf x})$",
#          fontsize=20)
axes[0].set
dir = np.sign(df.rent - f0) * 50
stage = 1
y_hat = np.array

axes[1].plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4)
axes[1].plot(df.sqfeet, f0+df.scaled1,
         linewidth=.8, linestyle='--', c=red)

axes[2].plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4)
axes[2].plot(df.sqfeet, f0+df.scaled1+df.scaled2,
         linewidth=.8, linestyle='--', c=red)

axes[3].plot(df.sqfeet,df.rent,'o', linewidth=.8, markersize=4)
axes[3].plot(df.sqfeet, f0+df.scaled1+df.scaled2+df.scaled3,
         linewidth=.8, linestyle='--', c=blue)
#axes[0].text(700, 1325, r"$f_0({\bf x})$")
axes[0].set_title(r"$F_0({\bf x}) = f_0({\bf x})$", fontsize=18)
axes[1].set_title(r"$F_1({\bf x})$", fontsize=18)
axes[2].set_title(r"$F_2({\bf x})$", fontsize=18)
axes[3].set_title(r"$F_3({\bf x})$", fontsize=18)
plt.savefig('/tmp/t.svg')
plt.show()
</pyfig>

$\hat y = f_0(\vec x) + \Delta_1(\vec x) + \Delta_2(\vec x) + ...  + \Delta_M(\vec x)$

That's easy enough, so what's the problem? How do we know this procedure is correct and terminates? Why do they call it gradient boosting?

we add together the results of multiple weak learners

How good is that model? To answer that, we need a loss or cost function, $L(y,\hat y)$, that computes the cost of predicting $\hat y$ instead of $y$.  The squared error, $L(y,\hat y) = (y-\hat y)^2$ is the most common, but sometimes we care more about the absolute difference, $L(y,\hat y) = |y-\hat y|$. The loss across all observations is just the sum (or the average if you want to divide by $n$) of all the individual observation losses:

\[
L(\vec y, X) = \sum_{i=1}^{n} L(y_i, F_M(\vec x_i))
\]

That gives this either $L(\vec y, X) = \sum_{i=1}^{n} (y_i - F_M(\vec x_i))^2$ or $L(\vec y, X) = \sum_{i=1}^{n} |y_i - F_M(\vec x_i)|$.


## The intuition behind gradient descent

what's a gradient?

Conceptually, gradient boosting is pretty easy to explain

Given $(X, \vec y)$ we want an $f(X)$ $f(X_i) = y_i$.

$\vec y = f(X)$

Treating a gradient boosting machine means an initial approximation to the target and then multiple trainings for models that map the x to a nudge.


typical use a gradient boosting is to use a single function that represents the model mapping predictors to target, such as  logistic regression or neural networks. In that case, we are tweaking the models to see how the resulting function output affects the cost function. With gradient boosting, we have multiple functions not just one. Of course when we train each model/function we must do some model-parameter fitting to learn, but that is a totally separate concept.  With gradient boosting, we are not tweaking the parameters of a single function but instead are asking how to tweak the output from a composite function. Each new tweak gets added into the composite function.

Game of golf. hit it once to get it close then nudge, nudge, nudge to get into hole.

$Cost(y, \hat y_0 + \Delta y_1 + \Delta y_2 + ...  + \Delta y_M)$.

$F_0(x) = \hat y$

$F_1(x) = \hat y + \Delta y_1$

$F_2(x) = \hat y + \Delta y_1 + \Delta y_2$

or, instead of nudging along we can try to jump from the initial approximation to the end:

$Cost(y, \hat y + \Delta y)$

The problem is that we can't make the Delta exactly and so we end up with, in both cases,

$Cost(y, \hat y + \hat{\Delta y}_1 + \hat{\Delta y}_2 + ...  + \hat{\Delta y}_M)$.



Want to minimize

$Cost(\vec y, f(X))$

where $f(X)$ is the set of predictions coming from our model $f$. To minimize that, we keep tweaking the parameters of $f$ until we get the minimum cost.  typically we expose those parameters, such as with linear regression model $y = \beta_0 + \beta_1 x$.

$Cost(\vec y, f(X;[\beta_0, \beta_1])$

or

$Cost(\vec y, \beta_0 + \beta_1 x)$

and then say to minimize that equation by shifting $[\beta_0, \beta_1]$ around.
 
$Cost(\vec y, \hat y)$

for some predictions from the model $\hat y$.

The typical iterative update procedure for gradient descent is:

$x_{t+1} = x_t - \eta \nabla f(x_t)$

to find $x$ that minimizes $f(x)$.   When using it to train a model, $x$ is the set of parameters for the model. In neural networks, the architecture is embodied by the $f$ function.

pikes peak metaphor: elevation function is model, x is location. shift x until elevation is minimized.  that is one model  called elevation. x is parameter.  with boosting, get a starting elevation (the summit?) then ask which direction we should take a step to head towards lower elevation at a specific x location.  hmm...maybe it's like we know the elevation at bottom, 6000' (vs 14110' at summit).  We ask, is current elevation 6000? No, need another step so drop elevation by 1 unit. this would be simulating one $\vec x$ and one $y$ target...too easy to move one person downhill. but it's like start at some elevation, now go down until bottom (cost is 0 when we reach bottom). duh. ok, extend so there are 20 kids on trip and all at different starting points. Now pretend they are all at average elevation.  Boosting tweaks not the x location of each kid, but sends a "go down or go up or stop moving signal".   The updated prediction of kid elevations is initial guess of average and then tweaks giving instructions to head up/down to reach same base camp at bottom. (some might have gotten lost and ended up below base camp). Hmm...actually, no. we send park rangers, one per kid, starting at basecamp looking to reach each kid. initial guess basecamp.  Then ask for ranger i, is kid i above/below/same. Tell ranger to move. Ranger elevation is initial + all tweaks. when tweak vector goes to all 0s, rangers have reached all kids. how does x location come into play?  training ranger to move elevation up/down might involve shifting in x/y plane to get single elevation value.

## Notation translation to Friedman's paper

foo

## Resources

Vincent and Bengio [http://www-labs.iro.umontreal.ca/~vincentp/Publications/kmp_mlj.pdf](http://www-labs.iro.umontreal.ca/~vincentp/Publications/kmp_mlj.pdf)

# Junk drawer

<!--
Let's perform some manual boosting using this diagram as a guide. Let the target $y$ be 100 yards distant from the tee and the initial approximation, $f_0(\vec x)$, be 70 yards: $F_0(\vec x) = f_0(\vec x) = 70$. The golfer still has $y-70 = 30$ yards to go and so we need to boost the current prediction in the positive direction. Let's use $\Delta_1 (\vec x) = 1$ to indicate a positive delta and, to avoid taking forever, let's scale the new weak learner by $w_1 = 10$: $F_1(\vec x) = 70 + 10$. The golfer is still short so we boost again with $\Delta_2 (\vec x) = 1$ and $w_2 = 10$: $F_2(\vec x) = 70 + 10 + 10$. Finally, we get $\hat y = F_3(\vec x) = 70 + 30 + 30 + 30 = 100$. We would not have to the same  weights, but that is convenient and works for this example. We could also use

 a constant amount

\latex{{
\begin{eqnarray*}
F_0 &=& f_0, ~~~~~~~~~~~~~~~~~~~~~y-\hat y = 30 \\
F_1 &=& F_0 + 10 \times 1\\
F_2 &=& F_1 + 10 \times 1\\
F_3 &=& F_2 + 10 \times 1\\
\end{eqnarray*}
}}
-->
