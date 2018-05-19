# Gradient boosting performs gradient descent

\author{[Terence Parr](http://parrt.cs.usfca.edu) and [Jeremy Howard](http://www.fast.ai/about/#jeremy)}

\todo{can't proof that it converges or is correct... it's just a way to show that it can work with any differentiable loss function or equivalently direction vector}

\todo{can't show that each direction vector gets us to lower cost because we might have a bad approximation of the week model}

\todo{weights not needed in the L2 case because the tree is already picking those out as minimum L2}

So far we've looked at GBMs that use two different direction vectors, the residual vector (<a href="L2-loss.html">Gradient boosting: Distance to target</a>) and the sign vector (<a href="L1.loss.html">Gradient boosting: Heading in the right direction</a>). It's natural to ask whether there are other direction vectors we can use and what effect they have on the final $F_M(X)$ predictions.  Your intuition probably told you that nudging the intermediate $\hat{\vec y}$ prediction towards target $\vec y$ gradually improves model performance, but would $\hat{\vec y}$ ever stop if we kept increasing $M$? If so, where would it stop? Is there something special or interesting about the sequence of vectors that $\hat{\vec y}$ passes through? Moreover, training a model on observations ($\vec x$, $y$) is a matter of finding a function, $F(\vec x)$, that optimizes some cost or loss function indicating how well $F$ performs. (The procedure is to tweak model $F$'s parameters until we minimize the loss function.) What is a GBM optimizing and what is the relationship with the choice of direction vector?

To answer these questions, we're going to employ a mathematician's favorite trick: showing how our current problem is a flavor of another problem with a known solution and lots of useful results, such as proofs of correctness. Specifically, this article shows how gradient boosting machines perform an optimization technique from numerical methods called [gradient or steepest descent](https://en.wikipedia.org/wiki/Gradient_descent). We'll see that a GBM using residual vectors optimizes the mean squared error (MSE), the $L_2$ loss, between the true target $\vec y$ and the intermediate predictions, $\hat{\vec y} = F_m(X)$. A GBM that uses sign vectors optimizes the mean absolute error (MAE), the $L_1$ loss. (*Dramatic foreshadowing*.) The residual vector is the gradient (vector of partial derivatives) of the $L_2$ loss function and the sign vector is the gradient of the $L_1$ loss function. GBMs nudge $\hat{\vec y}$ at each stage in the direction of reduced loss.

Before jumping into the mathematics, let's make some observations about the behavior of our gradient boosting machines based on the examples from the previous two articles.
 
## foo

Let's revisit the rent data and look at the sequence of approximate prediction vectors from the first article that used residual vectors to nudge $\hat{\vec y}$ towards $\vec y$:

\latex{{
{\small
\setlength{\tabcolsep}{0.5em}
\begin{tabular}[t]{rrrrrrr}
&$\vec x~~~$ & $y~~~$ & \multicolumn{4}{c}{$~\hat y$}\vspace{-1mm}\\
&{\bf SqFeet} & {\bf Rent} & $F_0(\vec x)$ & $F_1(\vec x)$ & $F_2(\vec x)$ & $F_3(\vec x)$\\
\hline
& 700 & 1125 & 1212 & 1159 & 1143 & 1137 \\
& 750 & 1150 & 1212 & 1159 & 1143 & 1137 \\
& 800 & 1135 & 1212 & 1159 & 1143 & 1137 \\
& 900 & 1300 & 1212 & 1291 & 1314 & 1308 \\
& 950 & 1350 & 1212 & 1291 & 1314 & 1339\\
\hline
\vspace{-4mm}\\
{\bf MSE}&\multicolumn{2}{l}{$\frac{1}{N}\sum_i^N(y_i-F_m(\vec x_i))^2$} & 8826 & 1079.5 & 382.3 & 100.9\\
\end{tabular}
}
}}

The units of the mean squared error (MSE) is squared rent-dollars and the units of each $\hat y$ is rent-dollars, so $\hat{\vec y}$ is a vector of dollar values in $N$-space (here, $N=5$). That means that the $F_m$ predictions are vectors sweeping through $N$-space as we increase $m$. Notice that the MSE drops monotonically as we add more stages. 

We get similar behavior from the GBM in the second article that used sign vectors to nudge $\hat{\vec y}$:

\latex{{
{\small
\setlength{\tabcolsep}{0.5em}
\begin{tabular}[t]{rrrrrrr}
&$\vec x~~~$ & $y~~~$ & \multicolumn{4}{c}{$~\hat y$}\vspace{-1mm}\\
& {\small\bf SqFeet} & {\bf Rent} & $F_0(\vec x)$ & $F_1(\vec x)$ & $F_2(\vec x)$ & $F_3(\vec x)$\\
\hline
& 700 & 1125 & 1150 & 1136 & 1135 & 1130 \\
& 750 & 1150 & 1150 & 1136 & 1135 & 1150 \\
& 800 & 1135 & 1150 & 1136 & 1135 & 1150 \\
& 900 & 1300 & 1150 & 1250 & 1280 & 1295 \\
& 950 & 1350 & 1150 & 1250 & 1280 & 1295 \\
\hline
\vspace{-4mm}\\
{\bf MAE}&\multicolumn{2}{l}{$\frac{1}{N}\sum_i^N|y_i-F_m(\vec x_i)|$} & 78 & 35.3 & 23 & 16 \\
\end{tabular}
}
}}

Here, again, the $F_m$ predictions are vectors sweeping through dollars $N$-space and the mean absolute error (MAE) falls as we add more weak models. Though it falls more slowly than the MSE, the MAE is also monotonically decreasing. Note that we can drop the "mean" part to get the same the total squared or total absolute error by dropping the $\frac{1}{N}$.

Minimizing some function $f(x)$ will always give the same $x$ as minimizing the same function scaled by constant, $cf(x)$.

The error measures in both cases never go up because we are specifically choosing to nudge the approximation in the direction of lower cost, which we call the direction vector.  Another way to say "direction of lower cost" is "gradient of the loss function," which is where the "gradient" in gradient boosting comes from. Shortly, we'll show that it's also the gradient in gradient descent.
 
Using the mean by dividing by $N$ is the same as measuring the total squared air or total absolute error because the number of observations is a constant once we start training. It's a constant scalar factor that can be ignored because it doesn't change the shape of the error curve. 

o key insight is the direction vector versus plain residual signal

o we can minimize any differentiable loss function

o units are dollars of rent. (target space for each dim)


\[
\frac{\partial L(y_i, F(\vec x_i))}{\partial F(\vec x_i)}
\]

\[
\frac{\partial L(\vec y, F(X))}{\partial F(X)}
\]

\[
\frac{\partial L(\vec y, \hat{\vec y})}{\partial \hat{\vec y}}
\]

How do you optimize a function? set the derivative to zero and solve for x.
 
Friedman says '*... consider F(x) evaluated at each point $\vec x$ to be a "parameter"*'

Students ask the most natural but hard-to-answer questions:

<ol>
<li>Why is it called <i>gradient</i> boosting?		</li>
<li>What is &ldquo;function space&rdquo;?</li>
<li>Why is it called &ldquo;gradient descent in function space&rdquo;?</li>
</ol>

The third question is the hardest to explain. As Gorman points out, &ldquo;<i>This is the part that gets butchered by a lot of gradient boosting explanations</i>.&rdquo; (His blog post does a good job of explaining it, but I thought I would give my own perspective here.)


How good is that model? To answer that, we need a loss or cost function, $L(y,\hat y)$, that computes the cost of predicting $\hat y$ instead of $y$.  The squared error, $L(y,\hat y) = (y-\hat y)^2$ is the most common, but sometimes we care more about the absolute difference, $L(y,\hat y) = |y-\hat y|$. The loss across all observations is just the sum (or the average if you want to divide by $N$) of all the individual observation losses:

\[
L(\vec y, X) = \sum_{i=1}^{N} L(y_i, F_M(\vec x_i))
\]

That gives this either $L(\vec y, X) = \sum_{i=1}^{N} (y_i - F_M(\vec x_i))^2$ or $L(\vec y, X) = \sum_{i=1}^{N} |y_i - F_M(\vec x_i)|$.

Have you ever wondered why this technique is called *gradient* boosting? We're boosting gradients because our weak models learn sign vectors, and the other common term for "direction vector" is, drumroll please, *gradient*.  that leads us to optimization via gradient descent.


## The intuition behind gradient descent

what's a gradient?


## Boosting as gradient descent in prediction space

we would choose absolute value difference when we are worried about outliers.

gradient descent does parameter optimization normally but we are now doing function space optimization. Gradient descent can't be doing parameter optimization because the different kinds of models would have different parameters.

## General algorithm

\latex{{
\setlength{\algomargin}{3pt}
\SetAlCapSkip{-10pt}
\begin{algorithm}[]
\LinesNumbered
\SetAlgorithmName{Algorithm}{List of Algorithms}
\SetAlgoSkip{}
\SetInd{.5em}{.5em}
\TitleOfAlgo{{\em boost}($X$,$\vec y$,$M$,$\eta$)}
\KwResult{model $F_M$}
Let $F_0(X)$ be value $v$ minimizing $\sum_{i=1}^N L(y_i, v)$, loss across all observations\\
\For{$m$ = 1 \KwTo $M$}{
	Let $\delta_m = \frac{\partial L(\vec y,~ \vec f(X))}{\partial \vec f(X)}\big\rvert_{\vec f=F_{m-1}}$, gradient of $L$ w.r.t. $\vec f$ evaluated at $F_{m-1}$\\
	Train regression tree $\Delta_m$ on $\delta_m$, minimizing squared error\\
	\ForEach{leaf $l \in \Delta_m$}{
		Let $w$ be value minimizing $L(y_i, F_{m-1}(\vec x_i) + w)$ for obs. $i$ in leaf $l$\\
		Alter $l$ to predict $w$ (not  $mean(y_i)$)\\
	}
	$F_m(X) = F_{m-1}(X) + \eta \Delta_m(X)$\\
}
\Return{$F_M$}\\
\end{algorithm}
}}

only do for trees; then we are just tweaking the predictions in the leaf nodes; use mean of leaf elements already in L2 but tweak to use median in L1 case. Min $L(y_i, F_{m-1}(\vec x_i) + w)$ for $\vec x_i$ in leaf $l$. L2 case, we have $(y_i - F_{m-1}(\vec x_i) - w)^2$ and $y_i - F_{m-1}(\vec x_i)$ is just the residual so $w$ as mean minimizes that. That residual is what we are training on and as we discussed before, the initial value should be either the mean or the median because that minimizes those cost functions.

$\tilde{\vec y} = - \nabla_{F_{m-1}(X)} L(\vec y, F_{m-1}(X))$

= $ \frac{\partial}{\partial F_{m-1}(X)} L(\vec y, F_{m-1}(X))$

$\nabla_{\hat{\vec y}} L(\vec y,\hat{\vec y})$

Let $F_0(X)$ be value $v$ that minimizes loss across all observations: $\sum_{i=1}^N L(y_i, v)$\\
**for**  $m$ = 1 **to**  $M$:\\



## Common points of confusion

picking weights is also an optimization problem.

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

\todo{ To get really fancy, we can even add momentum <a href="https://arxiv.org/abs/1803.02042">Accelerated Gradient Boosting</a> can mention momentum instead of or in addition to leaf weights.  see accelerated gradient boosting paper recently.}

## Summary

foo

<aside title="Q. Why is it called >gradient< boosting?">

**A**. We're boosting gradients because the weak models learn direction vectors, and the other common term for "direction vector" is, drumroll please, *gradient*. Yup, simple as that.
</aside>

## Notation translation to Friedman's paper

foo

## Resources

Vincent and Bengio [http://www-labs.iro.umontreal.ca/~vincentp/Publications/kmp_mlj.pdf](http://www-labs.iro.umontreal.ca/~vincentp/Publications/kmp_mlj.pdf)

# Junk drawer

The $w_m$ weights are computed by finding the weight that optimizes the mean squared error of the true target $\vec y$ and the proposed $F_m(X)$ model weighted by $w_m$.

The weights, $w_m$, don't appear in the model equations in these graphs because we used weight $w_m = 1$ for simplicity. But, a GBM implementation would choose the optimal weight $w_m$ so as to minimize the squared difference between $\vec y$ and the prediction of the new composite model $F_m(X)$ that used $w_m$. So, first we choose the new weak model $\Delta_m$ then try out different weights, looking for the one that makes best use of this new model.  See page 8 of <a href="https://statweb.stanford.edu/~jhf/ftp/trebst.pdf">Friedman's paper</a> for the form of the equation (Friedman's equation 12) that, when minimized, yields the appropriate weight for stage $m$. In our notation, the minimization is:

\[
w_m = argmin_w \sum_{i=1}^{N} (y_i - (F_{m-1}(\vec x_i) + w \Delta_m(\vec x_i)))^2
\]

In English, $argmin_w$ just says to find the $w$ such that the value of the summation to the right is minimized. The summation says to compute the loss across all observations between what we want, $y_i$, and what the model would give us if we used weight $w$: $F_{m-1}(\vec x_i) + w \Delta_m(\vec x_i)$.

foo

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
