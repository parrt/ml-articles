# Gradient boosting performs gradient descent

\author{[Terence Parr](http://parrt.cs.usfca.edu) and [Jeremy Howard](http://www.fast.ai/about/#jeremy)}

\preabstract{
(We teach in University of San Francisco's [MS in Data Science program](https://www.usfca.edu/arts-sciences/graduate-programs/data-science) and have other nefarious projects underway. You might know Terence as the creator of the [ANTLR parser generator](http://www.antlr.org). Jeremy is a founding researcher at [fast.ai](http://fast.ai), a research institute dedicated to making deep learning more accessible.)
}

Students ask the most natural but hard-to-answer questions:

<ol>
<li>Why is it called <i>gradient</i> boosting?		</li>
<li>What is &ldquo;function space&rdquo;?</li>
<li>Why is it called &ldquo;gradient descent in function space&rdquo;?</li>
</ol>

The third question is the hardest to explain. As Gorman points out, &ldquo;<i>This is the part that gets butchered by a lot of gradient boosting explanations</i>.&rdquo; (His blog post does a good job of explaining it, but I thought I would give my own perspective here.)



## The intuition behind gradient descent

what's a gradient?


## Boosting as gradient descent in prediction space

we would choose absolute value difference when we are worried about outliers.

gradient descent does parameter optimization normally but we are now doing function space optimization. Gradient descent can't be doing parameter optimization because the different kinds of models would have different parameters.

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
