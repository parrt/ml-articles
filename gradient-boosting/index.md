# How to understand gradient boosting

[Gradient boosting machines](https://en.wikipedia.org/wiki/Gradient_boosting) (GBMs) are currently very popular and so it's a good idea for machine learning practitioners to understand how GBMs work. The problem is that understanding all of the mathematical machinery is tricky and, unfortunately, these details are needed to tune the hyper parameters. (Tuning the hyper parameters is needed to getting a decent GBM model.) My purpose here is ...

Those with very strong mathematical backgrounds can start directly with the delicious 1999 paper by Friedman called [Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf).  To get the practical and implementation perspective, though, check out Gorman's excellent blog [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/) and Grover's [Gradient Boosting from scratch](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d) (written by a data science graduate students at the University of San Francisco).

As Gorman points out, "*This is the part that gets butchered by a lot of gradient boosting explanations*."

## The intuition behind boosting

The intuition behind GBM is straightforward. Rather than create a single, powerful model that learns to predict a target value $y$ given a feature vector $\vec x$, we start with a crappy model, $f_0(\vec x)$, that predicts an initial approximation of $y$ and then gradually nudge it towards the known target value $y$ by adding one or more tweaks to it.  Mathematicians call this "additive modeling" and electrical engineers call it "functional decomposition" (insert terrifying flashback to Fourier analysis here.) In the end, the model predicts $y$ using $\hat y = f_0(\vec x) + \Delta_1 + \Delta_2 + ...  + \Delta_M$. The $\Delta_i$ are the tweaks added to $f_0(\vec x)$ and are, technically, functions of the feature vector as well, $\Delta_i(\vec x)$, but I've left them off for the moment. The goal, of course, is to get $\hat y$ as close as possible to $y$ so we can extend the number of tweaks, $M$, to nudge closer and closer to $y$.  

This iterative refinement is a basic version of [boosting](https://en.wikipedia.org/wiki/Boosting_\(meta-algorithm\)). We can think of it as a golfer initially whacking a golf ball towards the hole at $y$ but only getting as far as $f_0(\vec x)$. The golfer[clipart] then repeatedly taps the ball more softly, working the ball towards the hole in stages to improve accuracy. 

\sidenote[clipart]{Golfer clipart from http://etc.usf.edu/clipart/}

<img src="images/golf-dir-vector.png" width="80%">

<pyeval label="examples" output="df" hide=true>
import pandas as pd
df = pd.DataFrame(data={"Bedrooms":[1,2,1], "SqFeet":[700,950,800]})
df["Rent"] = pd.Series([1000,1500,1200])
</pyeval>

$\hat y = f_0(\vec x) + \Delta_1(\vec x) + \Delta_2(\vec x) + ...  + \Delta_M(\vec x)$

That's easy enough, so what's the problem? How do we know this procedure is correct and terminates? Why do they call it gradient boosting?

we add together the results of multiple weak learners





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

## Resources

Vincent and Bengio [http://www-labs.iro.umontreal.ca/~vincentp/Publications/kmp_mlj.pdf](http://www-labs.iro.umontreal.ca/~vincentp/Publications/kmp_mlj.pdf)
