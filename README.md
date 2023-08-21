# ADO for CDN

We used the [ADOpy](https://github.com/adopy/adopy) to inspire our development of ADO for CDN. In this notebook we go over the mathematical foundation underlying ADO.

## Introduction to grid-based ADO algorithm

ADO works by first pre-computing the log-likelihood, entropy, and prior at step 0. Then the optimization takes place iteratively over three steps by updating the mutual information after a choice is made. This theoretical framework will be written in general for any given task design. In practice, a lot of computations can be written generally to work for any task design. The initial computation of the log-likelihood is based on each task and $SV$ defined by the model.


### Step 0: Pre-computation

In each task, or experiment, the participant makes a decision or a choice, $y$. This data is fit to a model defined by parameters, $\theta$. Finally, the design or choices presented to the participant is defined by the choice set space, $d$. In the code, these three values are defined first to subsequently compute the following distributions, etc. 


**1. Precompute the log-likelihood $log(p(y|\theta,d))$ for all values in $y$,$\theta$, and $d$:**


Given each possible combination ($y_i$,$\theta_j$,$d_k$), compute $SV_{reward}$ and $SV_{null}$ defined by the model. Use the difference between the two $\Delta SV = SV_{reward} - SV_{null}$ to get a probability, $p_l$, given a $\gamma_j$ (taken from $\theta_j$). Finally compare probability $p_l$ to choice $y_i$ and compute the $log(p(y_i|\theta_j,d_k))$. In Python you can compute `log_lik` using `choice` as $y_i$ `p_choose_reward` as $p_l$ by hand or using `bernoulli.logpmf` from `scipy.stats`:

```
    log_lik = (choice==1)*np.log(p_choose_reward) + (choice==0)*np.log(1-p_choose_reward)
    log_lik = bernoulli.logpmf(choice, p_choose_reward)
```



**2. Precompute the entropy $H(Y(d)|\theta) = -\sum_{y} p(y|\theta,d) log(p(y|\theta,d))$ for all values of $d$ and $\theta$:**

We use the log-likelihood (`log_lik`) to compute entropy as the second term in the summation is the `log_lik`, while the first term is the likelihood, so we exponentiate $p(y|\theta,d) = e^{log(p(y|\theta,d))}$. In the code this is computed using Numpy's function [`np.einsum()`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)

```
    ent = -1 * np.einsum('dpy,dpy->dp', np.exp(log_lik), log_lik)
```

Briefly, the function `np.einsum()` uses Einstein summation convention to sum over $y$. The first term is a string indicating what axis you intend to sum over. In this case `dpy` refers to design, parameters, output/response. Then the notation used `dpy,dpy-> dp` indicates summing along $y$, axis=$2$ from $(0,1,2)$.




**3. Initialize prior $p_t(\theta)$ for each discretized values of $\theta$:**

**NOTE:** the prior is used the first time around in the optimization phase as the posterior to compute the distributions.

If you have no priors then you can set this to a uniform distribution, normalized to $1.0$ as follows:

```
    log_prior = np.log(np.ones(n_p, dtype=np.float32) / n_p)
```

where `n_p` is the number of parameters ($\theta_j$). If you have a `prior` that you want to initialize, you can define an array with zeros, then set the parameter index you want to initialize to 1.0. Then convert this prior to a `log_prior` by trapping the prior into a `log_prior`. This is part of the code paraphrased here for illustration:


```python
import numpy as np
# here we think parameter index 12 is the one we want to initialize
pmid_idx = 12
n_p = 30 # this is number of parameters computed elsewhere
prior = np.zeros(n_p, dtype=np.float32)
prior[pmid_idx] = 1
noise_ratio = 1e-7
log_prior = np.log((1 - 2 * noise_ratio) * prior + noise_ratio)
print('log_prior is :\n {}'.format(log_prior))
```

    log_prior is :
     [-1.6118095e+01 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -1.6118095e+01 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -1.6118095e+01 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -5.9604645e-08 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -1.6118095e+01 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -1.6118095e+01 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -1.6118095e+01 -1.6118095e+01 -1.6118095e+01 -1.6118095e+01
     -1.6118095e+01 -1.6118095e+01]


### Step 1. Design Optimization

After pre-computing the above distributions, we begin the optimization phase. In this first step, we compute several different distributions to allow us to compute the mutual information which is maximized to find the new design.

**1. Compute the marginal likelihood $p(y|d) = \sum_{\theta} p(y|\theta,d)p_t(\theta)$ for all values of $y$ and $d$:**

In practice we compute the marginal log likelihood by taking the $\log$ -> $\log(p(y|d)) = \log(\sum_{\theta} p(y|\theta,d)p_t(\theta))$. This allows us to take advantage of the `logsumexp()` function from `scipy.special`:

```
    marg_log_lik = logsumexp(log_lik + log_post.reshape(1, -1, 1), axis=1)
```

From numpy.reshape: *One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.* The reshape makes `log_post` reshape from `(n_p,)` to `(1,n_p,1)` so it matches the dimensionbs of the `log_lik`

**NOTE:** We could have computed the marginal likelihood as : `marg_lik = np.einsum('dpy,dpy->dy', np.exp(log_lik), np.exp(log_post.reshape(1, -1, 1)))`


**2. Compute marginal entropy $H(Y(d)) = -\sum_{y}p(y|d)\log(p(y|d))$ for all values of $d$**:

```
    ent_marg = -1 * np.einsum('dy,dy->d', np.exp(marg_log_lik), marg_log_lik)
```
The marginal log likelihood, `marg_log_lik` is used to compute the marginal entropy, `ent_marg`. We use the function `np.einsum()` to sum over parameters `y`, or $y$, between marginal likelihood, $p(y|d)$, and marginal log likelihood, $\log(p(y|d))$. Note the string `'dy,dy->d'` is used to sum over `y`.




**3. Compute conditional entropy $H(Y(d)|\Theta) = \sum_{\theta}p_t(\theta)H(Y(d)|\theta)$ for all values of $d$:**

```
    ent_cond = np.einsum('p,dp->d', np.exp(log_post), ent)
```
We use the function `np.einsum()` to sum over parameters `p`, or $\theta$, between posterior, $p_t(\theta)$, and entropy, $H(Y(d)|\theta)$. Note the string `'p,dp->d'` is used to sum over `p`.




**4. Identify optimal design $d*$ to maximize mutual information $I(Y(d);\Theta) = H(Y(d)) - H(Y(d)|\Theta)$**

First we compute mutual information $I(Y(d);\Theta)$ as the difference of marginal entropy and conditional entropy: `mutual_info = ent_marg - ent_cond`. Next, we find the `cur_design` ($d*$) that maximizes this distributions by using `np.argmax()` along design dimension, $d$:

```
    # GET_DESIGN based on maximum Mutual information
    idx_design = np.argmax(mutual_info)
    cur_design = sets.choice.iloc[idx_design].to_dict()
```



### Step 2. Experimentation

**Use the `cur_design`, $d*$ to observe an outcome $y_{obs}(t)$**

In practice we implemented two options to observe an outcome, $y_{obs}(t)$. The outcome can be entered manually by a participant sitting in front of the computer or the observation can be simulated given a set of pre-defined true model parameters, `PARAM_TRUE`, $\theta_{TRUE}$. A snipet of the code is presented here to illustrate how these two functions work, given a `design`.



```python
import sys
from scipy.stats import bernoulli

# True parameter values to simulate responses, can select from prior distribution
PARAM_TRUE = {'kappa': 0.2, 'gamma': 1.0}

### Getting a response entered manually in ask_for_response(design) or simulated in get_simulated_response(design) ###

def ask_for_response(design):
    response = input('Please choose: >>>(0) {} in {} days<<< OR >>>(1) {} in {} days<<<'
                    .format(design['value_null'],design['time_null'],design['value_reward'],design['time_reward']))
    response = int(response)
    if response in [0,1]:
        pass
    else:
        print('ERROR, you selected {}, response needs to be 0 or 1'.format(response))
        print('Exiting now, try again')
        sys.exit()
    return response

def get_simulated_response(design):
    # Calculate the probability to choose a variable option
    tn, tr, vn, vr = (design['time_null'], design['time_reward'],design['value_null'], design['value_reward'])
    kappa, gamma = PARAM_TRUE['kappa'], PARAM_TRUE['gamma']
    
    SV_null = vn * (1. / (1 + kappa * tn))
    SV_reward = vr * (1. / (1 + kappa * tr))
    p_obs = 1. / (1 + np.exp(-gamma * (SV_reward - SV_null)))

    # Randomly sample a binary choice response from Bernoulli distribution
    return bernoulli.rvs(p_obs)

```

### Step 3. Bayesian Updating

**1. Compute (update) posterior $p(\theta|y_{obs}(t),d*) = \frac{p(y_{obs}(t)|\theta,d*)p_t(\theta)}{p(y_{obs}(t)|d*)}$ via Bayes rule for all values of $\theta$**

Given the `cur_design`, $d*$, and the observed outcome/response, $y_{obs}(t)$, we look for the point on the grid that is closest so we can compute `log_lik[i_d, :, i_y]`, $p(y_{obs}(t)|\theta,d*)$. We use `get_nearest_grid_index` function to get `(i_d,i_y)`. We use these indices to compute `log_lik` and update the `log_post`. This update, with adding the terms in log scale, gives us the numerator: $p(y_{obs}(t)|\theta,d*)p_t(\theta)$. Afterwards, the entire `log_post` is normalized by subtracting the `logsumexp(log_post)` from the `log_post`, this is equivalen to dividing by $p(y_{obs}(t)|d*)$. This update in log_posterior allows us to update mutual information to allow us to begin the next step again. Here is the snippet of code allowing us to do this



```python
from scipy.special import logsumexp

# This is essentially the engine.update where the mutual_info and log_posterior is updated after a response/choice is made
def update_MI(choice_set,response_set,cur_design,response,log_lik,ent,log_post):
    # Find the index of the best matching row vector to the given vector.
    def get_nearest_grid_index(design, design_set) -> int:
        design = design.reshape(1, -1)
        return np.square(design_set - design).sum(-1).argsort()[0]
    # in ADOpy they do some data_sort/prep to shape data into pandas series, we do this for now
    design_vals = np.fromiter(cur_design.values(), dtype=float)
    response_vals = np.array(response)
    # loop next three lines if there are multiple responses/designs to iterate through
    i_d = get_nearest_grid_index(design_vals, choice_set.values)
    i_y = get_nearest_grid_index(response_vals, response_set.values)
    # add log_likelihood (not pdf) to the log_posterior
    log_post = log_post + log_lik[i_d, :, i_y]

    # This seems to be a normalization so we can have a pdf for the log posterior, logsumexp() becomes 0 afterwards
    log_post = log_post - logsumexp(log_post)
    mutual_info = get_MI(log_lik,ent,log_post)
    return mutual_info,log_post
```


**2. Set the posterior to the new prior $p_{t+1}(\theta)=p(\theta|y_{obs}(t),d*)$, $t = t+1$ and go back to step 1 above. Rinse and repeat until convergence, number of trials is reached.**

We have already done this in the code above by assigning the `log_post` as `log_prior`. The three steps are put in a loop for the number of trials preset ahead of time.
