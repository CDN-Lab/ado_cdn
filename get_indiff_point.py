import numpy as np
import pandas as pd
import os,sys
import math
from scipy.stats import bernoulli
# expit(x) = 1/(1+exp(-x))
from scipy.special import expit as inv_logit
from scipy.special import logsumexp


def ask_for_response(design):
    response = input('Please choose: >>>(0) {} in {} days<<< OR >>>(1) {} in {} days<<<'.format(design['value_null'],design['time_null'],design['value_reward'],design['time_reward']))
    response = int(response)
    if response in [0,1]:
        pass
    else:
        print('ERROR, you selected {}, response needs to be 0 or 1'.format(response))
        print('Exiting now, try again')
        exit
    return response


def get_simulated_response(design):
    # Calculate the probability to choose a variable option
    t_ss, t_ll, r_ss, r_ll = (
        design['time_null'], design['time_reward'],
        design['value_null'], design['value_reward']
    )
    k, tau = PARAM_TRUE['k'], PARAM_TRUE['tau']
    
    u_ss = r_ss * (1. / (1 + k * t_ss))
    u_ll = r_ll * (1. / (1 + k * t_ll))
    p_obs = 1. / (1 + np.exp(-tau * (u_ll - u_ss)))

    # Randomly sample a binary choice response from Bernoulli distribution
    return bernoulli.rvs(p_obs)


def set_grids():
    grid_design = {
        'time_null': [0],
        'time_reward': [0.43, 0.714, 1, 2, 3,
                4.3, 6.44, 8.6, 10.8, 12.9,
                17.2, 21.5, 26, 52, 104,
                156, 260, 520],
        'value_null': [i*12.5 for i in range(1,64)],
        'value_reward': [800]
    }
    grid_param = {
        # 50 points on [10^-5, ..., 1] in a log scale
        'kappa': np.logspace(-5, 0, 50, base=10),
        # 10 points on (0, 5] in a linear scale
        'gamma': np.linspace(0, 5, 11)[1:]
    }
    grid_response = {
        'choice': [0, 1]
    }
    return grid_design,grid_param,grid_response


def insert_var(grid=[],var_nb=0,list_var=[]):
    if not var_nb:
        grid = list_var
    elif var_nb==1:
        grid = [[g]+[i] for g in grid for i in list_var]
    else:
        grid = [g+[i] for g in grid for i in list_var]
    return grid

def make_grid(design):
    grid = []
    labels = design.keys()
    for var_nb,row in enumerate(design.items()):
        grid = insert_var(grid=grid,var_nb=var_nb,list_var=row[1])
    grid_df = pd.DataFrame(grid,columns=labels)
    return grid_df


def populate_log_lik(choice_set,param_set,response_set):

    n_c = choice_set.shape[0]
    n_p = param_set.shape[0]
    n_r = response_set.shape[0]

    log_lik = np.zeros((n_c,n_p,n_r))
    for i_c,row_c in choice_set.iterrows():
        for i_p,row_p in param_set.iterrows():
            for i_r,row_r in response_set.iterrows():
                log_lik[i_c,i_p,i_r] = compute_log_lik(row_c,row_p,row_r)
    return log_lik

# this is equivalent to compute() in ADO, defined in each Model class: ModelHyp(Model) in dd.py
def compute_log_lik(row_c,row_p,row_r):
    tn,tr,vn,vr = row_c.values
    kappa,gamma = row_p.values
    choice = row_r.values[0]

    iSV_null = SV_discount(vn,tn,kappa=kappa,alpha=1)
    iSV_reward = SV_discount(vr,tr,kappa=kappa,alpha=1)
    p_choose_reward = inv_logit(gamma*(iSV_reward-iSV_null))
    return bernoulli.logpmf(choice, p_choose_reward)

def SV_discount(value,delay,kappa=0.005,alpha=1.0):
    def discount(delay):
        return np.divide(1, 1 + kappa * delay)
    SV = (value**alpha) * discount(delay)
    return SV


""" 
# extraneous function, using built-in functions deals with the special nan, -inf cases, OverflowError
def prob_softmax(SV1,SV0,gamma=0.5):
    # compute probability using softmax function, return 0 if OverlowError is thrown
    try: 
        p = inv_logit(gamma*(SV1-SV0))
        # p = 1 / (1 + math.exp(-gamma*(SV1 - SV0)))
    except OverflowError:
        print('We got OverflowError!')
        # It seems that now that we are using inv_logit() we are not getting OverflowError from p formula used before
        p = 0
    return p

# extraneous function, using built-in functions deals with the special nan, -inf cases, RuntimeWarning
def get_LL(choice,p_choose_reward):
    # eps = np.finfo(float).eps
    # smallest value LL can take
    try:
        # IDM_model method for calculating log-likelihood... possible we should change this
        # LL = (choice==1)*np.log(p_choose_reward) + ((choice==0))*np.log(1-p_choose_reward)
        LL = bernoulli.logpmf(choice, p_choose_reward)
        # we will deal with these cases when we return and do the np.log(np.exp()) trick
        # if math.isnan(LL):
        #     # print('log_lik is NaN with (choice,prob):({},{})'.format(choice,p_choose_reward))
        #     LL = LL_floor
        # elif LL == float("-inf"):
        #     # print('log_lik is -inf with (choice,prob):({},{})'.format(choice,p_choose_reward)) 
        #     LL = LL_floor
        # elif LL < LL_floor:
        #     # print('We got small LL: {}'.format(LL))
        #     LL = LL_floor
    except RuntimeWarning:
        # It seems that now that we are using bernoulli.logpmf() we are not getting RuntimeWarning from LL formula used before
        print('We got some RunTimeWarning up in here?')
        # LL = LL_floor
    return LL
 """

# Compute log_likelihood, entropy given the three design sets: choice_set, param_set, response_set
def compute_log_lik_ent(choice_set,param_set,response_set):
    lik_model = np.exp(populate_log_lik(choice_set,param_set,response_set))
    noise_ratio = 1e-7
    log_lik = np.log((1 - 2 * noise_ratio) * lik_model + noise_ratio)

    # dpy is design, parameters, output/response
    # notation dpy,dpy-> dp indicates summing along y, axis=2 from (0,1,2)
    ent = -1 * np.einsum('dpy,dpy->dp', np.exp(log_lik), log_lik)
    return log_lik,ent

# Compute Mutual Information, given log_likelihood, entropy, log_posterior
def compute_mutual_info(log_lik,ent,log_post):
    marg_log_lik = logsumexp(log_lik + log_post.reshape(1, -1, 1), axis=1)
    ent_marg = -1 * np.einsum('dy,dy->d', np.exp(marg_log_lik), marg_log_lik)
    post = np.exp(log_post)
    ent_cond = np.einsum('p,dp->d', post, ent)
    mutual_info = ent_marg - ent_cond
    return mutual_info

def get_nearest_grid_index(design, design_set) -> int:
    """
    Find the index of the best matching row vector to the given vector.
    """
    ds = design_set
    d = design.reshape(1, -1)
    return np.square(ds - d).sum(-1).argsort()[0]


def update_mutual_info(choice_set,response_set,cur_design,response,log_lik,ent,log_post):
    # in ADOpy they do some data_sort/prep to shape data into pandas series, we do this for now
    design_vals = np.fromiter(cur_design.values(), dtype=float)
    response_vals = np.array(response)
    # loop next three lines if there are multiple responses/designs to iterate through
    i_d = get_nearest_grid_index(design_vals, choice_set.values)
    i_y = get_nearest_grid_index(response_vals, response_set.values)
    log_post = log_post + log_lik[i_d, :, i_y]

    log_post = log_post - logsumexp(log_post)

    mutual_info = compute_mutual_info(log_lik,ent,log_post)
    return mutual_info,log_post


def get_post_mean(post,param_set):
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``num_params``.
    """
    return pd.Series(np.dot(post, param_set))

# def post_cov(self) -> np.ndarray:
#     """
#     An estimated covariance matrix for the posterior distribution.
#     Its shape is ``(num_grids, num_params)``.
#     """
#     # shape: (N_grids, N_param)
#     d = self.grid_param.values - self.post_mean.values
#     return np.dot(d.T, d * self.post.reshape(-1, 1))

# def post_sd(self) -> vector_like:
#     """
#     A vector of estimated standard deviations for the posterior
#     distribution. Its length is ``num_params``.
#     """
#     return pd.Series(np.sqrt(np.diag(self.post_cov)),
#                         index=self.model.params,
#                         name='Posterior SD')



# number of trials
N_TRIAL = 20

# 1 week, 2 weeks, 1 month, 6 months, 1 year, 2 years, 10 years
D_CAND = [1, 2, 4.3, 26, 52, 104, 520]

# DELTA_R_SS for the staircase method:
# The amount of changes on R_SS every 6 trials.
DELTA_R_SS = [400, 200, 100, 50, 25, 12.5]

# True parameter values to simulate responses
# after we get this to work, we can select from prior distribution
PARAM_TRUE = {'k': 0.12, 'tau': 1.5}



grid_design,grid_param,grid_response = set_grids()

choice_set = make_grid(grid_design)
param_set = make_grid(grid_param)
response_set = make_grid(grid_response)

log_lik,ent = compute_log_lik_ent(choice_set,param_set,response_set)

# this should be a default when there is no belief that any set should be preferred... i.e., all parameters are equally likely
n_p = param_set.shape[0]
log_prior = np.log(np.ones(n_p, dtype=np.float32) / n_p)

# this only for initialization, this needs to be updated when we go through the response, update sequence ... 
log_post = log_prior

'''
By accessing :code:`mutual_info` once, the engine computes log_lik,
marg_log_lik, ent, ent_marg, ent_cond, and mutual_info in a chain.
'''

# initialize
mutual_info = compute_mutual_info(log_lik,ent,log_post)



# Make an empty DataFrame to store data
columns = ['trial', 'response', 'mean_kappa', 'mean_gamma', 'sd_kappa', 'sd_gamma','time_null', 'time_reward', 'value_null', 'value_reward']
df_simul = pd.DataFrame(None, columns=columns)


for i in range(N_TRIAL):

    # GET_DESIGN based on maximum Mutual information

    idx_design = np.argmax(mutual_info)
    print(idx_design)
    cur_design = choice_set.iloc[idx_design].to_dict()
    print(cur_design)

    # get new response

    # Experiment
    # cur_response = ask_for_response(cur_design)
    cur_response = get_simulated_response(cur_design)    

    # UPDATE MI given a response
    mutual_info,log_post = update_mutual_info(choice_set,response_set,cur_design,cur_response,log_lik,ent,log_post)
    post_mean = get_post_mean(np.exp(log_post),param_set)
    post_sd = [0,0]
    # Save the information for updated posteriors
    dict_app = {
        'trial': i + 1,
        'response': cur_response,
        'mean_kappa': post_mean[0],
        'mean_gamma': post_mean[1],
        'sd_kappa': post_sd[0],
        'sd_gamma': post_sd[1],
    }
    dict_app.update(cur_design)
    df_app = pd.DataFrame(dict_app,index=[0])
    df_simul = pd.concat([df_simul,df_app],ignore_index=True)

print(df_simul)

# print(log_lik[0,:,0])
# print(np.min(np.min(log_lik)))
# print(np.log(np.finfo(float).eps))

'''
print(choice_set)

n_p = param_set.shape[0]

# log prior, initialization, all equally likely outcomes
print(np.log(np.ones(n_p) / n_p))


# figure out how to compute log likelihood!!!


# loop through each element in grid_design, grid_param, and grid_response
# compute log_lik for each combination of things


'''

