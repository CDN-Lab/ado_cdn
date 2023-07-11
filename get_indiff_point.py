#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------------

### Use adaptive design optimization (ADO) to compute the indifference point ###

This script runs the confidence delayed discounting (CDD) task form the Introspection 
and decision-making (IDM) project. After the user chooses between two values, the
script updates and a new design or choice is presented. Each update gets closer to the
user's indifference point, where it is more difficult for the user to make a choice.
Running this script for 15-20 trials should get to the indifference point.

Inputs: 
    selection by user of preference for money now or money later (CDD task).

Outputs: 
  	indifference-point .csv file: the scripts saves a user-specific behavior 
        spreadsheet that captures the choices presented, the response made by the user,
        and the model parameters.

Usage: $ python get_indiff_point.py

--------------------------------------------------------------------------------------
"""

# Built-in/Generic Imports
import sys

# Libs
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.special import expit as inv_logit
from scipy.special import logsumexp

# Own modules
from get_distributions import get_post_mean


__author__ = 'Ricardo Pizarro'
__copyright__ = 'Copyright 2023, Introspection and decision-making (IDM) project'
__credits__ = ['Ricardo Pizarro, Silvia Lopez-Guzman']
__license__ = 'ADO-CDN 1.0'
__version__ = '0.1.0'
__maintainer__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarro@nih.gov, silvia.lopezguzman@nih.gov'
__status__ = 'Dev'



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


### Defining the grids, and functions to make sets as dataframes: choice_set,param_set,response_set

def set_grids():
    grid_design = {
        'time_null': [0],
        'time_reward': [0.43, 0.714, 1, 2, 3,4.3, 6.44, 8.6, 10.8, 12.9,
                17.2, 21.5, 26, 52, 104,156, 260, 520],
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
    def discount(delay=1):
        return np.divide(1, 1 + kappa * delay)
    tn,tr,vn,vr = row_c.values
    kappa,gamma = row_p.values
    alpha = 1
    choice = row_r.values[0]

    iSV_null = (vn**alpha) * discount(delay=tn)
    iSV_reward = (vr**alpha) * discount(delay=tr)
    p_choose_reward = inv_logit(gamma*(iSV_reward-iSV_null))
    return bernoulli.logpmf(choice, p_choose_reward)

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

# This is essentially the engine.update where the mutual_info and log_posterior is updated after a response/choice is made
def update_mutual_info(choice_set,response_set,cur_design,response,log_lik,ent,log_post):
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

    # This seems to be a normalization so we can have a pdf for the log posterior
    print(log_post)
    print(logsumexp(log_post))
    log_post = log_post - logsumexp(log_post)
    print(logsumexp(log_post))
    print(log_post)
    mutual_info = compute_mutual_info(log_lik,ent,log_post)
    return mutual_info,log_post




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



### Global variables that can be accessed throughout the script ###

# number of trials
N_TRIAL = 2

# True parameter values to simulate responses
# after we get this to work, we can select from prior distribution
PARAM_TRUE = {'kappa': 0.12, 'gamma': 1.5}




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

