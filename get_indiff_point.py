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
import time

# Libs
import numpy as np
import pandas as pd
from scipy.stats import bernoulli

# Own modules
from get_distributions import get_LL_ent,get_MI,update_MI,get_post_mean,get_post_sd


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



### Global variables that can be accessed throughout the script ###

# number of trials
N_TRIAL = 200
# True parameter values to simulate responses, can select from prior distribution
PARAM_TRUE = {'kappa': 0.2, 'gamma': 1.0}


# Step 0, compute log_likelihood, entropy, assign log_prior to log_posterior

grid_design,grid_param,grid_response = set_grids()

choice_set = make_grid(grid_design)
param_set = make_grid(grid_param)
response_set = make_grid(grid_response)

tStep0 = time.time()

log_lik,ent = get_LL_ent(choice_set,param_set,response_set)

# this should be a default when there is no belief that any set should be preferred... i.e., all parameters are equally likely
n_p = param_set.shape[0]
log_prior = np.log(np.ones(n_p, dtype=np.float32) / n_p)

# this only for initialization, this needs to be updated when we go through the response, update sequence ... 
log_post = log_prior

print('Time to complete step 0 : {} minutes'.format((time.time() - tStep0)/60.0))



tStep123 = time.time()
'''
By accessing :code:`mutual_info` once, the engine computes log_lik,
marg_log_lik, ent, ent_marg, ent_cond, and mutual_info in a chain.
'''
# initialize
mutual_info = get_MI(log_lik,ent,log_post)



# Make an empty DataFrame to store data
columns = ['trial', 'response', 'mean_kappa', 'mean_gamma', 'sd_kappa', 'sd_gamma','time_null', 'time_reward', 'value_null', 'value_reward']
df_simul = pd.DataFrame(None, columns=columns)


for i in range(N_TRIAL):

    # GET_DESIGN based on maximum Mutual information
    idx_design = np.argmax(mutual_info)
    cur_design = choice_set.iloc[idx_design].to_dict()

    # Experiment
    # cur_response = ask_for_response(cur_design)
    cur_response = get_simulated_response(cur_design)    

    # UPDATE MI given a response
    mutual_info,log_post = update_MI(choice_set,response_set,cur_design,cur_response,log_lik,ent,log_post)
    post_mean = get_post_mean(np.exp(log_post),param_set)
    post_sd = get_post_sd(np.exp(log_post),param_set)
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

print('Time to complete step 1,2,3 : {} minutes'.format((time.time() - tStep123)/60.0))


