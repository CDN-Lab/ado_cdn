#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------------

### Use adaptive design optimization (ADO) to compute the indifference point ###

This script runs the confidence risky ambiguity (CRDM) task form the Introspection 
and decision-making (IDM) project. After the user chooses between two values, the
script updates and a new design or choice is presented. Each update gets closer to the
user's indifference point, where it is more difficult for the user to make a choice.
Running this script for 15-20 trials should get to the indifference point.

Inputs: 
    selection by user of preference for safe or lottery amount (CRDM task).

Outputs: 
  	indifference-point .csv file: the scripts saves a user-specific behavior 
        spreadsheet that captures the choices presented, the response made by the user,
        and the model parameters.

Usage: $ python CRDM_indiff_point.py

--------------------------------------------------------------------------------------
"""

# Built-in/Generic Imports
import os,sys
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


### Global variables that can be accessed throughout the script ###

# number of trials
N_TRIAL = 200
# True parameter values to simulate responses
PARAM_TRUE = {'alpha': 0.67, 'beta': 0.66, 'gamma': 1.5}
# machine precision
NOISE_RATIO = 1e-7

def alpha_beta(m,v):
    # use mean and variance to compute alpha and beta parameters for Beta distribution
    a = ((1.0-m)/(v) - 1/m)*(m**2)
    b = a*(1/m - 1.0)
    return max([NOISE_RATIO,a]),max([NOISE_RATIO,b])

def get_true_param(setting='default'):
    # True parameter values to simulate responses, can select from prior distribution
    print('The true parameters (ground truth) will be set with setting : {}'.format(setting))
    global PARAM_TRUE
    if setting=='default':
        PARAM_TRUE = {'alpha': 0.67, 'beta': 0.66, 'gamma': 1.5}
    elif setting=='plos_one':
        #  Lopez-Guzman 2018: The average RA parameter (α) was 0.7508, and the median was 0.6087
        #  Levy 2010: mean beta: session 1, 0.7 \pm 0.2; session 2, 0.6 \pm0.3; 
        PARAM_TRUE = {'alpha': 0.61, 'beta': 0.65, 'gamma': 1.0}
    elif setting=='beta_mle':
        utility_dir = '/Volumes/UCDN/datasets/IDM/utility'
        fn = os.path.join(utility_dir,'split_CRDM_analysis.csv')
        df = pd.read_csv(fn,index_col=0)
        
        mah,sah = df['alpha'].mean(),df['alpha'].std()
        mbh,sbh = df['beta'].mean(),df['beta'].std()
        mgh,sgh = df['gamma'].mean(),df['gamma'].std()

        alpha = np.random.normal(mah,sah)
        beta = np.random.normal(mbh,sbh)
        gamma = np.random.normal(mgh,sgh)

        PARAM_TRUE = {'alpha': max(alpha,0), 'beta':beta,'gamma': max(0,gamma)}

### Getting a response entered manually in ask_for_response(design) or simulated in get_simulated_response(design) ###

def ask_for_response(design):
    response = input('Please choose: >>>(0) ${0} at {1:0.1f}% <<< OR >>>(1) ${2} at {3:0.1f}% and {4:0.1f}% ambiguity<<<'
                     .format(design['value_null'],100*design['p_null'],design['value_reward'],100*design['p_reward'],
                             100*design['amb_level0']))
    response = int(response)
    if response in [0,1]:
        pass
    else:
        print('ERROR, you selected {}, response needs to be 0 or 1'.format(response))
        print('Exiting now, try again')
        sys.exit()
    return response

def get_simulated_response(d):
    # Calculate the probability to choose a variable option
    pn, pr, vn, vr, ambig = (d['p_null'],d['p_reward'],d['value_null'],d['value_reward'],d['amb_level'])
    alpha, beta, gamma = PARAM_TRUE['alpha'], PARAM_TRUE['beta'], PARAM_TRUE['gamma']
    
    SV_null = (vn**alpha) * pn
    SV_reward = (vr**alpha) * (pr - beta * ambig / 2)
    p_obs = 1. / (1 + np.exp(-gamma * (SV_reward - SV_null)))

    # Randomly sample a binary choice response from Bernoulli distribution
    return bernoulli.rvs(p_obs)


### Defining the grids, and functions to make sets as dataframes: choice_set,param_set,response_set

def set_grids():

    grid_design = {
        # safe: p_null = 1.0
        'p_null': [1.0],
        # probability reward (lottery winning probability)
        'p_reward': [0.13, 0.25, 0.38, 0.50, 0.75],
        # safe reward: $5
        'value_null': [5],
        # reward amount set to vary according to experiment
        'value_reward': [5, 8, 20, 40, 50],
        # ambiguity levels
        'amb_level':[0.0, 0.24, 0.50, 0.74]
    }

    grid_param = {
        'alpha': np.linspace(0, 3, 11)[1:],
        'beta': np.linspace(-3, 3, 11),
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

def insert_amb(grid=[],list_var=[]):
    grid_ambig = []
    for g in grid:
        if g[1]==0.50:
            grid_ambig += [g+[i] for i in list_var]
    return grid_ambig

def make_grid(grid_values,csp=False):
    grid = []
    labels = list(grid_values.keys())
    if not csp:
        for var_nb,row in enumerate(grid_values.items()):
            grid = insert_var(grid=grid,var_nb=var_nb,list_var=row[1])
    else:
        for var_nb,row in enumerate(grid_values.items()):
            if labels[var_nb] == 'amb_level':
                grid_unambig = insert_var(grid=grid,var_nb=var_nb,list_var=[row[1][0]])
                grid_ambig = insert_amb(grid=grid,list_var=row[1][1:])
                grid = grid_unambig + grid_ambig
            else:
                grid = insert_var(grid=grid,var_nb=var_nb,list_var=row[1])
    grid_df = pd.DataFrame(grid,columns=labels)
    return grid_df

def make_grids(gd,gp,gr):
    return make_grid(gd,csp=True),make_grid(gp),make_grid(gr)

class Gridset(object):
    choice = []
    param = []
    response = []

def get_khat(sets):
    vn = sets.choice['value_null'].unique()[0]
    tr_vals = sets.choice['time_reward'].unique()
    vr_vals = sets.choice['value_reward'].unique()
    tr_mid = tr_vals[len(tr_vals)//2]
    vr_mid = vr_vals[len(vr_vals)//2]
    return (vr_mid/vn - 1.0) / (tr_mid)

def set_to_logp(sets,k_hat):
    param_mid = sets.param.iloc[(sets.param['kappa']-k_hat).abs().argsort()[:1]]
    pmid_idx = param_mid.index.values.astype(int)[0]
    n_p = sets.param.shape[0]
    prior = np.zeros(n_p, dtype=np.float32)
    prior[pmid_idx] = 1
    noise_ratio = 1e-7
    log_prior = np.log((1 - 2 * noise_ratio) * prior + noise_ratio)
    return log_prior

def get_prior(sets,default=True):
    
    # this should be a default when there is no belief that any set should be preferred... i.e., all parameters are equally likely
    if default:
        n_p = sets.param.shape[0]
        return np.log(np.ones(n_p, dtype=np.float32) / n_p)
"""     else:
        k_hat = get_khat(sets)
        return set_to_logp(sets,k_hat) """

def step0():
    tStep0 = time.time()
    grid_design,grid_param,grid_response = set_grids()
    sets = Gridset()
    sets.choice,sets.param,sets.response = make_grids(grid_design,grid_param,grid_response)
    # choice_set,param_set,response_set = make_grids(grid_design,grid_param,grid_response)

    log_lik,ent = get_LL_ent(sets.choice,sets.param,sets.response)
    log_prior = get_prior(sets,default=True)

    # this only for initialization, this needs to be updated when we go through the response, update sequence ... 
    log_post = log_prior
    print('Time to complete step 0 : {} minutes'.format((time.time() - tStep0)/60.0))
    return log_lik,ent,log_post,sets

def step123(log_lik,ent,log_post,sets):
    tStep123 = time.time()
    '''
    By accessing :code:`mutual_info` once, the engine computes log_lik,
    marg_log_lik, ent, ent_marg, ent_cond, and mutual_info in a chain.
    '''
    # initialize
    mutual_info = get_MI(log_lik,ent,log_post)

    # ask what experiment we are doing ...
    # a - participant; b - simulated
    experiment = get_experiment_type()

    # Make an empty DataFrame to store data
    columns = ['trial','response','mean_alpha', 'mean_beta', 'mean_gamma','sd_alpha', 'sd_beta', 
               'sd_gamma','p_null', 'p_reward', 'value_null', 'value_reward', 'amb_level']
    df_simul = pd.DataFrame(None, columns=columns)

    for i in range(N_TRIAL):

        # GET_DESIGN based on maximum Mutual information
        idx_design = np.argmax(mutual_info)
        cur_design = sets.choice.iloc[idx_design].to_dict()
        
        # Experiment
        if experiment == 'participant':
            cur_response = ask_for_response(cur_design)
        elif experiment  == 'simulation':
            cur_response = get_simulated_response(cur_design)     

        # UPDATE MI given a response
        mutual_info,log_post = update_MI(sets.choice,sets.response,cur_design,cur_response,log_lik,ent,log_post)
        post_mean = get_post_mean(np.exp(log_post),sets.param)
        post_sd = get_post_sd(np.exp(log_post),sets.param)
        # Save the information for updated posteriors
        dict_app = {
            'trial': i + 1,
            'response': cur_response,
            'mean_alpha': post_mean[0],
            'mean_beta': post_mean[1],
            'mean_gamma': post_mean[2],
            'sd_alpha': post_sd[0],
            'sd_beta': post_sd[1],
            'sd_gamma': post_sd[2],
        }
        dict_app.update(cur_design)
        df_app = pd.DataFrame(dict_app,index=[0])
        df_simul = pd.concat([df_simul,df_app],ignore_index=True)

    print(df_simul)
    fn = '/tmp/ADO_crdm_simulation.csv'
    print('Saving to : {}'.format(fn))
    df_simul.to_csv(fn)
    print('Time to complete step 1,2,3 : {} minutes'.format((time.time() - tStep123)/60.0))

def get_experiment_type():
    experiment = input('Please select: >>>(a) participant responses<<< or >>>(b) simulate responses<<<\n')
    if experiment in ['a','b']:
        pass
    else:
        print('ERROR, you selected {}, response needs to be (a) or (b)'.format(experiment))
        print('Exiting now, try again')
        sys.exit()
    if experiment == 'a':
        print('You selected >>>(a) participant responses<<<\n')
        experiment = 'participant'
    elif experiment == 'b':
        print('You selected >>>(b) simulate responses<<<\n')
        experiment = 'simulation'
        # PARAM_TRUE are used for simulating a response
        get_true_param(setting='beta_mle')
        print(PARAM_TRUE)
    return experiment

def main():
    # Step 0, compute log_likelihood, entropy, assign log_prior to log_posterior
    log_lik,ent,log_post,sets = step0()
    step123(log_lik,ent,log_post,sets)



if __name__ == "__main__":
	# main will be executed after running the script
    main()
