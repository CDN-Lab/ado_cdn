import numpy as np
import pandas as pd
import os,sys
import math
from scipy.stats import bernoulli
# expit(x) = 1/(1+exp(-x))
from scipy.special import expit as inv_logit


def set_grids():

    grid_design = {
        # [Now]
        'time_null': [0,1],
        # [3 days, 5 days, 1 week, 2 weeks, 3 weeks,
        #  1 month, 6 weeks, 2 months, 10 weeks, 3 months,
        #  4 months, 5 months, 6 months, 1 year, 2 years,
        #  3 years, 5 years, 10 years] in a weekly unit
        'time_reward': [0.43, 0.714, 1, 2, 3,
                # 4.3, 6.44, 8.6, 10.8, 12.9,
                # 17.2, 21.5, 26, 52, 104,
                156, 260, 520],
        # [$12.5, $25, ..., $775, $787.5]
        # 'r_ss': np.arange(12.5, 800, 12.5),
        # 'r_ss': [i*12.5 for i in range(1,64)],
        'value_null': [i*12.5 for i in range(1,4)],
        # [$800]
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

def compute_log_lik(row_c,row_p,row_r):
    tn,tr,vn,vr = row_c.values
    kappa,gamma = row_p.values
    choice = row_r.values[0]

    iSV_null = SV_discount(vn,tn,kappa=kappa,alpha=1)
    iSV_reward = SV_discount(vr,tr,kappa=kappa,alpha=1)
    # print(gamma,iSV_null,iSV_reward,-gamma*(iSV_null - iSV_reward))
    p_choose_reward = prob_softmax(iSV_reward,iSV_null,gamma=gamma)
    LL = get_LL(choice,p_choose_reward)
    return LL

def SV_discount(value,delay,kappa=0.005,alpha=1.0):
    # SV = (value**alpha)/(1+kappa*delay)
    def discount(delay):
        return np.divide(1, 1 + kappa * delay)
    SV = (value**alpha) * discount(delay)
    return SV

def prob_softmax(SV1,SV0,gamma=0.5):
    # compute probability using softmax function, return 0 if OverlowError is thrown
    try: 
        p = inv_logit(gamma*(SV1-SV0))
        # p = 1 / (1 + math.exp(-gamma*(SV1 - SV0)))
    except OverflowError:
        p = 0
    return p

def get_LL(choice,p_choose_reward):
    # eps = np.finfo(float).eps
    # smallest value LL can take
    LL_floor = np.log(1e-7)
    try:
        LL = (choice==1)*np.log(p_choose_reward) + ((choice==0))*np.log(1-p_choose_reward)
        LL_bernie = bernoulli.logpmf(choice, p_choose_reward)
        print('LL is {} and bernie is {}'.format(LL,LL_bernie))

        if math.isnan(LL):
            # print('log_lik is NaN with (choice,prob):({},{})'.format(choice,p_choose_reward))
            LL = LL_floor
        elif LL == float("-inf"):
            # print('log_lik is -inf with (choice,prob):({},{})'.format(choice,p_choose_reward)) 
            LL = LL_floor
        elif LL < LL_floor:
            # print('We got small LL: {}'.format(LL))
            LL = LL_floor
    except RuntimeWarning:
        print('We got some RunTimeWarning up in here?')
        LL = LL_floor
    return LL


grid_design,grid_param,grid_response = set_grids()

choice_set = make_grid(grid_design)
param_set = make_grid(grid_param)
response_set = make_grid(grid_response)

log_lik = populate_log_lik(choice_set,param_set,response_set)


print(log_lik[0,:,0].shape)
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

