#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Libs
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.special import expit as inv_logit
from scipy.special import logsumexp


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
def get_LL_ent(choice_set,param_set,response_set):
    lik_model = np.exp(populate_log_lik(choice_set,param_set,response_set))
    noise_ratio = 1e-7
    log_lik = np.log((1 - 2 * noise_ratio) * lik_model + noise_ratio)

    # dpy is design, parameters, output/response
    # notation dpy,dpy-> dp indicates summing along y, axis=2 from (0,1,2)
    ent = -1 * np.einsum('dpy,dpy->dp', np.exp(log_lik), log_lik)
    return log_lik,ent

# Compute Mutual Information, given log_likelihood, entropy, log_posterior
def get_MI(log_lik,ent,log_post):
    marg_log_lik = logsumexp(log_lik + log_post.reshape(1, -1, 1), axis=1)
    ent_marg = -1 * np.einsum('dy,dy->d', np.exp(marg_log_lik), marg_log_lik)
    post = np.exp(log_post)
    ent_cond = np.einsum('p,dp->d', post, ent)
    mutual_info = ent_marg - ent_cond
    return mutual_info

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





def get_post_mean(post,param_set):
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``num_params``.
    """
    return pd.Series(np.dot(post, param_set))



def post_cov(post,param_set):
    """
    An estimated covariance matrix for the posterior distribution.
    Its shape is ``(num_grids, num_params)``.
    """
    # shape: (N_grids, N_param)
    d = param_set.values - get_post_mean(post,param_set).values
    return np.dot(d.T, d * post.reshape(-1, 1))

def get_post_sd(post,param_set):
    """
    A vector of estimated standard deviations for the posterior
    distribution. Its length is ``num_params``.
    """
    return pd.Series(np.sqrt(np.diag(post_cov(post,param_set))))





