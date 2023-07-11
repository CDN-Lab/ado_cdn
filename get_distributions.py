#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Libs
import numpy as np
import pandas as pd



def get_post_mean(post,param_set):
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``num_params``.
    """
    return pd.Series(np.dot(post, param_set))