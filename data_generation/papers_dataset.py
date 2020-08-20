import logging

import numpy as np


logger = logging.getLogger(__name__)


def dg_enet(n_obs=100, n_var=40, noise_group=True, random_state=None):
    """
    Regularization_and_variable_selection_via_elastic_net__hastie__zou__2015
    Example 4
    """
    if random_state is not None:
        np.random.seed(random_state)
    if n_var < 16:
        raise ValueError(f'n_var must be at least 16')
    if noise_group:
        group_index = np.asarray([1] * 5 + [2] * 5 + [3] * 5 + [4] * (n_var - 15))
    else:
        noise = list(range(16, (n_var + 1)))
        group_index = np.asarray([1] * 5 + [2] * 5 + [3] * 5 + noise)
    beta = np.r_[np.repeat(3, 15), np.zeros(n_var - 15)]
    # Generate 3 random variables following a normal distribution used as regressing variables
    tmp_variables = [np.random.normal(0, 1, n_obs) for i in range(3)]
    # Stack the variables into groups of 5 elements
    tmp_variables = [np.c_[elt, elt, elt, elt, elt] for elt in tmp_variables]
    x1, x2, x3 = [elt + np.random.multivariate_normal(np.repeat(0, 5), np.diag(np.repeat(0.05, 5)), n_obs) for
                  elt in tmp_variables]
    # Add random noise
    x4 = np.random.multivariate_normal(np.repeat(0, (n_var - 15)), np.diag(np.repeat(1, (n_var - 15))), n_obs)
    x = np.c_[x1, x2, x3, x4]
    error = np.random.normal(0, 15, n_obs)
    y = np.dot(x, beta) + error
    data = dict(x=x, y=y, true_beta=beta, group_index=group_index)
    logger.debug('Function finished without errors')
    return data


def dg_hierarchical(n_obs=200, n_var=100, ro=0.2, group_size=10, error_type='CHI4', beta_type='int', random_state=None):
    """
    Sparse_group_variable_selection_based_on_quantile_hierarchical_lasso__zhao__zhang__liu__2014
    Example 2
    """
    if random_state is not None:
        np.random.seed(random_state)
    error = None
    if error_type == 'N1':
        error = np.random.normal(0, 1, n_obs)
    elif error_type == 'N4':
        error = np.random.normal(0, 4, n_obs)
    elif error_type == 'DE':
        error = np.random.laplace(0, 1, n_obs)
    elif error_type == 'M':
        error = 0.1 * np.random.normal(0, 5, n_obs) + 0.9 * np.random.normal(0, 1, n_obs)
    elif error_type == 'CHI4':
        error = np.random.chisquare(df=4, size=n_obs)
    elif error_type == 'CHI2':
        error = np.random.chisquare(df=2, size=n_obs)
    group_levels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    num_non_zeros_per_group = np.array([10, 8, 6, 4, 2, 1, 0, 0, 0, 0]) * int((group_size / 10))
    group_index = np.repeat(group_levels, [group_size] * 10, axis=0)
    s = np.zeros((n_var, n_var))
    for level in group_levels:
        s[(level - 1) * group_size:level * group_size, (level - 1) * group_size:level * group_size] = ro
    np.fill_diagonal(s, 1)
    x = np.random.multivariate_normal(np.repeat(0, n_var), s, n_obs)
    betas = None
    if beta_type == 'int':
        betas = np.random.choice([-1, 1], size=n_var, replace=True)
    elif beta_type == 'N':
        betas = np.random.normal(0, 1, n_var)
    for i in range(len(group_levels)):
        betas[((group_levels[i] - 1) * group_size + num_non_zeros_per_group[i]):group_levels[i] * group_size] = 0
    y = np.dot(x, betas) + error
    data = dict(x=x, y=y, true_beta=betas, index=group_index)
    logger.debug('Function finished without errors')
    return data

# SIGNAL TO NOISE COMPUTATION ##########################################################################################

def stn_calculator(x, y, true_beta):
    error = y - np.dot(x, true_beta)
    s = np.cov(x.T)
    var_error = np.var(error)
    stn = np.dot(np.dot(true_beta.T, s), true_beta) / var_error
    return stn
