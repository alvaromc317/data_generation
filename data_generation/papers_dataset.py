import logging

import numpy as np

logger = logging.getLogger(__name__)


class PaperDataGenerator:
    def __init__(self, data_function_name='enet', n_obs=100, n_var=40, noise_group=True, random_state=None, ro=0.2,
                 group_size=10, error_type='CHI4', beta_type='int'):
        self.data_function_name = data_function_name
        self.valid_data_function_names = ['enet', 'hierarchical']
        self.n_obs = n_obs
        self.n_var = n_var
        self.noise_group = noise_group
        self.random_state = random_state
        self.ro = ro
        self.group_size = group_size
        self.error_type = error_type
        self.beta_type = beta_type

    def _enet(self):
        """
        Regularization_and_variable_selection_via_elastic_net__hastie__zou__2015
        Example 4
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.n_var < 16:
            raise ValueError(f'n_var must be at least 16')
        if self.noise_group:
            group_index = np.asarray([1] * 5 + [2] * 5 + [3] * 5 + [4] * (self.n_var - 15))
        else:
            noise = list(range(16, (self.n_var + 1)))
            group_index = np.asarray([1] * 5 + [2] * 5 + [3] * 5 + noise)
        beta = np.r_[np.repeat(3, 15), np.zeros(self.n_var - 15)]
        # Generate 3 random variables following a normal distribution used as regressing variables
        tmp_variables = [np.random.normal(0, 1, self.n_obs) for i in range(3)]
        # Stack the variables into groups of 5 elements
        tmp_variables = [np.c_[elt, elt, elt, elt, elt] for elt in tmp_variables]
        x1, x2, x3 = [elt + np.random.multivariate_normal(np.repeat(0, 5), np.diag(np.repeat(0.05, 5)), self.n_obs) for
                      elt in tmp_variables]
        # Add random noise
        x4 = np.random.multivariate_normal(np.repeat(0, (self.n_var - 15)),
                                           np.diag(np.repeat(1, (self.n_var - 15))), self.n_obs)
        x = np.c_[x1, x2, x3, x4]
        error = np.random.normal(0, 15, self.n_obs)
        y = np.dot(x, beta) + error
        data = dict(x=x, y=y, true_beta=beta, group_index=group_index)
        logger.debug('Function finished without errors')
        return data

    def _hierarchical(self):
        """
        Sparse_group_variable_selection_based_on_quantile_hierarchical_lasso__zhao__zhang__liu__2014
        Example 2
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        error = None
        if self.error_type == 'N1':
            error = np.random.normal(0, 1, self.n_obs)
        elif self.error_type == 'N4':
            error = np.random.normal(0, 4, self.n_obs)
        elif self.error_type == 'DE':
            error = np.random.laplace(0, 1, self.n_obs)
        elif self.error_type == 'M':
            error = 0.1 * np.random.normal(0, 5, self.n_obs) + 0.9 * np.random.normal(0, 1, self.n_obs)
        elif self.error_type == 'CHI4':
            error = np.random.chisquare(df=4, size=self.n_obs)
        elif self.error_type == 'CHI2':
            error = np.random.chisquare(df=2, size=self.n_obs)
        group_levels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        num_non_zeros_per_group = np.array([10, 8, 6, 4, 2, 1, 0, 0, 0, 0]) * int((self.group_size / 10))
        group_index = np.repeat(group_levels, [self.group_size] * 10, axis=0)
        s = np.zeros((self.n_var, self.n_var))
        for level in group_levels:
            s[(level - 1) * self.group_size:level * self.group_size, (level - 1) *
                self.group_size:level * self.group_size] = self.ro
        np.fill_diagonal(s, 1)
        x = np.random.multivariate_normal(np.repeat(0, self.n_var), s, self.n_obs)
        betas = None
        if self.beta_type == 'int':
            betas = np.random.choice([-1, 1], size=self.n_var, replace=True)
        elif self.beta_type == 'N':
            betas = np.random.normal(0, 1, self.n_var)
        for i in range(len(group_levels)):
            betas[((group_levels[i] - 1) * self.group_size +
                   num_non_zeros_per_group[i]):group_levels[i] * self.group_size] = 0
        y = np.dot(x, betas) + error
        data = dict(x=x, y=y, true_beta=betas, index=group_index)
        logger.debug('Function finished without errors')
        return data

    def __get_data_function_name(self):
        s = '_' + self.data_function_name
        return s

    def data_generation(self):
        if self.data_function_name not in self.valid_data_function_names:
            s = f'Invalid data generation function. Value provided:{self.data_function_name}.\nValid values:{self.valid_data_function_names}'
            logging.error(s)
            raise ValueError(s)
        data = getattr(self, self.__get_data_function_name())()
        return data

# SIGNAL TO NOISE COMPUTATION ##########################################################################################


def stn_calculator(x, y, true_beta):
    error = y - np.dot(x, true_beta)
    s = np.cov(x.T)
    var_error = np.var(error)
    stn = np.dot(np.dot(true_beta.T, s), true_beta) / var_error
    return stn
