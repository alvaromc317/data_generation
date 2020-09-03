import logging

import numpy as np
from scipy.stats import cauchy

logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self, n_obs=200, ro=0.5, error_distribution='student_t', e_df=3, e_loc=0, e_scale=3,
                 random_state=None):
        """
        :param error_distribution: error distribution
        :param e_df: degrees of freedom of the student t or chisq error distribution
        :param e_loc: location parameter of cauchy distribution
        :param e_scale: scale parameter of cauchy distribution
        :param random_state: random state value in case reproducible data is required
        """
        self.n_obs = n_obs
        self.ro = ro
        self.error_distribution = error_distribution
        self.e_df = e_df
        self.e_loc = e_loc
        self.e_scale = e_scale
        self.random_state = random_state

    def _compute_error(self):
        """
        return: array of errors for the model
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.error_distribution == 'student_t':
            error = np.random.standard_t(self.e_df, size=self.n_obs)
        elif self.error_distribution == 'cauchy':
            error = cauchy.rvs(loc=self.e_loc, scale=self.e_scale, size=self.n_obs)
        elif self.error_distribution == 'chi_square':
            error = np.random.chisquare(df=self.e_df, size=self.n_obs)
        elif self.error_distribution == 'normal':
            error = np.random.normal(loc=self.e_loc, scale=self.e_scale, size=self.n_obs)
        else:
            error = None
            logger.error(f'error_distribution provided:{self.error_distribution}')
        return error


# EQUAL GROUP SIZE DATA GENERATION ####################################################################################


class EqualGroupSize(DataGenerator):
    def __init__(self, n_obs=200, ro=0.5, error_distribution='student_t', e_df=3, e_loc=0, e_scale=3, random_state=None,
                 num_groups=20, group_size=20, non_zero_groups=7, non_zero_coef=8):
        """
        Generate a dataset formed by groups of variables. All the groups have the same size.
        :param n_obs: number of observations
        :param num_groups: number of groups
        :param group_size: size of the groups
        :param ro: correlation between variables inside a group
        :param non_zero_coef: controls how many significant variables we want
        :param non_zero_groups: controls how many groups we want to store significant variables
        :return: x, y, group_index
        If non_zero_coef=8 and non_zero_groups=7 we will have 8 groups with 7 significant variables,
        and num_groups-non_zero_coef groups with all non-significant variables
        The error distribution is a student t with 3 degrees of freedom
        """
        super().__init__(n_obs, ro, error_distribution, e_df, e_loc, e_scale, random_state)
        self.num_groups = num_groups
        self.group_size = group_size
        self.non_zero_groups = non_zero_groups
        self.non_zero_coef = non_zero_coef
        self.n_var = self.num_groups * self.group_size

    def __compute_predictors(self, group_levels):
        """
        :param group_levels: An array indicating the group index levels
        :return: matrix of predictors x
        """
        s = np.zeros((self.n_var, self.n_var))
        for level in group_levels:
            s[(level - 1) * self.group_size:level * self.group_size,
              (level - 1) * self.group_size:level * self.group_size] = self.ro
        np.fill_diagonal(s, 1)
        # Obtain matrix of predictors
        x = np.random.multivariate_normal(np.repeat(0, self.n_var), s, self.n_obs)
        return x

    def __compute_betas(self):
        """
        :return: array of beta coefficients
        """
        # Array of length num_groups indicating number of variables different than 0 per group
        betas = np.zeros(self.n_var)
        for i in range(self.non_zero_groups):
            betas[(i * self.group_size):(i * self.group_size + self.non_zero_coef)] = \
                np.arange(1, self.non_zero_coef + 1, 1)
        return betas

    def data_generation(self):
        group_levels = np.arange(1, (self.num_groups + 1), 1)
        group_index = np.repeat(group_levels, self.group_size)
        x = self.__compute_predictors(group_levels)
        betas = self.__compute_betas()
        error = self._compute_error()
        y = np.dot(x, betas) + error
        data = dict(x=x, y=y, true_beta=betas, group_index=group_index)
        logger.debug('Function finished without errors')
        return data

# UNEQUAL GROUP SIZE DATA GENERATION ###################################################################################


class UnequalGroupSize(DataGenerator):
    def __init__(self, n_obs=200, ro=0.5, error_distribution='student_t', e_df=3, e_loc=0, e_scale=3, random_state=None,
                 tuple_group_size=(5, 15, 30), tuple_number_of_groups=(15, 15, 15), tuple_non_zero_groups=(3, 3, 3),
                 tuple_non_zero_coef=(3, 6, 10)):
        """
        Generate a dataset formed by groups of variables. The groups have different sizes.
        :param n_obs: number of observations
        :param ro: correlation between variables inside each group
        :param tuple_group_size: A tuple with the group sizes of each type of group to be generated. If this is a
               length-2 tuple then we generate groups of the two sizes inputted in the tuple
        :param tuple_number_of_groups: How many groups of each size we want to generate
        :param tuple_non_zero_coef: How many significant variables we want inside each type of group
        :param tuple_non_zero_groups: How many groups with significant variables we want
        :return: x, y, group_index
        Using the predefined values, we would have 5 groups of size 15, including 3 groups with 3 significant variables
        and 12 groups of noise
        """
        super().__init__(n_obs, ro, error_distribution, e_df, e_loc, e_scale, random_state)
        self.tuple_group_size = tuple_group_size
        self.tuple_number_of_groups = tuple_number_of_groups
        self.tuple_non_zero_groups = tuple_non_zero_groups
        self.tuple_non_zero_coef = tuple_non_zero_coef
        if len(tuple_number_of_groups) != len(tuple_group_size):
            raise ValueError(f'All the tuples must have the same length')
        self.n_var = np.sum(np.asarray(tuple_group_size) * np.asarray(tuple_number_of_groups))

    def __one_step_compute_betas(self, n_var, group_size, non_zero_groups, non_zero_coef):
        """
        :return: array of beta coefficients
        """
        # Array of length num_groups indicating number of variables different than 0 per group
        betas = np.zeros(n_var)
        for i in range(non_zero_groups):
            betas[(i * group_size):(i * group_size + non_zero_coef)] = \
                np.arange(1, non_zero_coef + 1, 1)
        return betas

    def __compute_betas(self):
        # n_var is an array with the number of variables per group structure (number of variables in groups of each size
        # defined in the tuples)
        n_var = np.asarray(self.tuple_group_size) * np.asarray(self.tuple_number_of_groups)
        # betas is a list of arrays. Each array is linked to one group size
        betas = [self.__one_step_compute_betas(n_var, group_size, non_zero_groups, non_zero_coef) for
                 n_var, group_size, non_zero_groups, non_zero_coef in
                 zip(n_var, self.tuple_group_size, self.tuple_non_zero_groups, self.tuple_non_zero_coef)]
        # Convert list of arrays into one single array
        betas = np.concatenate(betas, axis=0)
        return betas

    def __compute_predictors(self, group_levels, group_sizes):
        s = np.zeros((self.n_var, self.n_var))
        inf_lim = 0
        sup_lim = 0
        for i in range(len(group_levels)):
            sup_lim = sup_lim + group_sizes[i]
            s[inf_lim:sup_lim, inf_lim:sup_lim] = self.ro
            inf_lim = sup_lim
        np.fill_diagonal(s, 1)
        x = np.random.multivariate_normal(np.repeat(0, self.n_var), s, self.n_obs)
        return x

    def data_generation(self):
        total_num_groups = int(np.sum(self.tuple_number_of_groups))
        group_levels = np.arange(1, total_num_groups + 1, 1)
        group_sizes = np.repeat(self.tuple_group_size, self.tuple_number_of_groups)
        betas = self.__compute_betas()
        group_index = np.repeat(group_levels, group_sizes)
        x = self.__compute_predictors(group_levels, group_sizes)
        error = self._compute_error()
        y = np.dot(x, betas) + error
        data = dict(x=x, y=y, true_beta=betas, group_index=group_index)
        logger.debug('Function finished without errors')
        return data
