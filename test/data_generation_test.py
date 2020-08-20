import numpy as np
import data_generation as dgen
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
file_handler = logging.FileHandler('testing_dgen_log.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def equal_array(array0, array1, tol=1e-6):
    n0 = array0.shape[0]
    n1 = array1.shape[0]
    if n0 != n1:
        logger.warning(f'Shapes do not match. Received: {n0} and {n1}')
    else:
        dif = array0 - array1
        if np.mean(dif) > tol:
            logger.warning(f'Arrays are not equal. Mean difference is {np.mean(dif)}')


if __name__ == '__main__':
    # Elastic net paper datasets
    logger.debug('Starting elastic net datasets generation')
    enet_data = dgen.dg_enet()
    enet_data1 = dgen.dg_enet(n_obs=200, n_var=100, noise_group=False)
    x1, y1, b1, g1 = dgen.dg_enet(n_obs=300, n_var=200, noise_group=False, random_state=1).values()
    x2, y2, b2, g2 = dgen.dg_enet(n_obs=300, n_var=200, noise_group=False, random_state=1).values()
    equal_array(x1, x2)
    equal_array(y1, y2)
    equal_array(b1, b2)
    equal_array(g1, g2)

    # Hierarchical dataset
    logger.debug('Hierarchical lasso datasets generation')
    hierarchical = dgen.dg_hierarchical()

    # Equal group sizes generation
    logger.debug('Equal group sizes datasets generation')
    data_equal = dgen.EqualGroupSize()
    data_equal_1 = data_equal.data_generation()

    data_equal = dgen.EqualGroupSize(n_obs=5000, ro=0.2, error_distribution='student_t', e_df=5, random_state=1)
    x1, y1, b1, g1 = data_equal.data_generation().values()
    x2, y2, b2, g2 = data_equal.data_generation().values()
    equal_array(x1, x2)
    equal_array(y1, y2)
    equal_array(b1, b2)
    equal_array(g1, g2)

    data_equal2 = dgen.EqualGroupSize(n_obs=5000, ro=0.2, error_distribution='cauchy', e_loc=1, e_scale=10)

    data_equal3 = dgen.EqualGroupSize(n_obs=5000, ro=0.2, error_distribution='chi_square', e_df=5)

    data_equal4 = dgen.EqualGroupSize(n_obs=5000, ro=0.2, error_distribution='normal', e_loc=1, e_scale=10)

    data_equal = dgen.EqualGroupSize(group_size=10, non_zero_groups=3, non_zero_coef=5, num_groups=7)
    x, y, b, g = data_equal.data_generation().values()
    if x.shape[1] != (data_equal.group_size * data_equal.num_groups):
        logger.warning('EqualGroup size non expected dimension')
    if np.sum(b > 0) != (data_equal.non_zero_groups * data_equal.non_zero_coef):
        logger.warning('EqualGroup size non expected dimension')

    # Different group sizes
    logger.debug('Different group sizes datasets generation')
    data_different = dgen.UnequalGroupSize()
    data_different_1 = data_different.data_generation()

    data_different = dgen.UnequalGroupSize(n_obs=5000, ro=0.8, error_distribution='normal', e_loc=1, e_scale=4,
                                           random_state=2)
    x1, y1, b1, g1 = data_different.data_generation().values()
    x2, y2, b2, g2 = data_different.data_generation().values()
    equal_array(x1, x2)
    equal_array(y1, y2)
    equal_array(b1, b2)
    equal_array(g1, g2)

    data_unequal = dgen.UnequalGroupSize(tuple_group_size=(2, 4, 6, 8),
                                         tuple_number_of_groups=(5, 10, 15, 20),
                                         tuple_non_zero_coef=(1, 2, 3, 4),
                                         tuple_non_zero_groups=(1, 3, 5, 7))
    x, y, b, g = data_unequal.data_generation().values()
    n_var = np.sum(np.asarray(data_unequal.tuple_group_size) * np.asarray(data_unequal.tuple_number_of_groups))
    non_zero_coef = np.sum(np.asarray(data_unequal.tuple_non_zero_coef) *
                           np.asarray(data_unequal.tuple_non_zero_groups))
    if x.shape[1] != n_var:
        logger.warning('UnequalGroup size non expected dimension')
    if np.sum(b > 0) != non_zero_coef:
        logger.warning('UnequalGroup size non expected dimension')

    logger.info('Test finished with no errors')
