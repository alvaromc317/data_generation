# `data_generation`
Are you looking for a package that can generate high dimensional synthetic datasets?

Do you need datasets with a grouped structure so that you can test your group lasso based formulation?

Dont look more. `data_generation` is precisely what you need.

## Usage example

```
data_equal = dgen.EqualGroupSize(n_obs=5000, ro=0.2, error_distribution='student_t', 
                                 e_df=5, random_state=1, group_size=10, non_zero_groups=3, 
                                 non_zero_coef=5, num_groups=7)

x, y, beta, group_index = data_equal.data_generation().values()
```

```
data_different = dgen.UnequalGroupSize(n_obs=5000, ro=0.8, error_distribution='normal', e_loc=1, e_scale=4,
                                       random_state=2, tuple_group_size=(2, 4, 6, 8),
                                       tuple_number_of_groups=(5, 10, 15, 20),
                                       tuple_non_zero_coef=(1, 2, 3, 4),
                                       tuple_non_zero_groups=(1, 3, 5, 7))

x, y, beta, group_index = data_different.data_generation().values()
```

For a deeper review and an explanation of the capabilities of this package we recommend to read the user_guide available in the repository.