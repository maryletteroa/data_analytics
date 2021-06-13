This notebook is in fulfillment of the **Exit Assessment** requirement for  **SP102: Designing and Building Data Products**, part of the Project SPARTA Learning Pathway.

Most of the code in this notebook has been taken from the sample notebook in Module 4.



## 🌟 Goal of this analysis

This analysis aims to build a model that will predict the median values of the owner-occupied homes.

The dependent variable of interest is `MEDV`. The input dataset is the **Boston Housing Dataset**, and the list of indepent variables are described below.



## Dataset

This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts.  

There are 14 attributes in each case of the dataset. They are:

- CRIM - per capita crime rate by town
- ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS - proportion of non-retail business acres per town.
- CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX - nitric oxides concentration (parts per 10 million)
- RM - average number of rooms per dwelling
- AGE - proportion of owner-occupied units built prior to 1940
- DIS - weighted distances to five Boston employment centres
- RAD - index of accessibility to radial highways
- TAX - full-value property-tax rate per \$10,000
- PTRATIO - pupil-teacher ratio by town
- BLACK / B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT - % lower status of the population
- MEDV - Median value of owner-occupied homes in $1000's

## 💡 tl;dr

###  Summary 

This notebook covers the following steps of a machine learning pipeline:

- ✔ Exploratory data analysis (EDA)
- ✔ Spliting and training the data
- ✔ Spot checking algorithms
- ✔ Evaluating the algorithms
- ✔ Model training
- ✔ Model selection

This does not cover activities that require GCP:

- ❌ Model deployment
- ❌ Batch prediction

In building the model, the features `tax`, `black`, `dis`, `chas`, and `zn` where removed. This is a deviation from the original notebook, and was done to improve the accuracy of the model.

Lastly, a model was built to predict the prices using **XGBRegressor**, with some hyperaramater tuning.

⭐ An explanation of the model using SHAP (SHapley Additive exPlanations) values has been included.



### 🎓 Conclusion

Model spotting resulted in **XGBoost** as the best model to use among those tested.

A model was subsequently trained with some parameter tuning, and has the following characteristics:

```
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=10,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=4, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
```
and score:

```
MAE: 0.037
MSE: 0.003 
RMSE: 0.054
R2: 0.88
```

Of the features, `rm` (average number of rooms per dwelling) and `lstat` (% lower status of the population) have the highest importance in predicting the median values of the houses. 


- As the number of rooms in the house increase, so does the house prices.
- As the percentage of the lower status of the population increase, the house prices decrease.

Other interesting interactions were observed, for example

- In the plot of `crim` vs `lstat`, criminal activities has a positive interaction with the lower status of the population, higher values of both affect the values of housing negatively.
- In the plot of `rm` vs `nox`, house with higher number of rooms tend to have lower amounts of nitric oxide while
- In the plot of `age` vs `nox`, higher levels of nitric oxide can be seen in properties built in areas with 80% or more units built prior to 1940.
- Lastly, in the plot `rad` vs `nox`, while radial access to highways does not dramatically affect the prices of houses, those with higher access (higher `rad` values), tend to also report a disproportional amounts of nitric oxide.


