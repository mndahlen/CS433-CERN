## Fast demo
Run scripts/run.py

## Modules
All python modules below are found in /scripts

### implementations_py
This module contains only the project mandatory functions such as *logistic_regression()* and its dependencies.

### run_py
This module is a script that executes the loading, feature engineering, training and testing which achieved the 80% result as presented in the report.

### algorithms_py
This module contains functions for more advanced machine learning methods than required by the project. These were an attempt to improve training time and accuracy.

The functions (algorithms) are:
* *ADAM()*
* *LSAGDR()*

### feature_engineering_py
Contains functions for engineering features. Primarily the function *create_features()* is used for getting features in *run.py*, the rest are helpers for this function.

Furthermore there are some functions for testing feasibility of features:
* *percentage_missing_data_per_feature()*
* *low_variance_feature_in_raw_data()*
* *feature_selection_pairwise_correlation()*


### helpers_py
Various helpers for loading and saving CSV data, predicting labels, calculating accuracy and score, splitting data etc. 

### train_py
Functions for creating lambda function for training models. Used in *run.py*

### cross_validation_py
This module for cross validation is only used for feature selection and not model evaluation. Performs cross validation on selected features. Note that all sets of feature must contain the same number of features. 
