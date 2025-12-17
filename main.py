from tests import (test_linear_regression, 
                   test_all_models, 
                   test_ga_param_optimizer, 
                   test_ga_lr_optimizer, 
                   test_scikit_learn_non_linear,
                   test_tab_pfn_non_linear,
                   test_nonlinear_ga_param_optimizer,
                   data_imports_test)

def main():
    #test_tab_pfn_non_linear()
    test_nonlinear_ga_param_optimizer()

if __name__ == '__main__':
    print(main())
    
#TO-DO
# Make template general linear models
# Get more Linear Regression test data-sets
# Get more Non-Linear test data-sets
# Write MAE metric
# Allow GA param optimizer to use multiple cost-functions (default should be MSE)
# Fix Logistic Regression Cost function
# possibly add a PER FEATURE parameter for whether y_pred is computed as func(w*x) or w*func(x) or some other configuration