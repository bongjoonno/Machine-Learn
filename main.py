from tests import test_linear_regression, test_all_models, test_ga_param_optimizer, test_ga_lr_optimizer, test_scikit_learn_linear_regression

def main():
    test_ga_param_optimizer()
    #test_scikit_learn_linear_regression()

if __name__ == '__main__':
    print(main())
    
#TO-DO
# Make template general linear models
# Get more Linear Regression test data-sets
# Write MAE metric
# Allow GA param optimizer to use multiple cost-functions (default should be MSE)
# Fix Logistic Regression Cost function