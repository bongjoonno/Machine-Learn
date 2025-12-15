from tests import test_linear_regression, test_all_models, test_ga_param_optimizer, test_ga_lr_optimizer, data_imports_test

def main():
    return test_linear_regression()
 

if __name__ == '__main__':
    print(main())
    
#TO-DO
# Make template for linear regression testing y_scale can be boolean so it will work with GA's as well.
# Base early stop for GA's on validation loss