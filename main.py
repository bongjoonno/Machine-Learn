from tests import (test_tab_pfn_non_linear,
                   test_nonlinear_ga_param_optimizer,
                   data_imports_test)

from src.machine_learn.imports import np, Path

save_path_desktop = Path(r'D:\code\repos\Machine-Learn\numpy_data')
save_path_laptop = Path(r'C:\main\code\repos\Machine-Learn\numpy_data')

def main():
    #np.save(save_path / 'tab_pfn_regression', test_tab_pfn_non_linear())
    #np.save(save_path / 'GANON_regression_sbx', test_nonlinear_ga_param_optimizer(crossover_method='sbx'))
    #np.save(save_path / 'GANON_regression_arithmetic_crossover', test_nonlinear_ga_param_optimizer(crossover_method='arithmetic'))
    #np.save(save_path / 'GANON_regression_arithmetic_crossover_sympy_2', test_nonlinear_ga_param_optimizer(crossover_method='arithmetic'))
    '''    #np.save(save_path_laptop / 'GANON_regression_sbx_function_crossover_sympy', test_nonlinear_ga_param_optimizer({'non_linearity' : True, 
                                                                                                                   'cost_function' : 'mae',
                                                                                                                   'selection_method' : 'tournament', 
                                                                                                                   'crossover_method' : 'sbx', 
                                                                                                                   'function_crossover_method' : 'arithmetic'}))
    
    '''
    pass
                                                                
                                    
    
if __name__ == '__main__':
    print(main())
    
#TO-DO
# DATA LEAKAGE in TABPFN testing vs. GANON Testing Validation set???
# Parallelize GANON
# Convert entire library to CuPY
# Make template general linear models
# Get more Linear Regression test data-sets
# Get more Non-Linear test data-sets
# Fix Logistic Regression Cost function
# possibly add a PER FEATURE parameter for whether y_pred is computed as func(w*x) or w*func(x) or some other configuration