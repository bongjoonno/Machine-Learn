from .breast_cancer_data import breast_cancer_x, breast_cancer_y
from .email_spam_data import email_df, email_ham, email_spam
from .titanic_data import titanic_x, titanic_y

from .insurance_costs_data import insurance_x, insurance_x_base_line, insurance_y, insurance_cols_to_scale
from .salary_data import salary_x, salary_y, salary_cols_to_scale
from .student_performance_data import student_x, student_y, student_cols_to_scale
from .car_price_data import car_price_x, car_price_y, car_price_x_base_line, car_price_y_base_line, car_price_cols_to_scale
from .energy_usage_data import energy_x, energy_y, energy_cols_to_scale
from .concrete_data import concrete_x, concrete_y, concrete_columns_to_scale
from .wine_data import wine_x, wine_y, wine_columns_to_scale

regression_test_data = [(wine_x, wine_y, wine_columns_to_scale),
                        (salary_x, salary_y, salary_cols_to_scale), 
                        (student_x, student_y, student_cols_to_scale) ,
                        (car_price_x, car_price_y, car_price_cols_to_scale),
                        (energy_x, energy_y, energy_cols_to_scale),
                        (insurance_x, insurance_y, insurance_cols_to_scale)]


baseline_model_regression_test_data = [(wine_x, wine_y, wine_columns_to_scale),
                                       (salary_x, salary_y, salary_cols_to_scale), 
                                       (student_x, student_y, student_cols_to_scale),
                                       (car_price_x_base_line, car_price_y_base_line, car_price_cols_to_scale),
                                       (energy_x, energy_y, energy_cols_to_scale),
                                       (insurance_x_base_line, insurance_y, insurance_cols_to_scale), 
                                       (concrete_x, concrete_y, concrete_columns_to_scale)]

