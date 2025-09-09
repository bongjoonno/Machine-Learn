from imports import np, pd, StandardScaler

#metrics
from r_squared import r_squared
from categorical_accuracy import categorical_acc

#models
from linear_regression import linear_regression
from logistic_regression import logistic_regression
from train_test_split import train_test_split
from naive_bayes import naive_bayes_prep, naive_bayes_predict

#data prep
from breast_cancer_data_prep import breast_cancer_df, breast_cancer_x, breast_cancer_y
from insurance_costs_data_prep import insurance_df, insurance_x, insurance_y
from email_spam_data_prep import email_df, email_ham, email_spam

from scale_data import scale_data