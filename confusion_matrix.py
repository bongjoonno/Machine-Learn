from imports import np, pd
from logistic_regression_test import breast_cancer_y_test, y_pred

def create_confusion_matrix(y_test, y_pred):
    y_test = pd.Series(y_test)
    y_pred = pd.Series(y_pred)

    y_test_value_counts = y_test.value_counts()
    y_pred_value_counts = y_pred.value_counts()

    errors = pd.DataFrame(y_test_value_counts - y_pred_value_counts).T

    positives = pd.DataFrame(y_test_value_counts + y_pred_value_counts).T

    positives = positives - errors
    errors = abs(errors)

    res = pd.concat([positives, errors])
    return res

res = create_confusion_matrix(breast_cancer_y_test, y_pred)
print(res)