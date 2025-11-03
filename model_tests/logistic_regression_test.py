from models.logistic_regression import LogisticRegression
from data_imports.data_imports import breast_cancer_x, breast_cancer_y
from data_manipulation.data_manipulation_imports import train_test_split, scale_data
from metrics.metrics_imports import categorical_accuracy


def logistic_regression_test():
    breast_cancer_x_train, breast_cancer_y_train, breast_cancer_x_test, breast_cancer_y_test = train_test_split(breast_cancer_x, breast_cancer_y)

    breast_cancer_x_train, breast_cancer_x_test = scale_data(breast_cancer_x_train, breast_cancer_x_test, breast_cancer_x_train.columns)

    logistic_regression_model = LogisticRegression()

    logistic_regression_model.train(breast_cancer_x_train, breast_cancer_y_train, 1_000, 0.01)
    y_pred = logistic_regression_model.predict(breast_cancer_x_test)

    logistic_regression_categorical_accuracy = categorical_accuracy(y_pred, breast_cancer_y_test)
    return logistic_regression_categorical_accuracy