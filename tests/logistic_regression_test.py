from src.machine_learn.models import LogisticRegression
from src.machine_learn.data_prep import breast_cancer_x, breast_cancer_y
from src.machine_learn.data_manipulation import train_test_split
from src.machine_learn.metrics import categorical_accuracy


def test_logistic_regression() -> float:
    breast_cancer_x_train, breast_cancer_y_train, breast_cancer_x_test, breast_cancer_y_test = train_test_split(breast_cancer_x, breast_cancer_y)
    
    logistic_regression_model = LogisticRegression()

    logistic_regression_model.train(breast_cancer_x_train, breast_cancer_y_train, 1_000, 0.01)
    y_pred = logistic_regression_model.predict(breast_cancer_x_test)

    logistic_regression_categorical_accuracy = categorical_accuracy(y_pred, breast_cancer_y_test)
    return logistic_regression_categorical_accuracy