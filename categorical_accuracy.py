def categorical_accuracy(y_pred, y_test):
    return sum(y_pred == y_test) / len(y_pred)