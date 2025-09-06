def categorical_acc(y_pred, y_test):
    return sum(y_pred == y_test) / len(y_pred)