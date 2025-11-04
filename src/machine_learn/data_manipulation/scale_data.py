from src.machine_learn.imports import StandardScaler

def scale_data(x_train, x_test, columns_to_scale):
    scaler = StandardScaler()

    x_train = x_train.copy()
    x_test = x_test.copy()
    
    x_train[columns_to_scale] = x_train[columns_to_scale].astype(float)
    x_test[columns_to_scale] = x_test[columns_to_scale].astype(float)

    x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
    x_test[columns_to_scale] = scaler.transform(x_test[columns_to_scale])

    return x_train, x_test