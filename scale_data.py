from imports import StandardScaler

def scale_data(x_train, x_test, columns_to_scale):
    scaler = StandardScaler()

    x_train.loc[:, columns_to_scale] = scaler.fit_transform(x_train.loc[:, columns_to_scale])
    x_test.loc[:, columns_to_scale] = scaler.transform(x_test.loc[:, columns_to_scale])

    return x_train, x_test