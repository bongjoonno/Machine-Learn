from src.machine_learn.imports import pd, StandardScaler
from src.machine_learn.types import DF

def scale_data(x_train: DF, *data_to_be_scaled_by_training_fit: DF, columns_to_scale: list[str]) -> tuple[DF, ...]:
    scaler = StandardScaler()

    x_train = x_train.copy()
    x_train[columns_to_scale] = x_train[columns_to_scale].astype(float)
    x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])
    
    scaled_non_training_data = []
    
    for data in data_to_be_scaled_by_training_fit:
        data_cop = data.copy()
        
        data_cop[columns_to_scale] = data_cop[columns_to_scale].astype(float)

        data_cop[columns_to_scale] = scaler.transform(data_cop[columns_to_scale])
        
        scaled_non_training_data.append(data_cop)

    return x_train, *scaled_non_training_data