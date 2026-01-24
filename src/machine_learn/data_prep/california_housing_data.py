from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY
from src.machine_learn.data_manipulation import split_k_folds, scale_data

california_housing_df = pd.read_csv(DATA_DIRECTORY / 'california_housing.csv')
california_housing_df = california_housing_df.sample(frac=1, random_state=42)

california_housing_df = pd.get_dummies(california_housing_df, columns='ocean_proximity', dtype=int)

california_housing_x = california_housing_df.drop(columns='median_house_value')

california_housing_y = california_housing_df['median_house_value']
cols_to_scale = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                    'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']


california_housing_data = split_k_folds(california_housing_x, california_housing_y, k=1)

for data in california_housing_data:
    california_housing_data[0]['total_bedrooms'] = california_housing_data[0]['total_bedrooms'].fillna(california_housing_data[0]['total_bedrooms'].median())
    california_housing_data[2]['total_bedrooms'] = california_housing_data[2]['total_bedrooms'].fillna(california_housing_data[0]['total_bedrooms'].median())

    california_housing_data[0], california_housing_data[2] = scale_data(california_housing_data[0], california_housing_data[2], cols_to_scale)    