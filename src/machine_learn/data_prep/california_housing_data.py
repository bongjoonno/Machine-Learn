from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

california_housing_df = pd.read_csv(DATA_DIRECTORY / 'california_housing.csv')
california_housing_df = california_housing_df.sample(frac=1, random_state=42)

california_housing_df['ocean_proximity'] = california_housing_df['ocean_proximity'].astype('category').cat.codes

california_housing_x = california_housing_df.drop(columns='median_house_value')

california_housing_y = california_housing_df['median_house_value']
california_housing_cols_to_scale = list(california_housing_x.columns)

print(california_housing_df.isna().sum())
print(california_housing_df)
