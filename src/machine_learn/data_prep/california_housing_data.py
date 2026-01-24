from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

california_housing_df = pd.read_csv(DATA_DIRECTORY / 'california_housing.csv')

print(california_housing_df)
print(california_housing_df.columns)