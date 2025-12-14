from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

energy_df = pd.read_csv(DATA_DIRECTORY / 'energy_usage/csv')

print(energy_df)