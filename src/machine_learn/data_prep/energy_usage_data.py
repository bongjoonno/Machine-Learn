from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

energy_df = pd.read_csv(DATA_DIRECTORY / 'energy_usage.csv')

energy_df = pd.get_dummies(energy_df, columns=['Building Type'], dtype=int)
energy_df['Day of Week'] = energy_df['Day of Week'].map({'Weekday' : 0, 'Weekend' : 1})

energy_x = energy_df.drop(columns=['Energy Consumption'])
energy_y = energy_df['Energy Consumption']