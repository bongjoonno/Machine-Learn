from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

salary_df = pd.read_csv(DATA_DIRECTORY / 'salary.csv')
print(salary_df)