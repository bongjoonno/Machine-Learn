from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

salary_df = pd.read_csv(DATA_DIRECTORY / 'salary.csv')
salary_df = salary_df.drop(columns = ['Unnamed: 0'])

salary_x = salary_df[['YearsExperience']]
print(salary_x, type(salary_x))