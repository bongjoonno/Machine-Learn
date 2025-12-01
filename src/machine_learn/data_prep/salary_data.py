from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

salary_df = pd.read_csv(DATA_DIRECTORY / 'salary.csv')
salary_df = salary_df.sample(frac=1, random_state=42)
salary_df = salary_df.drop(columns = ['Unnamed: 0'])

salary_x = salary_df[['YearsExperience']]
salary_y = salary_df['Salary']

salary_cols_to_scale = salary_x.columns