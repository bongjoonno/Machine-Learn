from src.machine_learn.imports import pd
from src.machine_learn.constants import PROJECT_DIRECTORY

student_df = pd.read_csv(PROJECT_DIRECTORY / 'data' / 'test_data' / 'student_performance.csv')
print(student_df)