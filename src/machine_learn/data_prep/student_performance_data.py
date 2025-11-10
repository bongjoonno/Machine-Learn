from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

student_df = pd.read_csv(DATA_DIRECTORY / 'student_performance.csv')
student_df = student_df.dropna()
student_x = student_df.drop(columns=['Performance Index', 'Extracurricular Activities'])
student_y = student_df['Performance Index']