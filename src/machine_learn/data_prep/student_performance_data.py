from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

student_df = pd.read_csv(DATA_DIRECTORY / 'student_performance.csv')

student_df = student_df.dropna()
student_x = student_df.drop(columns=['Performance Index'])
student_y = student_df['Performance Index']

student_x_base_line = student_x.copy()
student_y_base_line = student_y.copy()

student_x['Extracurricular Activities'] = student_x['Extracurricular Activities'].astype('category').cat.codes
student_cols_to_scale = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']