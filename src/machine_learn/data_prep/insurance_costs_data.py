from src.machine_learn.imports import pd
from src.machine_learn.constants import PROJECT_DIRECTORY

insurance_df = pd.read_csv(PROJECT_DIRECTORY / 'data' / 'test_data' / 'insurance.csv')

for col in ['sex', 'smoker']:
    insurance_df[col] = insurance_df[col].astype('category').cat.codes

insurance_x = insurance_df.drop(columns='charges')
insurance_y = insurance_df['charges']

insurance_x_base_line = insurance_x.copy()

insurance_x = pd.get_dummies(insurance_x, columns = ['region'], dtype = int)

insurance_cols_to_scale = ['age', 'bmi', 'children']