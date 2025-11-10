from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

breast_cancer_df = pd.read_csv(DATA_DIRECTORY / 'breast_cancer.csv')

breast_cancer_df = breast_cancer_df.drop(columns=['id'])
breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].map({'M': 1, 'B': 0})

breast_cancer_x = breast_cancer_df.drop(columns=['diagnosis'])
breast_cancer_y = breast_cancer_df['diagnosis']