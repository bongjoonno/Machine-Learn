from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

titanic_x = pd.read_csv(DATA_DIRECTORY / 'titanic_train.csv')
titanic_y = pd.read_csv(DATA_DIRECTORY / 'titanic_test.csv')

titanic_x.head()