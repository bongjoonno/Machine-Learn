from imports import pd
from constants import DATA_DIRECTORY

titanic_x = pd.read_csv(DATA_DIRECTORY / 'titanic_train.csv')
titanic_y = pd.read_csv(DATA_DIRECTORY / 'titanic_test.csv')

titanic_x.head()