from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

titanic_train = pd.read_csv(DATA_DIRECTORY / 'titanic_train.csv')
titanic_test = pd.read_csv(DATA_DIRECTORY / 'titanic_test.csv')

titanic_df = pd.concat([titanic_train, titanic_test])

titanic_x = titanic_df.drop(columns = ['Unnamed: 0', 'PassengerId', 'Survived'])
titanic_y = titanic_df['Survived']