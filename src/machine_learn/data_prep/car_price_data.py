from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

car_price_df = pd.read_csv(DATA_DIRECTORY / 'car_price.csv')
print(car_price_df)