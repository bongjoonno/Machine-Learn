from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

car_price_df = pd.read_csv(DATA_DIRECTORY / 'car_price.csv')
car_price_df = car_price_df.drop(columns = ['car_ID', 'CarName'])

binary_columns = ['fueltype', 'aspiration', 'doornumber', 'enginelocation']
multi_columns = ['carbody', 'drivewheel', 'enginetype', 'cylindernumber', 'fuelsystem']

for col in binary_columns:
    car_price_df[col] = car_price_df[col].astype('category').cat.codes

car_price_df = pd.get_dummies(car_price_df, columns = multi_columns, dtype = int)

car_price_x = car_price_df.drop(columns = ['price'])
car_price_y = car_price_df['price']