from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

car_price_df = pd.read_csv(DATA_DIRECTORY / 'car_price.csv')
car_price_df = car_price_df.drop(columns = ['car_ID'])

car_price_cols_to_scale = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

car_price_df['CarName'] = car_price_df['CarName'].str.lower().str.split().str[0]
car_price_df['CarName'] = car_price_df['CarName'].str.replace('maxda', 'mazda')
car_price_df['CarName'] = car_price_df['CarName'].str.replace('porcshce', 'porsche')
car_price_df['CarName'] = car_price_df['CarName'].str.replace('toyouta', 'toyota')
car_price_df['CarName'] = car_price_df['CarName'].str.replace('vokswagen', 'volkswagen').str.replace('vw', 'volkswagen')

binary_columns = ['fueltype', 'aspiration', 'doornumber', 'enginelocation']
multi_columns = ['CarName', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber', 'fuelsystem']

for col in binary_columns:
    car_price_df[col] = car_price_df[col].astype('category').cat.codes

car_price_df = pd.get_dummies(car_price_df, columns = multi_columns, dtype = int)

car_price_x = car_price_df.drop(columns = ['price'])
car_price_y = car_price_df['price']
