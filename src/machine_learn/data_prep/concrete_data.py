from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

concrete_df = pd.read_csv(DATA_DIRECTORY / 'concrete.csv')

concrete_columns_to_scale = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']

concrete_x = concrete_df.drop(columns=['concrete_compressive_strength'])
concrete_y = concrete_df['concrete_compressive_strength']