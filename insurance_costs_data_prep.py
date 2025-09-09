from imports import pd

insurance_df = pd.read_csv('/workspaces/first_repo/test_data/insurance.csv')

insurance_df['sex'] = insurance_df['sex'].map({'female' : 0, 'male' : 1})
insurance_df['smoker'] = insurance_df['smoker'].map({'yes' : 1, 'no' : 0})

regions = pd.get_dummies(insurance_df['region'], dtype=int)

insurance_df = insurance_df.drop(columns='region')

insurance_df = pd.concat([insurance_df, regions], axis=1)

insurance_x = insurance_df.drop(columns='charges')
insurance_y = insurance_df['charges']