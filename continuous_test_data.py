from imports import pd

df = pd.read_csv('/workspaces/first_repo/test_data/insurance.csv')

df['sex'] = df['sex'].map({'female' : 0, 'male' : 1})
df['smoker'] = df['smoker'].map({'yes' : 1, 'no' : 0})

regions = pd.get_dummies(df['region'], dtype=int)

df = pd.concat([df, regions], axis=1)

x_continuous = df.drop(columns='charges')
y_continuous = df['charges']
