from imports import pd

df = pd.read_csv('/workspaces/first_repo/test_data/insurance.csv')

df['sex'] = df['sex'].map({'female' : 0, 'male' : 1})
print(df)
