from imports import pd

df = pd.read_csv('/workspaces/first_repo/test_data/breast-cancer.csv')

df = df.drop(columns=['id'])
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

x_cat = df.drop(columns=['diagnosis'])
y_cat = df['diagnosis']