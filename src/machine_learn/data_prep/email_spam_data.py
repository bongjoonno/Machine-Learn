from src.machine_learn.imports import pd
from src.machine_learn.constants import DATA_DIRECTORY

email_df = pd.read_csv(DATA_DIRECTORY / 'email_spam.csv')

email_df = email_df[email_df['Category'].isin(['ham', 'spam'])]

email_ham = email_df[email_df['Category'] == 'ham']
email_spam = email_df[email_df['Category'] == 'spam']