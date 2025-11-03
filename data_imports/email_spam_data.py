from imports import pd
from constants import CUR_DIRECTORY

email_df = pd.read_csv(CUR_DIRECTORY / 'test_data' / 'email_spam.csv')

email_df = email_df[email_df['Category'].isin(['ham', 'spam'])]

email_ham = email_df[email_df['Category'] == 'ham']
email_spam = email_df[email_df['Category'] == 'spam']