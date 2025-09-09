from imports import pd

email_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/email_spam.csv')
email_df = email_df[email_df['Category'].isin(['ham', 'spam'])]

email_ham = email_df[email_df['Category'] == 'ham']
email_spam = email_df[email_df['Category'] == 'spam']