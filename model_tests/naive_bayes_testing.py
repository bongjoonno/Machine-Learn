from models.model_imports import NaiveBayes
from data_imports.data_imports import email_df, email_ham, email_spam
from metrics.metrics_imports import categorical_accuracy

naive_bayes_model = NaiveBayes()

class_labels = ['ham', 'spam']

train_split = 0.8
training_border = int(0.8 * len(email_ham))

email_ham_train = email_ham['Message'].iloc[:training_border]
email_spam_train = email_spam['Message'].iloc[:training_border]

email_df_test = email_df.iloc[training_border:].copy()

naive_bayes_model.train(class_labels, [email_ham_train, email_spam_train])

result = email_df_test['Message'].apply(lambda sentence: naive_bayes_model.predict(sentence))

categorizations = []

for row in result:
  categorizations.append(max(row, key=row.get))

email_df_test['Prediction'] = categorizations


accuracy = categorical_accuracy(email_df_test['Prediction'], email_df_test['Category'])
print(accuracy)