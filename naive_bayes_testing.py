from model_imports import NaiveBayes
from data_imports import email_df, email_ham, email_spam
from metrics_imports import categorical_accuracy

naive_bayes_model = NaiveBayes()

class_labels = ['ham', 'spam']
naive_bayes_model.train(class_labels, [email_ham['Message'], email_spam['Message']])

result = email_df['Message'].apply(lambda sentence: naive_bayes_model.predict(sentence))

categorizations = []

for row in result:
  categorizations.append(max(row, key=row.get))

email_df['Prediction'] = categorizations

accuracy = categorical_accuracy(email_df['Prediction'], email_df['Category'])
print(accuracy)