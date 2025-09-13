from model_imports import NaiveBayes
from data_imports import email_df, email_ham, email_spam
from metrics_imports import categorical_acc

class_labels = ['ham', 'spam']
word_counts_by_class, total_words_by_class, vocab_size = naive_bayes_prep(class_labels, [email_ham['Message'], email_spam['Message']])

  
normal_prob = len(email_ham) / len(email_df)
spam_prob = len(email_spam) / len(email_df)

prior_probs = [normal_prob, spam_prob]

naive_bayes_model = NaiveBayes()

result = email_df['Message'].apply(lambda sentence: naive_bayes_predict(sentence, word_counts_by_class, total_words_by_class, vocab_size, prior_probs))

categorizations = []

for row in result:
  categorizations.append(max(row, key=row.get))

email_df['Prediction'] = categorizations

accuracy = categorical_acc(email_df['Prediction'], email_df['Category'])
print(accuracy)