from main_testing_imports import *
'''
class_labels = ['ham', 'spam']
word_counts_by_class, total_words_by_class, vocab_size = naive_bayes_prep(class_labels, [email_ham['Message'], email_spam['Message']])

  
normal_prob = len(email_ham) / len(email_df)
spam_prob = len(email_spam) / len(email_df)

prior_probs = [normal_prob, spam_prob]

result = email_df['Message'].apply(lambda sentence: naive_bayes_predict(sentence, word_counts_by_class, total_words_by_class, vocab_size, prior_probs))

categorizations = []

for row in result:
  categorizations.append(max(row, key=row.get))

email_df['Prediction'] = categorizations

accuracy = (email_df['Category'] == email_df['Prediction']).sum() / len(email_df)
'''


#LinearRegression() Object
#train and test methods

insurance_x_train, insurance_y_train, insurance_x_test, insurance_y_test = train_test_split(insurance_x, insurance_y)

model = LinearRegression(insurance_x_train, insurance_y_train, insurance_x_test, insurance_y_test)

model.train()
accuracy = model.test()
print(accuracy)