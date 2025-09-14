from imports import np, pd
from naive_bayes_testing import email_df, class_labels

def create_confusion_matrix(y_test, y_pred):
    unique_classes = pd.unique(y_test)
    
    class_mapping = {unique_classes[i]: i for i in range(len(unique_classes))}

    y_test = y_test.map(class_mapping)
    y_pred = y_pred.map(class_mapping)

    return y_test, y_pred

res = create_confusion_matrix(email_df['Prediction'], email_df['Category'])
print(res)