from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

train_data = pd.read_csv('poker-hand-training-true.data', header=None, dtype=int)
test_data = pd.read_csv('poker-hand-testing.data.txt', header=None, dtype=int)

def transform(data):
    original_data = data.iloc[:, 0:-1]
    label = data.iloc[:, -1]
    card_value_std = original_data.iloc[:, 1:10:2].std(axis=1)
    print '0'
    card_type_count = original_data.iloc[:, 0:10:2].apply(pd.value_counts, axis=1).fillna(0)
    print '1'
    card_type_count = card_type_count.apply(pd.value_counts, axis=1).fillna(0)
    print '2'
    card_value_count = original_data.iloc[:, 1:10:2].apply(pd.value_counts, axis=1).fillna(0)
    print '3'
    card_value_count = card_value_count.apply(pd.value_counts, axis=1).fillna(0)
    print '4'

    return pd.concat([original_data, card_type_count, card_value_count, card_value_std], axis=1), label
    # return pd.concat([card_value_count], axis=1), label



train_data, train_label = transform(train_data)
print "Train Transform Finished"

# print train_data[train_label == 6]
test_data, test_label = transform(test_data)
print "Test Transform Finished"
# Test the data set
clf = RandomForestClassifier(n_estimators=100)
X_train = train_data.values
y_train = train_label.values
X_test = test_data.values
y_test = test_label.values

clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print confusion_matrix(y_test, y_hat)
print classification_report(y_test, y_hat)
print clf.feature_importances_
