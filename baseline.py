from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

train_data = pd.read_csv('poker-hand-training-true.data', header=None, dtype=int)
test_data = pd.read_csv('poker-hand-testing.data.txt', header=None, dtype=int)

clf = RandomForestClassifier(n_estimators=100)
X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values
clf.fit(X_train, y_train)

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values
y_hat = clf.predict(X_test)
print classification_report(y_test, y_hat)
print confusion_matrix(y_test, y_hat)
