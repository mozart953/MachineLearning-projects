import readFile as rF

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

log_model = LogisticRegression()

log_model = log_model.fit(rF.x_train, rF.y_train)

y_pred = log_model.predict(rF.x_test)

print(classification_report(rF.y_test, y_pred))