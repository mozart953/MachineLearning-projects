import readFile as rF

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb_model = GaussianNB()

nb_model = nb_model.fit(rF.x_train, rF.y_train)

y_pred = nb_model.predict(rF.x_test)

print(classification_report(rF.y_test, y_pred))