import readFile as rF
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm_model = SVC()
svm_model = svm_model.fit(rF.x_train, rF.y_train)

y_pred = svm_model.predict(rF.x_test)

print(classification_report(rF.y_test, y_pred))
