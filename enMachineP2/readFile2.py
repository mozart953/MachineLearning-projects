from sklearn.neighbors import KNeighborsClassifier
import readFile as rF

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(rF.x_train, rF.y_train)

KNeighborsClassifier(n_neighbors=1)

y_pred= knn_model.predict(rF.x_test)

y_pred