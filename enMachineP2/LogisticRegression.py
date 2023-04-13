import readFile as rF

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

log_model = LogisticRegression()

log_model = log_model.fit(rF.x_train, rF.y_train)