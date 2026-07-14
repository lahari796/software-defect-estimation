from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def train_bagging(X_train, y_train):
    model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
    model.fit(X_train, y_train)
    return model
