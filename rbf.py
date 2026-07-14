from sklearn.svm import SVC

def train_rbf(X_train, y_train):
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    return model
