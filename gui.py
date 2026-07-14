import tkinter as tk
from models import random_forest, naive_bayes, svm, mlp, rbf, bagging
from utils import preprocessing, evaluation

def run_model(model_func, label):
    X_train, X_test, y_train, y_test = preprocessing.load_and_preprocess('data/nasa_dataset.csv')
    model = model_func(X_train, y_train)
    acc = evaluation.evaluate_model(model, X_test, y_test)
    print(f"{label} Accuracy:", acc)

window = tk.Tk()
window.title("Software Defect Estimation")

tk.Button(window, text="Run Random Forest Algorithm", command=lambda: run_model(random_forest.train_random_forest, "Random Forest")).pack()
tk.Button(window, text="Run Naive Bayes Algorithm", command=lambda: run_model(naive_bayes.train_naive_bayes, "Naive Bayes")).pack()
tk.Button(window, text="Run SVM Algorithm", command=lambda: run_model(svm.train_svm, "SVM")).pack()
tk.Button(window, text="Run MLP Algorithm", command=lambda: run_model(mlp.train_mlp, "MLP")).pack()
tk.Button(window, text="Run RBF Algorithm", command=lambda: run_model(rbf.train_rbf, "RBF")).pack()
tk.Button(window, text="Run Bagging Algorithm", command=lambda: run_model(bagging.train_bagging, "Bagging")).pack()

window.mainloop()
