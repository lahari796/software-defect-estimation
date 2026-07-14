import matplotlib.pyplot as plt

def plot_accuracies(accuracy_dict):
    names = list(accuracy_dict.keys())
    scores = list(accuracy_dict.values())
    plt.figure(figsize=(10, 6))
    plt.bar(names, scores, color='skyblue')
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of ML Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
