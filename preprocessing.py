import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('defect', axis=1)
    y = df['defect']
    return train_test_split(X, y, test_size=0.2, random_state=42)
