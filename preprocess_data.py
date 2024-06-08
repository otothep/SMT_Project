import pandas as pd

def preprocess_data():
    df = pd.read_csv('pose_data.csv', header=None)
    df.iloc[:, -1] = df.iloc[:, -1].map({'linken': 0, 'rechten': 1})
    df.to_csv('pose_data_preprocessed.csv', index=False, header=False)

if __name__ == "__main__":
    preprocess_data()
