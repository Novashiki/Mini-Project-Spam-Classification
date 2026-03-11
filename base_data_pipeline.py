import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def load_dataset():
    df = pd.read_csv(DATA_URL, sep="\t", names=["label", "message"])
    return df

def preprocess_labels(df):
    df["label"] = df["label"].map({"ham":0, "spam":1})
    return df

def split_data(df, test_size=0.2, random_state=42):
    
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def get_processed_data():
    
    df = load_dataset()
    df = preprocess_labels(df)

    X_train, X_test, y_train, y_test = split_data(df)

    return X_train, X_test, y_train, y_test