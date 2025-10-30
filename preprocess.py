# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path=r"C:\Users\Computer Corner\Desktop\Breast Cancer DL\data.csv"):
    # Load dataset
    data = pd.read_csv(path)
    print("Before cleaning:", data.shape)

    # Drop 'id' if exists
    if 'id' in data.columns:
        data = data.drop(['id'], axis=1)

    # Drop completely empty columns
    data = data.dropna(axis=1, how='all')

    # Map target column
    if 'diagnosis' not in data.columns:
        raise ValueError("No 'diagnosis' column found. Please check dataset columns.")
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print("After cleaning & imputation:", X.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Quick test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)






