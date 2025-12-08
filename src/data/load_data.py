from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

## Note: loading is not file agnostic
def load():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.3, random_state=42
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
