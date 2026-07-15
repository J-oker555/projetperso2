"""Phase 1 - California Housing data pipeline.

Correct ML order: split first, then fit StandardScaler on X_train only.
Fitting the scaler before the split leaks information from validation/test
distributions into training preprocessing and makes evaluation optimistic.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_california_pipeline(random_state=42):
    data = fetch_california_housing()
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def main():
    x_train, x_val, x_test, y_train, y_val, y_test = load_california_pipeline()
    print("X_train:", x_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", x_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", x_test.shape, "y_test: ", y_test.shape)
    print("Train mean after scaling:", x_train.mean(axis=0).round(4))
    print("Test mean after train-only scaling:", x_test.mean(axis=0).round(4))


if __name__ == "__main__":
    main()
