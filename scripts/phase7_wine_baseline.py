"""Phase 7 - Wine Quality multiclass baseline.

Choice: aggregate quality into 3 nominal classes and use softmax +
sparse_categorical_crossentropy. We lose ordinal distance information: low vs
medium and low vs high are treated as independent class errors. Regression or
ordinal regression would preserve more order information but are separate tasks.
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def map_quality(value):
    if value <= 4:
        return 0
    if value <= 6:
        return 1
    return 2


def load_wine(random_state=42):
    df = pd.read_csv(WINE_URL, sep=";")
    df["quality_3class"] = df["quality"].apply(map_quality)
    x = df.drop(columns=["quality", "quality_3class"]).values
    y = df["quality_3class"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, df["quality_3class"].value_counts().sort_index()


def build_wine_model(input_dim=11):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main(epochs=100):
    tf.random.set_seed(42)
    x_train, x_test, y_train, y_test, distribution = load_wine()
    print("Aggregated class distribution:")
    print(distribution)
    model = build_wine_model(x_train.shape[1])
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    return history


if __name__ == "__main__":
    main()
