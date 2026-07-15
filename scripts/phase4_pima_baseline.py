"""Phase 4 - Pima Diabetes binary classification baseline.

Class distribution is roughly 65% non-diabetic and 35% diabetic, so a model
that always predicts the majority class reaches about 65% accuracy without
learning. The neural baseline must be interpreted against that naive score.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome",
]


def load_pima(random_state=42):
    df = pd.read_csv(PIMA_URL, names=COLUMNS)
    x = df.drop(columns=["outcome"]).values
    y = df["outcome"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def build_pima_model(input_dim):
    model = keras.Sequential(
        [
            layers.Dense(32, activation="relu", input_shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main(epochs=100):
    tf.random.set_seed(42)
    x_train, x_test, y_train, y_test = load_pima()
    counts = np.bincount(y_train)
    print("Train class counts:", counts, "majority baseline:", counts.max() / counts.sum())
    model = build_pima_model(x_train.shape[1])
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    return history


if __name__ == "__main__":
    main()
