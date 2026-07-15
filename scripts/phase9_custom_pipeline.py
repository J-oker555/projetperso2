"""Phase 9 - Personal dataset pipeline example.

Default personal dataset: sklearn breast_cancer, binary classification with
569 examples and 30 features. It reuses the full structured-data pipeline:
split, train-only scaling, Keras PMC, TensorBoard, EarlyStopping, final test.
"""

import datetime

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def load_personal_dataset(random_state=42):
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=random_state, stratify=data.target
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, data.target_names


def build_model(input_dim, optimizer="adam"):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001), input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run(optimizer="adam"):
    tf.random.set_seed(42)
    x_train, x_test, y_train, y_test, names = load_personal_dataset()
    print("Target names:", names)
    print("Train class distribution:", np.bincount(y_train))
    optimizer_name = optimizer if isinstance(optimizer, str) else optimizer.name
    log_dir = f"logs/custom/breast_cancer_{optimizer_name}_{datetime.datetime.now().strftime('%H%M%S')}"
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ]
    model = build_model(x_train.shape[1], optimizer=optimizer)
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    extreme = np.full((1, x_train.shape[1]), 99999.0)
    print(f"{optimizer_name}: test_accuracy={accuracy:.4f}, epochs={len(history.history['loss'])}, log_dir={log_dir}")
    print("Extreme out-of-distribution prediction:", model.predict(extreme, verbose=0).ravel())


def main():
    run("adam")
    run(keras.optimizers.SGD(learning_rate=0.01))


if __name__ == "__main__":
    main()
