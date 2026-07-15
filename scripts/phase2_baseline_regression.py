"""Phase 2 - Baseline PMC regression on California Housing.

The output layer must be Dense(1) without sigmoid. A sigmoid would constrain
predictions to [0, 1], while California Housing targets are continuous median
prices in hundreds of thousands of dollars and can exceed 1.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from phase1_pipeline_california import load_california_pipeline


def build_regression_model(input_dim):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main(epochs=100):
    tf.random.set_seed(42)
    x_train, x_val, x_test, y_train, y_val, y_test = load_california_pipeline()
    model = build_regression_model(x_train.shape[1])
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1,
    )
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final val_mae: {history.history['val_mae'][-1]:.4f}")
    print(f"Test MAE: {test_mae:.4f} ({test_mae * 100000:.0f} dollars approx.)")
    return history


if __name__ == "__main__":
    main()
