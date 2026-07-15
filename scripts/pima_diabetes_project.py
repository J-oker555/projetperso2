"""Dataset runner - Pima Diabetes binary classification.

This is the dataset-level entry point requested by the assignment. It runs the
baseline and regularized configurations used in the phase scripts.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from phase4_pima_baseline import build_pima_model, load_pima
from phase5_pima_regularisation import build_regularized_pima_model


def main(baseline_epochs=100, regularized_epochs=300):
    tf.random.set_seed(42)
    x_train, x_test, y_train, y_test = load_pima()
    counts = np.bincount(y_train)
    print("Dataset: Pima Diabetes")
    print("Train class counts:", counts, "majority baseline:", counts.max() / counts.sum())

    baseline = build_pima_model(x_train.shape[1])
    baseline_history = baseline.fit(
        x_train, y_train, validation_split=0.2, epochs=baseline_epochs, batch_size=32, verbose=0
    )
    baseline_metrics = baseline.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print(f"Baseline final val_accuracy: {baseline_history.history['val_accuracy'][-1]:.4f}")
    print("Baseline test metrics:", baseline_metrics)

    regularized = build_regularized_pima_model(x_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    regularized_history = regularized.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=regularized_epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    regularized_metrics = regularized.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print("Regularized test metrics:", regularized_metrics)
    print("Regularized epochs run:", len(regularized_history.history["loss"]))


if __name__ == "__main__":
    main()
