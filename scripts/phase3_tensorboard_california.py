"""Phase 3 - TensorBoard diagnostic on California Housing.

Expected diagnosis: with the correct split-then-scale pipeline, train_loss and
val_loss should generally descend together. If val_loss is suspiciously below
train_loss for long periods, re-check preprocessing and leakage.
"""

import datetime
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from phase2_baseline_regression import build_regression_model


def train_with_tensorboard(x_train, y_train, x_val, y_val, run_name, epochs=100):
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_{timestamp}"
    callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = build_regression_model(x_train.shape[1])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[callback],
        verbose=0,
    )
    print(f"{run_name}: log_dir={log_dir}, final val_mae={history.history['val_mae'][-1]:.4f}")
    return history


def main():
    data_home = Path(__file__).resolve().parents[1] / "data" / "sklearn"
    data = fetch_california_housing(data_home=data_home)
    x_train, x_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    train_with_tensorboard(x_train_scaled, y_train, x_val_scaled, y_val, "california_scaled")
    train_with_tensorboard(x_train, y_train, x_val, y_val, "california_unscaled")
    print("Open TensorBoard with: tensorboard --logdir logs/fit")


if __name__ == "__main__":
    main()
