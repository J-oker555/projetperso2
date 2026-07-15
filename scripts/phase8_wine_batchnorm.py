"""Phase 8 - Wine Quality regularisation with BatchNorm.

Order used for the main BN variant: Dense(linear) -> BatchNormalization -> ReLU.
This follows the original BatchNorm recommendation: normalize pre-activation
values, then apply the non-linearity. The script also compares BN after ReLU.
"""

import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from phase7_wine_baseline import load_wine


def build_wine_model(use_batchnorm=False, bn_before_activation=True, extra_layer=False):
    model = keras.Sequential()
    if use_batchnorm and bn_before_activation:
        model.add(layers.Dense(64, input_shape=(11,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
    else:
        model.add(layers.Dense(64, activation="relu", input_shape=(11,)))
        if use_batchnorm:
            model.add(layers.BatchNormalization())

    model.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    if extra_layer:
        model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    tf.random.set_seed(42)
    x_train, _, y_train, _, _ = load_wine()
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    configs = {
        "sans_bn": lambda: build_wine_model(use_batchnorm=False),
        "bn_avant_activation": lambda: build_wine_model(use_batchnorm=True, bn_before_activation=True),
        "bn_apres_activation": lambda: build_wine_model(use_batchnorm=True, bn_before_activation=False),
        "bn_extra_couche": lambda: build_wine_model(
            use_batchnorm=True, bn_before_activation=True, extra_layer=True
        ),
    }
    results = {}
    for name, build_fn in configs.items():
        log_dir = f"logs/wine/{name}_{datetime.datetime.now().strftime('%H%M%S')}"
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
        model = build_fn()
        history = model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, tensorboard],
            verbose=0,
        )
        results[name] = {
            "val_accuracy": max(history.history["val_accuracy"]),
            "val_loss_final": history.history["val_loss"][-1],
            "epochs": len(history.history["val_loss"]),
            "log_dir": log_dir,
        }
        print(f"{name}: {results[name]}")
    print("Open TensorBoard with: tensorboard --logdir logs/wine")
    return results


if __name__ == "__main__":
    main()
