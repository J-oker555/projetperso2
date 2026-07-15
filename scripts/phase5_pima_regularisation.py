"""Phase 5 - Pima regularisation.

Goal: improve over the phase 4 baseline while watching minority-class recall.
Order of levers: L2 first to reduce weight magnitude, EarlyStopping to keep the
best validation point, then Dropout if the train/val gap remains large.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from phase4_pima_baseline import load_pima


def build_regularized_pima_model(input_dim, l2_lambda=0.01, dropout_rate=0.2):
    model = keras.Sequential(
        [
            layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_lambda),
                input_shape=(input_dim,),
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Recall(name="recall"), keras.metrics.Precision(name="precision")],
    )
    return model


def main(epochs=300):
    tf.random.set_seed(42)
    x_train, x_test, y_train, y_test = load_pima()
    model = build_regularized_pima_model(x_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print(results)
    print("Epochs run:", len(history.history["loss"]))
    return history


if __name__ == "__main__":
    main()
