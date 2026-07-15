"""Phase 6 - Pima hyperparameter search with keras-tuner.

Protection against a misleading tuner: use enough epochs with EarlyStopping.
A very short max_epochs selects fast starters, not necessarily the best model.
Compare max_trials=1 with max_trials=15 to quantify search value.
"""

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from phase4_pima_baseline import load_pima


def build_model(hp):
    units = hp.Choice("units", [16, 32, 64, 128])
    second_units = hp.Choice("second_units", [8, 16, 32, 64])
    l2_lambda = hp.Choice("l2_lambda", [0.0, 0.001, 0.01])
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

    model = keras.Sequential(
        [
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_lambda),
                input_shape=(8,),
            ),
            layers.Dropout(dropout),
            layers.Dense(second_units, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda)),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(max_trials=15, seed=42):
    tf.random.set_seed(seed)
    x_train, x_test, y_train, y_test = load_pima(random_state=seed)
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="kt_dir",
        project_name=f"pima_random_search_seed_{seed}",
        overwrite=True,
        seed=seed,
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    tuner.search(x_train, y_train, validation_split=0.2, epochs=120, batch_size=32, callbacks=[early_stop])
    tuner.results_summary(5)
    best_model = tuner.get_best_models(1)[0]
    print("Best test metrics:", best_model.evaluate(x_test, y_test, verbose=0))


if __name__ == "__main__":
    main()
