"""Dataset runner - Wine Quality multiclass classification.

This is the dataset-level entry point requested by the assignment. It runs the
baseline and BatchNorm comparison used in the phase scripts.
"""

import tensorflow as tf

from phase7_wine_baseline import build_wine_model, load_wine
from phase8_wine_batchnorm import main as run_batchnorm_comparison


def main(baseline_epochs=100):
    tf.random.set_seed(42)
    x_train, x_test, y_train, y_test, distribution = load_wine()
    print("Dataset: Wine Quality")
    print("Aggregated class distribution:")
    print(distribution)

    baseline = build_wine_model(x_train.shape[1])
    history = baseline.fit(
        x_train, y_train, validation_split=0.2, epochs=baseline_epochs, batch_size=32, verbose=0
    )
    baseline_metrics = baseline.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print(f"Baseline final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("Baseline test metrics:", baseline_metrics)

    print("BatchNorm comparison:")
    run_batchnorm_comparison()


if __name__ == "__main__":
    main()
