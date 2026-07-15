"""Dataset runner - California Housing regression.

This is the dataset-level entry point requested by the assignment. It keeps the
phase scripts reusable while exposing one runnable script for the dataset.
"""

from phase1_pipeline_california import load_california_pipeline
from phase2_baseline_regression import build_regression_model
from phase3_tensorboard_california import train_with_tensorboard


def main(epochs=100):
    x_train, x_val, x_test, y_train, y_val, y_test = load_california_pipeline()
    print("Dataset: California Housing")
    print("Shapes:", x_train.shape, x_val.shape, x_test.shape)

    model = build_regression_model(x_train.shape[1])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Baseline final val_mae: {history.history['val_mae'][-1]:.4f}")
    print(f"Baseline test_mae: {test_mae:.4f}")

    train_with_tensorboard(x_train, y_train, x_val, y_val, "california_dataset_runner", epochs=epochs)
    print("TensorBoard: tensorboard --logdir logs/fit")


if __name__ == "__main__":
    main()
