import argparse
from src.data.load_data import load
from src.data.preprocess import preprocess
from src.models.dnn import build
from src.utils.callbacks import get_callbacks


def train(dry_run=False):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load()

    X_train, X_valid, X_test = preprocess(X_train, X_valid, X_test)

    model = build(X_train.shape[1:])

    if dry_run:
        print("Dry run completed. Model built successfully.")
        return

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_valid, y_valid),
        callbacks=get_callbacks(),
    )

    model.save("model.h5")

    loss, mae = model.evaluate(X_test, y_test)
    print("Test MAE:", mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.dry_run)
