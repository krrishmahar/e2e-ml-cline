from tensorflow import keras

def build(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model
