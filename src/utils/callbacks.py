from tensorflow import keras

def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(patience=5),
    ]
