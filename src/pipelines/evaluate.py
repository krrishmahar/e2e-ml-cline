from tensorflow import keras

def predict(data):
    model = keras.models.load_model("model.h5")
    return model.predict(data).tolist()
