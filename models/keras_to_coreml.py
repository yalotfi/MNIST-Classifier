import coremltools
from keras.models import load_model


def convert_model(filepath):
    # Convert Keras model to Core ML
    keras_model = load_model(filepath)
    coreml_model = coremltools.converters.keras.convert(keras_model)
    coreml_model.save('model.mlmodel')


if __name__ == "__main__":
    convert_model("model.h5")
