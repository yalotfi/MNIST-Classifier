import coremltools
from keras.models import model_from_json


def convert_model(json_path, weight_path):
    # Read model architecture configuration
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()

    # Read trained weights and compile keras model
    keras_model = model_from_json(model_json)
    keras_model.load_weights(weight_path)
    print("Loaded model architecture and weights from disk")

    # Convert Keras model to Core ML
    coreml_model = coremltools.converters.keras.convert(keras_model)
    print("Converted Keras model to Core ML model")
    coreml_model.save('DigitClassifier.mlmodel')
    print("Core ML model written to disk")


if __name__ == "__main__":
    convert_model("architecture.json", "weights.h5")
